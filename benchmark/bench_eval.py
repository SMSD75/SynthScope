"""Unified benchmark eval for AGID detectors — reproduce a paper's per-generator table.

Scores a detector over a benchmark's per-generator real/fake folders through the SAME verified
agid inference path the web app uses (engine + registry + per-model preprocessing + fake-prob),
computes per-generator + mean acc@0.5 / AP / AUROC, writes a JSON, and APPENDS a verified line to
``repro/MASTER_RESULTS.jsonl`` so every reported number traces to this run + its job log.

    python benchmark/bench_eval.py --model clip --benchmark forensynths \
        --data /hddstore/ssalehi/AGID/CNNDetection/test --tag clip_agid_forensynths

Benchmark profiles encode each dataset's correct generator grouping + real-pairing:
  forensynths  CNNDetection/test    per top-level generator; reals+fakes collected recursively
                                    (handles progan/<class>/{0_real,1_fake} nesting)
  ufd          UniversalFakeDetect  fakes per generator; reals paired Ojha-style
                                    (guided->imagenet, ldm/glide/dalle->laion)
  pergen       freqnet/aigibench/   each <gen>/{0_real,1_fake}; optional shared --reals
               cospybench/generic
  pool         GenImage/single-pool one group from <root>/{real,fake} or {0_real,1_fake}

Use --ckpt to override the checkpoint (official vs AGID-trained, for side-by-side runs).
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch
from PIL import Image, ImageFile

import agid
from models.image.cure.preprocess import SRC_FORMAT_KEY

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
REAL_DIRNAMES = ("0_real", "real", "reals")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "ai")
SHARED_REAL_DIRS = {"imagenet", "laion", "_real_pool", "real", "0_real", "reals"}
RESULTS_LOG = Path(__file__).resolve().parents[1] / "repro" / "MASTER_RESULTS.jsonl"
# Ojha UniversalFakeDetect real pairing (verified against the official dataset_paths.py).
UFD_REAL = {"guided": "imagenet"}  # everything else -> laion
UFD_DEFAULT_REAL = "laion"


def _images(d: Path) -> list[Path]:
    if not d or not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in {e.lower() for e in IMAGE_EXTS})


def _collect(gen_dir: Path, dirnames: tuple[str, ...]) -> list[Path]:
    """All images under any subdir of gen_dir whose name matches one of dirnames (any depth)."""
    out: list[Path] = []
    for name in dirnames:
        for sub in gen_dir.rglob(name):
            if sub.is_dir():
                out += [p for p in sub.rglob("*") if p.suffix.lower() in {e.lower() for e in IMAGE_EXTS}]
    # de-dup while keeping order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return sorted(uniq)


def discover(benchmark: str, root: Path, shared_reals: Path | None, select: set[str] | None):
    """Return [(generator, real_paths, fake_paths)] for the benchmark."""
    shared = _images(shared_reals) if shared_reals else []

    if benchmark == "pool":
        reals = _collect(root, REAL_DIRNAMES) or shared
        fakes = _collect(root, FAKE_DIRNAMES)
        return [("all", reals, fakes)] if reals and fakes else []

    if benchmark == "ufd":
        out = []
        gens = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in SHARED_REAL_DIRS)
        for gen in gens:
            if select and gen not in select:
                continue
            real_src = root / UFD_REAL.get(gen, UFD_DEFAULT_REAL)
            reals = _collect(real_src, REAL_DIRNAMES)
            fakes = _collect(root / gen, FAKE_DIRNAMES)
            if reals and fakes:
                out.append((gen, reals, fakes))
            else:
                print(f"  skip {gen}: reals={len(reals)} fakes={len(fakes)}")
        return out

    if benchmark == "forensynths":
        out = []
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if select and gen_dir.name not in select:
                continue
            reals = _collect(gen_dir, REAL_DIRNAMES) or shared
            fakes = _collect(gen_dir, FAKE_DIRNAMES)
            if reals and fakes:
                out.append((gen_dir.name, reals, fakes))
            else:
                print(f"  skip {gen_dir.name}: reals={len(reals)} fakes={len(fakes)}")
        return out

    # pergen / generic: each top-level dir (excluding shared-real dirs) is a generator
    out = []
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if gen_dir.name in SHARED_REAL_DIRS and gen_dir.name != root.name:
            continue
        if select and gen_dir.name not in select:
            continue
        reals = _collect(gen_dir, REAL_DIRNAMES) or shared
        fakes = _collect(gen_dir, FAKE_DIRNAMES)
        if reals and fakes:
            out.append((gen_dir.name, reals, fakes))
        else:
            print(f"  skip {gen_dir.name}: reals={len(reals)} fakes={len(fakes)}")
    return out


def native_res_transform(tf):
    """Drop the up-scaling Resize/CenterCrop from a model's eval transform, keeping
    ToTensor + Normalize, so images are scored at NATIVE resolution.

    Some methods (e.g. NPR on GANGen-Detection/freqnet) detect up-sampling artifacts and
    MUST NOT be fed a bilinearly up-scaled image — resizing a 128x128 GAN crop to 256x256
    injects the exact artifact they key on, collapsing the real class. The official NPR
    GANGen recipe runs with ``no_resize=True``; this reproduces it. Safe only when every
    image in a per-generator batch shares one resolution (true for freqnet's per-gen sets),
    since bench_eval stacks a batch into a single tensor."""
    import torchvision.transforms as T
    if not isinstance(tf, T.Compose):
        return tf
    kept = [t for t in tf.transforms if not isinstance(t, (T.Resize, T.CenterCrop))]
    return T.Compose(kept)


@torch.no_grad()
def _open_rgb(path: Path) -> Image.Image:
    """Open ``path`` as RGB, recording the source file format in ``img.info`` so format-aware eval
    transforms (CuRe's PNG->JPEG96 step) see what the official loader sees."""
    source = Image.open(path)
    source_format = source.format
    image = source.convert("RGB")
    image.info[SRC_FORMAT_KEY] = source_format
    return image


def score(det, model, spec, paths: list[Path], bs: int, transform=None) -> torch.Tensor:
    device = det._engine.device
    tf = transform if transform is not None else spec.transform
    # Native-resolution specs (SPAI any-res) produce variable-size tensors that cannot be stacked,
    # so they MUST be scored one image at a time.
    if getattr(spec, "native_res", False):
        bs = 1
    probs = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                batch.append(tf(_open_rgb(p)))
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        probs.append(det._engine._fake_prob(model, spec, x).detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def metrics(fake_probs: torch.Tensor, labels: torch.Tensor) -> dict | None:
    from torchmetrics.functional import auroc, average_precision
    if labels.unique().numel() < 2:
        return None
    ap = float(average_precision(fake_probs, labels, task="binary")) * 100
    au = float(auroc(fake_probs, labels, task="binary")) * 100
    preds = (fake_probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "auroc": round(au, 2),
            "real_acc": round(real_acc, 2), "fake_acc": round(fake_acc, 2),
            "acc": round((real_acc + fake_acc) / 2, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="AGID detector key (see `agid list`)")
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--ckpt", default=None, help="checkpoint override (official vs AGID weights)")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--native-res", action="store_true",
                    help="score at native resolution (strip up-scaling Resize/CenterCrop); "
                         "for artifact-based methods like NPR on low-res GAN sets (freqnet)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", required=True, help="run tag (used for the output filename + log)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    load_kw = {}
    if args.ckpt:
        load_kw["checkpoint"] = args.ckpt
    det = agid.load(args.model, device=args.device, **load_kw)
    det._ensure_weights()
    spec = det.spec
    model = det._engine._get_model(spec)
    eval_transform = native_res_transform(spec.transform) if args.native_res else spec.transform
    if args.native_res:
        print(f"[native-res] scoring at native resolution; transform={eval_transform}", flush=True)

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== bench_eval {args.tag} | model={args.model} ckpt={args.ckpt or 'registry-default'} "
          f"| {args.benchmark} | {len(gens)} generators | device={det._engine.device} ===", flush=True)

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs = score(det, model, spec, reals + fakes, args.bs, transform=eval_transform)
        if probs.numel() != labels.numel():
            labels = labels[:probs.numel()]
        m = metrics(probs, labels)
        if m is None:
            print(f"    skip {name}: one class only")
            continue
        per_gen[name] = m
        for k in agg:
            agg[k].append(m[k])
        print(f"    {name}: acc={m['acc']} AP={m['ap']} AUROC={m['auroc']} "
              f"(real_acc={m['real_acc']} fake_acc={m['fake_acc']})", flush=True)

    if not per_gen:
        print("No generator produced valid metrics.")
        return 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2)

    means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "mauroc": mean(agg["auroc"])}
    result = {
        "tag": args.tag,
        "model": args.model,
        "ckpt": args.ckpt or "registry-default",
        "benchmark": args.benchmark,
        "data_root": str(root),
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "agid_version": getattr(agid, "__version__", "?"),
    }
    out = Path(args.out).expanduser() if args.out else (
        RESULTS_LOG.parent / "bench_results" / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOG.open("a") as f:
        f.write(json.dumps({k: result[k] for k in
                            ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                             "n_generators", "means")}) + "\n")

    print(f"\n=== RESULT {args.tag}: mAP={means['map']} mAUROC={means['mauroc']} acc={means['acc']} "
          f"over {len(per_gen)} generators -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
