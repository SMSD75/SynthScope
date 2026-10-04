"""Standalone, faithful eval of the OFFICIAL SPAI detector (Karageorgiou et al., CVPR'25).

This vendors the official ``PatchBasedMFViT`` (github mever-team/spai, built via the official
``build_mf_vit`` from the checkpoint's own stored config), so its official-weights numbers sit
side by side with the SynthScope re-implementation in the same ``repro/bench_results`` /
``repro/MASTER_RESULTS.jsonl`` schema as ``benchmark/bench_eval.py``.

Model build + preprocessing + P(fake) are COPIED from the verified vendored eval
``/hddstore/ssalehi/SynthScope/repro/spai_eval.py`` (kept faithful):
  - ``build_mf_vit(cfg)`` from the ckpt's stored config; the released ``spai_official.pth``
    loads with 0 missing / 0 unexpected keys (verified).
  - Fixed-resolution variant: resize shorter side -> 448, center-crop 448x448, raw [0,1];
    ``forward_batch`` patchifies into a fixed 2x2 grid of four 224x224 patches, aggregated by
    the Spectral Context Attention (MINIMUM_PATCHES=4). The model applies the circular
    frequency mask + ImageNet normalization internally.
  - P(fake) = sigmoid(logit); label 1 = synthetic (fake).

discover()/_images()/_collect()/metrics() are COPIED (not imported) from benchmark/bench_eval.py
so discovery is byte-identical (incl. the Ojha UFD real pairing: guided->imagenet, else->laion).

CLI::
    --benchmark {forensynths,ufd,pergen,pool} --data <root> [--reals] [--select] [--limit]
    --bs N --device cuda|cpu --ckpt <spai_official.pth> --tag TAG

Default ckpt: /hddstore/ssalehi/SynthScope/repro/ckpts/spai_official.pth (any-res, LDM-trained).
For GenImage_full: --benchmark forensynths --data /hddstore/ssalehi/SynthScope/GenImage_full/test.
SPAI on CPU is slow; use --limit 6 for the validation probe.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Vendored official SPAI package + caches.
SPAI_VENDOR = "/hddstore/ssalehi/SynthScope/repro/spai_vendor"
if SPAI_VENDOR not in sys.path:
    sys.path.insert(0, SPAI_VENDOR)
os.environ.setdefault("TORCH_HOME", "/hddstore/ssalehi/.cache/torch")
os.environ.setdefault("HF_HOME", "/hddstore/ssalehi/.cache/huggingface")

from spai.models.sid import build_mf_vit  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants (mirror benchmark/bench_eval.py)
# ---------------------------------------------------------------------------
REPRO_DIR = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro")
RESULTS_LOG = REPRO_DIR / "MASTER_RESULTS.jsonl"
BENCH_RESULTS_DIR = REPRO_DIR / "bench_results"
DEFAULT_CKPT = Path("/hddstore/ssalehi/SynthScope/repro/ckpts/spai_official.pth")
# resize shorter side -> 448, center-crop 448 => fixed 2x2 grid of 224 patches.
PATCH_GRID_SIZE = 448

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
REAL_DIRNAMES = ("0_real", "real", "reals")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "ai")
SHARED_REAL_DIRS = {"imagenet", "laion", "_real_pool", "real", "0_real", "reals"}
# Ojha UniversalFakeDetect real pairing (verified against the official dataset_paths.py).
UFD_REAL = {"guided": "imagenet"}  # everything else -> laion
UFD_DEFAULT_REAL = "laion"


# ===========================================================================
# COPIED VERBATIM from benchmark/bench_eval.py (identical discovery + metrics).
# ===========================================================================
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


# ===========================================================================
# Official SPAI model build (copied from repro/spai_eval.py).
# ===========================================================================
def build_model(ckpt: Path, device: torch.device):
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cfg.defrost()
    try:
        cfg.set_new_allowed(True)
    except Exception:
        pass
    if "MODEL_WEIGHTS" not in cfg:
        cfg.MODEL_WEIGHTS = "mfm"
    if "DISABLE_RECONSTRUCTION_SIMILARITY" not in cfg.MODEL.FRE:
        cfg.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY = False
    # Fixed-resolution variant: require >= 4 patches (the 2x2 grid yields exactly 4).
    cfg.MODEL.PATCH_VIT.MINIMUM_PATCHES = 4
    cfg.freeze()
    print(f"config: MODEL_WEIGHTS={cfg.MODEL_WEIGHTS} RESOLUTION_MODE={cfg.MODEL.RESOLUTION_MODE} "
          f"ATTN_EMBED_DIM={cfg.MODEL.PATCH_VIT.ATTN_EMBED_DIM} "
          f"MASKING_RADIUS={cfg.MODEL.FRE.MASKING_RADIUS} IMG_SIZE={cfg.DATA.IMG_SIZE} "
          f"CLS_HEAD.MLP_RATIO={cfg.MODEL.CLS_HEAD.MLP_RATIO}", flush=True)
    model = build_mf_vit(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing, f"MISSING keys: {missing}"
    assert not unexpected, f"UNEXPECTED keys: {unexpected}"
    print(f"loaded {ckpt.name} into {type(model).__name__}: {len(ck['model'])} tensors, "
          f"0 missing / 0 unexpected (ckpt epoch={ck.get('epoch')} "
          f"max_acc={ck.get('max_accuracy')})", flush=True)
    return model.to(device).eval()


def build_transform() -> T.Compose:
    # Fixed-resolution preprocessing: shorter side -> 448, center-crop 448 (uniform for both
    # classes). Raw [0,1] (model normalizes internally). forward_batch then tiles into 2x2 = 4
    # patches of 224, aggregated by the Spectral Context Attention.
    return T.Compose([
        T.Resize(PATCH_GRID_SIZE),
        T.CenterCrop(PATCH_GRID_SIZE),
        T.ToTensor(),
    ])


@torch.no_grad()
def score(model, transform, device, paths: list[Path], bs: int) -> torch.Tensor:
    probs = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                batch.append(transform(Image.open(p).convert("RGB")))
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        logits = model.forward_batch(x).squeeze(-1)  # fixed-resolution grid + SCA
        probs.append(torch.sigmoid(logits).detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="official SPAI checkpoint .pth")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--tag", required=True, help="run tag (output filename + log line)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # This node's old kernel can transiently make CUDA unavailable at startup; fail fast so
        # the launcher can relaunch rather than silently fall back to a useless CPU run.
        if not torch.cuda.is_available():
            print("[FATAL] CUDA requested but unavailable at startup (transient init failure) "
                  "— relaunch", flush=True)
            return 3
        torch.zeros(1, device="cuda")  # force-init CUDA; crash loudly if context can't be created

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        print(f"ckpt not found: {ckpt}")
        return 1

    model = build_model(ckpt, device)
    transform = build_transform()

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== spai_official_eval {args.tag} | model=spai (official-vendored) ckpt={ckpt} "
          f"| {args.benchmark} | {len(gens)} generators | device={device} ===", flush=True)

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs = score(model, transform, device, reals + fakes, args.bs)
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
        "model": "spai",
        "ckpt": str(ckpt),
        "benchmark": args.benchmark,
        "data_root": str(root),
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "synthscope_version": "official-vendored",
    }
    out = Path(args.out).expanduser() if args.out else (BENCH_RESULTS_DIR / f"{args.tag}.json")
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
