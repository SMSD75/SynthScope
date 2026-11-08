"""Unified benchmark eval for SynthScope detectors — reproduce a paper's per-generator table.

Scores a detector over a benchmark's per-generator real/fake folders through the SAME verified
synthscope inference path the web app uses (engine + registry + per-model preprocessing + fake-prob),
computes per-generator + mean acc@0.5 / AP / AUROC, writes a JSON, and APPENDS a verified line to
``repro/MASTER_RESULTS.jsonl`` so every reported number traces to this run + its job log.

    python benchmark/bench_eval.py --model clip --benchmark forensynths \
        --data /hddstore/ssalehi/SynthScope/CNNDetection/test --tag clip_synthscope_forensynths

    # one step: download the dataset if needed (its built-in loader arranges it), pick the profile, evaluate
    python benchmark/bench_eval.py --model npr --dataset commgen15 --root /data/CommGen15

Benchmark profiles encode each dataset's correct generator grouping + real-pairing:
  forensynths  CNNDetection/test    per top-level generator; reals+fakes collected recursively
                                    (handles progan/<class>/{0_real,1_fake} nesting)
  ufd          UniversalFakeDetect  fakes per generator; reals paired Ojha-style
                                    (guided->imagenet, ldm/glide/dalle->laion)
  pergen       freqnet/aigibench/   each <gen>/{0_real,1_fake}; optional shared --reals
               cospybench/generic
  pool         GenImage/single-pool one group from <root>/{real,fake} or {0_real,1_fake}

Use --ckpt to override the checkpoint (official vs SynthScope-trained, for side-by-side runs).
Use --model-config NAME with --ckpt for a checkpoint trained by a recipe whose model config (conf/model/NAME.yaml) has its own
test preprocessing, e.g. Effort's UFD/ProGAN recipe (image/effort_ufd: center crop 224, not the served model's squash).
Use --jpeg-fakes Q for protocols that JPEG-compress only the synthetic images (DDA: Q=96 on GenImage,
ForenSynths and AIGCDetect, to remove the PNG-fake/JPEG-real format bias).
Use --degrade SPEC for a robustness slice: one deterministic degradation applied to EVERY image (reals and fakes) after it is
loaded and before --jpeg-fakes (fakes only) and the detector's own transform, in that order. SPEC is one token:
jpeg<Q> | down<F> | blur<S> | social | social2 (see ``parse_degrade``). Use --results-log PATH to append the run's summary line
to PATH instead of repro/MASTER_RESULTS.jsonl.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import dataclasses
import datetime
import io
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageFilter

import synthscope
from interface.backend.registry import get_spec, model_transform
from synthscope.api import Detector
from models.image.cure.preprocess import SRC_FORMAT_KEY
from datasets_util import dataset_config, ensure_dataset, find_eval_root

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None   # benchmark data is trusted; some real photos (e.g. RealChain) exceed PIL's 179 MP bomb guard


def _limit_heap_growth() -> None:
    """Serve large allocations from mmap and give freed heap back to the OS (glibc ``mallopt``; the same as
    ``MALLOC_MMAP_THRESHOLD_`` / ``MALLOC_TRIM_THRESHOLD_`` = 128 KiB). Scoring one native-resolution image after another
    (B-Free, SPAI: a differently sized decode and tensor each time) otherwise fragments the heap and the process grows by
    ~1 GB a minute until it is killed (B-Free on EvalGEN); with it the run stays flat and the scores are identical."""
    libc = ctypes.util.find_library("c")
    if libc:
        mallopt = ctypes.CDLL(libc).mallopt
        mallopt(-3, 128 * 1024)    # M_MMAP_THRESHOLD
        mallopt(-1, 128 * 1024)    # M_TRIM_THRESHOLD


_limit_heap_growth()

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
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
    """Drop the up-scaling resize/center-crop steps from a model's eval transform, keeping
    to_dtype + normalize, so images are scored at NATIVE resolution (ValueError for eval lists
    whose own steps resize, see ``AIGITransformations.without_resize``).

    Some methods (e.g. NPR on GANGen-Detection/freqnet) detect up-sampling artifacts and
    MUST NOT be fed a bilinearly up-scaled image — resizing a 128x128 GAN crop to 256x256
    injects the exact artifact they key on, collapsing the real class. The official NPR
    GANGen recipe runs with ``no_resize=True``; this reproduces it. Safe only when every
    image in a per-generator batch shares one resolution (true for freqnet's per-gen sets),
    since bench_eval stacks a batch into a single tensor."""
    return tf.without_resize()


@torch.no_grad()
def _open_rgb(path: Path) -> Image.Image:
    """Open ``path`` as RGB, recording the source file format in ``img.info`` so format-aware eval
    transforms (CuRe's PNG->JPEG96 step) see what the official loader sees."""
    source = Image.open(path)
    source_format = source.format
    image = source.convert("RGB")
    image.info[SRC_FORMAT_KEY] = source_format
    return image


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    """In-memory JPEG round trip, as DDA's ``JPEG_Compression`` (Training/data/custom_transforms.py)."""
    buf = io.BytesIO()
    image.save(buf, format="jpeg", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.info[SRC_FORMAT_KEY] = "JPEG"
    return out


SOCIAL_LONG_SIDE = 1080     # "social": the long side an image is resized to (when larger) before the JPEG q75 re-encode
SOCIAL_QUALITY = 75


def _resized(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Bicubic resize that keeps the source-format hint."""
    out = image.resize(size, Image.BICUBIC)
    out.info.update(image.info)
    return out


def _down(image: Image.Image, factor: float) -> Image.Image:
    """Downscale by ``factor`` (bicubic), then upsample back to the original size (bicubic)."""
    w, h = image.size
    small = _resized(image, (max(1, round(w * factor)), max(1, round(h * factor))))
    return _resized(small, (w, h))


def _blur(image: Image.Image, sigma: float) -> Image.Image:
    out = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    out.info.update(image.info)
    return out


def _social(image: Image.Image) -> Image.Image:
    """A social-network upload: long side to 1080 if larger (bicubic), then JPEG q75 (4:2:0)."""
    w, h = image.size
    long_side = max(w, h)
    if long_side > SOCIAL_LONG_SIDE:
        scale = SOCIAL_LONG_SIDE / long_side
        size = (SOCIAL_LONG_SIDE if w >= h else max(1, round(w * scale)),
                SOCIAL_LONG_SIDE if h > w else max(1, round(h * scale)))
        image = _resized(image, size)
    return _jpeg(image, SOCIAL_QUALITY)


def parse_degrade(spec: str):
    """The deterministic degradation (``PIL image -> RGB PIL image``) a ``--degrade`` token names; ValueError if malformed.

        jpeg<Q>   in-memory PIL JPEG re-encode at quality Q (1-100), default 4:2:0 chroma subsampling
        down<F>   downscale by factor F (0 < F < 1, e.g. down0.5) with bicubic, then back to the original size with bicubic
        blur<S>   Gaussian blur, sigma S > 0 (PIL ``ImageFilter.GaussianBlur(radius=S)``)
        social    long side resized to 1080 if larger (bicubic), then JPEG q75 (4:2:0)
        social2   ``social`` applied twice

    No RNG. Source-format hint (``SRC_FORMAT_KEY``, read by CuRe / MIRROR for their official PNG->JPEG step): a degradation
    that JPEG-encodes (jpeg, social, social2) leaves a JPEG, so the hint becomes "JPEG"; down and blur keep the file's own."""
    number = r"(\d+(?:\.\d+)?)"
    if spec == "social":
        return _social
    if spec == "social2":
        return lambda image: _social(_social(image))
    if m := re.fullmatch(r"jpeg(\d+)", spec):
        quality = int(m.group(1))
        if not 1 <= quality <= 100:
            raise ValueError(f"--degrade {spec}: JPEG quality must be in 1..100")
        return lambda image: _jpeg(image, quality)
    if m := re.fullmatch(rf"down{number}", spec):
        factor = float(m.group(1))
        if not 0 < factor < 1:
            raise ValueError(f"--degrade {spec}: downscale factor must be in (0, 1)")
        return lambda image: _down(image, factor)
    if m := re.fullmatch(rf"blur{number}", spec):
        sigma = float(m.group(1))
        if sigma <= 0:
            raise ValueError(f"--degrade {spec}: blur sigma must be > 0")
        return lambda image: _blur(image, sigma)
    raise ValueError(f"--degrade {spec!r}: expected jpeg<Q>, down<F>, blur<S>, social or social2")


def score(det, model, spec, paths: list[Path], bs: int, transform=None,
          jpeg_q: int | None = None, jpeg_from: int = 0, degrade=None) -> tuple[torch.Tensor, list[int]]:
    """Fake probabilities for the images that could be read, and the indices (into ``paths``) they belong to.
    ``degrade`` (a ``parse_degrade`` callable) is applied to every image right after loading; then, with ``jpeg_q``,
    images from index ``jpeg_from`` on (the fakes) are JPEG-compressed at that quality; then the detector's transform."""
    device = det._engine.device
    tf = transform if transform is not None else spec.transform
    # Native-resolution specs (SPAI any-res) produce variable-size tensors that cannot be stacked,
    # so they MUST be scored one image at a time.
    if getattr(spec, "native_res", False):
        bs = 1
    probs, kept = [], []
    for i in range(0, len(paths), bs):
        batch = []
        for j, p in enumerate(paths[i:i + bs], start=i):
            try:
                image = _open_rgb(p)
                if degrade is not None:
                    image = degrade(image)
                if jpeg_q is not None and j >= jpeg_from and jpeg_q < 100:
                    image = _jpeg(image, jpeg_q)
                batch.append(tf(image))
                kept.append(j)
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        probs.append(det._engine._fake_prob(model, spec, x).detach().float().cpu())
    return (torch.cat(probs) if probs else torch.empty(0)), kept


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
    ap.add_argument("-m", "--model", required=True, help="SynthScope detector key (see `synthscope list`)")
    ap.add_argument("--benchmark", choices=["forensynths", "ufd", "pergen", "pool"],
                    help="how the folders are grouped (chosen automatically with --dataset)")
    ap.add_argument("--data", help="benchmark root (not needed with --dataset)")
    ap.add_argument("--dataset", help="a supported dataset (see `python benchmark/download_dataset.py --list`): downloads it "
                    "if missing, finds the folder to evaluate and the right --benchmark profile")
    ap.add_argument("--root", help="with --dataset: where the dataset lives / should be downloaded (default: root_dir in its yaml)")
    ap.add_argument("--no-download", action="store_true", help="with --dataset: fail instead of downloading missing data")
    ap.add_argument("--dry-run", action="store_true", help="with --dataset: only show the resolved folder and profile")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback; with --dataset also the real pool of loaders that ship "
                    "fakes only, e.g. Synthbuster)")
    ap.add_argument("--ckpt", default=None, help="checkpoint override (official vs SynthScope weights)")
    ap.add_argument("--model-config", default=None, metavar="NAME",
                    help="score --ckpt with another model config of the same detector (conf/model/NAME.yaml, e.g. image/effort_ufd: "
                         "a training recipe whose own test preprocessing differs from the served model's); needs --ckpt")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--native-res", action="store_true",
                    help="score at native resolution (strip up-scaling Resize/CenterCrop); "
                         "for artifact-based methods like NPR on low-res GAN sets (freqnet)")
    ap.add_argument("--jpeg-fakes", type=int, default=None, metavar="Q",
                    help="JPEG-compress only the fake images at quality Q before scoring (DDA protocol: 96)")
    ap.add_argument("--degrade", default=None, metavar="SPEC",
                    help="robustness slice: degrade EVERY image (reals and fakes) after loading, before --jpeg-fakes and the "
                         "detector's transform. SPEC: jpeg<Q> | down<F> | blur<S> | social | social2, e.g. jpeg75, down0.5, "
                         "blur2 (deterministic; recorded as `degrade` in the output)")
    ap.add_argument("--results-log", default=None, metavar="PATH",
                    help="append the run's summary line to PATH instead of repro/MASTER_RESULTS.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", help="run tag (output filename + log); default <model>_<dataset> (+ _<degrade> with --degrade and no --out)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.model_config and not args.ckpt:
        ap.error("--model-config needs --ckpt")
    degrade = None
    if args.degrade is not None:     # "" is rejected by parse_degrade, not silently ignored
        try:
            degrade = parse_degrade(args.degrade)
        except ValueError as e:
            ap.error(str(e))

    if args.dataset:
        overrides = []
        if args.reals and "real_pool" in dataset_config(args.dataset, "image", args.root).DatasetConfig:
            # loaders that ship fakes only (Synthbuster) pair them with this pool when building the layout
            overrides.append(f"DatasetConfig.real_pool={Path(args.reals).expanduser().resolve()}")
        root = ensure_dataset(args.dataset, "image", args.root, download=not args.no_download and not args.dry_run,
                              overrides=overrides)
        eval_root, profile = find_eval_root(args.dataset, root)
        args.data = args.data or str(eval_root)
        args.benchmark = args.benchmark or profile
        # a degraded run must never overwrite the clean default repro/bench_results/<model>_<dataset>.json
        degrade_suffix = f"_{args.degrade}" if args.degrade is not None and not args.out else ""
        args.tag = args.tag or f"{args.model}_{args.dataset}{degrade_suffix}"
        print(f"[dataset] {args.dataset}: evaluating {args.data} with the '{args.benchmark}' profile", flush=True)
        if args.dry_run:
            return 0
    if not (args.benchmark and args.data):
        ap.error("give --dataset NAME, or both --benchmark and --data")
    if not args.tag:
        ap.error("--tag is required without --dataset")

    load_kw = {}
    if args.ckpt:
        load_kw["checkpoint"] = args.ckpt
    if args.model_config:
        spec = get_spec(args.model)
        # only the model config changes; the spec's other overrides (e.g. backbone options) stay
        overrides = [o for o in spec.overrides if not o.startswith("model@ModelConfig=")]
        spec = dataclasses.replace(spec, overrides=overrides + [f"model@ModelConfig={args.model_config}"],
                                   transform=model_transform(args.model_config))
        det = Detector(spec, device=args.device, **load_kw)
    else:
        det = synthscope.load(args.model, device=args.device, **load_kw)
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
          f"| {args.benchmark} | {len(gens)} generators | device={det._engine.device}"
          + (f" | fakes JPEG q{args.jpeg_fakes}" if args.jpeg_fakes else "")
          + (f" | degrade {args.degrade}" if args.degrade is not None else "") + " ===", flush=True)

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs, kept = score(det, model, spec, reals + fakes, args.bs, transform=eval_transform,
                            jpeg_q=args.jpeg_fakes, jpeg_from=len(reals), degrade=degrade)
        if len(kept) != labels.numel():     # unreadable images are skipped: keep each score paired with its own label
            print(f"    {labels.numel() - len(kept)} unreadable image(s) left out of {name}", flush=True)
            labels = labels[kept]
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
        "model_config": args.model_config,
        "benchmark": args.benchmark,
        "data_root": str(root),
        "jpeg_fakes": args.jpeg_fakes,
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "synthscope_version": getattr(synthscope, "__version__", "?"),
    }
    if args.degrade is not None:
        result["degrade"] = args.degrade
    out = Path(args.out).expanduser() if args.out else (
        RESULTS_LOG.parent / "bench_results" / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    results_log = Path(args.results_log).expanduser() if args.results_log else RESULTS_LOG
    results_log.parent.mkdir(parents=True, exist_ok=True)
    with results_log.open("a") as f:
        f.write(json.dumps({k: result[k] for k in
                            ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                             "n_generators", "means") + (("model_config",) if args.model_config else ())
                            + (("degrade",) if args.degrade is not None else ())}) + "\n")

    print(f"\n=== RESULT {args.tag}: mAP={means['map']} mAUROC={means['mauroc']} acc={means['acc']} "
          f"over {len(per_gen)} generators -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
