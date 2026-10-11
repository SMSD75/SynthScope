"""Evaluate an SynthScope detector on a benchmark track and write a schema-valid submission JSON.

    python benchmark/evaluate.py --model fat_former --track forensynths \
        --data /path/to/forensynths_test --out benchmark/submissions/fat_former.json

Reuses the verified ``synthscope`` inference path (model build + checkpoint load + per-model
preprocessing + fake-probability), then computes per-generator + mean metrics with torchmetrics —
the same metrics the trainer reports. Image tracks only in v1 (video harness is future work; the
video seed entries come from their reproduction eval).

Expected data layout (SynthScope common layout):

    <data>/<generator>/real/*.png   <data>/<generator>/fake/*.png

Use ``--reals DIR`` for benchmarks that share one real set across per-generator fake folders.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch
from PIL import Image, ImageFile

import synthscope

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_NAMES = {"real", "0_real", "reals"}
FAKE_NAMES = {"fake", "1_fake", "fakes", "ai"}


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if d and d.is_dir() else []


def _find_split(gen_dir: Path, names: set[str]) -> Path | None:
    for n in names:
        if (gen_dir / n).is_dir():
            return gen_dir / n
    return None


def discover_generators(root: Path, shared_reals: Path | None):
    """Return [(generator_name, real_paths, fake_paths)] from the benchmark root."""
    shared = _images(shared_reals) if shared_reals else []
    out = []
    # Single pooled set: <root>/real, <root>/fake
    if _find_split(root, REAL_NAMES) or _find_split(root, FAKE_NAMES):
        reals = _images(_find_split(root, REAL_NAMES)) or shared
        fakes = _images(_find_split(root, FAKE_NAMES))
        if reals and fakes:
            return [("all", reals, fakes)]
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        reals = _images(_find_split(gen_dir, REAL_NAMES)) or shared
        fakes = _images(_find_split(gen_dir, FAKE_NAMES))
        if reals and fakes:
            out.append((gen_dir.name, reals, fakes))
        else:
            print(f"  skip generator '{gen_dir.name}': need both real and fake images")
    return out


@torch.no_grad()
def score(det: "synthscope.Detector", paths: list[Path], batch_size: int) -> torch.Tensor:
    """Return P(fake) for a list of image paths, using the detector's own preprocessing."""
    det._ensure_weights()
    model = det._engine._get_model(det.spec)          # build + load (verified path)
    spec, device = det.spec, det._engine.device
    probs = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i + batch_size]:
            batch.append(spec.transform(Image.open(p).convert("RGB")))
        x = torch.stack(batch).to(device)
        probs.append(det._engine._fake_prob(model, spec, x).detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def _per_generator_metrics(fake_probs: torch.Tensor, labels: torch.Tensor) -> dict | None:
    from torchmetrics.functional import auroc, average_precision

    if labels.unique().numel() < 2:
        return None  # AUROC/AP need both classes
    ap = float(average_precision(fake_probs, labels, task="binary")) * 100
    au = float(auroc(fake_probs, labels, task="binary")) * 100
    preds = (fake_probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "auroc": round(au, 2),
            "real_acc": round(real_acc, 2), "fake_acc": round(fake_acc, 2),
            "balanced_acc": round((real_acc + fake_acc) / 2, 2)}


def _infer_eval_type(model_type: str) -> str:
    from interface.backend.registry import get_spec

    spec = get_spec(model_type)
    if str(spec.ckpt_kind).startswith("official"):
        return "official-weights"
    if spec.ckpt_kind == "none":
        return "training-free"
    return "from-scratch"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a detector and emit a benchmark submission.")
    ap.add_argument("-m", "--model", required=True, help="SynthScope detector key (see `synthscope list`)")
    ap.add_argument("--track", required=True, choices=["forensynths", "genimage-sdv1.4"],
                    help="benchmark track (image tracks only in v1)")
    ap.add_argument("--data", required=True, help="benchmark root (per-generator real/fake dirs)")
    ap.add_argument("--reals", default=None, help="shared real-image dir (for shared-real layouts)")
    ap.add_argument("--out", required=True, help="output submission JSON path")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--device", default=None)
    ap.add_argument("--method", default=None, help="display name (default: registry name)")
    ap.add_argument("--eval-type", default=None,
                    help="override provenance (official-weights/from-scratch/frozen-probe/...)")
    ap.add_argument("--submitted-by", default="SynthScope")
    args = ap.parse_args(argv)

    det = synthscope.load(args.model, device=args.device)
    root = Path(args.data).expanduser()
    gens = discover_generators(root, Path(args.reals).expanduser() if args.reals else None)
    if not gens:
        print(f"No generators with real+fake images under {root}")
        return 1

    per_gen, means = {}, {"ap": [], "auroc": [], "balanced_acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        paths = reals + fakes
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs = score(det, paths, args.batch_size)
        m = _per_generator_metrics(probs, labels)
        if m is None:
            print(f"    (skipped metrics: only one class)")
            continue
        per_gen[name] = m
        for k in means:
            means[k].append(m[k])

    if not per_gen:
        print("No generator produced valid metrics.")
        return 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2)

    from interface.backend.registry import get_spec
    spec = get_spec(args.model)
    submission = {
        "method": args.method or spec.name,
        "track": args.track,
        "modality": "image",
        "model_type": args.model,
        "paper": spec.paper,
        "url": "https://github.com/SMSD75/SynthScope",
        "evaluation": args.eval_type or _infer_eval_type(args.model),
        "metrics": {
            "acc": mean(means["balanced_acc"]),
            "map": mean(means["ap"]),
            "ap": None,
            "auroc": mean(means["auroc"]),
        },
        "per_generator": per_gen,
        "submitted_by": args.submitted_by,
        "date": datetime.date.today().isoformat(),
        "verified": True,
        "source": f"synthscope evaluate.py on {root} ({len(per_gen)} generators)",
        "protocol_notes": None,
        "synthscope_version": synthscope.__version__,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(submission, indent=2) + "\n")
    print(f"Wrote {out}")
    print(f"  mAP={submission['metrics']['map']}  AUROC={submission['metrics']['auroc']}  "
          f"Acc={submission['metrics']['acc']}  over {len(per_gen)} generators")
    print("Run `python benchmark/leaderboard.py` to refresh LEADERBOARD.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
