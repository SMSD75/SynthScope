"""Run the OFFICIAL D3 pipeline (Zheng et al., ICCV 2025; github.com/Zig-HS/D3) on a SynthScope video dataset.

SynthScope's D3 class is identical to the official model (same scores on the same frames, max |diff| = 0), but the
official numbers also depend on the official frame extraction: ``utils/video2frame.py`` writes JPEG frames with ffmpeg
(8 fps over 3 s), and ``data/datasets.py`` reads them back with ``cv2.imread`` (BGR), crops 10% of the longer side,
resizes to 224 and normalizes with ImageNet statistics; ``eval.py`` takes the first ``max_len=1000`` clips per class
and reports ``average_precision_score(1 - y, std)`` (real videos as the positive class). This script imports those
official functions from a clone of the repository and runs them end to end, so its AP is the like-for-like comparison
with the paper.

Two unavoidable differences from the authors' runs: their CSVs list clips in file-system order and their frame
extraction uses a random start (from a multiprocessing pool) for clips longer than 3 s; here clips are taken in sorted
order and longer clips start at a per-clip seeded random offset.

    python repro/official_evals/d3_official_eval.py --dataset-root /data/GenVideo/test --name genvideo \\
        --official-repo /path/to/D3 --albumentations-path /path/to/site-packages --ffmpeg /path/to/ffmpeg \\
        --frames /scratch/d3_frames --limit 1000

``--dataset-root`` is a ``<generator>/{0_real,1_fake}`` tree (what the genvideo / videophy / vidprom loaders write) or,
with ``--reals``, a ``<generator>/*.mp4`` fake-only tree (EvalCrafter). Output: repro/bench_results/<tag>.json and a line
in repro/MASTER_RESULTS.jsonl, in the same schema as benchmark/bench_eval_video.py.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPRO = Path(__file__).resolve().parents[1]
RESULTS_LOG = REPRO / "MASTER_RESULTS.jsonl"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".m4v"}


def videos(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS) if d.is_dir() else []


def duration(path: Path) -> float:
    """Clip length in seconds (the official code uses moviepy); 0 when decord cannot measure it (e.g. some GIFs),
    which makes the frame extraction start at 0, as it does for every clip up to 3 s."""
    from decord import VideoReader, cpu
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
        return len(vr) / (float(vr.get_avg_fps()) or 30.0)
    except Exception:  # noqa: BLE001
        return 0.0


def extract(path: Path, out: Path, ffmpeg: str) -> Path:
    """Official utils/video2frame.py: 3 s at 8 fps as JPEG frames; start 0 for clips <= 3 s, random otherwise."""
    if out.is_dir() and any(out.iterdir()):
        return out
    out.mkdir(parents=True, exist_ok=True)
    length = duration(path)
    seed = int(hashlib.md5(path.name.encode()).hexdigest()[:8], 16)
    start = 0 if length <= 3 else int(np.floor(random.Random(seed).uniform(0, length - 3)))
    subprocess.run([ffmpeg, "-loglevel", "quiet", "-ss", str(start), "-t", "3", "-i", str(path), "-vf", "fps=8",
                    f"{out}/%d.jpg"], check=False)
    return out


@torch.no_grad()
def scores(model, read_video, trans, paths, frames_root: Path, ffmpeg: str, device) -> tuple[list[float], list[Path]]:
    out, kept = [], []
    for p in paths:
        folder = extract(p, frames_root / hashlib.md5(str(p.resolve()).encode()).hexdigest()[:16], ffmpeg)
        try:
            x = read_video(str(folder), trans=trans).unsqueeze(0).to(device)    # official reader: needs >= 8 frames
        except Exception as e:  # noqa: BLE001
            print(f"    skip {p.name}: {e}")
            continue
        out.append(float(model(x)[2].item()))
        kept.append(p)
    return out, kept


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--name", required=True, help="dataset name for the tag, e.g. genvideo")
    ap.add_argument("--reals", default=None, help="real-video folder for a fake-only tree (EvalCrafter)")
    ap.add_argument("--official-repo", required=True, help="clone of github.com/Zig-HS/D3")
    ap.add_argument("--albumentations-path", default=None, help="extra sys.path entry providing albumentations")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--frames", required=True, help="scratch folder for the extracted JPEG frames")
    ap.add_argument("--limit", type=int, default=1000, help="clips per class per generator (official max_len=1000)")
    ap.add_argument("--encoder", default="XCLIP-16")
    ap.add_argument("--loss", default="l2")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    sys.path[:0] = [args.official_repo] + ([args.albumentations_path] if args.albumentations_path else [])
    from data.datasets import read_video, set_preprocessing      # official D3
    from models.D3_model import D3_model                         # official D3

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device).eval()
    trans = set_preprocessing(None, None)
    root, frames_root = Path(args.dataset_root), Path(args.frames)
    tag = args.tag or f"d3_official_{args.name}"
    gens = sorted(p for p in root.iterdir() if p.is_dir()
                  and (not args.select or p.name in args.select.split(",")))
    print(f"=== d3_official_eval {tag} | {len(gens)} generators | {args.encoder}/{args.loss} | device={device} ===",
          flush=True)

    shared = videos(Path(args.reals))[:args.limit] if args.reals else None
    cache: dict = {}
    per_gen = {}
    for g in gens:
        reals = shared if shared is not None else videos(g / "0_real")[:args.limit]
        fakes = videos(g) if shared is not None else videos(g / "1_fake")
        fakes = fakes[:args.limit]
        if not reals or not fakes:
            continue
        key = (len(reals), reals[0].name, reals[-1].name)
        if key not in cache:
            cache[key] = scores(model, read_video, trans, reals, frames_root, args.ffmpeg, device)
        rs, _ = cache[key]
        fs, _ = scores(model, read_video, trans, fakes, frames_root, args.ffmpeg, device)
        y = np.array([0] * len(rs) + [1] * len(fs)); s = np.array(rs + fs)
        m = {"ap_real": round(average_precision_score(1 - y, s) * 100, 2),     # official metric
             "ap": round(average_precision_score(y, -s) * 100, 2),
             "auroc": round(roc_auc_score(1 - y, s) * 100, 2), "n_real": len(rs), "n_fake": len(fs)}
        per_gen[g.name] = m
        print(f"  {g.name}: {len(rs)} real + {len(fs)} fake  AP_real={m['ap_real']} AP={m['ap']} AUROC={m['auroc']}",
              flush=True)

    mean = lambda k: round(sum(v[k] for v in per_gen.values()) / len(per_gen), 2)  # noqa: E731
    means = {"map_real": mean("ap_real"), "map": mean("ap"), "mauroc": mean("auroc")}
    result = {"tag": tag, "model": "d3", "ckpt": f"official D3 code ({args.encoder}, {args.loss})",
              "benchmark": "video:official-d3", "data_root": str(root), "n_generators": len(per_gen), "means": means,
              "per_generator": per_gen, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    out = REPRO / "bench_results" / f"{tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    with RESULTS_LOG.open("a") as f:
        f.write(json.dumps({k: result[k] for k in ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                                                   "n_generators", "means")}) + "\n")
    print(f"\n=== RESULT {tag}: mAP_real={means['map_real']} mAP={means['map']} mAUROC={means['mauroc']} "
          f"over {len(per_gen)} generators -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
