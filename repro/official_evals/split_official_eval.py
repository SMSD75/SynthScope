"""Run the OFFICIAL SPLIT pipeline (Hyun & Kim, ECCV 2026; github.com/mldljyh/SPLIT) on a SynthScope video dataset.

SynthScope's SPLIT gives the official ``SPLIT_model`` scores on the same frames (relative difference ~1e-6), but the
official numbers also depend on the official frame extraction, which SPLIT shares with D3: ``utils/video2frame.py``
writes JPEG frames with ffmpeg (8 fps over 3 s, random start for longer clips), ``data/datasets.py`` reads them back
with ``cv2.imread`` (BGR), crops 10% of the longer side, resizes to 224 and normalizes with ImageNet statistics. The
frames are extracted exactly as ``d3_official_eval.py`` does (same seeded start, same frame folders, so the two share a
cache), and the metrics come from the official ``eval.py``'s ``evaluate_split`` unchanged: fake recall at FPR 0.1 / 1 /
5 % with the threshold set on the real scores alone (``np.quantile(real, 1 - fpr, method="higher")``).

Like the paper's GenVideo table, every real clip (GenVideo-Val's MSR-VTT reals) calibrates the threshold and every fake
clip is scored, per generator and pooled over all generators ("Overall"); there is no per-class limit unless ``--limit``
is given.

    python repro/official_evals/split_official_eval.py --dataset-root /data/GenVideo/test --name genvideo \\
        --official-repo /path/to/SPLIT --albumentations-path /path/to/site-packages --ffmpeg /path/to/ffmpeg \\
        --frames /scratch/d3_frames/genvideo

Output: repro/bench_results/<tag>.json and a line in repro/MASTER_RESULTS.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d3_official_eval import extract, videos  # noqa: E402  (the shared official frame extraction)

REPRO = Path(__file__).resolve().parents[1]
RESULTS_LOG = REPRO / "MASTER_RESULTS.jsonl"
FPRS = (0.001, 0.01, 0.05)


def load_official_eval(repo: str):
    """The official ``eval.py`` as a module (its ``evaluate_split`` computes the paper's metrics)."""
    spec = importlib.util.spec_from_file_location("split_official_eval_py", Path(repo) / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def scores(model, read_video, trans, paths, frames_root: Path, ffmpeg: str, device) -> list[float]:
    out = []
    for p in paths:
        folder = extract(p, frames_root / hashlib.md5(str(p.resolve()).encode()).hexdigest()[:16], ffmpeg)
        try:
            x = read_video(str(folder), trans=trans).unsqueeze(0).to(device)    # official reader: needs >= 8 frames
        except Exception as e:  # noqa: BLE001
            print(f"    skip {p.name}: {e}")
            continue
        out.append(float(model.forward_score(x).item()))
    return out


def summarize(official, real: list[float], fake: list[float]) -> dict:
    m = official.evaluate_split(np.array(real), np.array(fake))
    return {"fake_recall_at_fpr": {f"{a * 100:g}": round(m["fake_recall_at_fpr"][a] * 100, 2) for a in FPRS},
            "ap": round(m["ap_fake"] * 100, 2), "ap_real": round(m["ap_real"] * 100, 2),
            "auroc": round(m["roc_auc"] * 100, 2), "n_real": len(real), "n_fake": len(fake)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, help="<generator>/{0_real,1_fake} tree")
    ap.add_argument("--name", required=True, help="dataset name for the tag, e.g. genvideo")
    ap.add_argument("--official-repo", required=True, help="clone of github.com/mldljyh/SPLIT")
    ap.add_argument("--albumentations-path", default=None, help="extra sys.path entry providing albumentations")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--frames", required=True, help="scratch folder for the extracted JPEG frames")
    ap.add_argument("--limit", type=int, default=0, help="clips per class per generator (0 = all)")
    ap.add_argument("--encoder", default="XCLIP-16")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    sys.path[:0] = [args.official_repo] + ([args.albumentations_path] if args.albumentations_path else [])
    from data.datasets import read_video, set_preprocessing      # official SPLIT
    from models.SPLIT_model import SPLIT_model                   # official SPLIT
    official = load_official_eval(args.official_repo)

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SPLIT_model(encoder_type=args.encoder, loss_type="l2").to(device).eval()
    trans = set_preprocessing(None, None)
    root, frames_root = Path(args.dataset_root), Path(args.frames)
    tag = args.tag or f"split_official_{args.name}"
    cap = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    gens = sorted(p for p in root.iterdir() if p.is_dir()
                  and (not args.select or p.name in args.select.split(",")))
    print(f"=== split_official_eval {tag} | {len(gens)} generators | {args.encoder} | device={device} ===", flush=True)

    cache: dict = {}
    per_gen, all_real, all_fake = {}, None, []
    for g in gens:
        reals, fakes = cap(videos(g / "0_real")), cap(videos(g / "1_fake"))
        if not reals or not fakes:
            continue
        key = tuple(p.name for p in reals)     # GenVideo copies the same MSR-VTT reals into every generator folder
        if key not in cache:
            cache[key] = scores(model, read_video, trans, reals, frames_root, args.ffmpeg, device)
        rs = cache[key]
        fs = scores(model, read_video, trans, fakes, frames_root, args.ffmpeg, device)
        per_gen[g.name] = summarize(official, rs, fs)
        all_real = rs if all_real is None or len(rs) > len(all_real) else all_real
        all_fake += fs
        print(f"  {g.name}: {len(rs)} real + {len(fs)} fake  fake recall @ FPR 0.1/1/5% = "
              f"{list(per_gen[g.name]['fake_recall_at_fpr'].values())}  AP={per_gen[g.name]['ap']}", flush=True)

    overall = summarize(official, all_real, all_fake)
    mean = lambda f: round(sum(v["fake_recall_at_fpr"][f] for v in per_gen.values()) / len(per_gen), 2)  # noqa: E731
    means = {"overall_fake_recall_at_fpr": overall["fake_recall_at_fpr"],
             "mean_fake_recall_at_fpr": {f: mean(f) for f in overall["fake_recall_at_fpr"]},
             "map": round(sum(v["ap"] for v in per_gen.values()) / len(per_gen), 2)}
    result = {"tag": tag, "model": "split", "ckpt": f"official SPLIT code ({args.encoder}), training-free",
              "benchmark": "video:official-split", "data_root": str(root), "n_generators": len(per_gen),
              "means": means, "overall": overall, "per_generator": per_gen,
              "date": datetime.datetime.now().isoformat(timespec="seconds")}
    out = REPRO / "bench_results" / f"{tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    with RESULTS_LOG.open("a") as f:
        f.write(json.dumps({k: result[k] for k in ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                                                   "n_generators", "means")}) + "\n")
    print(f"\n=== RESULT {tag}: Overall fake recall @ FPR 0.1/1/5% = {list(overall['fake_recall_at_fpr'].values())} "
          f"over {len(per_gen)} generators -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
