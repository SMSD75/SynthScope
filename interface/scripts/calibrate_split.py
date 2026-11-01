"""Calibrate the training-free SPLIT video detector so its output is a usable probability.

SPLIT's raw score (TTR ** 8 * motion incoherence) spans many orders of magnitude, so the calibration works on its
logarithm and follows the paper's real-only protocol: the decision threshold is the score at a target false-positive
rate on real videos alone (default 1%, numpy's ``quantile(..., method="higher")`` as the official ``eval.py``), and the
logistic scale puts the median real video at P(fake) = 0.05. The engine then reports
``sigmoid((log(score) - threshold) / scale)`` as P(fake). Fakes are scored only to report the recall at that threshold.

  conda run -n SynthScope python interface/scripts/calibrate_split.py [--n-real 2000] [--n-fake 1000] [--fpr 0.01]
      [--data DIR] [--out FILE]

Clips go through the engine's own SPLIT preprocessing (``Engine.video_clip``), so the calibration matches what the web
app and benchmarks feed the model. ``--data`` takes GenVideo-Val as released (``Real/`` + ``Fake/<gen>/``) or as laid out
by the ``genvideo`` loader (``test/<gen>/{0_real,1_fake}/``).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from backend import config                     # noqa: E402
from backend.engine import get_engine           # noqa: E402
from backend.registry import get_spec            # noqa: E402
from calibrate_d3 import GENVIDEO, collect, score_paths   # noqa: E402

EPS = 1e-30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--n-fake", type=int, default=1000)
    ap.add_argument("--fpr", type=float, default=0.01, help="target false-positive rate on real videos")
    ap.add_argument("--data", default=str(GENVIDEO), help="GenVideo-Val root (released layout or the genvideo loader's)")
    ap.add_argument("--out", default=None, help="where to write the JSON (default: <CKPT_DIR>/split_calibration.json)")
    args = ap.parse_args()

    engine = get_engine()
    spec = get_spec("split")
    model = engine._get_model(spec)
    reals, _ = collect(args.n_real, args.data)
    _, fakes = collect(args.n_fake, args.data)
    print(f"scoring {len(reals)} real + {len(fakes)} fake clips on {engine.device}...")
    rs = np.log(np.maximum(score_paths(engine, spec, model, reals), EPS))
    fs = np.log(np.maximum(score_paths(engine, spec, model, fakes), EPS))

    threshold = float(np.quantile(rs, 1 - args.fpr, method="higher"))
    scale = max((threshold - float(np.median(rs))) / math.log(19), 1e-6)    # median real -> sigmoid(-ln 19) = 0.05
    cal = {"threshold": threshold, "scale": scale, "higher_is_fake": True, "log_score": True,
           "target_fpr": args.fpr, "real_fpr": float((rs >= threshold).mean()),
           "fake_recall": float((fs >= threshold).mean()), "median_log_real": float(np.median(rs)),
           "median_log_fake": float(np.median(fs)), "n_real": int(len(rs)), "n_fake": int(len(fs))}
    out = Path(args.out) if args.out else config.CHECKPOINTS_DIR / "split_calibration.json"
    out.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"threshold(log)={threshold:.4f} scale={scale:.4f} real FPR={cal['real_fpr']:.4f} "
          f"fake recall={cal['fake_recall']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
