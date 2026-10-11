"""Calibrate the training-free D3 video detector so its output is a usable probability.

D3's raw score is the per-video second-order feature volatility (std). The model's argmax
threshold defaults to 0.0, which is not a real decision boundary, so every video reads as
"real". This script estimates, on a balanced real/fake sample of GenVideo-Val:
  * threshold  -- midpoint between the class-mean scores,
  * scale      -- pooled std (logistic temperature), and
  * higher_is_fake -- which class has the larger score,
and writes them to ``<CKPT_DIR>/d3_calibration.json``. The engine then reports
``sigmoid(sign * (score - threshold) / scale)`` as P(fake).

  conda run -n SynthScope python interface/scripts/calibrate_d3.py [--n 200] [--data DIR] [--out FILE]

Clips go through the engine's own D3 preprocessing (``Engine.video_clip``), so the calibration always matches what the
web app and benchmarks feed the model. ``--data`` takes GenVideo-Val either as released (``Real/`` + ``Fake/<gen>/``) or
as laid out by the ``genvideo`` loader (``test/<gen>/{0_real,1_fake}/``).
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import config                 # noqa: E402
from backend.engine import get_engine       # noqa: E402
from backend.registry import get_spec        # noqa: E402

GENVIDEO = Path("/hddstore/ssalehi/SynthScope/d3_repro/genvideo_val/GenVideo-Val")


def _unique(paths):
    """Drop repeats of the same file name (the genvideo loader hard-links one real pool into every generator)."""
    seen, out = set(), []
    for p in sorted(paths):
        if p.name not in seen:
            seen.add(p.name); out.append(p)
    return out


def collect(n, root=GENVIDEO):
    root = Path(root)
    if (root / "Real").is_dir():
        reals, fakes = sorted((root / "Real").rglob("*.mp4")), sorted((root / "Fake").rglob("*.mp4"))
    else:
        reals, fakes = _unique(root.rglob("0_real/*.mp4")), sorted(root.rglob("1_fake/*.mp4"))
    random.Random(0).shuffle(reals)
    random.Random(1).shuffle(fakes)
    return reals[:n], fakes[:n]


@torch.no_grad()
def score_paths(engine, spec, model, paths):
    out = []
    for p in paths:
        try:
            clip = engine.video_clip(spec, p).unsqueeze(0).to(engine.device)
            out.append(float(model.score(clip).item()))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default=str(GENVIDEO), help="GenVideo-Val root (released layout or the genvideo loader's)")
    ap.add_argument("--out", default=None, help="where to write the JSON (default: <CKPT_DIR>/d3_calibration.json)")
    args = ap.parse_args()

    engine = get_engine()
    spec = get_spec("d3")
    model = engine._get_model(spec)
    reals, fakes = collect(args.n, args.data)
    print(f"scoring {len(reals)} real + {len(fakes)} fake clips on {engine.device}...")
    rs = score_paths(engine, spec, model, reals)
    fs = score_paths(engine, spec, model, fakes)

    mr, mf = sum(rs) / len(rs), sum(fs) / len(fs)
    var = (sum((x - mr) ** 2 for x in rs) + sum((x - mf) ** 2 for x in fs)) / (len(rs) + len(fs))
    scale = max(math.sqrt(var), 1e-6)
    threshold = (mr + mf) / 2.0
    higher_is_fake = mf > mr

    # quick sanity accuracy at this threshold
    def pred_fake(s):
        return (s > threshold) if higher_is_fake else (s < threshold)
    acc = (sum(not pred_fake(s) for s in rs) + sum(pred_fake(s) for s in fs)) / (len(rs) + len(fs))

    cal = {"threshold": threshold, "scale": scale, "higher_is_fake": higher_is_fake,
           "mean_real": mr, "mean_fake": mf, "n": args.n, "balanced_acc": acc}
    out = Path(args.out) if args.out else config.CHECKPOINTS_DIR / "d3_calibration.json"
    out.write_text(json.dumps(cal, indent=2))
    print(f"mean_real={mr:.5f} mean_fake={mf:.5f} threshold={threshold:.5f} scale={scale:.5f} "
          f"higher_is_fake={higher_is_fake} acc={acc:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
