"""Calibrate the training-free D3 video detector so its output is a usable probability.

D3's raw score is the per-video second-order feature volatility (std). The model's argmax
threshold defaults to 0.0, which is not a real decision boundary, so every video reads as
"real". This script estimates, on a balanced real/fake sample of GenVideo-Val:
  * threshold  -- midpoint between the class-mean scores,
  * scale      -- pooled std (logistic temperature), and
  * higher_is_fake -- which class has the larger score,
and writes them to ``<CKPT_DIR>/d3_calibration.json``. The engine then reports
``sigmoid(sign * (score - threshold) / scale)`` as P(fake).

  conda run -n SynthScope python interface/scripts/calibrate_d3.py [--n 200]
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
from backend.registry import get_spec, CLIP_MEAN, CLIP_STD  # noqa: E402
from backend import media                   # noqa: E402

GENVIDEO = Path("/hddstore/ssalehi/AGID/d3_repro/genvideo_val/GenVideo-Val")


def collect(n):
    reals = sorted((GENVIDEO / "Real").rglob("*.mp4"))
    fakes = sorted((GENVIDEO / "Fake").rglob("*.mp4"))
    random.Random(0).shuffle(reals)
    random.Random(1).shuffle(fakes)
    return reals[:n], fakes[:n]


@torch.no_grad()
def score_paths(engine, spec, model, paths):
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    out = []
    for p in paths:
        try:
            clip = media.video_clip_tensor(p, spec.num_frames, spec.fps, spec.frame_size)
            clip = ((clip - mean) / std).unsqueeze(0).to(engine.device)
            out.append(float(model.score(clip).item()))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    engine = get_engine()
    spec = get_spec("d3")
    model = engine._get_model(spec)
    reals, fakes = collect(args.n)
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
    out = config.CHECKPOINTS_DIR / "d3_calibration.json"
    out.write_text(json.dumps(cal, indent=2))
    print(f"mean_real={mr:.5f} mean_fake={mf:.5f} threshold={threshold:.5f} scale={scale:.5f} "
          f"higher_is_fake={higher_is_fake} acc={acc:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
