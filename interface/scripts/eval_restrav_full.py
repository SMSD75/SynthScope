"""Evaluate the ReStraV video detector on the full (staged) GenVideo-Val test set, through the
same engine the web app uses. Reports acc@0.5, real/fake accuracy, AP and AUROC.

  conda run -n SynthScope python interface/scripts/eval_restrav_full.py [--n 800]
--n caps videos PER CLASS (videos are decode-heavy); 0 = all.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.engine import get_engine     # noqa: E402
from backend.registry import get_spec     # noqa: E402
from backend import media                 # noqa: E402

ROOT = Path("/hddstore/ssalehi/SynthScope/restrav_data_fulltest/test/genvideo")


def ap_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p, kind="mergesort"); y = y[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan"), float("nan")
    prec = tp / np.maximum(tp + fp, 1); rec = tp / npos
    rec = np.concatenate([[0.0], rec]); prec = np.concatenate([[1.0], prec])
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))
    tpr = np.concatenate([[0.0], tp / npos]); fpr = np.concatenate([[0.0], fp / nneg])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return ap, auc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()
    engine = get_engine()
    spec = get_spec("restrav")
    model = engine._get_model(spec)
    mean = torch.tensor(spec.video_norm[0]).view(1, 3, 1, 1)
    std = torch.tensor(spec.video_norm[1]).view(1, 3, 1, 1)

    reals = sorted((ROOT / "0_real").glob("*.mp4"))
    fakes = sorted((ROOT / "1_fake").glob("*.mp4"))
    if args.n > 0:
        random.Random(0).shuffle(reals); random.Random(1).shuffle(fakes)
        reals, fakes = reals[:args.n], fakes[:args.n]
    items = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
    print(f"restrav full eval: {len(reals)} real + {len(fakes)} fake on {engine.device}")

    probs, labels = [], []
    for i, (p, y) in enumerate(items):
        try:
            clip = media.video_clip_tensor(p, spec.num_frames, spec.fps, spec.frame_size,
                                           window_center=spec.window_center, interp=spec.resize_interp)
            clip = ((clip - mean) / std)
            probs.append(float(engine._video_fake_prob(model, spec, clip).item()))
            labels.append(y)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)}")
    probs = np.array(probs); labels = np.array(labels)
    pred = (probs > 0.5).astype(int)
    acc = float((pred == labels).mean())
    racc = float((pred[labels == 0] == 0).mean()); facc = float((pred[labels == 1] == 1).mean())
    apv, aucv = ap_auc(labels, probs)
    print(f"restrav: acc@0.5={acc*100:.2f} real_acc={racc*100:.2f} fake_acc={facc*100:.2f} "
          f"AP={apv*100:.2f} AUC={aucv*100:.2f}")


if __name__ == "__main__":
    main()
