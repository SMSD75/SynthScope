"""Verify a (trained or official) image detector by scoring a balanced real/fake sample
from the local ForenSynths/CNNDetection test set, through the SAME engine the web app uses.

Reports accuracy@0.5, real/fake accuracy, AP and AUC so the checkpoint's quality is a real,
reproducible number (not a training-log artifact).

  conda run -n SynthScope python interface/scripts/verify_model.py <key> [--n 200] [--gens progan,biggan]
"""
import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.engine import get_engine     # noqa: E402
from backend.registry import get_spec     # noqa: E402

TEST_ROOT = "/hddstore/ssalehi/SynthScope/CNNDetection/test"


def list_split(gen, sub):
    d = os.path.join(TEST_ROOT, gen, sub)
    fs = []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.webp", "*.bmp"):
        fs += glob.glob(os.path.join(d, "**", e), recursive=True)
    return sorted(set(fs))


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
    ap.add_argument("key")
    ap.add_argument("--n", type=int, default=200, help="images per class")
    ap.add_argument("--gens", default="progan,biggan,stylegan,cyclegan")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    engine = get_engine()
    spec = get_spec(args.key)
    model = engine._get_model(spec)

    items = []
    for gen in args.gens.split(","):
        reals = list_split(gen, "0_real"); fakes = list_split(gen, "1_fake")
        random.Random(0).shuffle(reals); random.Random(1).shuffle(fakes)
        per = max(1, args.n // len(args.gens.split(",")))
        items += [(p, 0) for p in reals[:per]] + [(p, 1) for p in fakes[:per]]
    random.Random(2).shuffle(items)
    print(f"{args.key}: scoring {len(items)} images from [{args.gens}] on {engine.device}")

    probs, labels = [], []
    for i in range(0, len(items), args.bs):
        chunk = items[i:i + args.bs]
        batch = torch.stack([spec.transform(Image.open(p).convert("RGB")) for p, _ in chunk]).to(engine.device)
        probs += engine._fake_prob(model, spec, batch).float().cpu().tolist()
        labels += [l for _, l in chunk]

    probs = np.array(probs); labels = np.array(labels)
    pred = (probs > 0.5).astype(int)
    acc = float((pred == labels).mean())
    racc = float((pred[labels == 0] == 0).mean())
    facc = float((pred[labels == 1] == 1).mean())
    apv, aucv = ap_auc(labels, probs)
    print(f"  acc@0.5={acc*100:.2f}  real_acc={racc*100:.2f}  fake_acc={facc*100:.2f}  "
          f"AP={apv*100:.2f}  AUC={aucv*100:.2f}  (mean P(fake)={probs.mean():.3f})")


if __name__ == "__main__":
    main()
