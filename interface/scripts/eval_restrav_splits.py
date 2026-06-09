"""Evaluate the DEPLOYED ReStraV (restrav.pth, the serving path) on restrav_data val vs test,
to check whether the 'awful test' performance matches the AUROC numbers we've quoted.

val  = in-distribution (VidProM: VSC reals + VidProM generators) -> expect high.
test = cross-dataset    (GenVideo: msrvtt reals + GenVideo generators) -> expect ~chance.
"""
import glob, random, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.engine import get_engine          # noqa: E402
from backend.registry import get_spec          # noqa: E402
from backend import media                       # noqa: E402

ROOT = "/hddstore/ssalehi/AGID/restrav_data"
N = 200  # per class per split


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p, kind="mergesort"); y = y[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], tp / npos]); fpr = np.concatenate([[0.0], fp / nneg])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


@torch.no_grad()
def main():
    eng = get_engine(); spec = get_spec("restrav"); model = eng._get_model(spec)
    mean = torch.tensor(spec.video_norm[0]).view(1, 3, 1, 1)
    std = torch.tensor(spec.video_norm[1]).view(1, 3, 1, 1)
    for split in ["val", "test"]:
        reals = sorted(glob.glob(f"{ROOT}/{split}/genvideo/0_real/*.mp4"))
        fakes = sorted(glob.glob(f"{ROOT}/{split}/genvideo/1_fake/*.mp4"))
        random.Random(0).shuffle(reals); random.Random(1).shuffle(fakes)
        items = [(p, 0) for p in reals[:N]] + [(p, 1) for p in fakes[:N]]   # AGID: real=0, fake=1
        probs, labels = [], []
        for p, y in items:
            try:
                clip = media.video_clip_tensor(p, spec.num_frames, spec.fps, spec.frame_size,
                                               window_center=spec.window_center, interp=spec.resize_interp)
                clip = (clip - mean) / std
                probs.append(float(eng._video_fake_prob(model, spec, clip).item())); labels.append(y)
            except Exception:
                pass
        probs = np.array(probs); labels = np.array(labels)
        pred = (probs > 0.5).astype(int)
        racc = (pred[labels == 0] == 0).mean() if (labels == 0).any() else float("nan")
        facc = (pred[labels == 1] == 1).mean() if (labels == 1).any() else float("nan")
        # AUROC with fake as positive (label 1 = fake, score = P(fake))
        a = auroc((labels == 1).astype(int), probs)
        acc = (pred == labels).mean() * 100
        # best-threshold balanced accuracy (what "Auto-calibrate" finds, fit IN-SAMPLE)
        best_bacc, best_t = 0.0, 0.5
        for t in np.linspace(0.01, 0.99, 99):
            p2 = (probs > t).astype(int)
            r = (p2[labels == 0] == 0).mean() if (labels == 0).any() else 0
            f = (p2[labels == 1] == 1).mean() if (labels == 1).any() else 0
            if (r + f) / 2 > best_bacc:
                best_bacc, best_t = (r + f) / 2, t
        print(f"{split:5s} (n={len(labels)}): acc@0.5={acc:.1f}  real_acc={racc*100:.1f}  fake_acc={facc*100:.1f}  "
              f"AUROC={a*100:.1f}  | best-threshold(t={best_t:.2f}) balanced_acc={best_bacc*100:.1f}", flush=True)


if __name__ == "__main__":
    main()
