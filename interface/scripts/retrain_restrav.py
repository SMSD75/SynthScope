"""Retrain the deployed ReStraV checkpoint with the corrected (official) recipe.

The previous restrav.pth was trained on ImageNet-normalized frames but is *served* raw [0,1]
(registry video_norm = identity) -- a train/serve mismatch that collapsed it (AUROC ~0.50,
real_acc ~2%). This retrains the SynthScope `SynthScopeReStraVModel` head end-to-end on raw [0,1] frames,
so training and serving use the exact same DINOv2 features.

Key fidelity choices:
  * Features come from the model's OWN ``_encode`` + ``_geometry_features`` (so they are
    bit-identical to what the deployed forward computes, incl. the official [mean,min,max,var]
    summary order), with frames decoded by the SAME serving pipeline (media.video_clip_tensor,
    center window + BICUBIC, raw [0,1]).
  * Official training recipe: standardize (BatchNorm1d running stats, train mode), MLP
    21->64->32->1, BCEWithLogits, plain Adam lr 1e-3 (no weight decay), 10 epochs, batch 32.
  * Save the FINAL-epoch state_dict (no val-loss checkpoint selection).

Run: conda run -n SynthScope python interface/scripts/retrain_restrav.py [--n 3000]
"""
import argparse
import glob
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # interface/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root (config_utils)
from backend.registry import get_spec           # noqa: E402
from backend import media                        # noqa: E402
from backend.engine import _build_model          # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_ROOT = "/hddstore/ssalehi/SynthScope/restrav_data/train/genvideo"
OUT_DIR = Path("/hddstore/ssalehi/SynthScope/checkpoints/_train/restrav_fixed")
DEPLOY_LINK = Path("/hddstore/ssalehi/SynthScope/checkpoints/restrav.pth")


def build_model(spec):
    from config_utils import compose_config
    cfg = compose_config(spec.overrides)
    model = _build_model(cfg).to(DEV).float()
    model.eval()                      # frozen backbone stays in eval
    return model


@torch.no_grad()
def extract_features(model, spec, items, tag):
    X, y = [], []
    t0 = time.time()
    for i, (p, lab) in enumerate(items):
        try:
            clip = media.video_clip_tensor(
                p, spec.num_frames, spec.fps, spec.frame_size,
                window_center=spec.window_center, interp=spec.resize_interp,
            )                                                # [T,3,H,W] raw [0,1]
            z = model._encode(clip.unsqueeze(0).to(DEV))     # [1,T,D]
            feat = model._geometry_features(z)[0]            # [21]
            if torch.isfinite(feat).all():
                X.append(feat.cpu()); y.append(lab)
        except Exception as e:
            if i < 5:
                print(f"  skip {Path(p).name}: {e}", flush=True)
        if (i + 1) % 300 == 0:
            print(f"  [{tag}] {i+1}/{len(items)}  ({(time.time()-t0)/(i+1):.2f}s/vid)", flush=True)
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="videos per class")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    spec = get_spec("restrav")
    model = build_model(spec)
    print(f"model built on {DEV}; serving recipe: num_frames={spec.num_frames} "
          f"window_center={spec.window_center} interp={spec.resize_interp} video_norm={spec.video_norm}", flush=True)

    reals = sorted(glob.glob(TRAIN_ROOT + "/0_real/*.mp4"))
    fakes = sorted(glob.glob(TRAIN_ROOT + "/1_fake/*.mp4"))
    random.Random(0).shuffle(reals); random.Random(1).shuffle(fakes)
    # SynthScope label convention: 0 = real, 1 = fake.
    items = [(p, 0) for p in reals[:args.n]] + [(p, 1) for p in fakes[:args.n]]
    random.Random(2).shuffle(items)
    print(f"extracting features for {len(items)} videos (raw [0,1], model._encode)...", flush=True)
    X, y = extract_features(model, spec, items, "train")
    X = X.to(DEV); y = y.to(DEV)
    print(f"features: {tuple(X.shape)}  reals={(y==0).sum().item()} fakes={(y==1).sum().item()}", flush=True)

    # --- train head (official recipe). feature_norm is BatchNorm1d(affine=False): no params,
    #     just running stats accumulated in train mode. classifier holds the trainable weights. ---
    model.feature_norm.train(); model.classifier.train()
    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)   # wd=0 (plain Adam)
    crit = nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), device=DEV)
        tot = 0.0
        for i in range(0, len(X), 32):
            b = perm[i:i+32]
            opt.zero_grad()
            logit = model.classifier(model.feature_norm(X[b])).squeeze(-1)   # fake logit
            loss = crit(logit, y[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"  epoch {ep+1}/{args.epochs}  loss={tot/len(X):.4f}", flush=True)

    # --- save FINAL state_dict (matches what engine._load_checkpoint expects) ---
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "model_final.pth"
    torch.save(model.state_dict(), out)
    print(f"saved {out}", flush=True)

    # quick in-sample sanity (real_acc / fake_acc on the train features)
    with torch.no_grad():
        p_fake = torch.sigmoid(model.classifier(model.feature_norm(X)).squeeze(-1))
        pred = (p_fake > 0.5).float()
        racc = (pred[y == 0] == 0).float().mean().item()
        facc = (pred[y == 1] == 1).float().mean().item()
    print(f"in-sample: real_acc={racc*100:.1f} fake_acc={facc*100:.1f}", flush=True)

    # relink the deployed checkpoint
    if DEPLOY_LINK.is_symlink() or DEPLOY_LINK.exists():
        DEPLOY_LINK.unlink()
    DEPLOY_LINK.symlink_to(out)
    print(f"relinked {DEPLOY_LINK} -> {out}", flush=True)


if __name__ == "__main__":
    main()
