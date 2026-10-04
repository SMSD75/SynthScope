"""Faithful ReStraV reproduction (mirrors github.com/ChristianInterno/ReStraV).

Matches the official pipeline exactly, except torchcodec (unavailable here) is replaced by
decord with identical frame selection:
  * decode: 24 frames over a CENTER 2s window, repeat_last clamp                (dinov2_features.decode_clip)
  * preprocess: float/255 + BICUBIC resize to 224, NO ImageNet normalization    (dinov2_features.preprocess)
  * DINOv2 ViT-S/14 forward_features -> [CLS ; patches] flattened -> Z[T,D]      (extract_dinov2_embeddings)
  * 21-d features: d[:7], theta[:6], [mu_d,min_d,max_d,var_d, mu_t,min_t,max_t,var_t]  (features_from_Z)
  * label REAL=1, FAKE=0                                                         (extract_training_features)
  * standardize with train mean/std; MLP 21->64->32->1, BCEWithLogits, Adam 1e-3, 10 epochs (train.py)
  * report ROC-AUC (and AP for fake=positive)

Runs TWO protocols:
  IN-DISTRIBUTION : GenVideo-Val pool, balanced, random 50/50 stratified split (shared generators)
  CROSS-GENERATOR : train on d3_repro/extracted generators, test on disjoint GenVideo-Val generators
"""
import os, sys, glob, re, random
os.environ.setdefault("HF_HOME", "/hddstore/ssalehi/SynthScope/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/hddstore/ssalehi/SynthScope/.cache/torch")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import decord
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

EXTRACTED = "/hddstore/ssalehi/SynthScope/d3_repro/extracted"
GV = "/hddstore/ssalehi/SynthScope/d3_repro/genvideo_val/GenVideo-Val"
DEV = "cuda"
N = int(os.environ.get("RESTRAV_REPRO_N", "500"))   # per class, per pool
T, WIN = 24, 2.0

dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEV)


def decode_clip(path):
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    n = len(vr); fps = float(vr.get_avg_fps()) or 30.0; dur = n / fps
    center = dur / 2.0; start = max(0.0, center - WIN / 2)
    delta = WIN / max(T - 1, 1)
    idxs = [min(int(round((start + i * delta) * fps)), n - 1) for i in range(T)]  # repeat_last clamp
    frames = vr.get_batch(idxs).asnumpy()                      # [T,H,W,3] uint8
    return torch.from_numpy(frames).permute(0, 3, 1, 2)        # [T,3,H,W]


def preprocess(frames):
    frames = frames.float() / 255.0
    return torch.stack([TF.resize(x, [224, 224], InterpolationMode.BICUBIC, antialias=True) for x in frames])


@torch.no_grad()
def feats_of(path):
    clip = preprocess(decode_clip(path)).to(DEV)
    f = dinov2.forward_features(clip)
    tokens = torch.cat([f["x_norm_clstoken"].unsqueeze(1), f["x_norm_patchtokens"]], dim=1)
    Z = tokens.flatten(1).unsqueeze(0)                         # [1,T,D]
    dz = Z[:, 1:] - Z[:, :-1]; d = dz.norm(dim=-1)
    cos = F.cosine_similarity(dz[:, :-1], dz[:, 1:], dim=-1)
    th = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    def m4(x): return [x.mean(-1), x.amin(-1), x.amax(-1), x.var(-1, unbiased=False)]
    stats = torch.stack(m4(d) + m4(th), dim=1)                 # [1,8] (mu_d,min,max,var, mu_t,min,max,var)
    return torch.cat([d[:, :7], th[:, :6], stats], dim=1)[0].cpu().numpy()  # [21]


def build(items, tag):
    X, y = [], []
    for i, (p, lab) in enumerate(items):
        try:
            f = feats_of(p)
            if np.isfinite(f).all(): X.append(f); y.append(lab)
        except Exception:
            pass
        if (i + 1) % 200 == 0: print(f"  [{tag}] {i+1}/{len(items)}", flush=True)
    return np.array(X, np.float32), np.array(y, np.int64)


def train_eval(Xtr, ytr, Xte, yte, tag):
    # standardize with TRAIN stats (official)
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    net = nn.Sequential(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, dtype=torch.float32, device=DEV).unsqueeze(1)
    for ep in range(10):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 32):
            b = perm[i:i+32]; opt.zero_grad()
            loss = crit(net(Xt[b]), yt[b]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p_real = torch.sigmoid(net(torch.tensor(Xte, device=DEV))).cpu().numpy().ravel()
    # label real=1; AUC on real. For fake-detection AP, positive=fake => score 1-p_real, label (1-yte)
    auc = roc_auc_score(yte, p_real)
    ap_fake = average_precision_score(1 - yte, 1 - p_real)
    print(f"  {tag}:  ROC-AUC={auc:.3f}   AP(fake)={ap_fake:.3f}   (n_test={len(yte)})", flush=True)


def main():
    random.seed(0)
    # GenVideo-Val pool (label real=1, fake=0)
    gvr = sorted(glob.glob(GV + "/Real/*.mp4")); gvf = sorted(glob.glob(GV + "/Fake/*/*.mp4"))
    random.Random(0).shuffle(gvr); random.Random(1).shuffle(gvf)
    print(f"Extracting GenVideo-Val features ({N}/class)...", flush=True)
    Xg, yg = build([(p, 1) for p in gvr[:N]] + [(p, 0) for p in gvf[:N]], "GV")

    print("\n=== IN-DISTRIBUTION (GenVideo-Val, random 50/50 split, shared generators) ===")
    Xtr, Xte, ytr, yte = train_test_split(Xg, yg, test_size=0.5, stratify=yg, random_state=42)
    train_eval(Xtr, ytr, Xte, yte, "in-distribution")

    # extracted pool (label real=1, fake=0)
    ex = sorted(glob.glob(EXTRACTED + "/*.mp4"))
    exr = [p for p in ex if re.search(r"_real_", os.path.basename(p))]
    exf = [p for p in ex if not re.search(r"_real_", os.path.basename(p))]
    random.Random(2).shuffle(exr); random.Random(3).shuffle(exf)
    print(f"\nExtracting train-pool (extracted) features ({N}/class)...", flush=True)
    Xe, ye = build([(p, 1) for p in exr[:N]] + [(p, 0) for p in exf[:N]], "EX")
    print("\n=== CROSS-GENERATOR (train=extracted generators, test=disjoint GenVideo-Val) ===")
    train_eval(Xe, ye, Xg, yg, "cross-generator")


if __name__ == "__main__":
    main()
