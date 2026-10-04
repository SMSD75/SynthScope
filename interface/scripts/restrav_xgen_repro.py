"""Decisive ReStraV cross-generator reproduction.

Question: our README reports a weak/unstable ReStraV "cross-generator" number, and the deployed
restrav.pth collapses to AUROC ~0.50. Does the OFFICIAL ReStraV recipe (github.com/ChristianInterno/
ReStraV) reproduce the paper's cross-generator generalization (Table 3: unseen-generator mAP 97.3,
future/Sora mAP 92.9), or does it also collapse?

This script extracts the official 21-d trajectory-geometry features (verbatim official recipe:
center 2s / 24 frames / BICUBIC-224 / DINOv2 ViT-S/14 / [CLS;patches]) and runs the official MLP
training recipe (21->64->32->1, BCE, Adam 1e-3, 10 epochs, standardized by train mean/std) under
THREE evaluation protocols, to separate *method/code* faithfulness from *protocol* mismatch:

  (A) IN-DISTRIBUTION   official train.py protocol: GenVideo-Val pool, balanced, stratified 50/50.
  (B) UNSEEN-GENERATOR  paper Table-3 protocol: SAME real source, leave-one-fake-generator-out
                        (train on the other generators, test on the held-out one).
  (C) CROSS-DATASET     our README "cross-generator": train on `extracted` (VidProM/DeMamba pool,
                        VSC reals), test on GenVideo-Val (DIFFERENT real source + different fakes).

To settle the normalization question, every video's features are computed TWICE in one decode pass:
  * raw  = official preprocessing (frames/255, NO ImageNet normalization)  <- official recipe
  * norm = ImageNet-normalized (normalize_dinov2)                          <- what restrav.sh trains with

Label convention (official): REAL=1, FAKE=0. AUROC is positive-class-invariant; AP is fake-positive.
"""
import os, sys, glob, re, random, time
os.environ.setdefault("HF_HOME", "/hddstore/ssalehi/SynthScope/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/hddstore/ssalehi/SynthScope/.cache/torch")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import decord
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

GV = "/hddstore/ssalehi/SynthScope/d3_repro/genvideo_val/GenVideo-Val"
EXTRACTED = "/hddstore/ssalehi/SynthScope/d3_repro/extracted"
CACHE = "/hddstore/ssalehi/SynthScope/restrav_xgen_cache"
os.makedirs(CACHE, exist_ok=True)
DEV = "cuda"
T, WIN = 24, 2.0
K_FAKE_PER_GEN = int(os.environ.get("K_FAKE_PER_GEN", "200"))
N_REAL_GV = int(os.environ.get("N_REAL_GV", "2000"))
N_EX = int(os.environ.get("N_EX", "1500"))   # per class for the extracted (cross-dataset) train pool

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)

print("loading DINOv2 ViT-S/14...", flush=True)
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEV)


def decode_clip(path):
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    n = len(vr); fps = float(vr.get_avg_fps()) or 30.0; dur = n / fps
    center = dur / 2.0; start = max(0.0, center - WIN / 2)
    delta = WIN / max(T - 1, 1)
    idxs = [min(int(round((start + i * delta) * fps)), n - 1) for i in range(T)]
    frames = vr.get_batch(idxs).asnumpy()
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


def preprocess(frames):
    frames = frames.float() / 255.0
    return torch.stack([TF.resize(x, [224, 224], InterpolationMode.BICUBIC, antialias=True) for x in frames])


def geom21(Z):
    dz = Z[:, 1:] - Z[:, :-1]; d = dz.norm(dim=-1)
    cos = F.cosine_similarity(dz[:, :-1], dz[:, 1:], dim=-1)
    th = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    def m4(x): return [x.mean(-1), x.amin(-1), x.amax(-1), x.var(-1, unbiased=False)]
    stats = torch.stack(m4(d) + m4(th), dim=1)
    return torch.cat([d[:, :7], th[:, :6], stats], dim=1)[0].cpu().numpy()


@torch.no_grad()
def feats_both(path):
    clip = preprocess(decode_clip(path)).to(DEV)

    def run(x):
        f = dinov2.forward_features(x)
        tokens = torch.cat([f["x_norm_clstoken"].unsqueeze(1), f["x_norm_patchtokens"]], dim=1)
        return geom21(tokens.flatten(1).unsqueeze(0))
    raw = run(clip)
    norm = run((clip - IMNET_MEAN) / IMNET_STD)
    return raw, norm


def build(items, tag):
    """items: list of (path, label, gen). Returns Xraw, Xnorm, y, gens (np arrays), caching to npz."""
    cache = os.path.join(CACHE, f"feat_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[{tag}] loaded {len(z['y'])} cached features", flush=True)
        return z["Xraw"], z["Xnorm"], z["y"], z["gens"]
    Xr, Xn, y, gens = [], [], [], []
    t0 = time.time()
    for i, (p, lab, g) in enumerate(items):
        try:
            r, n = feats_both(p)
            if np.isfinite(r).all() and np.isfinite(n).all():
                Xr.append(r); Xn.append(n); y.append(lab); gens.append(g)
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            print(f"  [{tag}] {i+1}/{len(items)}  ({(time.time()-t0)/(i+1):.2f}s/vid)", flush=True)
    Xr = np.array(Xr, np.float32); Xn = np.array(Xn, np.float32)
    y = np.array(y, np.int64); gens = np.array(gens)
    np.savez(cache, Xraw=Xr, Xnorm=Xn, y=y, gens=gens)
    print(f"[{tag}] extracted+cached {len(y)} features", flush=True)
    return Xr, Xn, y, gens


def train_eval(Xtr, ytr, Xte, yte, seed=0):
    """Official recipe. Label real=1. Returns (AUROC, AP_fake). Averaged over a few seeds outside."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    net = nn.Sequential(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, dtype=torch.float32, device=DEV).unsqueeze(1)
    for ep in range(10):
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), 32):
            b = perm[i:i+32]; opt.zero_grad()
            loss = crit(net(Xt[b]), yt[b]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p_real = torch.sigmoid(net(torch.tensor(Xte, device=DEV))).cpu().numpy().ravel()
    auc = roc_auc_score(yte, p_real)
    ap_fake = average_precision_score(1 - yte, 1 - p_real)
    return auc, ap_fake


def mean_eval(Xtr, ytr, Xte, yte, seeds=(0, 1, 2)):
    rs = [train_eval(Xtr, ytr, Xte, yte, s) for s in seeds]
    auc = np.mean([r[0] for r in rs]); ap = np.mean([r[1] for r in rs])
    return auc, ap


def balance(X, y, rng):
    """Subsample to equal priors (official train.py)."""
    ir = np.where(y == 1)[0]; ifk = np.where(y == 0)[0]
    k = min(len(ir), len(ifk))
    sel = np.concatenate([rng.choice(ir, k, replace=False), rng.choice(ifk, k, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]


def rperm_split(perm):
    h = len(perm) // 2
    return perm[:h], perm[h:]


def main():
    random.seed(0)
    # ---- assemble GenVideo-Val item list (real=1, fake=0), tagged by generator ----
    gvr = sorted(glob.glob(GV + "/Real/*.mp4")); random.Random(0).shuffle(gvr)
    gv_items = [(p, 1, "real") for p in gvr[:N_REAL_GV]]
    for gdir in sorted(glob.glob(GV + "/Fake/*/")):
        g = os.path.basename(gdir.rstrip("/"))
        fk = sorted(glob.glob(gdir + "*.mp4")); random.Random(hash(g) % 100).shuffle(fk)
        gv_items += [(p, 0, g) for p in fk[:K_FAKE_PER_GEN]]
    print(f"GenVideo-Val items: {len(gv_items)}", flush=True)
    Xr_gv, Xn_gv, y_gv, g_gv = build(gv_items, "genvideo")

    # ---- extracted (VidProM/DeMamba) pool for cross-dataset ----
    ex = sorted(glob.glob(EXTRACTED + "/*.mp4"))
    exr = [p for p in ex if re.search(r"_real_", os.path.basename(p))]
    exf = [p for p in ex if not re.search(r"_real_", os.path.basename(p))]
    random.Random(2).shuffle(exr); random.Random(3).shuffle(exf)
    ex_items = [(p, 1, "real") for p in exr[:N_EX]] + [(p, 0, "vidprom") for p in exf[:N_EX]]
    print(f"extracted items: {len(ex_items)}", flush=True)
    Xr_ex, Xn_ex, y_ex, g_ex = build(ex_items, "extracted")

    for variant, Xgv, Xex in [("raw[0,1] (OFFICIAL)", Xr_gv, Xr_ex), ("imagenet-norm (restrav.sh)", Xn_gv, Xn_ex)]:
        print(f"\n{'='*78}\n### FEATURE VARIANT: {variant}\n{'='*78}", flush=True)

        # (A) IN-DISTRIBUTION
        rng = np.random.default_rng(42)
        Xb, yb = balance(Xgv, y_gv, rng)
        Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.5, stratify=yb, random_state=42)
        auc, ap = mean_eval(Xtr, ytr, Xte, yte)
        print(f"(A) IN-DISTRIBUTION (official 50/50, shared gens):   AUROC={auc:.3f}  AP(fake)={ap:.3f}  n_test={len(yte)}", flush=True)

        # (B) UNSEEN-GENERATOR  (leave-one-generator-out, real source fixed)
        print("(B) UNSEEN-GENERATOR (paper Table-3 style, leave-one-gen-out, SAME real source):", flush=True)
        real_idx = np.where(y_gv == 1)[0]
        rng2 = np.random.default_rng(7); rperm = rng2.permutation(real_idx)
        tr_real, te_real = rperm_split(rperm)
        gens = sorted(set(g_gv[y_gv == 0]))
        per = []
        for held in gens:
            tr_f = np.where((y_gv == 0) & (g_gv != held))[0]
            te_f = np.where((y_gv == 0) & (g_gv == held))[0]
            Xtr = np.concatenate([Xgv[tr_real], Xgv[tr_f]]); ytr = np.concatenate([np.ones(len(tr_real)), np.zeros(len(tr_f))]).astype(np.int64)
            Xte = np.concatenate([Xgv[te_real], Xgv[te_f]]); yte = np.concatenate([np.ones(len(te_real)), np.zeros(len(te_f))]).astype(np.int64)
            rngb = np.random.default_rng(1); Xtr, ytr = balance(Xtr, ytr, rngb)
            auc, ap = mean_eval(Xtr, ytr, Xte, yte)
            per.append((held, auc, ap, len(te_f)))
            print(f"      held-out={held:<12} AUROC={auc:.3f}  AP(fake)={ap:.3f}  (n_fake_test={len(te_f)})", flush=True)
        print(f"      >>> MEAN over unseen generators: AUROC={np.mean([p[1] for p in per]):.3f}  AP(fake)={np.mean([p[2] for p in per]):.3f}", flush=True)

        # (C) CROSS-DATASET (our README "cross-generator")
        rngc = np.random.default_rng(5); Xtr, ytr = balance(Xex, y_ex, rngc)
        auc, ap = mean_eval(Xtr, ytr, Xgv, y_gv)
        print(f"(C) CROSS-DATASET (train=extracted/VidProM -> test=GenVideo-Val, real source SHIFTS):  AUROC={auc:.3f}  AP(fake)={ap:.3f}  n_test={len(y_gv)}", flush=True)


if __name__ == "__main__":
    main()
