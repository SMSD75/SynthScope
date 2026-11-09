"""CuRe stage-1 investigation (Task 1): analysis of the features saved by ``feature_diag.py`` (CPU).

1. Ratio regression with a FRESH linear ratio head per encoder and per mixing variant (after / before / noq70): fit on
   mix_train (BCE with the soft target a, L2 1e-4 on standardised features, fixed a priori), report on mix_val the BCE,
   its excess over the floor (the mean binary entropy of the val ratios, reached by a perfect predictor), MAE and
   Pearson / Spearman correlation. Also our trained stage-1 ratio head as is.
2. Linear-probe separability (real vs generated endpoints): logistic regression (C = 1, standardised; fixed) fitted on
   train_clean, evaluated on test_clean, the perturbed test copies and the wild set (diagnostic only); plus a 5-fold
   CV probe fitted on the wild set itself (intrinsic separability, diagnostic only).
3. Detector scores at the 0.5 threshold: the released detector (released LoRA + its mu / W / head), our detector (our
   LoRA + our stage-2/3 mu / W / head), our stage-1 ratio head used directly (paper Table 4 "Direct"), and a fresh ratio
   head (fitted in 1. on the "after" variant) used directly, for every encoder.
4. Feature geometry: drift from the frozen base (cosine, relative L2), the real-vs-generated mean-difference direction
   (cosine between encoders, Cohen's d along it on test / wild), linear CKA between encoders, and the relative feature
   change under each perturbation.

Nothing here selects or tunes anything; every hyper-parameter is fixed above.
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ENCODERS = ("base", "released", "ours")
VARIANTS = ("after", "before", "noq70")
PERTS = ("jpeg50", "blur1", "blur2", "resize05")


def block(data, name, enc):
    m = data["meta"][name]
    return data["features"][enc][m["start"]:m["start"] + m["n"]].float()


def binary_metrics(score, label, thr=0.5):
    score, label = np.asarray(score, dtype=np.float64), np.asarray(label)
    pred = score >= thr
    real_acc = float((~pred[label == 0]).mean()) if (label == 0).any() else float("nan")
    fake_rec = float(pred[label == 1].mean()) if (label == 1).any() else float("nan")
    out = {"real_acc": real_acc, "fake_recall": fake_rec, "bacc": (real_acc + fake_rec) / 2}
    out["auroc"] = float(roc_auc_score(label, score)) if len(set(label.tolist())) == 2 else float("nan")
    out["mean_score_real"] = float(score[label == 0].mean()) if (label == 0).any() else float("nan")
    out["mean_score_fake"] = float(score[label == 1].mean()) if (label == 1).any() else float("nan")
    return {k: round(v * 100, 2) if k in ("real_acc", "fake_recall", "bacc", "auroc") else round(v, 4) for k, v in out.items()}


def fit_ratio_head(X, a, l2=1e-4, iters=500):
    """Linear head on standardised X, BCE with soft target a; returns (mean, std, w, b)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    Z, a = Z.double(), a.double()
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w + b, a) + 0.5 * l2 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mu.double(), sd.double(), w.detach(), b.detach()


def apply_head(head, X):
    mu, sd, w, b = head
    return torch.sigmoid(((X.double() - mu) / sd) @ w + b)


def ratio_metrics(pred_logit_or_prob, a, is_prob):
    a = a.double()
    if is_prob:
        p = pred_logit_or_prob.double().clamp(1e-7, 1 - 1e-7)
        bce = float(F.binary_cross_entropy(p, a))
    else:
        p = torch.sigmoid(pred_logit_or_prob.double())
        bce = float(F.binary_cross_entropy_with_logits(pred_logit_or_prob.double(), a))
    ac = a.clamp(1e-12, 1 - 1e-12)
    floor = float((-(ac * ac.log() + (1 - ac) * (1 - ac).log())).mean())
    return {"bce": round(bce, 5), "excess_bce": round(bce - floor, 5), "floor": round(floor, 5),
            "mae": round(float((p - a).abs().mean()), 4), "pearson": round(float(pearsonr(p.numpy(), a.numpy())[0]), 4),
            "spearman": round(float(spearmanr(p.numpy(), a.numpy())[0]), 4)}


def linear_cka(X, Y):
    X, Y = X.double() - X.double().mean(0), Y.double() - Y.double().mean(0)
    return float((Y.T @ X).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def cohen_d(proj, label):
    r, f = proj[label == 0], proj[label == 1]
    pooled = math.sqrt((r.var() + f.var()) / 2)
    return float((f.mean() - r.mean()) / pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--released", required=True, help="released cure_adapter.pt (mu / W / head)")
    ap.add_argument("--ours-stage1", required=True, help="our stage-1 model_last.pth (ratio_head)")
    ap.add_argument("--ours-adapter", required=True, help="our cure_trained.pth (mu / W / head)")
    ap.add_argument("--extra", default=None, help="features_extra.pt (extra_fake set, scored against the test_clean reals)")
    ap.add_argument("--aug", default=None, help="features_aug.pt (test_jpeg95 + mix_val_aug_* sets)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = torch.load(args.features, map_location="cpu", weights_only=False)
    meta = data["meta"]
    if args.extra:
        ex = torch.load(args.extra, map_location="cpu", weights_only=False)
        tc = meta["test_clean"]
        n_r = sum(1 for v in tc["label"] if v == 0)
        # extra_vs_real: the test_clean reals followed by the extra fakes (one contiguous block appended to the features)
        reals = {k: data["features"][k][tc["start"]:tc["start"] + n_r] for k in ENCODERS}
        base_n = data["features"]["base"].shape[0]
        for k in ENCODERS:
            data["features"][k] = torch.cat([data["features"][k], reals[k], ex["features"][k]])
        em = ex["meta"]["extra_fake"]
        meta["extra_vs_real"] = {"start": base_n, "n": n_r + em["n"], "label": [0] * n_r + em["label"],
                                 "generator": ["real"] * n_r + em["generator"]}
    if args.aug:
        ag = torch.load(args.aug, map_location="cpu", weights_only=False)
        off = data["features"]["base"].shape[0]
        for k in ENCODERS:
            data["features"][k] = torch.cat([data["features"][k], ag["features"][k]])
        for name, m in ag["meta"].items():
            meta[name] = {**m, "start": m["start"] + off}
    res = {"counts": {k: v["n"] for k, v in meta.items()}}

    # ---- 1. ratio regression
    s1 = torch.load(args.ours_stage1, map_location="cpu", weights_only=True, mmap=True)
    s1 = s1.get("model", s1)
    rw, rb = s1["ratio_head.weight"].float().view(-1), s1["ratio_head.bias"].float().view(())
    ratio, heads = {}, {}
    for enc in ENCODERS:
        for v in VARIANTS:
            Xtr, atr = block(data, f"mix_train_{v}", enc), torch.tensor(meta[f"mix_train_{v}"]["alpha"])
            Xva, ava = block(data, f"mix_val_{v}", enc), torch.tensor(meta[f"mix_val_{v}"]["alpha"])
            head = fit_ratio_head(Xtr, atr)
            heads[(enc, v)] = head
            ratio[f"{enc}/{v}/fresh_head_val"] = ratio_metrics(apply_head(head, Xva), ava, True)
            ratio[f"{enc}/{v}/fresh_head_train"] = ratio_metrics(apply_head(head, Xtr), atr, True)
            if enc == "ours":
                ratio[f"ours/{v}/trained_ratio_head_val"] = ratio_metrics(Xva @ rw + rb, ava, False)
        # cross-variant transfer of the head fitted on "after"
        for v in VARIANTS[1:]:
            Xva, ava = block(data, f"mix_val_{v}", enc), torch.tensor(meta[f"mix_val_{v}"]["alpha"])
            ratio[f"{enc}/head_after_on_{v}_val"] = ratio_metrics(apply_head(heads[(enc, "after")], Xva), ava, True)
    for enc in ENCODERS:
        for k in ("gray", "blur1", "jpeg60"):
            if f"mix_val_aug_{k}" in meta:
                Xa, aa = block(data, f"mix_val_aug_{k}", enc), torch.tensor(meta[f"mix_val_aug_{k}"]["alpha"])
                ratio[f"{enc}/head_after_on_aug_{k}_val"] = ratio_metrics(apply_head(heads[(enc, "after")], Xa), aa, True)
                if enc == "ours":
                    ratio[f"ours/trained_ratio_head_on_aug_{k}_val"] = ratio_metrics(Xa @ rw + rb, aa, False)
    res["ratio_regression"] = ratio

    # ---- 2. linear probes
    EVAL = ["test_clean", "wild"] + [f"test_{k}" for k in PERTS] + (["extra_vs_real"] if "extra_vs_real" in meta else [])
    lab = {n: np.asarray(meta[n]["label"]) for n in ["train_clean"] + EVAL}
    probes = {}
    for enc in ENCODERS:
        sc = StandardScaler().fit(block(data, "train_clean", enc).numpy())
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(block(data, "train_clean", enc).numpy()), lab["train_clean"])
        for n in EVAL:
            probes[f"{enc}/genimage_probe_on_{n}"] = binary_metrics(clf.predict_proba(sc.transform(block(data, n, enc).numpy()))[:, 1], lab[n])
        Xw, yw = block(data, "wild", enc).numpy(), lab["wild"]
        scores = np.zeros(len(yw))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xw, yw):
            s = StandardScaler().fit(Xw[tr])
            scores[te] = LogisticRegression(C=1.0, max_iter=3000).fit(s.transform(Xw[tr]), yw[tr]).predict_proba(s.transform(Xw[te]))[:, 1]
        probes[f"{enc}/wild_5fold_cv_probe"] = binary_metrics(scores, yw)
    res["linear_probes"] = probes

    # ---- 3. detectors
    def subspace_scorer(adapter_path):
        st = torch.load(adapter_path, map_location="cpu", weights_only=True)
        W = st["W"].float()
        W = W if W.shape[0] == 1024 else W.T
        mu, hw, hb = st["mu"].float(), st["head.weight"].float(), st["head.bias"].float()
        return lambda X: torch.softmax(((X - mu) @ W) @ hw.T + hb, dim=1)[:, 1]
    det_rel, det_ours = subspace_scorer(args.released), subspace_scorer(args.ours_adapter)
    dets = {}
    for n in EVAL:
        dets[f"released_detector/{n}"] = binary_metrics(det_rel(block(data, n, "released")).numpy(), lab[n])
        dets[f"ours_detector/{n}"] = binary_metrics(det_ours(block(data, n, "ours")).numpy(), lab[n])
        dets[f"ours_stage1_ratio_head_direct/{n}"] = binary_metrics(torch.sigmoid(block(data, n, "ours") @ rw + rb).numpy(), lab[n])
        for enc in ENCODERS:
            dets[f"{enc}_fresh_ratio_head_after_direct/{n}"] = binary_metrics(apply_head(heads[(enc, "after")], block(data, n, enc)).numpy(), lab[n])
    plat = np.asarray(meta["wild"]["platform"])
    for p in sorted(set(plat.tolist())):
        sel = plat == p
        dets[f"released_detector/wild_{p}"] = binary_metrics(det_rel(block(data, "wild", "released")).numpy()[sel], lab["wild"][sel])
        dets[f"ours_detector/wild_{p}"] = binary_metrics(det_ours(block(data, "wild", "ours")).numpy()[sel], lab["wild"][sel])
    if "extra_vs_real" in meta:
        gen = np.asarray(meta["extra_vs_real"]["generator"])
        y = lab["extra_vs_real"]
        sr = det_rel(block(data, "extra_vs_real", "released")).numpy()
        so = det_ours(block(data, "extra_vs_real", "ours")).numpy()
        for gname in sorted(set(gen.tolist()) - {"real"}):
            sel = (gen == gname) | (gen == "real")
            dets[f"released_detector/extra_{gname}"] = binary_metrics(sr[sel], y[sel])
            dets[f"ours_detector/extra_{gname}"] = binary_metrics(so[sel], y[sel])
    res["detectors"] = dets

    # ---- 4. geometry
    geo = {}
    Tb = block(data, "test_clean", "base")
    for enc in ("released", "ours"):
        T = block(data, "test_clean", enc)
        cos = F.cosine_similarity(T, Tb, dim=1)
        rel = (T - Tb).norm(dim=1) / Tb.norm(dim=1)
        y = lab["test_clean"]
        geo[f"{enc}/drift_from_base"] = {"cos_real": round(float(cos[y == 0].mean()), 4), "cos_fake": round(float(cos[y == 1].mean()), 4),
                                         "relL2_real": round(float(rel[y == 0].mean()), 4), "relL2_fake": round(float(rel[y == 1].mean()), 4)}
    dirs = {}
    for enc in ENCODERS:
        X, y = block(data, "train_clean", enc), lab["train_clean"]
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
        dirs[enc] = d / d.norm()
        geo[f"{enc}/mean_diff_norm_train"] = round(float(d.norm()), 4)
        for n in ["train_clean"] + EVAL:
            geo[f"{enc}/cohen_d_along_train_meandiff/{n}"] = round(cohen_d((block(data, n, enc) @ dirs[enc]).numpy(), lab[n]), 3)
    geo["cos_meandiff_released_ours"] = round(float(dirs["released"] @ dirs["ours"]), 4)
    geo["cos_meandiff_released_base"] = round(float(dirs["released"] @ dirs["base"]), 4)
    geo["cos_meandiff_ours_base"] = round(float(dirs["ours"] @ dirs["base"]), 4)
    for a, b in (("released", "ours"), ("released", "base"), ("ours", "base")):
        geo[f"cka_test_clean/{a}_vs_{b}"] = round(linear_cka(block(data, "test_clean", a), block(data, "test_clean", b)), 4)
        geo[f"cka_wild/{a}_vs_{b}"] = round(linear_cka(block(data, "wild", a), block(data, "wild", b)), 4)
    n_r = sum(1 for v in meta["test_clean"]["label"] if v == 0)
    for k in PERTS:
        npert = meta[f"test_{k}"]["n"] // 2
        idx = list(range(npert)) + [n_r + i for i in range(npert)]
        for enc in ENCODERS:
            C, P = block(data, "test_clean", enc)[idx], block(data, f"test_{k}", enc)
            geo[f"{enc}/rel_feature_change/{k}"] = round(float(((P - C).norm(dim=1) / C.norm(dim=1)).mean()), 4)
    # stage-2 mean mu under mixing after / before q70 vs the released mu (paired: same pairs and grid)
    st = torch.load(args.released, map_location="cpu", weights_only=True)
    mu_rel = st["mu"].double()
    for enc in ENCODERS:
        for v in ("after", "before"):
            X = block(data, f"rf_grid_{v}", enc).double()
            pair = torch.tensor(meta[f"rf_grid_{v}"]["pair"])
            mu = X.mean(0)
            h1, h2 = X[pair % 2 == 0].mean(0), X[pair % 2 == 1].mean(0)
            geo[f"{enc}/stage2_mu_{v}"] = {"rel_dist_to_released_mu_pct": round(float((mu - mu_rel).norm() / mu_rel.norm() * 100), 4),
                                         "cos_to_released_mu": round(float(F.cosine_similarity(mu, mu_rel, dim=0)), 7),
                                         "split_half_rel_dist_pct": round(float((h1 - h2).norm() / mu_rel.norm() * 100), 4)}
        Xa, Xb = block(data, "rf_grid_after", enc).double(), block(data, "rf_grid_before", enc).double()
        geo[f"{enc}/stage2_mu_after_vs_before_rel_dist_pct"] = round(float((Xa.mean(0) - Xb.mean(0)).norm() / mu_rel.norm() * 100), 4)
    # paper Fig. 2 "format" groups: the same test images at JPEG q95 vs q50; how linearly separable is the JPEG quality
    if "test_jpeg95" in meta:
        from sklearn.model_selection import GroupKFold
        n = meta["test_jpeg95"]["n"]
        for enc in ENCODERS:
            X = torch.cat([block(data, "test_jpeg95", enc), block(data, "test_jpeg50", enc)]).numpy()
            y = np.r_[np.zeros(n), np.ones(n)]
            grp = np.r_[np.arange(n), np.arange(n)]
            sc_ = np.zeros(2 * n)
            for tr, te in GroupKFold(5).split(X, y, grp):
                s_ = StandardScaler().fit(X[tr])
                sc_[te] = LogisticRegression(C=1.0, max_iter=3000).fit(s_.transform(X[tr]), y[tr]).predict_proba(s_.transform(X[te]))[:, 1]
            geo[f"{enc}/format_probe_q95_vs_q50_5fold"] = binary_metrics(sc_, y)
            d = X[y == 1].mean(0) - X[y == 0].mean(0)
            rf = block(data, "train_clean", enc)
            lab_tr = np.asarray(meta["train_clean"]["label"])
            dsrc = (rf[lab_tr == 1].mean(0) - rf[lab_tr == 0].mean(0)).numpy()
            geo[f"{enc}/format_shift_norm_over_source_shift_norm"] = round(float(np.linalg.norm(d) / np.linalg.norm(dsrc)), 4)
    res["geometry"] = geo
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
