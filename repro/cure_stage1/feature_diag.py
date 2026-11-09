"""CuRe stage-1 investigation (Task 1, diagnostics 1 and 2): encode one fixed set of images with three frozen encoders
(PE Core base without LoRA, the released stage-1 LoRA, our trained stage-1 LoRA) and save the 1024-d features. The
analysis (fresh ratio heads, linear probes, detector scores, drift) is ``feature_analysis.py`` on the saved file.

Image sets (all labelled; none is used to tune or select anything):
  train_clean  GenImage SD v1.4 train subset, clean endpoints (the official test-time pipeline)       -> probe fitting
  test_clean   GenImage SD v1.4 test subset, clean endpoints                                          -> held-out val
  test_pert    the first ``--n-pert`` reals and fakes of test_clean under one perturbation each (JPEG q50, Gaussian
               blur sigma 1 / 2, bicubic x0.5), applied to the format-aligned image BEFORE the pipeline, as the paper's
               robustness protocol ("before method-specific preprocessing")
  mix_train / mix_val  pixel mixtures x_a = (1 - a) x_r + a x_g, a ~ U(0, 1), of random real / generated pairs (train
               subset / test subset), each pair in three preprocessing variants:
                 after   mix the two preprocessed endpoints after the final JPEG q70 (our stage-1 recipe)
                 before  mix the resized uint8 endpoints, round, then JPEG q70 on the mixture, then normalise
                 noq70   mix the resized endpoints, no JPEG q70 at all
  rf_grid_after / rf_grid_before  ``--n-rf-grid`` real->generated trajectories on the stage-2 grid {0, .2, ..., 1}
               (mixing after / before q70): the stage-2 mean mu of each variant is compared with the released mu
  wild         a small in-the-wild set (WildRF test), a DIAGNOSTIC ONLY, never used for selection
  extra_fake   (``--only-extra``, a separate run -> features_extra.pt) generated images of ``--extra-fake-dir``
               (<generator>/1_fake; Synthbuster, not a CuRe paper benchmark), scored against the test_clean reals
  aug sets     (``--only-aug``, a separate run -> features_aug.pt) test_jpeg95: the n_pert test images JPEG q95 (with
               test_jpeg50 the paper's Fig. 2 "format" groups q95 / q50); mix_val_aug_<k>: the mix_val pairs and ratios
               ("after" variant) with both endpoints perturbed first (k = gray, blur1, jpeg60), to compare how the
               ratio read-out of each encoder degrades under training-style augmentations

The pipeline is the official ``cure/preprocess.py`` (PNG source -> JPEG q96; SmallestMaxSize 390 cubic; CenterCrop 390;
Resize 336 cubic; JPEG q70; mean = std = 0.5), split into its steps here so the mixture can be placed before or after
the q70 step; at start-up the split pipeline is checked bit-exact against ``CuReTransform``.
"""
import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.image.cure.preprocess import CuReTransform, SRC_FORMAT_KEY
from models.image.cure.train_stages import build_model

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}
VARIANTS = ("after", "before", "noq70")
PERTS = ("jpeg50", "blur1", "blur2", "resize05")
GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def load_aligned(path):
    """Decode -> RGB; a PNG source is re-encoded as JPEG q96 (the paper's format alignment / official step 1)."""
    with Image.open(path) as src:
        fmt = src.format
        img = src.convert("RGB")
        img.load()
    if fmt == "PNG":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=96)
        buf.seek(0)
        with Image.open(buf) as dec:
            img = dec.convert("RGB")
            img.load()
    return img


def geom(img):
    """SmallestMaxSize(390, cubic) -> CenterCrop(390) -> Resize(336, cubic): uint8 [336, 336, 3]."""
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    s = 390 / min(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    if (nh, nw) != (h, w):
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top, left = (arr.shape[0] - 390) // 2, (arr.shape[1] - 390) // 2
    arr = np.ascontiguousarray(arr[top:top + 390, left:left + 390])
    return cv2.resize(arr, (336, 336), interpolation=cv2.INTER_CUBIC)


def q70(arr):
    _, enc = cv2.imencode(".jpg", arr, (int(cv2.IMWRITE_JPEG_QUALITY), 70))
    return cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)


def norm(arr):
    return torch.from_numpy((arr.astype(np.float32) - 127.5) / 127.5).permute(2, 0, 1).contiguous()


def perturb(img, kind):
    if kind == "jpeg50":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        with Image.open(buf) as dec:
            out = dec.convert("RGB")
            out.load()
        return out
    if kind in ("blur1", "blur2"):
        sigma = 1.0 if kind == "blur1" else 2.0
        return Image.fromarray(cv2.GaussianBlur(np.asarray(img), (0, 0), sigmaX=sigma, sigmaY=sigma))
    if kind == "jpeg95" or kind == "jpeg60":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(kind[4:]))
        buf.seek(0)
        with Image.open(buf) as dec:
            out = dec.convert("RGB")
            out.load()
        return out
    if kind == "gray":
        return img.convert("L").convert("RGB")
    if kind == "resize05":
        w, h = img.size
        return img.resize((max(1, round(w * 0.5)), max(1, round(h * 0.5))), Image.BICUBIC)
    raise ValueError(kind)


class Jobs(Dataset):
    """``jobs[i]`` is ("clean", path) | ("pert", path, kind) | ("mix", real_path, gen_path, alpha, variant)."""

    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        j = self.jobs[i]
        if j[0] == "clean":
            x = norm(q70(geom(load_aligned(j[1]))))
        elif j[0] == "pert":
            x = norm(q70(geom(perturb(load_aligned(j[1]), j[2]))))
        elif j[0] == "mixp":
            _, pr, pg, a, kind = j
            x = (1 - a) * norm(q70(geom(perturb(load_aligned(pr), kind)))) + a * norm(q70(geom(perturb(load_aligned(pg), kind))))
        else:
            _, pr, pg, a, var = j
            gr, gg = geom(load_aligned(pr)), geom(load_aligned(pg))
            if var == "after":
                x = (1 - a) * norm(q70(gr)) + a * norm(q70(gg))
            elif var == "before":
                m = np.clip(np.rint((1 - a) * gr.astype(np.float32) + a * gg.astype(np.float32)), 0, 255).astype(np.uint8)
                x = norm(q70(m))
            else:
                x = (1 - a) * norm(gr) + a * norm(gg)
        return x, i


def check_pipeline(paths):
    """The split pipeline must equal the repo's CuReTransform (== official preprocess) bit for bit."""
    t = CuReTransform()
    for p in paths:
        with Image.open(p) as src:
            fmt = src.format
            img = src.convert("RGB")
        img.info[SRC_FORMAT_KEY] = fmt
        ref = t(img)
        mine = norm(q70(geom(load_aligned(p))))
        d = float((ref - mine).abs().max())
        if d != 0.0:
            raise RuntimeError(f"pipeline mismatch on {p}: max|diff| {d}")
    return len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True, help="released cure_adapter.pt (LoRA source)")
    ap.add_argument("--ours", required=True, help="our stage-1 model_last.pth")
    ap.add_argument("--train-dir", required=True, help="GenImage SD1.4 train subset with real/ fake/")
    ap.add_argument("--test-dir", required=True, help="GenImage SD1.4 test subset with real/ fake/")
    ap.add_argument("--wild-dir", required=True, help="WildRF test (<platform>/{0_real,1_fake}); diagnostic only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mix-train", type=int, default=4000)
    ap.add_argument("--n-mix-val", type=int, default=2000)
    ap.add_argument("--n-pert", type=int, default=1000)
    ap.add_argument("--n-rf-grid", type=int, default=2000)
    ap.add_argument("--extra-fake-dir", default=None, help="<generator>/1_fake tree (Synthbuster subset)")
    ap.add_argument("--only-extra", action="store_true", help="encode only the extra fakes (features_extra.pt)")
    ap.add_argument("--only-aug", action="store_true", help="encode only the format / augmented-mixture sets (features_aug.pt)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr_r, tr_f = list_images(Path(args.train_dir) / "real"), list_images(Path(args.train_dir) / "fake")
    te_r, te_f = list_images(Path(args.test_dir) / "real"), list_images(Path(args.test_dir) / "fake")
    wild = [(p, 0 if "0_real" in p.parts else 1, p.relative_to(args.wild_dir).parts[0]) for p in list_images(args.wild_dir)]
    print(f"[diag] train {len(tr_r)}+{len(tr_f)} test {len(te_r)}+{len(te_f)} wild {len(wild)}", flush=True)
    print(f"[diag] pipeline check: {check_pipeline(te_r[:6] + te_f[:6] + [w[0] for w in wild[:6]])} images bit-exact", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    meta, jobs = {}, []

    def add(name, js, **cols):
        meta[name] = {"start": len(jobs), "n": len(js), **{k: v for k, v in cols.items()}}
        jobs.extend(js)

    add("train_clean", [("clean", p) for p in tr_r + tr_f], label=[0] * len(tr_r) + [1] * len(tr_f))
    add("test_clean", [("clean", p) for p in te_r + te_f], label=[0] * len(te_r) + [1] * len(te_f))
    pr, pf = te_r[:args.n_pert], te_f[:args.n_pert]
    for k in PERTS:
        add(f"test_{k}", [("pert", p, k) for p in pr + pf], label=[0] * len(pr) + [1] * len(pf))
    for split, (R, F_, n) in {"mix_train": (tr_r, tr_f, args.n_mix_train), "mix_val": (te_r, te_f, args.n_mix_val)}.items():
        ri = torch.randint(len(R), (n,), generator=g).tolist()
        fi = torch.randint(len(F_), (n,), generator=g).tolist()
        al = torch.rand(n, generator=g, dtype=torch.float64).tolist()
        for v in VARIANTS:
            add(f"{split}_{v}", [("mix", R[a], F_[b], c, v) for a, b, c in zip(ri, fi, al)], alpha=al)
    ri = torch.randint(len(tr_r), (args.n_rf_grid,), generator=g).tolist()
    fi = torch.randint(len(tr_f), (args.n_rf_grid,), generator=g).tolist()
    for v in ("after", "before"):
        js, pair, alpha = [], [], []
        for k, (a, b) in enumerate(zip(ri, fi)):
            for al in GRID:
                js.append(("mix", tr_r[a], tr_f[b], al, v))
                pair.append(k)
                alpha.append(al)
        add(f"rf_grid_{v}", js, pair=pair, alpha=alpha)
    add("wild", [("clean", w[0]) for w in wild], label=[w[1] for w in wild], platform=[w[2] for w in wild])
    if args.only_aug:
        mv = jobs[meta["mix_val_after"]["start"]:meta["mix_val_after"]["start"] + meta["mix_val_after"]["n"]]
        mv_alpha = meta["mix_val_after"]["alpha"]
        meta, jobs = {}, []
        add("test_jpeg95", [("pert", p, "jpeg95") for p in pr + pf], label=[0] * len(pr) + [1] * len(pf))
        for k in ("gray", "blur1", "jpeg60"):
            add(f"mix_val_aug_{k}", [("mixp", j[1], j[2], j[3], k) for j in mv], alpha=mv_alpha)
    if args.only_extra:
        meta, jobs = {}, []
        fakes = list_images(args.extra_fake_dir)
        add("extra_fake", [("clean", p) for p in fakes], label=[1] * len(fakes),
            generator=[p.relative_to(args.extra_fake_dir).parts[0] for p in fakes])
    print(f"[diag] {len(jobs)} items: " + ", ".join(f"{k} {v['n']}" for k, v in meta.items()), flush=True)

    dev = "cuda"
    encoders = {"base": build_model(None, dev), "released": build_model(args.released, dev), "ours": build_model(args.ours, dev)}
    feats = {k: torch.zeros(len(jobs), 1024, dtype=torch.float16) for k in encoders}
    loader = DataLoader(Jobs(jobs), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    t0 = time.time()
    with torch.no_grad():
        for step, (x, idx) in enumerate(loader):
            x = x.to(dev, non_blocking=True)
            for k, m in encoders.items():
                feats[k][idx] = m.features(x).half().cpu()
            if step % 50 == 0:
                done = (step + 1) * args.bs
                print(f"[diag] {done}/{len(jobs)} {done / (time.time() - t0):.1f} items/s", flush=True)
    for k in feats:
        if not torch.isfinite(feats[k].float()).all():
            raise RuntimeError(f"non-finite features from {k}")
    torch.save({"features": feats, "meta": meta, "args": {k: v for k, v in vars(args).items()
                                                          if k not in ("released", "ours", "train_dir", "test_dir", "wild_dir", "out",
                                                                       "extra_fake_dir")}},
               out / ("features_extra.pt" if args.only_extra else "features_aug.pt" if args.only_aug else "features.pt"))
    print(f"[diag] saved {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
