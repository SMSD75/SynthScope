"""Why does NPR (official ProGAN weights) score so badly on CommGen15? Is it a bug or a property of NPR?

Checks, all with the same SynthScope NPR detector:
  1. what the images are: format (JPEG/PNG), size, JPEG quality of reals vs fakes in CommGen15;
  2. NPR AUROC under different input handling (the paper-faithful square resize vs native resolution, crop only, ...);
  3. controls: NPR on a ForenSynths subset (must stay high) and on our own PNG FLUX.2 images, plain and JPEG-compressed.
Run: python repro/investigate/npr_commgen15_check.py --commgen /path/CommGen15/test --forensynths /path/CNNDetection/test
"""
import argparse, io, json, os, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark"))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
import synthscope
import bench_eval as be
from interface.backend.registry import IMAGENET_MEAN, IMAGENET_STD

NORM = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
VARIANTS = {
    "paper-faithful: square resize 256": None,                       # the registered transform
    "native resolution (no resize)": (T.Compose([T.ToTensor(), NORM]), 1),
    "center-crop 256, no resize": (T.Compose([T.CenterCrop(256), T.ToTensor(), NORM]), 1),
    "shorter side 256 + crop 224": (T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]), 32),
}


def qual(path):
    """Rough JPEG quality from the quantization table (None for non-JPEG)."""
    im = Image.open(path)
    if im.format != "JPEG" or not getattr(im, "quantization", None):
        return None
    q = np.mean(list(im.quantization[0]))
    return round(float(q), 1)          # mean luma quantization step: ~1-2 = near lossless, ~10+ = visible compression


def stats(paths):
    fmts, sizes, quants = Counter(), [], []
    for p in paths:
        try:
            im = Image.open(p); fmts[im.format] += 1; sizes.append(max(im.size)); quants.append(qual(p))
        except Exception:
            pass
    qs = [q for q in quants if q is not None]
    return {"formats": dict(fmts), "long_side_median": int(np.median(sizes)) if sizes else None,
            "long_side_min_max": [min(sizes), max(sizes)] if sizes else None,
            "mean_jpeg_quant_step": round(float(np.mean(qs)), 2) if qs else None}


def auroc(det, model, spec, reals, fakes, variant):
    tf, bs = (None, 32) if VARIANTS[variant] is None else VARIANTS[variant]
    paths = reals + fakes
    probs = be.score(det, model, spec, paths, bs, transform=tf or spec.transform)
    y = torch.tensor([0] * len(reals) + [1] * len(fakes))[: probs.numel()]
    return float(roc_auc_score(y.numpy(), probs.numpy())) * 100, float(probs[: len(reals)].mean()), float(probs[len(reals):].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commgen", required=True); ap.add_argument("--forensynths"); ap.add_argument("--flux-fakes")
    ap.add_argument("--gens", default="akool,kling,flux,midjourney,sora,nano_banana")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", default=str(ROOT / "repro/investigate/npr_commgen15_check.json"))
    a = ap.parse_args()
    random.seed(0)
    det = synthscope.load("npr"); det._ensure_weights(); spec = det.spec; model = det._engine._get_model(spec)
    out = {"commgen": {}, "controls": {}}

    print("== 1. what the images are (CommGen15)")
    root = Path(a.commgen)
    for gen in a.gens.split(","):
        r = be._collect(root / gen, be.REAL_DIRNAMES); f = be._collect(root / gen, be.FAKE_DIRNAMES)
        random.shuffle(r); random.shuffle(f); r, f = r[: a.n], f[: a.n]
        out["commgen"][gen] = {"real": stats(r), "fake": stats(f), "auroc": {}}
        print(f"  {gen}: real {out['commgen'][gen]['real']}\n  {' ' * len(gen)}  fake {out['commgen'][gen]['fake']}")
        for v in VARIANTS:
            au, mr, mf = auroc(det, model, spec, r, f, v)
            out["commgen"][gen]["auroc"][v] = {"auroc": round(au, 2), "mean_p_real": round(mr, 3), "mean_p_fake": round(mf, 3)}
            print(f"    {v:38s} AUROC={au:6.2f}  mean P(fake): real={mr:.3f} fake={mf:.3f}")

    if a.forensynths:
        print("== 3a. control: ForenSynths (what NPR was trained for)")
        for gen in ("progan", "stylegan2", "cyclegan", "biggan", "gaugan"):
            d = Path(a.forensynths) / gen
            if not d.exists():
                continue
            r = be._collect(d, be.REAL_DIRNAMES); f = be._collect(d, be.FAKE_DIRNAMES); random.shuffle(r); random.shuffle(f)
            au, mr, mf = auroc(det, model, spec, r[:100], f[:100], "paper-faithful: square resize 256")
            out["controls"][f"forensynths/{gen}"] = round(au, 2)
            print(f"    {gen:10s} AUROC={au:6.2f}  mean P(fake): real={mr:.3f} fake={mf:.3f}")

    if a.flux_fakes:
        print("== 3b. control: our FLUX.2 [klein] PNGs vs the same images re-saved as JPEG")
        fk = sorted(Path(a.flux_fakes).glob("*.png"))[:60]; rl = sorted((Path(a.flux_fakes).parent / "real").glob("*.png"))[:60]
        tmp = Path("/tmp/npr_jpeg_ctrl"); tmp.mkdir(exist_ok=True)
        for q in (None, 95, 75):
            def prep(paths, tag):
                outp = []
                for p in paths:
                    if q is None: outp.append(p); continue
                    t = tmp / f"{tag}_{q}_{p.stem}.jpg"; Image.open(p).convert("RGB").save(t, quality=q); outp.append(t)
                return outp
            au, mr, mf = auroc(det, model, spec, prep(rl, "r"), prep(fk, "f"), "paper-faithful: square resize 256")
            out["controls"][f"flux_klein_jpeg_{q}"] = round(au, 2)
            print(f"    {'PNG' if q is None else 'JPEG q=%d' % q:10s} AUROC={au:6.2f}  mean P(fake): real={mr:.3f} fake={mf:.3f}")
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
