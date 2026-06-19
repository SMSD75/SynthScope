"""Does AGID's 3-ch-internal residual path equal the official 6-ch-transform residual path?

Official (PGC_official eval_dataset): ToTensor -> residual = extractor(x01) -> Normalize /0.5.
AGID (engine): ToTensor -> Normalize(ImageNet) -> [model de-normalizes] -> residual(x01') -> /0.5.
The de-norm round-trip is the only mathematical difference. Quantify it on a real seeingdark crop.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, torchvision.transforms as T
from PIL import Image
from repro.official_evals.pgc_official_eval import QuantizationResidualExtractor

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

M_YUV = QuantizationResidualExtractor()  # identical matrices/extractor

def _residual_from(xp_int):
    """Residual exactly like the extractor but from an explicit [B,3,H,W] *255-scale tensor."""
    import torch as _t
    xp = xp_int.permute(0, 2, 3, 1)
    x_yuv = _t.matmul(xp, M_YUV.M_t.T)
    x_rec = _t.matmul(_t.round(x_yuv), M_YUV.M_t_inv.T)
    return _t.clamp((xp - x_rec).permute(0, 3, 1, 2), -1.0, 1.0)

def agid_residual(x01, snap=False):
    """AGID engine path: normalize, then model de-normalizes back to [0,1] (lossy round-trip).
    snap=True applies the proposed fix: round(x01*255) recovers the exact uint8 pixel grid."""
    x_norm = (x01 - MEAN) / STD
    x01p = (x_norm * STD + MEAN).clamp(0.0, 1.0)   # de-norm round-trip (model does this)
    xp = torch.round(x01p * 255.0) if snap else (x01p * 255.0)
    return _residual_from(xp)

CROP = T.Compose([T.CenterCrop(224), T.ToTensor()])
samples = {
    "seeingdark-real": "/ssdstore/ssalehi/AGID/CNNDetection/test/seeingdark/0_real/10003_00_100_gt.png",
    "san-real":        "/ssdstore/ssalehi/AGID/CNNDetection/test/san/0_real",
}
import glob
for tag, p in samples.items():
    if os.path.isdir(p): p = sorted(glob.glob(p + "/*"))[0]
    x01 = CROP(Image.open(p).convert("RGB")).unsqueeze(0)
    off = QuantizationResidualExtractor()(x01)        # official: residual from exact [0,1]
    cur = agid_residual(x01, snap=False)              # CURRENT AGID: de-norm round-trip
    fix = agid_residual(x01, snap=True)               # PROPOSED FIX: snap to uint8 grid
    def stats(a, b):
        d = (a - b).abs()
        return d.max().item(), d.mean().item(), (a.sign() != b.sign()).float().mean().item()
    cm, cme, cf = stats(off, cur)
    fm, fme, ff = stats(off, fix)
    print(f"{tag}:")
    print(f"   current vs official: max|Δ|={cm:.3e} mean|Δ|={cme:.3e} sign-flip={cf:.3e}")
    print(f"   FIXED   vs official: max|Δ|={fm:.3e} mean|Δ|={fme:.3e} sign-flip={ff:.3e}")
