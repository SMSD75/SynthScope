"""Verify the SPAI per-patch spectral interpretability map end-to-end via agid.predict."""
import sys, os, base64, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
import agid

OUT = os.path.join(os.path.dirname(__file__), "interp_smoketest_out"); os.makedirs(OUT, exist_ok=True)
FAKE = os.path.join(os.path.dirname(__file__), "ideogram_bench", "fake", "000.png")
REAL = os.path.join(os.path.dirname(__file__), "ideogram_bench", "real", "000.png")

dev = "cuda"
try:
    import torch
    if not torch.cuda.is_available(): dev = "cpu"
except Exception:
    dev = "cpu"
det = agid.load("spai", device=dev)
ok = True
for label, p in [("fake", FAKE), ("real", REAL)]:
    if not os.path.exists(p):
        print(f"  {label}: missing, skip"); continue
    res = det.predict(p)
    amap = res.get("interpret_map")
    print(f"spai {label}: verdict={res['verdict']} P(fake)={res['fake_probability']:.3f} "
          f"interpret_kind={res.get('interpret_kind')}")
    if not amap:
        print("   interpret_map=None (FAIL: five-crop fallback or grid too large?)"); ok = False; continue
    raw = base64.b64decode(amap.split(",", 1)[1]); png = Image.open(io.BytesIO(raw)); png.load()
    out = os.path.join(OUT, f"spai_{label}.png"); png.save(out)
    print(f"   OK: valid PNG {png.size} {png.mode}, {len(raw)} bytes -> {out}")
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
