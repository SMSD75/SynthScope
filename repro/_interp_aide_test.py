"""Verify the AIDE native-res DCT energy interpretability map (weight-independent transform path)."""
import sys, os, base64, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from interface.backend import engine as E
from interface.backend.registry import get_spec

OUT = os.path.join(os.path.dirname(__file__), "interp_smoketest_out"); os.makedirs(OUT, exist_ok=True)
FAKE = os.path.join(os.path.dirname(__file__), "ideogram_bench", "fake", "000.png")
REAL = os.path.join(os.path.dirname(__file__), "ideogram_bench", "real", "000.png")

eng = E.Engine(device="cpu")
spec = get_spec("aide")
print(f"aide interpret={spec.interpret!r}")
ok = True
for label, p in [("fake", FAKE), ("real", REAL)]:
    if not os.path.exists(p):
        print(f"  {label}: missing, skip"); continue
    pil = Image.open(p).convert("RGB")
    # aide_dct branch uses spec.transform.dct_energy_map(pil); model is unused -> pass None.
    amap = eng._interpret_map(None, spec, None, pil)
    if not amap:
        print(f"  {label}: interpret_map=None (FAIL)"); ok = False; continue
    raw = base64.b64decode(amap.split(",", 1)[1]); png = Image.open(io.BytesIO(raw)); png.load()
    em = spec.transform.dct_energy_map(pil)
    out = os.path.join(OUT, f"aide_{label}.png"); png.save(out)
    print(f"  {label}: energy_map shape={tuple(em.shape)} -> PNG {png.size} {png.mode} {len(raw)}B -> {out}")
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
