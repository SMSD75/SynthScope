"""One-off check (run in an environment that has vLLM): ``models/vlm/image.py`` ``vllm_rgb`` vs vLLM's own
``ImageMediaIO(image_mode="RGB").load_bytes`` on synthetic files that exercise every branch (EXIF orientations, RGBA,
LA, palette with tRNS, L, CMYK, RGB). Writes repro/vlm_image_vllm_parity.json.

    /path/to/vllm-env/bin/python repro/vlm_image_vllm_parity.py
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import vllm
from PIL import Image
from vllm.multimodal.media.image import ImageMediaIO

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("vlm_image", ROOT / "models" / "vlm" / "image.py")
vlm_image = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vlm_image)


def files() -> dict:
    rng = np.random.RandomState(0)
    rgb = Image.fromarray((rng.rand(37, 53, 3) * 255).astype("uint8"))
    out = {}
    for o in range(1, 9):                                   # every EXIF orientation
        exif = Image.Exif()
        exif[0x0112] = o
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", exif=exif.tobytes(), quality=95)
        out[f"jpeg_exif_orientation_{o}"] = buf.getvalue()
    rgba = Image.fromarray((rng.rand(31, 45, 4) * 255).astype("uint8"), "RGBA")
    la = Image.fromarray((rng.rand(31, 45, 2) * 255).astype("uint8"), "LA")
    pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=16)
    for name, image, fmt, kw in [("png_rgba", rgba, "PNG", {}), ("png_la", la, "PNG", {}),
                                 ("png_palette_trns", pal, "PNG", {"transparency": 3}), ("png_palette", pal, "PNG", {}),
                                 ("png_l", rgb.convert("L"), "PNG", {}), ("jpeg_cmyk", rgb.convert("CMYK"), "JPEG", {}),
                                 ("png_rgb", rgb, "PNG", {})]:
        buf = io.BytesIO()
        image.save(buf, fmt, **kw)
        out[name] = buf.getvalue()
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    rgba.save(buf, "PNG", exif=exif.tobytes())
    out["png_rgba_exif_6"] = buf.getvalue()
    return out


def main() -> int:
    io_ = ImageMediaIO(image_mode="RGB")
    rows = {}
    for name, data in files().items():
        theirs = io_.load_bytes(data).media
        ours = vlm_image.vllm_rgb(Image.open(io.BytesIO(data)))
        a, b = np.asarray(theirs), np.asarray(ours)
        rows[name] = {"mode": Image.open(io.BytesIO(data)).mode, "vllm_size": list(theirs.size),
                      "ours_size": list(ours.size), "identical": theirs.mode == ours.mode == "RGB" and a.shape == b.shape
                      and bool((a == b).all())}
    result = {"vllm": vllm.__version__, "all_identical": all(r["identical"] for r in rows.values()), "files": rows}
    Path(ROOT / "repro" / "vlm_image_vllm_parity.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
