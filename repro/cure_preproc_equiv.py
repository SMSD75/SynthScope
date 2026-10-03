"""Check that AGID's cv2/PIL CuReTransform equals the OFFICIAL CuRe preprocessing.

The official ``cure/preprocess.py`` needs ``albumentations==2.0.8`` (not an AGID dependency). Run
this with the official repo and albumentations on the path, e.g.:

  PYTHONPATH=/path/to/albumentations-2.0.8-target:/path/to/CuRe \\
    python repro/cure_preproc_equiv.py --images <dir-or-files...>

It feeds the same files through the official ``preprocess_image(path)`` and through
``CuReTransform`` (with the source-format hint a loader would set) and reports the max |diff|.
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cure import preprocess_image as official_preprocess  # official CuRe repo (on PYTHONPATH)
from models.image.cure.preprocess import SRC_FORMAT_KEY, CuReTransform

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def agid_preprocess(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        fmt = source.format
        image = source.convert("RGB")
        image.info[SRC_FORMAT_KEY] = fmt
    return CuReTransform()(image)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    paths = []
    for item in args.images:
        p = Path(item)
        paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in EXTS) if p.is_dir() else [p]
    paths = paths[: args.limit]
    worst, n_exact = 0.0, 0
    for path in paths:
        ref, ours = official_preprocess(path), agid_preprocess(path)
        assert ref.shape == ours.shape == (3, 336, 336), (ref.shape, ours.shape)
        diff = (ref - ours).abs().max().item()
        worst = max(worst, diff)
        n_exact += diff == 0.0
        with Image.open(path) as im:
            print(f"{diff:.3e}  {im.format:5s} {im.size}  {path.name}")
    print(f"\n{len(paths)} images | bit-exact: {n_exact} | worst max|diff| = {worst:.3e}")
    sys.exit(0 if worst < 1e-5 else 1)


if __name__ == "__main__":
    main()
