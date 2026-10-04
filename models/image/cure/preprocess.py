"""CuRe's deterministic eval preprocessing (official ``cure/preprocess.py``), reimplemented with
cv2 + PIL so SynthScope needs no ``albumentations`` dependency.

Official pipeline (albumentations 2.0.8), per image:
  1. decode -> RGB; if the SOURCE FILE is a PNG, re-encode as JPEG q96 and decode again
  2. SmallestMaxSize(390, cubic): scale = 390 / min(H, W); new size = max(1, round(.)) per side
  3. CenterCrop(390)           (offsets (dim - 390) // 2)
  4. Resize(336, cubic)
  5. ImageCompression(q=70)    JPEG round-trip applied DIRECTLY to the RGB array via cv2
     (cv2 treats the array as BGR; the official code does the same, so we must not swap channels)
  6. Normalize(mean=std=0.5)   -> (x - 127.5) / 127.5, float32, CHW

Step 1 depends on the source file's format, which a PIL image loaded + ``convert("RGB")`` has lost.
Loaders therefore record it in ``img.info["synthscope_src_format"]`` (see ``SRC_FORMAT_KEY``); without the
hint the image is treated as non-PNG. Equivalence with the official albumentations pipeline is
checked by ``repro/cure_preproc_equiv.py``: identical up to 1 float32 ulp (max |diff| 5.96e-8, from
the normalize arithmetic) on PNG/JPEG images from 200 px to 4256 px, square and non-square.
"""
import io

import cv2
import numpy as np
import torch
from PIL import Image

INPUT_SIZE = 336
RESIZE_SIZE = 390
PNG_REENCODE_QUALITY = 96
EVAL_JPEG_QUALITY = 70
SRC_FORMAT_KEY = "synthscope_src_format"


def _cubic_resize(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    if arr.shape[:2] == (height, width):
        return arr
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_CUBIC)


class CuReTransform:
    """PIL image -> normalized float32 tensor ``[3, 336, 336]`` (picklable, stateless)."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        source_format = img.info.get(SRC_FORMAT_KEY) or img.format
        image = img.convert("RGB")
        if source_format == "PNG":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=PNG_REENCODE_QUALITY)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert("RGB")
                image.load()
        arr = np.asarray(image)
        height, width = arr.shape[:2]
        scale = RESIZE_SIZE / min(height, width)
        arr = _cubic_resize(arr, max(1, round(height * scale)), max(1, round(width * scale)))
        top = (arr.shape[0] - RESIZE_SIZE) // 2
        left = (arr.shape[1] - RESIZE_SIZE) // 2
        arr = np.ascontiguousarray(arr[top:top + RESIZE_SIZE, left:left + RESIZE_SIZE])
        arr = _cubic_resize(arr, INPUT_SIZE, INPUT_SIZE)
        _, encoded = cv2.imencode(".jpg", arr, (int(cv2.IMWRITE_JPEG_QUALITY), EVAL_JPEG_QUALITY))
        arr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        arr = (arr.astype(np.float32) - 127.5) / 127.5
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
