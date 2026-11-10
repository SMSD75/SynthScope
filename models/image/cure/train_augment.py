"""CuRe stage-1 investigation, R3: DDA-style training augmentation for the stage-1 endpoints (an ASSUMPTION under test, not
part of the paper's stated recipe: CuRe, arXiv 2609.14316, Sec. 4.1, follows "the setting of DDA" for format alignment and
states no augmentation).

Per endpoint (a real or a generated image), in this order:
  1. format alignment, as the eval pipeline: a PNG source is re-encoded as JPEG q96 (CuRe Appendix A)
  2. DDA's training augmentations, exactly DDA's ``create_train_transforms`` list (vendored in
     ``models/image/dda/official_train_data.py``) without its last three steps: RandomJPEG q55-100 p 0.15, horizontal and
     vertical flips p 0.5, Gaussian noise (std 55) p 0.1, pepper noise p 0.1, one of {Gaussian blur k3, k5, median 3,
     motion 5} p 0.2, sharpen p 0.1, colour jitter (0.4, 0.4, 0.4, 0.175) p 0.2, grayscale p 0.1
  3. CuRe's fixed geometry with DDA's random crop in place of the centre crop: SmallestMaxSize(390, cubic) ->
     RandomCrop(390) (replaces CenterCrop(390); DDA's ``PadRandomCrop`` never pads here because the short side is 390)
     -> Resize(336, cubic) -> JPEG q70 -> mean = std = 0.5
Steps 2-3 draw from one seed per (pair, epoch) that is SHARED by the two endpoints of a pair, as DDA's
``ComposedTransforms`` replays one RNG state for the real and the fake of a pair: both endpoints get the same decisions
(whether each step applies, flips, blur kind, jitter factors, grayscale, JPEG quality, relative crop position), so the
augmentation cannot vary with the mixing ratio. Unlike DDA's pairs (a real and its same-size reconstruction) CuRe's
endpoints differ in size, so the global RNGs are re-seeded before every step (size-dependent draws such as the noise field
would otherwise desynchronise the later decisions); only the noise / pepper realisations differ between the endpoints. Not used: DDA's extra re-scaled
copies (``build_item``'s 0.2-1.0 / 1.0-3.5 resized samples) and its pixel / frequency blending of reconstructions, which are
separate DDA-specific samples rather than per-image augmentations. Validation, stage 2 and stage 3 stay un-augmented.
With ``augment=False`` the output equals the eval pipeline ``CuReTransform`` bit for bit (tests/test_cure_recipe.py).
"""
import io
import random

import cv2
import numpy as np
import torch
from PIL import Image

from models.image.dda.official_train_data import create_train_transforms
from .preprocess import EVAL_JPEG_QUALITY, INPUT_SIZE, PNG_REENCODE_QUALITY, RESIZE_SIZE, SRC_FORMAT_KEY


def _dda_augmentations():
    """DDA's train transform list without PadRandomCrop / ToTensor / Normalize (parameters unchanged)."""
    steps = create_train_transforms(size=RESIZE_SIZE).transforms
    names = [type(t).__name__ for t in steps[-3:]]
    assert names == ["PadRandomCrop", "ToTensor", "Normalize"], names
    return steps[:-3]


class CuReDDAEndpoint:
    """PIL image (+ source-format hint) and a seed -> normalized float32 tensor [3, 336, 336]."""

    def __init__(self, augment: bool = True):
        self.augment = augment
        self.steps = _dda_augmentations() if augment else []

    def __call__(self, img: Image.Image, seed: int | None = None) -> torch.Tensor:
        source_format = img.info.get(SRC_FORMAT_KEY) or img.format
        image = img.convert("RGB")
        if source_format == "PNG":
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=PNG_REENCODE_QUALITY)
            buf.seek(0)
            with Image.open(buf) as dec:
                image = dec.convert("RGB")
                image.load()
        crop_rng = None
        if self.augment:
            # replay one RNG state per pair (DDA's ComposedTransforms); restore the caller's global states afterwards
            states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            try:
                # re-seed before EVERY step: the two endpoints differ in size, so size-dependent draws (noise, pepper
                # coordinates) would otherwise desynchronise all later decisions; per-step seeding keeps every decision
                # (apply?, which blur, jitter factors, flips, JPEG quality, ...) identical for the pair
                for k, step in enumerate(self.steps):
                    s = (seed * 1_000_003 + k) % 2 ** 31
                    random.seed(s); np.random.seed(s); torch.manual_seed(s)
                    image = step(image)
                image = image.convert("RGB")
            finally:
                random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
            crop_rng = random.Random(seed + 1)
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        s = RESIZE_SIZE / min(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        if (nh, nw) != (h, w):
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_CUBIC)
        if crop_rng is None:
            top, left = (arr.shape[0] - RESIZE_SIZE) // 2, (arr.shape[1] - RESIZE_SIZE) // 2
        else:
            # the same RELATIVE crop position for both endpoints (their sizes differ)
            fy, fx = crop_rng.random(), crop_rng.random()
            top, left = round(fy * (arr.shape[0] - RESIZE_SIZE)), round(fx * (arr.shape[1] - RESIZE_SIZE))
        arr = np.ascontiguousarray(arr[top:top + RESIZE_SIZE, left:left + RESIZE_SIZE])
        if arr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            arr = cv2.resize(arr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        _, enc = cv2.imencode(".jpg", arr, (int(cv2.IMWRITE_JPEG_QUALITY), EVAL_JPEG_QUALITY))
        arr = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy((arr.astype(np.float32) - 127.5) / 127.5).permute(2, 0, 1).contiguous()
