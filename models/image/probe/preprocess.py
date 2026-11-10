"""PROBE's test-time preprocessing; ``pad_image`` is vendored unchanged and ``sliding_patches`` is the
``EvaluateDataset.generate_sliding_patches`` of github.com/Amamiya-C/PROBE-AIGI-Detection (commit b145f71,
``Detector/util.py`` and ``Detector/evaluate_dino.py`` / ``evaluate_resnet.py``; licence unclear: no LICENSE file -
see LICENSES.md).

The official evaluation (``EvaluateDataset.__getitem__`` + ``evaluate``), per image:
  1. ``Image.open(path).convert('RGB')``,
  2. ``data_augment(image, params, mode='fix', crop_size)``: every perturbation probability is 0 at test time (the
     README commands set none), so this is ``Image.fromarray(np.array(image))`` (an identity) followed by
     ``pad_image(image, size=crop_size, mode=pad_mode)``: an image with a side under ``crop_size`` is tiled with copies
     of itself (``'repeat'``, the DINOv2 script) or pasted on a black ``crop_size`` square (``'pad'``, the ResNet-50
     script; a side longer than ``crop_size`` is then cut, offset ``(crop_size - side) // 2``),
  3. ``ToTensor`` + ImageNet ``Normalize`` on the whole image,
  4. non-overlapping ``crop_size`` patches (``unfold`` with stride = size; pixels past the last full patch on the
     right / bottom are dropped) -> ``[N, 3, crop_size, crop_size]``,
  5. the model scores every patch; the image's logit is the mean of its patch logits, P(fake) = sigmoid(mean).

Steps 2 and 4 are the transform steps ``probe_pad_repeat_336`` / ``probe_pad_224`` and ``sliding_patches_336`` /
``sliding_patches_224`` (``transformations.py``); 3 is ``to_dtype`` + ``normalize_imagenet``; 5 is in the model's
``forward``. The patch count depends on the image size, so images are scored one at a time (``native_res``).
"""
from __future__ import annotations

import torch
from PIL import Image


# --- official Detector/util.py, unchanged ----------------------------------------------------------------------------
def pad_image(img, size=224, mode='pad'):
    width, height = img.size

    if width>=size and height>=size:
        return img

    if mode == 'repeat':
        repeat_times_width = max(1, (size + width - 1) // width)
        repeat_times_height = max(1, (size + height - 1) // height)
        new_width = width * repeat_times_width
        new_height = height * repeat_times_height
        new_img = Image.new('RGB', (new_width, new_height))

        for i in range(repeat_times_height):
            for j in range(repeat_times_width):
                new_img.paste(img, (j * width, i * height))
        return new_img
    if mode == 'pad':
        new_img = Image.new('RGB', (size, size), (0,0,0))
        paste_x = (size - width) // 2
        paste_y = (size - height) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img
# ----------------------------------------------------------------------------------------------------------------------


def sliding_patches(img_tensor: torch.Tensor, ps: int) -> torch.Tensor:
    """The official ``generate_sliding_patches`` after its ``self.transform``: [3, H, W] -> [N, 3, ps, ps], row-major
    non-overlapping patches (an image smaller than ``ps`` gives N = 0, as the official code; ``pad_image`` runs first,
    so that does not happen in the pipeline)."""
    C, H, W = img_tensor.shape
    stride = ps

    if H < ps or W < ps:
        return torch.zeros(0, 3, ps, ps)

    # unfold on tensor (much faster than PIL crop)
    patches = img_tensor.unfold(1, ps, stride).unfold(2, ps, stride)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.view(-1, C, ps, ps)

    return patches


class ProbePad:
    """``probe_pad_repeat_<n>`` / ``probe_pad_<n>``: the official ``pad_image`` on an RGB PIL image (PIL out)."""

    def __init__(self, size: int, mode: str):
        if mode not in ("repeat", "pad"):
            raise ValueError(f"pad mode must be 'repeat' or 'pad', not {mode!r}")
        self.size, self.mode = size, mode

    def __call__(self, image: Image.Image) -> Image.Image:
        return pad_image(image.convert("RGB"), size=self.size, mode=self.mode)

    def __repr__(self) -> str:
        return f"ProbePad({self.size}, {self.mode!r})"


class SlidingPatches:
    """``sliding_patches_<n>``: a normalized [3, H, W] tensor -> its non-overlapping [N, 3, n, n] patches."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return sliding_patches(torch.as_tensor(image), self.size)

    def __repr__(self) -> str:
        return f"SlidingPatches({self.size})"
