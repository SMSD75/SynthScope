# Vendored from github.com/HuiGuanLab/LTF (MIT License, Copyright (c) 2026 HuiGuanLab), commit a12bab9, file ltf/data/images.py.
# Unchanged below this header: the official tiling policy and test-time preprocessing.
"""Original-resolution tiling and the released checkpoint's CLIP preprocessing."""

from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def get_start_positions(length, size, bias=0):
    """Use full non-overlapping blocks, then append one boundary-aligned block.

    Only the final block can overlap its predecessor. Short axes keep the
    original single padded crop; bias translates all starts equally.
    """
    if length <= size:
        return [bias]
    positions = list(range(0, length - size + 1, size))
    if positions[-1] != length - size:
        positions.append(length - size)
    return [position + bias for position in positions]


def tile_boxes(width, height, tile_size=224):
    return [
        (x, y, x + tile_size, y + tile_size)
        for y in get_start_positions(height, tile_size)
        for x in get_start_positions(width, tile_size)
    ]


def crop_image_to_tiles(image, tile_size=224):
    return [image.crop(box) for box in tile_boxes(*image.size, tile_size)]


def transform_image(size=224, blur=None, blur_probability=0.0):
    operations = [
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
    ]
    if blur is not None and blur_probability:
        operations.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=blur)], p=blur_probability
            )
        )
    operations += [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
    return transforms.Compose(operations)


def prepare_image(path, tile_size=224):
    with Image.open(Path(path).expanduser()) as original:
        image = original.convert("RGB")
    transform = transform_image(tile_size)
    return torch.stack(
        [transform(image)]
        + [transform(tile) for tile in crop_image_to_tiles(image, tile_size)]
    )


def image_paths(root):
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
