"""LTF's official test-time preprocessing: one global view plus native-resolution 224 tiles.

The official ``prepare_image`` (``official/images.py``) opens the file, converts it to RGB and stacks
``[global] + tiles``, every view through the same CLIP transform (Resize(224, bicubic) on the short side,
CenterCrop(224), ToTensor, CLIP normalization). The global view is the whole image resized; the tiles are 224x224
crops of the original pixels, so the tile itself is not rescaled. Tile starts per axis are 0, 224, 448, ... while a
full block fits, plus one block aligned to the far edge if pixels remain (500 px -> 0, 224, 276); an axis shorter than
224 keeps one crop padded with black. The number of views therefore depends on the image size (images are scored one
at a time).

``LTFViews`` runs exactly that code on an already opened RGB image (the registry / benchmark loaders open it the same
way), so the tensor is the official one bit for bit.
"""

from __future__ import annotations

import torch
from PIL import Image

from models.image.ltf.official.images import crop_image_to_tiles, get_start_positions, transform_image

TILE_SIZE = 224   # CLIP ViT-L/14 input resolution (the official model.input_resolution)


class LTFViews:
    """PIL image -> [1 + tiles, 3, 224, 224] float tensor, exactly as the official ``prepare_image``."""

    def __init__(self, tile_size: int = TILE_SIZE):
        self.tile_size = tile_size
        self.transform = transform_image(tile_size)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        return torch.stack([self.transform(image)]
                           + [self.transform(tile) for tile in crop_image_to_tiles(image, self.tile_size)])

    @staticmethod
    def start_positions(length: int, tile_size: int = TILE_SIZE) -> list[int]:
        """Tile start offsets along one axis (the official ``get_start_positions``)."""
        return get_start_positions(length, tile_size)

    def __repr__(self) -> str:
        return f"LTFViews({self.tile_size})"
