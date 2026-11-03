"""MoE-JEPA's official eval preprocessing: the V-JEPA 2 video processor on one image.

The official ``preprocess_image`` runs ``AutoVideoProcessor`` (facebook/vjepa2-vitl-fpc64-256: shortest edge to 292,
bilinear; center crop 256; ImageNet normalization) on the RGB image and repeats the frame into a clip; the model
(``SynthScopeMoEJEPAModel.clip``) does the repeating here, so the transform returns one [3, 256, 256] frame. The
processor itself is called, so the pixels are the official ones exactly.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoVideoProcessor

from models.image.moejepa.moejepa import BACKBONE


class VJEPA2ImageTransform:
    """PIL image -> [3, 256, 256] float tensor, exactly as the official ``preprocess_image`` (one frame)."""

    def __init__(self, backbone: str = BACKBONE):
        self.backbone = backbone
        self._processor = None                      # loaded on first use (downloads the small processor config)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self._processor is None:
            self._processor = AutoVideoProcessor.from_pretrained(self.backbone)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self._processor(img, return_tensors="pt")["pixel_values_videos"][0, 0]

    def __repr__(self) -> str:
        return f"VJEPA2ImageTransform({self.backbone})"
