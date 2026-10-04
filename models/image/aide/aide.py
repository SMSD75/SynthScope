"""AIDE: AI-generated Image DEtector with Hybrid Features (Yan et al., ICLR 2025).

Faithful SynthScope wrapper around the GENUINE official AIDE network (``official_arch.AIDE_Model``):
a 5-view ``[B,5,3,256,256]`` model = 4 DCT-selected native-resolution patches (SRM/HPF + two
custom ResNet-50 streams) + the full image (frozen OpenCLIP ConvNeXt-XXL semantic branch).

The DCT patch selection runs at NATIVE resolution inside the preprocessing transform
(``official_arch.AIDEFiveView``), NOT inside ``forward`` on a resized batch — this is the fidelity
detail that matters (resizing to 256 before the DCT costs ~12 mAP on the same weights). Because the
preprocessing emits a fixed ``[5,3,256,256]`` stack per image, batches stack normally.

The official released checkpoints (progan_train / sd14_train / GenImage_train) load key-for-key into
``AIDE_Model``; for training from scratch set ``convnext_pretrained`` to the laion2b soup so the
frozen ConvNeXt trunk has real weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .official_arch import AIDE_Model


class SynthScopeAIDEModel(nn.Module):
    """AIDE adapted to SynthScope's ``compute_loss(batch) -> {loss, outputs}`` API."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        cfg = config.ModelConfig
        pretrained = cfg.get("convnext_pretrained", None)
        if pretrained in ("", "none", "None", False):
            pretrained = None
        self.net = AIDE_Model(convnext_pretrained=pretrained,
                              convnext_model_name=str(cfg.get("convnext_model_name", "convnext_xxlarge")))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(config.TrainerConfig.get("label_smoothing", 0.0))
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 5, 3, 256, 256] from official_arch.AIDEFiveView.
        return self.net(images)

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return {"loss": loss, "outputs": logits}
