import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .official_arch import Wrapper5crops

# The released model's arch string is "timm_c5i504_<timm name>": the timm model with one output, wrapped to five 504 crops.
TIMM_ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"


class SynthScopeBFreeModel(Wrapper5crops):
    """B-Free (Guillaro et al., CVPR 2025): DINOv2 ViT-B/14 with registers, trained end to end on bias-free data
    (self-conditioned inpainted fakes paired with their real images). The official wrapper takes an image at its native
    resolution, cuts five 504x504 crops (center and corners) at the patch-embedding level, replicating the embedding
    grid when the image is smaller, and averages the five logits; it is used unchanged (``official_arch.py``).

    The official ``load_state_dict`` splits a checkpoint into the patch embedding and the rest; SynthScope loads the
    standard nn.Module state dict instead (``registry.remap_bfree`` maps a released checkpoint onto it)."""

    load_state_dict = nn.Module.load_state_dict

    def __init__(self, config: DictConfig):
        mc = config.ModelConfig
        backbone = timm.create_model(mc.get("timm_arch", TIMM_ARCH), num_classes=1,
                                     pretrained=bool(mc.get("pretrained_backbone", False)))
        super().__init__(backbone, int(mc.get("crop_size", 504)))
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return super().forward(images).squeeze(1)

    def compute_loss(self, batch):
        images, labels, _ = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())
        return {"loss": loss, "outputs": torch.stack([-logits, logits], dim=1)}
