import os
import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.image.adapted_dino.adapted_dino import Adapter

# DINOv3 local repo + gated weights dir (env-overridable). The weights were downloaded to
# /hddstore on cn023 (the original /nvmestore path is not mounted here).
_DINOV3_REPO = os.environ.get("DINO_V3_REPO_DIR", "/home/ssalehi/.cache/torch/hub/facebookresearch_dinov3/")
_DINOV3_WEIGHTS_DIR = os.environ.get("DINO_V3_WEIGHTS_DIR", "/hddstore/ssalehi/SynthScope/dinov3_checkpoints")


class SynthScopeDINOModel(nn.Module):
    MODEL_DICT = {
        "dinov2": lambda backbone_name: torch.hub.load("facebookresearch/dinov2", backbone_name),
        "adapted_dino": lambda backbone_name: torch.hub.load("facebookresearch/dinov2", backbone_name),
        "dinov3": lambda backbone_name: torch.hub.load(_DINOV3_REPO, source="local", model=backbone_name, weights=f"{_DINOV3_WEIGHTS_DIR}/{backbone_name}_pretrain_lvd1689m-8aa4cbdd.pth"),
    }
    def __init__(self, config: DictConfig):
        super().__init__()
        self.backbone = self.MODEL_DICT[config.ModelConfig.model_type](config.ModelConfig.backbone_name)
        if config.ModelConfig.freeze_backbone:
            self._freeze_backbone()
        self.config = config
        embed_dim = self.backbone.embed_dim
        num_blocks = len(self.backbone.blocks)

        if config.ModelConfig.use_adapter:
            self.adapters = nn.ModuleList(
                [Adapter(embed_dim, config.ModelConfig.bottleneck_dim) for _ in range(num_blocks)]
            )
        else:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.head = nn.Linear(embed_dim, 2)
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config.TrainerConfig.get("label_smoothing", 0.0)
        )
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)

        for blk, adapter in zip(self.backbone.blocks, self.adapters):
            x = blk(x)
            x = adapter(x)

        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

    def compute_loss(self, batch):
        images, labels, folder_ids = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        return {"loss": loss, "outputs": outputs}

class SynthScopeDINOv3Model(SynthScopeDINOModel):
    def forward(self, x):
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)

        for blk, adapter in zip(self.backbone.blocks, self.adapters):
            if self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            x = adapter(x)

        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)
    
    def compute_loss(self, batch):
        images, labels, folder_ids = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        return {"loss": loss, "outputs": outputs}