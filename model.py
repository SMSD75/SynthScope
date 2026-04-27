import torch
import torch.nn as nn
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
import torchvision
from torchvision.models import ResNet50_Weights
import clip

class Adapter(nn.Module):
    """Bottleneck adapter with residual connection and learnable scale."""

    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(x)))


class AGIDDINOModel(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", config.ModelConfig.backbone_name
        )
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


class AGIDCNNModel(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if config.ModelConfig.freeze_backbone:
            self._freeze_backbone()
        self.config = config
        self.head = nn.Linear(self.backbone.fc.in_features, 2)
        self.backbone.fc = nn.Identity()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        return self.head(out)


class AGIDCLIPModel(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        backbone_name = config.ModelConfig.backbone_name
        if backbone_name.startswith("CLIP:"):
            backbone_name = backbone_name.removeprefix("CLIP:")

        self.backbone, _ = clip.load(backbone_name)
        if config.ModelConfig.freeze_backbone:
            self._freeze_backbone()
        self.config = config
        self.head = nn.Linear(self.backbone.visual.output_dim, 2)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        image_features = self.backbone.encode_image(images)
        return self.head(image_features.float())
