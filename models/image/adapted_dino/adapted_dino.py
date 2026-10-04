import torch
import torch.nn as nn
from omegaconf import DictConfig


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


class SynthScopeAdaptedDINOModel(nn.Module):
    def __init__(self, config: DictConfig):
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