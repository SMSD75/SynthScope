import torch
import torch.nn as nn


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


class AGIDModel(nn.Module):
    def __init__(self, backbone_name="dinov2_vitb14", bottleneck_dim=64):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = self.backbone.embed_dim
        num_blocks = len(self.backbone.blocks)

        self.adapters = nn.ModuleList(
            [Adapter(embed_dim, bottleneck_dim) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)

        for blk, adapter in zip(self.backbone.blocks, self.adapters):
            x = blk(x)
            x = adapter(x)

        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)