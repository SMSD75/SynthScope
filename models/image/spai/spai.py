from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision.transforms import Normalize

from .vision_transformer import VisionTransformer
from .filters import filter_image_frequencies, generate_circular_mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MFM_CKPT = "/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth"


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class Projector(nn.Module):
    """LayerNorm -> [Linear, GELU/Identity, Dropout] x layers -> LayerNorm."""

    def __init__(self, in_dim, dim, layers=2, last_act=nn.Identity, dropout=0.5, input_norm=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim) if input_norm else nn.Identity()
        seq = [nn.Dropout(dropout)]
        for i in range(layers):
            seq += [nn.Linear(in_dim if i == 0 else dim, dim),
                    nn.GELU() if i < layers - 1 else last_act(),
                    nn.Dropout(dropout)]
        self.proj = nn.Sequential(*seq)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm2(self.proj(self.norm1(x)))


class SpectralContext(nn.Module):
    """SCV: learnable softmax-weighted sum over blocks of per-block (mean, std) token stats."""

    def __init__(self, num_blocks, dim, layers=2, dropout=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, num_blocks, dim))
        self.proj1 = Projector(2 * dim, dim, layers, nn.GELU, dropout, input_norm=False)
        self.proj2 = Projector(dim, dim, layers, nn.GELU, dropout, input_norm=False)

    def forward(self, x):                                       # x: B x N x L x D
        x = self.proj1(torch.cat([x.mean(2), x.std(2)], dim=-1))  # B x N x D
        x = (torch.softmax(self.alpha, dim=1) * x).sum(1)         # B x D
        return self.proj2(x)


class SPAI(nn.Module):
    """Fixed-resolution SPAI (https://arxiv.org/abs/2411.19417): spectral reconstruction
    similarity + spectral context vector over an MFM-pretrained ViT-B/16. Expects images in [0, 1];
    ImageNet normalization happens inside."""

    def __init__(self, config: DictConfig):
        super().__init__()
        cfg = config.ModelConfig
        self.frozen = cfg.freeze_backbone
        self.backbone = self._build_backbone()
        if self.frozen:
            self.backbone.requires_grad_(False)

        n, dim, d = len(self.backbone.intermediate_layers), cfg.projection_dim, self.backbone.embed_dim
        self.proj = nn.ModuleList([Projector(d, dim) for _ in range(n)])
        self.context = SpectralContext(n, dim)
        z_dim, hidden = 6 * n + dim, 3 * (6 * n + dim)
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, 1),
        )
        for mod in (self.proj, self.context, self.classifier):
            mod.apply(_init)

        self.register_buffer("mask", generate_circular_mask(cfg.input_size, cfg.mask_radius))
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_backbone(self):
        vit = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_abs_pos_emb=True, use_mean_pooling=True, use_intermediate_layers=True,
            intermediate_layers=tuple(range(12)), return_features=True,
        )
        sd = torch.load(MFM_CKPT, map_location="cpu")
        sd = sd.get("model", sd)
        msg = vit.load_state_dict(sd, strict=False)
        if msg.unexpected_keys:
            print(f"[SPAI] unexpected checkpoint keys ignored: {msg.unexpected_keys}")
        return vit

    def _branch_features(self, img):                            # img: B x C x H x W in [0, 1]
        with torch.no_grad() if self.frozen else nullcontext():
            f = self.backbone(self.norm(img.clamp(0, 1)))       # B x N x L x 768
        return torch.stack([p(f[:, i]) for i, p in enumerate(self.proj)], dim=1)  # B x N x L x dim

    @staticmethod
    def _srs(a, b):                                             # cosine-similarity token stats
        s = F.cosine_similarity(a, b, dim=-1)                   # B x N x L
        return torch.cat([s.mean(-1), s.std(-1)], dim=1)        # B x 2N

    def forward(self, x):
        low, hi = filter_image_frequencies(x, self.mask)
        orig, low, hi = (self._branch_features(t) for t in (x, low, hi))
        srs = torch.cat([self._srs(orig, low), self._srs(orig, hi), self._srs(low, hi)], dim=1)
        z = torch.cat([srs, self.context(orig)], dim=1)         # B x (6N + dim)
        return self.classifier(z).squeeze(-1)                   # B

    def compute_loss(self, batch):
        images, labels, _ = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels.float())
        return {"loss": loss, "outputs": torch.stack([-logits, logits], dim=1)}
