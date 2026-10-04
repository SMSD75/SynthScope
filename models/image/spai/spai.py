import os
import _synthscope_env  # noqa: F401  (maps legacy AGID_* variables to SYNTHSCOPE_*)
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
# Path to the MFM-pretrained ViT-B/16 backbone weights. Override per-node (e.g. on a non-cn023 node
# with a local copy) via the SYNTHSCOPE_MFM_CKPT env var.
MFM_CKPT = os.environ.get("SYNTHSCOPE_MFM_CKPT", "/hddstore/ssalehi/SynthScope/mfm/mfm_pretrain_vit_base.pth")


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


def patchify_image(img, patch=224, stride=224):
    """Split B x C x H x W into a grid of patches -> B x L x C x patch x patch.

    Ported verbatim from the official SPAI ``spai.models.utils.patchify_image`` (unfold-based).
    A 448x448 input with patch=stride=224 yields the fixed 2x2 grid (L=4) used by the official
    fixed-resolution variant; a 224x224 input yields L=1 (degenerates to the single-patch path)."""
    img = img.unfold(2, patch, stride).unfold(3, patch, stride)      # B x C x nH x nW x patch x patch
    img = img.permute(0, 2, 3, 1, 4, 5).contiguous()                 # B x nH x nW x C x patch x patch
    return img.view(img.size(0), -1, img.size(3), patch, patch)      # B x L x C x patch x patch


class SpectralContextAttention(nn.Module):
    """SCA: cross-attention between a learnable aggregator vector and the per-patch feature vectors,
    aggregating the L grid patches into one descriptor. Faithful port of the official
    ``PatchBasedMFViT.patches_attention`` (plain-torch multi-head, no einops)."""

    def __init__(self, cls_vector_dim, num_heads=12, attn_embed_dim=1536, dropout=0.0):
        super().__init__()
        self.heads = num_heads
        self.dim_head = attn_embed_dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_kv = nn.Linear(cls_vector_dim, attn_embed_dim * 2, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros(num_heads, 1, self.dim_head))
        nn.init.trunc_normal_(self.patch_aggregator, std=0.02)
        self.to_out = nn.Sequential(nn.Linear(attn_embed_dim, cls_vector_dim, bias=False),
                                    nn.Dropout(dropout))

    def _split(self, t):                                             # B x L x (H*dh) -> B x H x L x dh
        b, l, _ = t.shape
        return t.view(b, l, self.heads, self.dim_head).permute(0, 2, 1, 3)

    def forward(self, x):                                            # x: B x L x cls_vector_dim
        b = x.size(0)
        agg = self.patch_aggregator.expand(b, -1, -1, -1)           # B x H x 1 x dh
        k, v = (self._split(t) for t in self.to_kv(x).chunk(2, dim=-1))
        dots = torch.matmul(agg, k.transpose(-1, -2)) * self.scale  # B x H x 1 x L
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)                                 # B x H x 1 x dh
        out = out.permute(0, 2, 1, 3).reshape(b, 1, self.heads * self.dim_head)
        return self.to_out(out).squeeze(1)                          # B x cls_vector_dim


class SPAI(nn.Module):
    """SPAI (https://arxiv.org/abs/2411.19417): spectral reconstruction similarity + spectral
    context over an MFM-pretrained ViT-B/16. Expects images in [0, 1]; ImageNet normalization
    happens inside.

    With ``use_sca`` (default), the input is tiled into a grid of 224x224 patches (a 448 input ->
    the official 2x2 grid) and the per-patch descriptors are aggregated by the Spectral Context
    Attention before classification, matching the official any-/fixed-resolution variant. With
    ``use_sca=False`` it reduces to the previous single-patch fixed-resolution model."""

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

        # Spectral Context Attention over grid patches (the official cross-patch aggregator).
        self.use_sca = bool(cfg.get("use_sca", True))
        self.patch_size = cfg.get("patch_size", cfg.input_size)
        self.patch_stride = cfg.get("patch_stride", self.patch_size)
        # Official any-/native-resolution variant: patchify each image at its native size; if it yields
        # fewer than MINIMUM_PATCHES (4) 224-patches (small images), fall back to a 5-crop (sid.py:198-202).
        self.minimum_patches = int(cfg.get("minimum_patches", 4))
        if self.use_sca:
            self.sca = SpectralContextAttention(
                z_dim, num_heads=cfg.get("num_heads", 12),
                attn_embed_dim=cfg.get("attn_embed_dim", 1536), dropout=cfg.get("sca_dropout", 0.0))
            self.attn_norm = nn.LayerNorm(z_dim)

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, 1),
        )
        mods = [self.proj, self.context, self.classifier]
        if self.use_sca:
            mods += [self.sca, self.attn_norm]
        for mod in mods:
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

    def _patch_z(self, x):                                      # x: B x C x 224 x 224 in [0, 1]
        """Per-patch spectral descriptor z = [SRS, SCV], shape B x (6N + dim)."""
        low, hi = filter_image_frequencies(x, self.mask)
        orig, low, hi = (self._branch_features(t) for t in (x, low, hi))
        srs = torch.cat([self._srs(orig, low), self._srs(orig, hi), self._srs(low, hi)], dim=1)
        return torch.cat([srs, self.context(orig)], dim=1)      # B x (6N + dim)

    def forward(self, x):
        if not self.use_sca:
            return self.classifier(self._patch_z(x)).squeeze(-1)                       # B
        # Native-resolution images smaller than one patch cannot be unfolded (nor five-cropped):
        # upscale the short side up to the patch size, preserving aspect, so >=1 full patch exists.
        h, w = x.shape[-2], x.shape[-1]
        if min(h, w) < self.patch_size:
            s = self.patch_size / min(h, w)
            x = F.interpolate(x, size=(max(self.patch_size, round(h * s)),
                                       max(self.patch_size, round(w * s))),
                              mode="bilinear", align_corners=False)
        patches = patchify_image(x, self.patch_size, self.patch_stride)                # B x L x C x p x p
        if patches.size(1) < self.minimum_patches:                                     # native-res small image
            from torchvision.transforms.functional import five_crop
            crops = five_crop(x, [self.patch_size, self.patch_size])                   # 5 x (B x C x p x p)
            patches = torch.stack(crops, dim=1)                                        # B x 5 x C x p x p
        z = torch.stack([self._patch_z(patches[:, i]) for i in range(patches.size(1))], dim=1)  # B x L x z
        z = self.attn_norm(self.sca(z))                                                # B x z
        return self.classifier(z).squeeze(-1)                                          # B

    @torch.no_grad()
    def patch_fake_map(self, x):
        """Per-patch fake-evidence map ``[nH,nW]`` for interpretability (heatmap only, not scoring).

        Runs the model's *own* classifier on each 224 window's spectral descriptor (attn-normalized
        for consistency with the SCA path) — i.e. which spatial regions look synthetic. Uses a dense
        overlapping grid (stride = patch/2) so a clean heatmap exists for any image (the model's own
        decision uses the stride-224 grid / five-crop; this is purely a denser visualization). Falls
        back to the stride-224 grid if the dense one exceeds ~64 windows; None if even that is too big."""
        if not self.use_sca:
            return None
        h, w = x.shape[-2], x.shape[-1]
        if min(h, w) < self.patch_size:                       # upscale tiny images like forward()
            s = self.patch_size / min(h, w)
            x = F.interpolate(x, size=(max(self.patch_size, round(h * s)),
                                       max(self.patch_size, round(w * s))),
                              mode="bilinear", align_corners=False)
        H, W = x.shape[-2], x.shape[-1]
        for stride in (max(1, self.patch_size // 2), self.patch_size):
            nH = (H - self.patch_size) // stride + 1
            nW = (W - self.patch_size) // stride + 1
            if nH * nW <= 64:
                break
        else:
            return None
        patches = patchify_image(x, self.patch_size, stride)             # B x L x C x p x p
        if patches.size(1) != nH * nW:
            return None
        scores = [self.classifier(self.attn_norm(self._patch_z(patches[:, i]))).squeeze(-1)
                  for i in range(patches.size(1))]
        return torch.stack(scores, dim=1)[0].reshape(nH, nW)            # [nH, nW]

    def compute_loss(self, batch):
        images, labels, _ = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels.float())
        return {"loss": loss, "outputs": torch.stack([-logits, logits], dim=1)}
