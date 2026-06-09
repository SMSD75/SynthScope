"""AIDE: AI-generated Image DEtector with Hybrid Features (Yan et al., ICLR 2025).

Faithful port of https://github.com/shilinyan99/AIDE adapted to AGID's
``compute_loss(batch) -> {loss, outputs}`` API. The dataset hands the model a
``[B, 3, 224, 224]`` CLIP-normalized tensor (the AGID convention); all
AIDE-specific preprocessing -- DCT scoring, SRM, two-branch ResNet-50, the
frozen ConvNeXt-XXL OpenCLIP trunk -- is owned by the model:

* **Semantic (SFE)**: the CLIP-normalized input goes straight into the frozen
  OpenCLIP ``convnext_xxlarge`` trunk, then mean-pooled and projected to 256-d.
* **Patchwise (PFE)**: undo CLIP normalization back to pixel space, resize to
  256x256, DCT-score 32x32 patches with K=6 band-pass filters weighted by
  ``2^k`` over ``log(|DCT| + 1)``, pick the 2 highest- and 2 lowest-scoring
  patches, upsample each to 256x256, apply 30 fixed SRM kernels and two
  ResNet-50 streams (max-freq pair on ``model_max``, min-freq pair on
  ``model_min``). The mean of the 4 pooled features is the 2048-d artifact
  descriptor.
* **Head**: channel-concat ``[v_sem ; v_art]`` -> Linear -> GELU -> Linear -> 2.
"""

from __future__ import annotations

import math

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision.models import ResNet50_Weights, resnet50

from .srm_kernels import all_normalized_hpf_list

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _HPF(nn.Module):
    """30 fixed SRM high-pass kernels packed into a single ``Conv2d(3 -> 30)``.

    Each output channel applies the same 5x5 kernel to all three input channels
    and sums the responses, matching the official AIDE formulation.
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = []
        for k in all_normalized_hpf_list:
            if k.shape[0] == 3:
                k = np.pad(k, ((1, 1), (1, 1)))
            kernels.append(k)
        weight = torch.from_numpy(np.stack(kernels)).float().view(30, 1, 5, 5).repeat(1, 3, 1, 1)
        self.conv = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DCTPatchSelector(nn.Module):
    """DCT scoring + top/bottom patch picker, batched over ``[B, C, H, W]``.

    Selects, per sample, the two patches with the highest and the two with the
    lowest log-DCT energy weighted across ``K`` band-pass filters with weights
    ``{2^k}``. The level filter from the official module is the all-pass band
    when ``level_fliter=[0]`` (the default), so we skip its DCT/IDCT round-trip
    and pick patches directly in pixel space.
    """

    def __init__(self, window_size: int = 32, stride: int = 16, grade_N: int = 6) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.grade_N = grade_N

        n = window_size
        idx = torch.arange(n, dtype=torch.float32)
        dct = torch.cos((idx[None, :] + 0.5) * math.pi * idx[:, None] / n) * math.sqrt(2.0 / n)
        dct[0] *= math.sqrt(0.5)
        self.register_buffer("dct", dct, persistent=False)
        self.register_buffer("dct_t", dct.t().contiguous(), persistent=False)

        bands = torch.zeros(grade_N, n, n)
        ij = torch.arange(n)[:, None] + torch.arange(n)[None, :]
        for k in range(grade_N):
            lo = 2.0 * n * k / grade_N
            hi = 2.0 * n * (k + 1) / grade_N
            mask = ((ij >= lo) & (ij <= hi)).float()
            denom = mask.sum().clamp(min=1.0)
            bands[k] = mask / denom
        self.register_buffer("bands", bands, persistent=False)
        self.register_buffer(
            "band_weights", torch.tensor([2.0**k for k in range(grade_N)]), persistent=False
        )
        self.unfold = nn.Unfold(kernel_size=window_size, stride=stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, _, _ = x.shape
        ws = self.window_size

        patches_flat = self.unfold(x)
        L = patches_flat.shape[-1]
        patches = patches_flat.transpose(1, 2).reshape(B, L, C, ws, ws)

        x_dct = self.dct @ patches @ self.dct_t
        log_abs = torch.log(x_dct.abs() + 1.0)
        per_band = torch.einsum("blcij,kij->kbl", log_abs, self.bands)
        grade = (self.band_weights.view(-1, 1, 1) * per_band).sum(0)

        idx_sorted = torch.argsort(grade, dim=-1)
        min0 = idx_sorted[:, 0]
        min1 = idx_sorted[:, 1] if L > 1 else idx_sorted[:, 0]
        max0 = idx_sorted[:, -1]
        max1 = idx_sorted[:, -2] if L > 1 else idx_sorted[:, -1]

        batch_idx = torch.arange(B, device=x.device)
        x_minmin = patches[batch_idx, min0]
        x_maxmax = patches[batch_idx, max0]
        x_minmin1 = patches[batch_idx, min1]
        x_maxmax1 = patches[batch_idx, max1]
        return x_minmin, x_maxmax, x_minmin1, x_maxmax1


def _build_pfe_resnet50(pretrained: bool = True) -> nn.Module:
    """ResNet-50 with the first conv replaced to accept 30 SRM channels.

    Returns a 2048-d feature vector for ``[B, 30, H, W]`` inputs.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.fc = nn.Identity()
    return model


class AGIDAIDEModel(nn.Module):
    """AIDE adapted to AGID's ``compute_loss(batch) -> {loss, outputs}`` API."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        cfg = config.ModelConfig

        dct_cfg = cfg.get("dct", {}) or {}
        self.dct_size = int(dct_cfg.get("output_size", 256))
        self.hpf = _HPF()
        self.dct = _DCTPatchSelector(
            window_size=int(dct_cfg.get("window_size", 32)),
            stride=int(dct_cfg.get("stride", 16)),
            grade_N=int(dct_cfg.get("grade_N", 6)),
        )

        resnet_pretrained = bool(cfg.get("resnet_pretrained", True))
        self.model_max = _build_pfe_resnet50(pretrained=resnet_pretrained)
        self.model_min = _build_pfe_resnet50(pretrained=resnet_pretrained)

        clip_model, _, _ = open_clip.create_model_and_transforms(
            str(cfg.get("convnext_model_name", "convnext_xxlarge")),
            pretrained=str(cfg.get("convnext_pretrained", "laion2b_s34b_b82k_augreg_soup")),
        )
        self.convnext = clip_model.visual.trunk
        self.convnext.head.global_pool = nn.Identity()
        self.convnext.head.flatten = nn.Identity()

        self._freeze_convnext = bool(cfg.get("freeze_convnext", True))
        if self._freeze_convnext:
            self.convnext.eval()
            for p in self.convnext.parameters():
                p.requires_grad = False

        sem_dim = int(cfg.get("semantic_dim", 3072))
        proj_dim = int(cfg.get("semantic_proj_dim", 256))
        self.semantic_pool = nn.AdaptiveAvgPool2d(1)
        self.semantic_proj = nn.Linear(sem_dim, proj_dim)

        hidden = int(cfg.get("head_hidden", 1024))
        num_classes = int(cfg.get("num_classes", 2))
        self.head = nn.Sequential(
            nn.Linear(2048 + proj_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(config.TrainerConfig.get("label_smoothing", 0.0))
        )

        self.register_buffer("clip_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1))
        # ImageNet/DINOv2 stats used by the official AIDE for the SRM/HPF patch branch
        # (transform_train = Resize(256) + Normalize(ImageNet)). The semantic branch instead
        # gets CLIP normalization (official applies ImageNet then remaps ImageNet->CLIP, which
        # is algebraically identical to CLIP-normalizing the raw image directly).
        self.register_buffer("imnet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("imnet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # noqa: D401  (override)
        super().train(mode)
        if self._freeze_convnext:
            self.convnext.eval()
        return self

    def _semantic_features(self, x_pixel: torch.Tensor) -> torch.Tensor:
        # Official: tokens are ImageNet-normalized then remapped ImageNet->CLIP inside the model,
        # which equals CLIP-normalizing the raw image directly. ConvNeXt-XXL on 256 -> [B,3072,8,8].
        x_clip = (x_pixel - self.clip_mean) / self.clip_std
        if self._freeze_convnext:
            with torch.no_grad():
                feat = self.convnext(x_clip)
        else:
            feat = self.convnext(x_clip)
        feat = self.semantic_pool(feat).flatten(1)
        return self.semantic_proj(feat)

    def _patchwise_features(self, x_pixel: torch.Tensor) -> torch.Tensor:
        # DCT patch selection on the raw [0,1] image (ToTensor space, matching the official
        # DCT_base_Rec_Module which runs on the un-normalized tensor).
        size = (self.dct_size, self.dct_size)
        x = F.interpolate(x_pixel, size=size, mode="bilinear", align_corners=False)
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(x)

        def _view(p: torch.Tensor) -> torch.Tensor:
            # Official transform_train applied to each patch: Resize(256) + Normalize(ImageNet),
            # then HPF (30 SRM kernels). The patch branch is normalized with ImageNet stats.
            p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            p = (p - self.imnet_mean) / self.imnet_std
            return self.hpf(p)

        f_min = self.model_min(_view(x_minmin))
        f_max = self.model_max(_view(x_maxmax))
        f_min1 = self.model_min(_view(x_minmin1))
        f_max1 = self.model_max(_view(x_maxmax1))
        return (f_min + f_max + f_min1 + f_max1) / 4.0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # ``images`` is a raw [0,1] tensor (the AIDE transform is now Resize(256)+ToTensor, no
        # normalization) so the DCT runs in pixel space and each branch applies its own
        # normalization (CLIP for semantic, ImageNet for the SRM patch branch), as in official AIDE.
        x_pixel = images.clamp(0.0, 1.0)
        v_sem = self._semantic_features(x_pixel)
        v_art = self._patchwise_features(x_pixel)
        return self.head(torch.cat([v_sem, v_art], dim=1))

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return {"loss": loss, "outputs": logits}
