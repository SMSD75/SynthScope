"""D3 integrated into SynthScope.

Paper: "D3: Training-Free AI-Generated Video Detection Using Second-Order
Features" (Zheng et al., ICCV 2025). https://github.com/Zig-HS/D3

D3 is a *training-free* detector. A frozen pre-trained visual encoder turns each
frame into a feature vector (zero-order feature ``F0``). First-order features are
the inter-frame L2 distances (or cosine similarities); second-order features are
the central difference of the first-order features. Real videos exhibit large
volatility in the second-order features while AI-generated videos are flatter, so
the per-video score is the standard deviation of the second-order features:

    F0[b, t]            = encoder(frame_{b,t})                       # [B, T, N]
    F1[b, k]            = ||F0[b, k+1] - F0[b, k]||_2 / dt           # [B, T-1]
    F2[b, k]            = (F1[b, k] - F1[b, k-1]) / dt               # [B, T-2]
    s[b]                = std(F2[b, :])                              # [B]

Real -> high ``s``; AI-generated -> low ``s``. We map ``s`` to two-class outputs
so the SynthScope metrics work:

    outputs = stack([s - tau, tau - s], dim=1)      # col 0 = real, col 1 = fake

so ``softmax(outputs)[:, 1]`` (the fake score) ranks AI-generated videos by *low*
``s``. Because D3 does not train, the threshold-free metrics (AP / AUROC) are the
meaningful ones; argmax accuracy depends on the ``threshold`` (``tau``).

The trainer's ``AdamW`` needs at least one trainable parameter, so the module
carries a single dummy parameter ``_noop`` and returns ``loss = (_noop * 0).sum()``
-- a true no-op with a valid gradient. "Training" has zero effect; evaluation
produces the real result.

Normalization note
------------------
This model does NOT normalize internally. It expects frames already normalized by
the dataset transform pipeline, matched to the chosen ``backbone_name`` -- exactly
like the image CLIP/DINO detectors. Pair the backbone with the right transform in
the experiment script / dataset config: ``normalize_clip`` for the ``clip`` backbone,
``normalize_dinov2`` (ImageNet stats) for ``dinov2`` / ``resnet18``. The frozen encoder
will silently degrade if fed un-normalized (or mismatched) frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel
from torchvision.models import ResNet18_Weights, resnet18


class _ClipEncoder(nn.Module):
    """HF CLIP vision tower -> per-frame pooled CLS feature ``[N, D]`` (matches official D3)."""

    def __init__(self, hf_name: str):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(hf_name)

    def encode_frames(self, x):
        return self.model(pixel_values=x).pooler_output


class _XClipEncoder(nn.Module):
    """HF X-CLIP vision tower -> per-frame pooled feature ``[N, D]`` (the paper's headline encoder).
    Used exactly like CLIP per-frame (official D3 ``D3_model.py`` flattens B*T and takes ``pooler_output``);
    X-CLIP's video-pretrained vision tower gives stronger per-frame features (paper 98.46 vs CLIP ~97)."""

    def __init__(self, hf_name: str):
        super().__init__()
        self.model = XCLIPVisionModel.from_pretrained(hf_name)

    def encode_frames(self, x):
        return self.model(pixel_values=x, output_hidden_states=True).pooler_output


class _DinoEncoder(nn.Module):
    """HF DINOv2 -> per-frame pooled feature ``[N, D]`` (matches official D3)."""

    def __init__(self, hf_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_name)

    def encode_frames(self, x):
        return self.model(pixel_values=x).pooler_output


class _ResnetEncoder(nn.Module):
    """ImageNet ResNet-18 -> global-avg-pooled feature ``[N, 512]``."""

    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        self.model = model

    def encode_frames(self, x):
        return self.model(x)


class SynthScopeD3Model(nn.Module):
    """Training-free second-order video detector (D3)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        cfg = config.ModelConfig

        self.backbone_name = str(cfg.get("backbone_name", "clip")).lower()
        self.first_order = str(cfg.get("first_order", "l2")).lower()
        if self.first_order not in ("l2", "cos"):
            raise ValueError(f"first_order must be 'l2' or 'cos', got {self.first_order!r}")
        self.threshold = float(cfg.get("threshold", 0.0))
        self.dt = float(cfg.get("dt", 1.0))
        self.frozen = bool(cfg.get("freeze_backbone", True))

        self.backbone = self._build_backbone(self.backbone_name)
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # AdamW needs >=1 trainable parameter; this one is a true no-op.
        self._noop = nn.Parameter(torch.zeros(1))

    def _build_backbone(self, name: str):
        """Return the visual-encoder module for ``name``.

        The returned module exposes ``encode_frames(x) -> [N, D]`` for input
        ``x`` of shape ``[N, 3, H, W]`` (already normalized by the dataset transform
        pipeline, matched to this backbone -- see the module docstring).
        """
        if name in ("clip", "clip-16", "clip-b/16", "clip_vit_b16"):
            return _ClipEncoder("openai/clip-vit-base-patch16")
        if name in ("clip-32", "clip-b/32"):
            return _ClipEncoder("openai/clip-vit-base-patch32")
        if name in ("xclip", "xclip-16", "xclip-b/16"):
            return _XClipEncoder("microsoft/xclip-base-patch16")
        if name in ("xclip-32", "xclip-b/32"):
            return _XClipEncoder("microsoft/xclip-base-patch32")
        if name in ("dinov2", "dinov2-base", "dino-base"):
            return _DinoEncoder("facebook/dinov2-base")
        if name in ("dinov2-large", "dino-large"):
            return _DinoEncoder("facebook/dinov2-large")
        if name in ("resnet18", "resnet-18"):
            return _ResnetEncoder()
        raise ValueError(f"Unknown D3 backbone_name: {name!r}")

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` [B, T, 3, H, W] (already normalized by the dataset transform
        pipeline) -> zero-order features ``F0`` [B, T, N]."""
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:])
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feats = self.backbone.encode_frames(x).float()
        return feats.reshape(b, t, -1)

    def _first_order(self, f0: torch.Tensor) -> torch.Tensor:
        """Inter-frame first-order feature ``F1`` [B, T-1]."""
        if self.first_order == "cos":
            f1 = F.cosine_similarity(f0[:, 1:], f0[:, :-1], dim=-1)
        else:
            f1 = torch.linalg.vector_norm(f0[:, 1:] - f0[:, :-1], dim=-1)
        return f1 / self.dt

    def score(self, frames: torch.Tensor) -> torch.Tensor:
        """Per-video second-order volatility score ``s`` [B]."""
        f0 = self._encode(frames)
        f1 = self._first_order(f0)
        f2 = (f1[:, 1:] - f1[:, :-1]) / self.dt          # [B, T-2]
        # Unbiased std over the T-2 second-order values == sqrt(1/(T-3) * sum (.)^2), i.e. Eq. 8.
        return f2.std(dim=1, unbiased=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        s = self.score(frames)
        return torch.stack([s - self.threshold, self.threshold - s], dim=1)

    def compute_loss(self, batch):
        frames, labels, _folder_ids = batch
        outputs = self.forward(frames)
        loss = (self._noop * 0.0).sum()
        return {"loss": loss, "outputs": outputs}
