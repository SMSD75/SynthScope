"""ReStraV integrated into AGID.

Paper: "AI-Generated Video Detection via Perceptual Straightening" (Internò
et al., 2025). https://github.com/ChristianInterno/ReStraV

ReStraV is built on the *perceptual straightening* hypothesis: in a frozen
self-supervised feature space (DINOv2 ViT-S/14), natural video trajectories are
"straighter" (lower curvature) than AI-generated ones. The detector:

1. Encodes each frame with a frozen DINOv2 ViT-S/14 and forms a per-frame vector
   ``z`` by concatenating the final-block CLS token (384-d) with the flattened
   patch tokens, giving a temporal trajectory ``Z = (z_1, ..., z_T)``.
2. Measures the trajectory geometry:
     - displacement ``dz_i = z_{i+1} - z_i``; stepwise distance ``d_i = ||dz_i||_2``.
     - curvature ``theta_i = arccos(cos(dz_i, dz_{i+1}))`` in degrees (cosine clamped
       to [-1, 1] for numerical safety).
3. Builds a 21-d per-video feature ``y`` = first 7 distances + first 6 curvatures
   + (mean, var, min, max) of all distances + (mean, var, min, max) of all
   curvatures.
4. Trains a small MLP classifier (default 64 -> 32) on ``y`` with cross-entropy.
   The DINOv2 backbone stays frozen; only the MLP is trained.

``compute_loss`` returns ``{"loss": CE(logits, labels), "outputs": logits}`` with
``outputs`` of shape ``[B, 2]`` (col 0 = real, col 1 = fake).

Normalization note
------------------
This model does NOT normalize internally. It expects frames already normalized by the
dataset transform pipeline -- use ``normalize_dinov2`` (ImageNet stats, what DINOv2 was
trained with) in the experiment script / dataset config, consistent with the image
CLIP/DINO detectors. The frozen backbone will silently degrade on un-normalized frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# Direct (un-aggregated) trajectory signals kept in the feature vector y (paper Section 6).
NUM_DIRECT_DISTANCES = 7
NUM_DIRECT_CURVATURES = 6


class AGIDReStraVModel(nn.Module):
    """Perceptual-straightening video detector (ReStraV)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        cfg = config.ModelConfig

        backbone_name = str(cfg.get("backbone_name", "dinov2_vits14"))
        self.frozen = bool(cfg.get("freeze_backbone", True))
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.feature_dim = NUM_DIRECT_DISTANCES + NUM_DIRECT_CURVATURES + 4 + 4  # 21
        hidden = list(cfg.get("mlp_hidden", [64, 32]))
        self.classifier = self._build_mlp([self.feature_dim, *hidden, 2])
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.TrainerConfig.get("label_smoothing", 0.0)
        )

    @staticmethod
    def _build_mlp(dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` [B, T, 3, H, W] (already normalized by the dataset transform
        pipeline) -> trajectory ``Z`` [B, T, D]."""
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:])
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.backbone.forward_features(x)
            cls = out["x_norm_clstoken"]                       # [B*T, 384]
            patch = out["x_norm_patchtokens"]                  # [B*T, P, 384]
        z = torch.cat([cls, patch.reshape(patch.shape[0], -1)], dim=1).float()
        return z.reshape(b, t, -1)

    @staticmethod
    def _moment_stats(values: torch.Tensor) -> torch.Tensor:
        """``[mean, var, min, max]`` over the time axis of ``values`` [B, L] -> [B, 4]."""
        return torch.stack(
            [
                values.mean(dim=1),
                values.var(dim=1, unbiased=False),
                values.amin(dim=1),
                values.amax(dim=1),
            ],
            dim=1,
        )

    def _geometry_features(self, z: torch.Tensor) -> torch.Tensor:
        """Trajectory geometry feature ``y`` [B, 21] from ``Z`` [B, T, D]."""
        dz = z[:, 1:] - z[:, :-1]                              # [B, T-1, D]
        distance = torch.linalg.vector_norm(dz, dim=-1)       # [B, T-1]

        cos = F.cosine_similarity(dz[:, :-1], dz[:, 1:], dim=-1).clamp(-1.0, 1.0)  # [B, T-2]
        curvature = torch.rad2deg(torch.arccos(cos))          # degrees [B, T-2]

        direct_d = distance[:, :NUM_DIRECT_DISTANCES]
        direct_theta = curvature[:, :NUM_DIRECT_CURVATURES]
        return torch.cat(
            [direct_d, direct_theta, self._moment_stats(distance), self._moment_stats(curvature)],
            dim=1,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        z = self._encode(frames)
        y = self._geometry_features(z)
        return self.classifier(y)

    def compute_loss(self, batch):
        frames, labels, _folder_ids = batch
        logits = self.forward(frames)
        loss = self.criterion(logits, labels)
        return {"loss": loss, "outputs": logits}
