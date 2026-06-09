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
   + (mean, min, max, var) of all distances + (mean, min, max, var) of all
   curvatures (this 8-stat order matches the official ``moment4``).
4. Trains a small MLP classifier (default 64 -> 32) on ``y`` with cross-entropy.
   The DINOv2 backbone stays frozen; only the MLP is trained.

``compute_loss`` returns ``{"loss": CE(logits, labels), "outputs": logits}`` with
``outputs`` of shape ``[B, 2]`` (col 0 = real, col 1 = fake).

Normalization note
------------------
This model does NOT normalize internally, and -- unlike the image CLIP/DINO detectors -- it
must be fed **raw [0,1] frames with NO ImageNet normalization**. The official ReStraV
``preprocess`` is exactly ``frames/255`` + BICUBIC resize (no ``Normalize``), so the
experiment script and the deployed serving spec both use raw [0,1] (do NOT add
``normalize_dinov2``). ReStraV's signal is the *relative* geometry of the DINOv2 trajectory,
which the authors compute on un-normalized inputs; training on ImageNet-normalized frames
while serving raw [0,1] (or vice versa) is a train/serve mismatch that destroys transfer.

Fidelity vs the official repo
-----------------------------
This class is a *deployable* re-implementation, faithful to the official recipe except for a
few framework-imposed differences (the bit-exact reproduction is ``interface/scripts/
restrav_repro.py``): standardization uses an in-model ``BatchNorm1d(affine=False)`` (running
stats) rather than fixed ``mean.npy``/``std.npy`` saved from the train set, and the head
emits the AGID 2-column ``[-logit, logit]`` (col 1 is the single BCE logit -- functionally
identical for ranking metrics). Feed it with plain Adam (no weight decay), 10 epochs.
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
        # Standardize the 21-d geometry features before the MLP (official ReStraV does
        # (X-mean)/std with saved train stats; BatchNorm1d(affine=False) is the running-stat
        # equivalent). The 21 features have very different scales (distance norms vs degrees vs
        # variances), so without this the MLP is dominated by large-magnitude features and fails
        # to transfer across generators.
        self.feature_norm = nn.BatchNorm1d(self.feature_dim, affine=False)
        # Official ReStraV head: a single-logit MLP trained with BCE (+ standardized features).
        self.classifier = self._build_mlp([self.feature_dim, *hidden, 1])
        self.criterion = nn.BCEWithLogitsLoss()

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
        """``[mean, min, max, var]`` over the time axis of ``values`` [B, L] -> [B, 4].

        Order matches the official ReStraV ``moment4`` (``dinov2_features.py``):
        ``(mean, amin, amax, var)``. The order is otherwise arbitrary for a from-scratch
        MLP, but keeping it exact lets official feature vectors / weights interoperate.
        """
        return torch.stack(
            [
                values.mean(dim=1),
                values.amin(dim=1),
                values.amax(dim=1),
                values.var(dim=1, unbiased=False),
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
        logit = self.classifier(self.feature_norm(y)).squeeze(-1)   # [B], P(fake) logit
        return torch.stack([-logit, logit], dim=1)                  # [B, 2] (col 1 = fake) for AGID metrics

    def compute_loss(self, batch):
        frames, labels, _folder_ids = batch
        outputs = self.forward(frames)
        loss = self.criterion(outputs[:, 1], labels.float())        # BCE on the fake logit (label 1 = fake)
        return {"loss": loss, "outputs": outputs}
