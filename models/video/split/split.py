"""SPLIT integrated into SynthScope.

Paper: "SPLIT: Training-Free AI-Generated and Partially Edited Video Detection via Spatial Patch-Level Incoherence and
Temporal Roughness" (Hyun & Kim, ECCV 2026). https://github.com/mldljyh/SPLIT

Training-free. A frozen vision encoder (X-CLIP B/16 by default) turns every frame into patch tokens P[t, n] (CLS
dropped). Two signals are computed on them:

* Two-step Temporal Roughness (TTR), per patch: the path length of its token trajectory with 1-frame steps,
  L1 = sum_t ||P[t+1] - P[t]||, against the path length with 2-frame steps rescaled to the same number of steps,
  L2 = (sum_t ||P[t+2] - P[t]|| / 2) * (T-1) / (T-2); TTR = log2(L1 / L2). Real motion is smooth (TTR near 0); generated
  video jitters from frame to frame, so 1-frame steps are long compared with 2-frame steps. Averaged over patches.
* Spatial patch-level motion incoherence: the motion field M[t] = P[t+1] - P[t] on the patch grid and its spatial
  differences between neighbouring patches; the mean of their L2 norms over time and space, averaged over x and y.

score = TTR ** gamma * incoherence (gamma = 8); higher means more likely generated.

This is SynthScope's own implementation of the paper's equations (written before the official repository was MIT-licensed); it is
checked against the official ``SPLIT_model`` on the same frames (repro/split parity script, identical scores).
The released code reads frames exactly as D3 does (the same ``video2frame`` / ``read_video``), so the registry entry
reuses D3's preprocessing: 16 frames at 8 fps, BGR, 10% crop of the long side, resize 224, ImageNet normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import XCLIPVisionModel

ENCODERS = {"xclip-16": "microsoft/xclip-base-patch16", "xclip-32": "microsoft/xclip-base-patch32"}


class SynthScopeSPLITModel(nn.Module):
    """Training-free SPLIT detector on a frozen X-CLIP vision tower."""

    higher_is_fake = True     # orientation of ``score`` (benchmark/bench_eval_video.py fake recall at fixed FPR)

    def __init__(self, config: DictConfig):
        super().__init__()
        cfg = config.ModelConfig
        self.encoder = XCLIPVisionModel.from_pretrained(ENCODERS[str(cfg.get("encoder", "xclip-16")).lower()])
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.gamma = float(cfg.get("gamma", 8.0))
        self.threshold = float(cfg.get("threshold", 0.0))
        self._noop = nn.Parameter(torch.zeros(1))      # the trainer's AdamW needs one parameter; training is a no-op

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()                            # the encoder stays frozen in eval mode
        return self

    def patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` [B, T, 3, H, W] (normalized) -> patch tokens [B, T, N, D] (CLS dropped)."""
        b, t = frames.shape[:2]
        with torch.no_grad():
            tokens = self.encoder(pixel_values=frames.reshape(b * t, *frames.shape[2:])).last_hidden_state[:, 1:]
        return tokens.reshape(b, t, *tokens.shape[1:]).float()

    @staticmethod
    def temporal_roughness(p: torch.Tensor) -> torch.Tensor:
        """Mean over patches of TTR = log2(L1 / L2); ``p`` [B, T, N, D] -> [B]."""
        t = p.shape[1]
        l1 = (p[:, 1:] - p[:, :-1]).norm(dim=-1).sum(dim=1)                        # [B, N]
        l2 = (p[:, 2:] - p[:, :-2]).norm(dim=-1).sum(dim=1) / 2 * (t - 1) / (t - 2)
        return ((torch.log(l1 + 1e-8) - torch.log(l2 + 1e-8)) / math.log(2)).mean(dim=1)

    @staticmethod
    def motion_incoherence(p: torch.Tensor) -> torch.Tensor:
        """Mean L2 norm of the spatial differences of the motion field on the patch grid; ``p`` [B, T, N, D] -> [B]."""
        b, t, n, d = p.shape
        g = int(round(n ** 0.5))
        m = (p[:, 1:] - p[:, :-1]).reshape(b, t - 1, g, g, d)                        # motion field on the grid
        dx = (m[:, :, :, :-1] - m[:, :, :, 1:]).norm(dim=-1).mean(dim=(1, 2, 3))
        dy = (m[:, :, :-1] - m[:, :, 1:]).norm(dim=-1).mean(dim=(1, 2, 3))
        return (dx + dy) / 2

    def score(self, frames: torch.Tensor) -> torch.Tensor:
        """Per-video SPLIT score [B] (higher = more likely generated)."""
        p = self.patch_tokens(frames)
        return self.temporal_roughness(p) ** self.gamma * self.motion_incoherence(p)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        s = self.score(frames)
        return torch.stack([self.threshold - s, s - self.threshold], dim=1)

    def compute_loss(self, batch):
        frames, _labels, _folder_ids = batch
        return {"loss": (self._noop * 0.0).sum(), "outputs": self.forward(frames)}
