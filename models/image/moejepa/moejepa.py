"""MoE-JEPA integrated into SynthScope.

Paper: "Unifying Semantic Priors and High-Frequency Traces: Enhancing V-JEPA with Mixture-of-Experts for Robust
Synthetic Image Forensics" (Teglia & Amerini, AI4MFDD workshop @ ECCV 2026; arXiv 2609.16778).
Code: https://github.com/ALCOR-Lab-DIAG/MoE-JEPA (MIT).

A frozen V-JEPA 2 ViT-L video encoder sees the image as a static 4-frame clip. In its last three blocks the MLP becomes
a residual mixture of experts (a frozen shared expert = the pretrained MLP, plus top-2 of 6 trained routed experts);
the patch tokens are pooled by gated-attention MIL. A BayarConv noise branch (constrained high-pass conv + small CNN)
adds a high-frequency feature through a learned per-image gate, and a linear head classifies real / fully synthetic /
tampered.

``SynthScopeMoEJEPAModel`` is the official ``MoEJEPADeepfakeDetector`` without Lightning: the same attribute names
(``model``, ``pooler``, ``noise_stream``, ``noise_proj``, ``ln_semantic``, ``ln_noise``, ``fusion_gate``, ``fc``), so the
released ``moejepa_sidset.safetensors`` loads key for key. The checkpoint holds only the trained tensors; the frozen
V-JEPA 2 weights and the shared experts are rebuilt from ``facebook/vjepa2-vitl-fpc64-256`` (``pretrained_prefixes``).

SynthScope is binary: ``forward`` returns two logits, [real, log(exp(fake) + exp(tampered))], so that
``softmax(.)[:, 1]`` is the official P(fully synthetic) + P(tampered). ``forward_3way`` returns the official three logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel

from models.image.moejepa.official_arch import GatedAttentionPooling, MoEJEPALayer, NoiseStream

BACKBONE = "facebook/vjepa2-vitl-fpc64-256"


class SynthScopeMoEJEPAModel(nn.Module):
    """MoE-JEPA on a frozen V-JEPA 2 ViT-L (hyperparameters of the released SID-Set checkpoint)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        cfg = config.ModelConfig
        self.num_frames = int(cfg.get("num_frames", 4))
        self.model = AutoModel.from_pretrained(str(cfg.get("backbone", BACKBONE)))
        hidden = self.model.config.hidden_size
        intermediate = int(getattr(self.model.config, "intermediate_size",     # 4096 for ViT-L (mlp_ratio 4)
                                   getattr(self.model.config, "mlp_ratio", 4) * hidden))
        for p in self.model.parameters():
            p.requires_grad = False
        # Frozen backbone tensors present before the MoE layers replace the last MLPs (the shared experts copy them).
        frozen = {f"model.{n}" for n, _ in self.model.named_parameters()} | \
                 {f"model.{n}" for n, _ in self.model.named_buffers()}
        for layer in self.model.encoder.layer[-int(cfg.get("moe_layers", 3)):]:
            layer.mlp = MoEJEPALayer(weights=layer.mlp.state_dict(), num_trainable_experts=int(cfg.get("num_experts", 6)),
                                     top_k=int(cfg.get("top_k", 2)),
                                     router_temperature=float(cfg.get("router_temperature", 0.5)),
                                     hidden_dim=hidden, intermediate_dim=intermediate)
        self.pooler = GatedAttentionPooling(in_dim=hidden, hidden_dim=512)
        self.noise_stream = NoiseStream()
        self.noise_proj = nn.Linear(128, hidden)
        self.ln_semantic = nn.LayerNorm(hidden)
        self.ln_noise = nn.LayerNorm(hidden)
        self.fusion_gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())
        self.fc = nn.Linear(hidden, int(cfg.get("out_classes", 3)))
        self.criterion = nn.CrossEntropyLoss()
        # Tensors the released checkpoint does not store: the frozen V-JEPA 2 weights (incl. the shared experts, built
        # from them). Routers and routed experts are trained, so they must load from the checkpoint.
        names = [n for n, _ in self.named_parameters()] + [n for n, _ in self.named_buffers()]
        self.pretrained_prefixes = tuple(n for n in names if n in frozen or ".mlp.shared_expert." in n)

    def clip(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] (one image) -> [B, T, 3, H, W], the static clip of the official ``preprocess_image``."""
        return images.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1) if images.dim() == 4 else images

    def forward_3way(self, images: torch.Tensor) -> torch.Tensor:
        """Official logits [B, 3]: real, fully synthetic, tampered."""
        x = self.clip(images)
        semantic, _ = self.pooler(self.model(x).last_hidden_state[:, 1:, :])
        semantic = self.ln_semantic(semantic)
        noise = self.ln_noise(self.noise_proj(self.noise_stream(x)))
        alpha = self.fusion_gate(torch.cat([semantic, noise], dim=1))
        return self.fc(semantic + alpha * noise)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward_3way(images)
        return torch.stack([logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)], dim=1)

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        outputs = self.forward(images)
        return {"loss": self.criterion(outputs, labels.long()), "outputs": outputs}
