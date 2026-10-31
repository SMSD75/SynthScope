import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import CLIPVisionModel

from .official_arch import apply_svd_residual_to_self_attn


class SynthScopeEffortModel(nn.Module):
    """Effort (Yan et al., ICML 2025): CLIP ViT-L/14 whose self-attention projections are split by SVD into a frozen
    principal part (rank ``r`` = 1023) and a trainable residual (the remaining singular direction), so fine-tuning only
    moves the minor subspace and keeps the pretrained knowledge; the pooled feature goes to ``Linear(1024, 2)``. Same
    module layout as the official DeepfakeBench ``EffortDetector`` (``backbone`` = the CLIP vision tower, ``head``), so
    its released checkpoints load key for key once the ``module.`` prefix is removed (``registry.remap_effort``)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        mc = config.ModelConfig
        vision = CLIPVisionModel.from_pretrained(mc.get("clip_name", "openai/clip-vit-large-patch14")).vision_model
        self.backbone = apply_svd_residual_to_self_attn(vision, r=int(mc.get("svd_rank", 1023)))
        self.head = nn.Linear(1024, 2)
        self.criterion = nn.CrossEntropyLoss()
        # The principal SVD factors (U_r, S_r, V_r of every projection) are recomputed here from the CLIP weights, as in the
        # official code, and only feed the orthogonality loss in training; the released checkpoints do not store them
        # (the forward pass uses weight_main + the residual). The engine accepts exactly these keys as missing.
        self.pretrained_prefixes = tuple(n for n, _ in self.named_parameters() if n.rsplit(".", 1)[-1] in ("U_r", "S_r", "V_r"))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)["pooler_output"])

    def compute_loss(self, batch):
        images, labels, _ = batch
        logits = self(images)
        return {"loss": self.criterion(logits, labels), "outputs": logits}
