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
        clip_vision = CLIPVisionModel.from_pretrained(mc.get("clip_name", "openai/clip-vit-large-patch14"))
        # transformers <= 5.5.4 wraps the tower as ``.vision_model``; 5.22.2 made CLIPVisionModel the tower itself (the
        # tensor names under it, and the pooled output, are the same either way).
        vision = getattr(clip_vision, "vision_model", clip_vision)
        self.backbone = apply_svd_residual_to_self_attn(vision, r=int(mc.get("svd_rank", 1023)))
        self.head = nn.Linear(1024, 2)
        # Training recipe only (conf/model/image/effort_ufd.yaml): the official UniversalFakeDetect trainer re-draws the
        # classifier weight from N(0, 0.02) (models/trainer.py ``torch.nn.init.normal_(self.model.fc.weight.data, 0.0,
        # opt.init_gain)``); the bias keeps the nn.Linear default, as there. Unset (the served models): default init, and
        # a loaded checkpoint overwrites the head either way.
        if mc.get("head_init_std", None) is not None:
            nn.init.normal_(self.head.weight, 0.0, float(mc.head_init_std))
        # Training recipe only: the official classifier is ``Linear(1024, 1)`` + ``BCEWithLogitsLoss``. With
        # ``head_single_logit`` row 0 of the 2-logit head is pinned at zero (zeroed, and its gradient zeroed by hooks, so
        # AdamW never moves it), the logits are [0, z] and ``CrossEntropyLoss`` equals ``BCEWithLogitsLoss(z)`` exactly
        # (softmax([0, z])[1] = sigmoid(z)); row 1 is the official single logit. The checkpoint keeps the 2-logit format.
        # Default off: the served ``effort`` / ``effort_chameleon`` heads are untouched.
        if mc.get("head_single_logit", False):
            with torch.no_grad():
                self.head.weight[0].zero_()
                self.head.bias[0].zero_()
            self.head.weight.register_hook(self._zero_row0)
            self.head.bias.register_hook(self._zero_row0)
        self.criterion = nn.CrossEntropyLoss()
        # The principal SVD factors (U_r, S_r, V_r of every projection) are recomputed here from the CLIP weights, as in the
        # official code, and only feed the orthogonality loss in training; the released checkpoints do not store them
        # (the forward pass uses weight_main + the residual). The engine accepts exactly these keys as missing.
        self.pretrained_prefixes = tuple(n for n, _ in self.named_parameters() if n.rsplit(".", 1)[-1] in ("U_r", "S_r", "V_r"))

    @staticmethod
    def _zero_row0(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[0] = 0
        return grad

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)["pooler_output"])

    def compute_loss(self, batch):
        images, labels, _ = batch
        logits = self(images)
        return {"loss": self.criterion(logits, labels), "outputs": logits}
