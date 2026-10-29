"""OmniAID (Decoupling Semantics and Artifacts for Universal AI-Generated Image Detection in the Wild, arXiv 2511.08423).

Official repo: github.com/yunncheng/OmniAID (MIT). A CLIP ViT-L/14@336 whose attention projections are split by SVD into
a shared main subspace plus small expert subspaces (a hybrid mixture of experts):

    route   = frozen CLIP vision tower (pooler output) → router → top-k SEMANTIC experts (e.g. human/animal, object/scene)
    tokens  = MoE ViT: every layer uses the routed semantic experts + the always-on ARTIFACT expert
    logits  = Linear(ln_post(CLS)) → [B, 2];  P(fake) = softmax(logits)[:, 1]   (the official ``prob``)

``SynthScopeOmniAIDModel`` *is* the official ``OmniAID`` class (vendored unchanged in ``official_arch.py``); the released
checkpoints hold the full model and load key-for-key with ``strict=True`` as in the official eval. The constructor, as in
the official code, first builds the experts from the CLIP weights (``openai/clip-vit-large-patch14-336``), which the
checkpoint then overwrites. Each release has its own expert configuration (number of experts, rank, router width,
top-k), set per checkpoint in the model configs.

Training: the official recipe trains in two stages (each semantic expert with hard routing, then the router with the
artifact expert fixed) with orthogonality, load-balancing and gating losses; ``compute_loss`` here is only the
classification loss and does NOT reproduce that recipe.
"""
import types

import torch.nn as nn
from omegaconf import DictConfig

from .official_arch import OmniAID

CONFIG_KEYS = ("CLIP_path", "num_experts", "rank_per_expert", "moe_lambda_orth", "moe_lambda_balance",
               "moe_lambda_gating_cls", "moe_router_hidden_dim", "moe_top_k", "dropout",
               "gradient_checkpointing_enable", "is_hybrid")


class SynthScopeOmniAIDModel(OmniAID):
    """The official OmniAID model, configured from SynthScope's ModelConfig."""

    def __init__(self, config: DictConfig):
        mc = config.ModelConfig
        super().__init__(types.SimpleNamespace(**{k: mc[k] for k in CONFIG_KEYS}))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images):
        return super().forward(images)["cls"]

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        logits = self.forward(images)
        return {"loss": self.criterion(logits, labels), "outputs": logits}
