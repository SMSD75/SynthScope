"""LTF (Learning to Falsify: Hierarchical Multiple Instance Learning for High Resolution AI-Generated Image Detection,
arXiv 2611.04529).

Official repo: github.com/HuiGuanLab/LTF (MIT). A frozen CLIP ViT-L/14 reads every view of the image (one global view
plus native-resolution 224 tiles, see ``preprocess.py``); its block-24 hidden tokens, before the final LayerNorm and
projection, feed a hierarchical MIL head:

    L1     = gated-attention MIL over the patch tokens of each tile -> tile bag embeddings
             -> 8-head attention with a learned query over the tiles      -> level1_emb [B, 1024]
    L2     = gated-attention MIL over the CLS tokens of all views (global + tiles) -> level2_emb [B, 1024]
    gate   = sigmoid(MLP([attention-pooled L1 / L2 instance logits]))
    logits = MLP([gate * level1_emb, (1 - gate) * level2_emb]) -> [B, 2];  P(fake) = softmax(logits)[:, 1]

``SynthScopeLTFModel`` *is* the official ``LTF`` class (vendored unchanged in ``official/``). ``forward`` returns the
fusion logits, whose softmax is the official ``score_fake`` (``Detector.predict`` with the default fusion branch).
The released GenImage checkpoint (HF ``MuskAI/LTF-v3.3`` ``weights/epoch_20.pth``, loaded as ``ltf.pth``) holds the MIL
head only; ``remap_ltf`` (registry) drops its 27 disconnected baseline tensors, as the official ``load_core``, and the
37 retained tensors load strictly. The frozen CLIP comes from the pinned ``backbones/ViT-L-14.pt`` of the same Hub
revision (byte-identical to OpenAI's ViT-L-14.pt), loaded with the OpenAI ``clip`` package as in the official code.

Training: ``compute_loss`` is the official objective (cross-entropy on the level1, level2 and fusion logits, weight 1
each); the official data pipeline (reconstruction pairs with a per-step tile schedule) is not ported.
"""

import hashlib
import os
from pathlib import Path

import clip
import torch.nn.functional as F
from omegaconf import DictConfig

from .official.model import LTF

HUB_REPO = "MuskAI/LTF-v3.3"                                  # ltf_ref/assets.json
HUB_REVISION = "a2a2e29c4c64d201ba073f22bfcce2d3f5d287a1"
CORE_FILENAME = "weights/epoch_20.pth"
CORE_SHA256 = "32e9fa2e38e04d6d0221c844d650f0d45f7235bbf898b5ad665543d8be580031"
BACKBONE_FILENAME = "backbones/ViT-L-14.pt"
BACKBONE_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
# LTFCore options (the official config.DEFAULT_MODEL keys other than feature_layer).
CORE_KEYS = ("dropout", "mil_heads", "fusion_hidden", "l1_cross_shuffle_p", "l1_replace_ratio_range",
             "l1_with_replacement_p", "l2_cross_shuffle_p", "l2_replace_ratio_range", "l2_with_replacement_p",
             "bag_token_shuffle_p", "cross_shuffle_min_keep_original", "use_mil_cross_shuffle", "pool_gate_hidden",
             "use_pool_gate_fusion")
BRANCHES = ("level1", "level2", "fusion")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_backbone(path=None) -> Path:
    """The pinned CLIP ViT-L/14 file: explicit path > $SYNTHSCOPE_LTF_BACKBONE > an OpenAI ``clip`` cache copy
    ($SYNTHSCOPE_CLIP_DIR or ~/.cache/clip, the same bytes) > HF ``MuskAI/LTF-v3.3`` at the pinned revision.
    The SHA-256 is checked as in the official ``fetch_assets``."""
    path = path or os.environ.get("SYNTHSCOPE_LTF_BACKBONE")
    if not path:
        cached = Path(os.environ.get("SYNTHSCOPE_CLIP_DIR") or Path.home() / ".cache" / "clip") / "ViT-L-14.pt"
        if cached.is_file() and sha256(cached) == BACKBONE_SHA256:
            return cached
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HUB_REPO, BACKBONE_FILENAME, revision=HUB_REVISION)
    path = Path(path).expanduser()
    if sha256(path) != BACKBONE_SHA256:
        raise ValueError(f"{path}: SHA-256 differs from the pinned LTF backbone {BACKBONE_SHA256}")
    return path


class SynthScopeLTFModel(LTF):
    """The official LTF model, configured from SynthScope's ModelConfig."""

    # The frozen CLIP comes from its own file; the released checkpoint holds the MIL head (``core.``) only.
    pretrained_prefixes = ("visual.",)

    def __init__(self, config: DictConfig):
        mc = config.ModelConfig
        pretrained, _ = clip.load(str(resolve_backbone(mc.get("backbone_checkpoint", None))), device="cpu", jit=False)
        options = {k: (list(mc[k]) if k.endswith("_range") else mc[k]) for k in CORE_KEYS}
        super().__init__(pretrained.visual, feature_layer=mc.feature_layer, **options)
        del pretrained
        self.eval_chunk_size = int(mc.eval_chunk_size)
        self.loss_weights = {b: float(mc.loss_weights[b]) for b in BRANCHES}

    def forward(self, images):
        """[B, views, 3, 224, 224] (global view first) -> fusion logits [B, 2]."""
        return super().forward(images)["logits_fusion"]

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        out = super().forward(images, labels=labels)
        loss = sum(F.cross_entropy(out["logits_" + b], labels) * self.loss_weights[b] for b in BRANCHES)
        return {"loss": loss, "outputs": out["logits_fusion"]}
