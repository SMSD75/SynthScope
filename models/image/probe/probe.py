"""PROBE (Where Detectors Fail: Probing Generative Space for Generalizable AI-Generated Image Detection, ICML 2026,
arXiv 2605.24906).

Official repo: github.com/Amamiya-C/PROBE-AIGI-Detection (licence unclear: no LICENSE file). PROBE is a training
method: the detector steers a generator through manifold-level edits to find realistic samples it misclassifies, and
is then refined on them. The released detectors are plain classifiers with one logit:

    PROBE-DINOv2   = DINOv2 ViT-L/14 with registers (facebook/dinov2-with-registers-large) -> CLS -> Linear(1024, 1)
    PROBE-ResNet50 = ResNet-50 -> global average pool -> Linear(2048, 1)

and the official evaluation scores an image as the mean logit of its non-overlapping crop_size patches at native
resolution (336 for DINOv2, 224 for ResNet-50; ``preprocess.py``), P(fake) = sigmoid(mean logit).

``SynthScopePROBEModel`` *is* the official ``dino_classifier`` and ``SynthScopePROBEResNetModel`` the official
``resnet50(num_classes=1, classifier_type='linear')`` (vendored in ``official/``); the released
``DINOv2_best_model_step_34999.pth`` / ``ResNet50_best_model_step_19343.pth`` ({"model_state_dict": ...}) load
key-for-key as ``probe.pth`` / ``probe_resnet.pth``. The DINOv2 backbone is built from its config, so inference needs
no backbone download.

Input contract: ``forward`` takes the patch bag of one image per sample, [B, N, 3, S, S] from the eval transform
(``probe_pad_*``, ``to_dtype``, ``normalize_imagenet``, ``sliding_patches_*``), and returns the mean patch logit [B, 1],
as the official ``evaluate``; a [B, 3, S, S] batch is one patch per sample. Patches go through the network in chunks
of ``patch_chunk_size`` (0 = all at once, as the official code does per batch) to bound memory on large images.

Training: the official fine-tuning and boundary-exploration code is not released (README TODO), so no recipe is
reproduced; ``compute_loss`` is BCE with logits on one patch per image, outputs [-logit, logit] (the repo convention
for single-logit detectors).
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .official.dino_classifier import DINOV2_REG_LARGE, dino_classifier
from .official.resnet50 import Bottleneck, ResNet


def mean_patch_logits(net_forward, images: torch.Tensor, chunk: int) -> torch.Tensor:
    """[B, N, 3, S, S] (or [B, 3, S, S] = one patch each) -> the mean logit over each sample's patches, [B, 1]."""
    if images.dim() == 4:
        images = images.unsqueeze(1)
    b, n = images.shape[:2]
    patches = images.flatten(0, 1)
    step = chunk if chunk > 0 else len(patches)
    logits = torch.cat([net_forward(patches[i:i + step]) for i in range(0, len(patches), step)])   # [B*N, 1]
    return logits.view(b, n, -1).mean(dim=1)


def bce_loss(model, batch):
    images, labels = batch[0], batch[1]
    logits = model.forward(images)[:, 0]
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
    # Two-column [-logit, logit] so the trainer's metrics (argmax / softmax[:, 1]) read the single logit right.
    return {"loss": loss, "outputs": torch.stack([-logits, logits], dim=1)}


class SynthScopePROBEModel(dino_classifier):
    """PROBE-DINOv2: the official ``dino_classifier``, configured from SynthScope's ModelConfig."""

    def __init__(self, config: DictConfig):
        mc = config.ModelConfig
        # backbone_path null: build DINOv2-reg ViT-L/14 from its config (the weights come from probe.pth); a HF-format
        # directory or hub id loads pretrained weights, as the official model_name.
        backbone_path = mc.get("backbone_path")
        # backbone_config: overrides of the ViT-L config fields (e.g. a tiny backbone for the smoke test).
        backbone_config = dict(DINOV2_REG_LARGE, **dict(mc.get("backbone_config") or {}))
        super().__init__(model_name=str(backbone_path) if backbone_path else backbone_config,
                         classifier_type=mc.get("classifier_type", "linear"))
        self.patch_chunk_size = int(mc.get("patch_chunk_size", 0) or 0)

    def forward(self, images):
        """[B, N, 3, S, S] normalized patches (or [B, 3, S, S]) -> mean patch logit [B, 1]."""
        return mean_patch_logits(super().forward, images, self.patch_chunk_size)

    def compute_loss(self, batch):
        return bce_loss(self, batch)


class SynthScopePROBEResNetModel(ResNet):
    """PROBE-ResNet50: the official ``resnet50(num_classes=1, classifier_type='linear')``."""

    def __init__(self, config: DictConfig):
        mc = config.ModelConfig
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=1, classifier_type=mc.get("classifier_type", "linear"))
        self.patch_chunk_size = int(mc.get("patch_chunk_size", 0) or 0)

    def forward(self, images):
        """[B, N, 3, S, S] normalized patches (or [B, 3, S, S]) -> mean patch logit [B, 1]."""
        return mean_patch_logits(super().forward, images, self.patch_chunk_size)

    def compute_loss(self, batch):
        return bce_loss(self, batch)

