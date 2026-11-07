"""VINA (Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection, arXiv 2605.21977).

Official repo: github.com/mgiant/VINA (licence unclear: no LICENSE file). One detector for images and videos: a
DINOv3 ViT-L/16 fine-tuned end to end with a linear head on its CLS (pooler) output. Training mixes images with video
frames, which act as a natural augmentation; at test time a video is scored frame by frame:

    feat   = DINOv3(ImageNet-normalized x).pooler_output     # [B*T, 1024]
    logit  = head(feat)                                       # [B*T, 1]
    logit  = mean over the T frames of a video                # [B, 1] (official util.eval_model / test.py)
    P(fake) = sigmoid(logit)

``SynthScopeVINAModel`` *is* the official ``DINOv3`` detector (vendored in ``official_arch.py``); the released
``VINA_DINOv3-L_140k+DDA.pth`` holds the full model (backbone and head) and loads key-for-key as ``vina.pth``. The
backbone is built from its config, so inference needs no gated DINOv3 download.

Input contract: ``forward`` takes ImageNet-normalized [B, 3, 256, 256] images from the official eval transform
(``resize_short_256_bicubic``, ``center_crop_256``, ``to_image``, ``to_dtype``, ``normalize_imagenet``), or
[B, T, 3, 256, 256] clips, whose per-frame logits are averaged as in the official eval. It runs fp32 with eager
attention, the official defaults (``use_bf16`` and ``attn_implementation`` are unset in configs/VINA_DINOv3.yaml).

Training: ``compute_loss`` is the official ``util._classification_loss`` for one output (BCE with logits). The official
training code is not released (README TODO), so no training recipe is reproduced here.
"""
import torch.nn.functional as F
from omegaconf import DictConfig

from .official_arch import DINOV3_VITL16, DINOv3


class SynthScopeVINAModel(DINOv3):
    """The official VINA ``DINOv3`` detector, configured from SynthScope's ModelConfig."""

    def __init__(self, config: DictConfig):
        mc = config.ModelConfig
        # backbone_path null: build DINOv3 ViT-L/16 from its config (the weights come from vina.pth); a HF-format
        # directory loads pretrained DINOv3 weights (the official `weights/dinov3-vitl16-pretrain-lvd1689m`).
        backbone_path = mc.get("backbone_path")
        # backbone_config: overrides of the ViT-L config fields (e.g. a tiny backbone for the smoke test).
        backbone_config = dict(DINOV3_VITL16, **dict(mc.get("backbone_config") or {}))
        super().__init__(num_classes=1, backbone_source=str(backbone_path) if backbone_path else backbone_config,
                         attn_implementation=mc.get("attn_implementation", "eager"),
                         use_bf16=bool(mc.get("use_bf16", False)))

    def forward(self, images):
        """[B, 3, H, W] or [B, T, 3, H, W] normalized pixels -> fake logits [B, 1] (mean over T, as the official eval)."""
        clip = images.dim() == 5
        if clip:
            b, t = images.shape[:2]
            images = images.flatten(0, 1)
        logits = super().forward(images)["cls"]
        if clip:
            logits = logits.view(b, t, -1).mean(dim=1)
        return logits.float()

    def compute_loss(self, batch):
        images, labels = batch[0], batch[1]
        logits = self.forward(images)
        loss = F.binary_cross_entropy_with_logits(logits[:, 0], labels.to(logits.dtype))
        return {"loss": loss, "outputs": logits}
