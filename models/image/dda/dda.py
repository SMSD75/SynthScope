"""DDA (Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable, NeurIPS 2025 Spotlight).

Official repo: github.com/roy-ch/Dual-Data-Alignment (paper: arXiv 2505.14359). The released detector is DINOv2
ViT-L/14 with LoRA (rank 8, alpha 1) on every ``attn.qkv``, ``attn.proj``, ``mlp.fc1`` and ``mlp.fc2`` linear layer,
and a linear head on the normalized CLS token:

    cls    = DINOv2(images).forward_features()["x_norm_clstoken"]    # [B, 1024]
    logit  = fc(cls)                                                  # [B, 1]; P(fake) = sigmoid(logit)

What makes DDA generalize is its training data (VAE-reconstructed fakes aligned with their real images in pixel and
frequency space), not the architecture. ``SynthScopeDDAModel`` keeps the official module names
(``base_model.model.*``, ``base_model.fc.*``), so the released ``DDA_ckpt.pth`` (Hugging Face
``Junwei-Xi/Dual-Data-Alignment``; full model, 307.5M parameters) loads key-for-key as ``dda.pth``.

Input contract: ``forward`` takes the tensor produced by the official test transform (center-crop 336 with no
resize, ToTensor, CLIP mean/std), i.e. registry ``tf_crop(336, CLIP_MEAN, CLIP_STD)``.

Training: ``compute_loss`` is the official objective's form (BCE on the single logit) with the official trainable
set (LoRA tensors + head); DDA's data construction is not part of this class.
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .official_arch import apply_lora_to_linear_layers

CHANNELS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
LORA_TARGETS = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]


class DDABase(nn.Module):
    """The official ``DINOv2Model``: a torch.hub DINOv2 (``model``) and a linear head (``fc``)."""

    def __init__(self, name: str, pretrained: bool, num_classes: int = 1):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x):
        return self.fc(self.model.forward_features(x)["x_norm_clstoken"])


class SynthScopeDDAModel(nn.Module):
    """DINOv2 + LoRA + linear head, with the official ``base_model.*`` key layout."""

    def __init__(self, config: DictConfig):
        super().__init__()
        mc = config.ModelConfig
        name = str(mc.get("backbone_name", "dinov2_vitl14"))
        # The released checkpoint holds the full model, so the DINOv2 weights are only needed for training.
        self.base_model = DDABase(name, pretrained=bool(mc.get("pretrained_backbone", True)))
        self.base_model.model = apply_lora_to_linear_layers(
            self.base_model.model, rank=int(mc.get("lora_rank", 8)), alpha=float(mc.get("lora_alpha", 1.0)),
            target_modules=list(mc.get("lora_targets", LORA_TARGETS)), trainable_orig=False)
        # Official trainable set (DINOv2ModelWithLoRA.get_trainable_params): LoRA tensors + head only.
        for pname, param in self.named_parameters():
            param.requires_grad = ".lora." in pname or pname.startswith("base_model.fc.")
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images):
        return self.base_model(images)

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        logits = self.forward(images).squeeze(1)
        loss = self.criterion(logits, labels.float())
        # Two-column [-logit, logit] so the trainer's metrics (argmax / softmax[:, 1]) read the single logit right.
        return {"loss": loss, "outputs": torch.stack([-logits, logits], dim=1)}
