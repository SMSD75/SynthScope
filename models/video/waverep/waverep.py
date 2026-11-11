"""WaveRep (Corvi et al., "Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented
Augmentation", NeurIPS 2025).

Official repo: github.com/grip-unina/WaveRep-SyntheticVideoDetection (GRIP-UNINA license: informational and nonprofit
use only). The detector is a DINOv2 ViT-B/14 with four registers (timm ``vit_base_patch14_reg4_dinov2.lvd142m``) at a
504x504 input, fine-tuned end to end frame by frame with one output logit; the method's contribution is the training
data and a wavelet-band augmentation, not the architecture. A video is scored frame by frame (official demo/utils.py):

    x_t    = CenterCrop(504)(frame_t) -> ToTensor -> ImageNet Normalize   # no resize; smaller frames are zero-padded
    logit  = vit(x_t)[:, -1]                                               # one logit per frame
    score  = mean over the frames of the video;  P(fake) = sigmoid(score)

``SynthScopeWaveRepModel`` holds the timm model exactly as the official ``create_model`` builds it
(``timm.create_model(arc, num_classes=1, img_size=504)``) as ``self.model``; the released ``weights_dinov2_G{1,4}.ckpt``
are flat timm state dicts that the official code loads strictly, and ``registry.remap_waverep`` puts them under
``model.``. ``pretrained_backbone`` (default false) only matters for training: the official code starts from the timm
DINOv2 weights, which every checkpoint key then overwrites.

Input contract: ``forward`` takes ImageNet-normalized [B, 3, 504, 504] frames (the registry transform:
``center_crop_504``, ``to_dtype``, ``normalize_imagenet``) -> [B] logits, or [B, T, 3, 504, 504] clips -> the mean of
their T frame logits, computed ``frame_batch`` frames at a time.

Training: the official training code is not released (only the augmentation, augmentation/pytorch_wt.py);
``compute_loss`` is binary cross-entropy with logits on single frames, and its ``outputs`` are the two columns
[-logit, logit], the repo convention for single-logit detectors. No training recipe is reproduced here.
"""
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

# The official demo's arch string (demo/main_avideo.py: arc, cropping).
TIMM_ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"
CROP_SIZE = 504


class SynthScopeWaveRepModel(nn.Module):
    """WaveRep's frame-level detector, configured from SynthScope's ModelConfig."""

    def __init__(self, config: DictConfig):
        super().__init__()
        mc = config.ModelConfig
        self.model = timm.create_model(mc.get("timm_arch", TIMM_ARCH), num_classes=1,
                                       pretrained=bool(mc.get("pretrained_backbone", False)),
                                       img_size=int(mc.get("crop_size", CROP_SIZE)))
        self.frame_batch = int(mc.get("frame_batch", 16))   # frames per forward pass when scoring a clip
        self.criterion = nn.BCEWithLogitsLoss()

    def frame_logits(self, frames: torch.Tensor) -> torch.Tensor:
        """[N, 3, H, W] normalized frames -> [N] logits: the official ``model(x)[:, -1]``."""
        return self.model(frames)[:, -1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B] frame logits, or [B, T, 3, H, W] -> [B] mean frame logit per clip (official evaluate +
        ``tab['logit'].mean()``)."""
        if images.dim() == 4:
            return self.frame_logits(images)
        b, t = images.shape[:2]
        frames = images.flatten(0, 1)
        logits = torch.cat([self.frame_logits(chunk) for chunk in frames.split(self.frame_batch)])
        return logits.view(b, t).double().mean(dim=1).to(logits.dtype)

    def compute_loss(self, batch):
        images, labels = batch[0], batch[1]
        logits = self.forward(images)
        loss = self.criterion(logits, labels.to(logits.dtype))
        # Two-column [-logit, logit] so the trainer's metrics (argmax / softmax[:, 1]) read the single logit right.
        return {"loss": loss, "outputs": torch.stack([-logits, logits], dim=1)}
