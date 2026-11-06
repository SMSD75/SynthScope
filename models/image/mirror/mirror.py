"""MIRROR (Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection, NeurIPS 2026,
arXiv 2602.02222).

Official repo: github.com/handsome-rich/MIRROR (licence unclear: README badge MIT, no LICENSE file). A frozen DINOv3
ViT-H+/16 with LoRA (rank 8, alpha 16) on the attention q/k/v projections, a frozen memory bank of 4096 "real"
prototypes read by sparse top-128 attention (8 heads), and a dual-branch head:

    tokens, cls = DINOv3(ImageNet-normalized x)            # 196 patch + 4 register tokens, CLS
    recon, attn = MemoryBank(tokens)                       # the "ideal reference" of the input
    logits      = head(perplexity(max, entropy of attn), MLP(mean(tokens - recon)), cls)   # [B, 2]
    P(fake)     = softmax(logits)[:, 1]                    # official inference.py

``SynthScopeMIRRORModel`` *is* the official ``MIRROR_Detector`` (vendored in ``official_arch.py``); the released
phase-2 ``checkpoint-h-cur.pth`` holds the full model (backbone, LoRA, memory bank, head) and loads key-for-key as
``mirror.pth``. The backbone is built from its config, so inference needs no gated DINOv3 download, and no separate
phase-1 file: the official inference loads ``mirror_phase1.pth`` first, but the strict phase-2 load then overwrites
every memory-bank tensor. (The two memory banks differ: the released phase-2 one has the statistics of its default
initialization, randn memory and kaiming-uniform projections, and the checkpoint's training args have
memory_path=None; see the README.)

Input contract: ``forward`` takes [0, 1] pixels from the official eval transform (``mirror_eval`` + ``to_dtype``: short
side > 512 resized to 512, center crop 224, JPEG q96 for PNG/BMP/TIFF sources) and normalizes inside, like the official
model. It runs fp32 by default, the official default (``inference.py --use_amp`` is opt-in, ``eval.sh`` does not set it,
and the released checkpoint was trained with ``use_amp=False``). ``use_amp: true`` runs inference under CUDA fp16 autocast
exactly as the official ``evaluate --use_amp``; it never applies on CPU or in training mode.

Training: ``compute_loss`` is the official phase-2 loss (timm LabelSmoothingCrossEntropy, smoothing 0.1 =
CrossEntropyLoss(label_smoothing=0.1)) on the logits; the phase-1 memory-bank training (mmbank.py) and the official
data pipeline are not reproduced here.
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .official_arch import DINOV3_VITH16PLUS, MIRROR_Detector


class SynthScopeMIRRORModel(MIRROR_Detector):
    """The official MIRROR detector, configured from SynthScope's ModelConfig."""

    def __init__(self, config: DictConfig):
        mc = config.ModelConfig
        backbone_path = mc.get("backbone_path")       # null: build DINOv3 ViT-H+ from its config (weights from mirror.pth)
        # backbone_config: overrides of the ViT-H+ config fields (e.g. a tiny backbone for the smoke test).
        backbone_config = dict(DINOV3_VITH16PLUS, **dict(mc.get("backbone_config") or {}))
        super().__init__(dino_path=str(backbone_path) if backbone_path else backbone_config,
                         memory_path=mc.get("memory_path"))
        self.use_amp = bool(mc.get("use_amp", False))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(mc.get("label_smoothing", 0.1)))

    def forward(self, images):
        """[B, 3, 224, 224] pixels in [0, 1] -> 2-class logits [B, 2] (index 1 = fake)."""
        amp = self.use_amp and images.is_cuda and not self.training     # inference-only autocast, as the official evaluate
        with torch.amp.autocast(device_type=images.device.type, enabled=amp):
            logits, _, _ = super().forward(images)
        return logits.float()

    def compute_loss(self, batch):
        images, labels = batch[0], batch[1]
        logits = self.forward(images)
        return {"loss": self.criterion(logits, labels), "outputs": logits}
