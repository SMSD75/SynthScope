"""D3QE (Discrete Distribution Discrepancy-aware Quantization Error, ICCV 2025) — detector for images made by
visual autoregressive (AR) generators.

Official repo: github.com/Zhangyr2022/D3QE (paper: arXiv 2510.05891, MIT). Two frozen encoders see the image:

    vq   = LlamaGen VQ-16 tokenizer (codebook 16384) on the raw [0, 1] image
    res  = z_q - z                                 quantization error, [B, 8, 16, 16]
    tok  = D3AT(Conv(res), codebook-frequency bias) → pooled → [B, 256]
    clip = OpenAI CLIP ViT-L/14 on the center 224 crop (CLIP normalization) → [B, 768] → [B, 256]
    logit = MLP([tok, clip]) → [B, 1];  P(fake) = sigmoid(logit)

``SynthScopeD3QEModel`` *is* the official ``D3QE`` class (vendored unchanged in ``official/``), so the released
``model_epoch_best.pth`` (Hugging Face ``Yanran21/D3QE``) loads key-for-key as ``d3qe.pth``. That checkpoint holds only
the trained part (81 tensors, 7.7M parameters, plus the codebook counts); the frozen CLIP ViT-L/14 and LlamaGen VQ-16
come from their own releases, exactly as in the official code (``pretrained_prefixes`` tells the loader so).

Two properties of the released code that this port keeps on purpose:
* the VQ tokenizer receives the [0, 1] image (``forward`` builds a [-1, 1] copy but does not use it), and
* the codebook-frequency bias is zero at inference: ``freq_log_counter`` is a plain attribute (not saved), so a
  freshly loaded model is below its 200-step warm-up and uses ``freq_diff = 0``; the saved codebook counts only act
  during training.

Input contract: ``forward`` takes the tensor of the official test transform — Resize(256) on the short side
(bilinear), CenterCrop(256), ToTensor, no normalization — i.e. the registry transform.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .official.D3QE import D3QE

VQ_REPO_ID = "FoundationVision/LlamaGen"   # MIT
VQ_FILENAME = "vq_ds16_c2i.pt"


def resolve_vq_checkpoint(path=None) -> Path:
    """Local path to the frozen LlamaGen VQ-16 tokenizer: explicit path > $SYNTHSCOPE_LLAMAGEN_VQ > HF Hub."""
    path = path or os.environ.get("SYNTHSCOPE_LLAMAGEN_VQ")
    if path:
        return Path(path)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(VQ_REPO_ID, VQ_FILENAME))


class SynthScopeD3QEModel(D3QE):
    """The official D3QE model, built from SynthScope's config."""

    # Frozen pretrained parts that the released checkpoint does not contain (loaded in ``D3QE.__init__``).
    pretrained_prefixes = ("clip_model.", "vq_model.")

    def __init__(self, config: DictConfig):
        super().__init__(vqvae_path=str(resolve_vq_checkpoint(config.ModelConfig.get("vq_checkpoint", None))))
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_loss(self, batch):
        # Official objective (networks/trainer.py): BCE on the single logit; passing the labels lets the forward pass
        # update the codebook-frequency statistics during training, as in the official code.
        images, labels, _folder_ids = batch
        logits = self(images, labels).squeeze(1)
        loss = self.criterion(logits, labels.float())
        # Two-column [-logit, logit] so the trainer's metrics (argmax / softmax[:, 1]) read the single logit right.
        return {"loss": loss, "outputs": torch.stack([-logits, logits], dim=1)}
