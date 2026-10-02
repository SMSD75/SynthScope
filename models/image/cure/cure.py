"""CuRe (Learning Continuous Source Responses for Generalizable AI-Generated Image Detection).

Official repo: github.com/manic-cui/CuRe (paper: arXiv 2609.14316). The released model is a frozen
Perception Encoder Core L/14 (336 px) visual tower with Q/V LoRA, a "source-response" subspace
(``mu`` [1024], ``W`` [1024, 128]) and a linear 128 -> 2 head:

    features = PECore(images)                       # mean of the 576 patch tokens, [B, 1024]
    logits   = head((features - mu) @ W)            # [B, 2]; P(fake) = softmax(logits)[:, 1]

``AGIDCuReModel`` keeps the official module/key names (``backbone.*``, ``mu``, ``W``, ``head.*``) so
the released 25 MB adapter (``cure_adapter.pt``) loads key-for-key as ``cure.pth``. The frozen base
weights (``PE-Core-L14-336.pt``, Apache-2.0) are fetched from the Hugging Face Hub on first use (or
from ``ModelConfig.base_checkpoint`` / ``$AGID_PE_CORE_CKPT``), exactly as the official code does.

Input contract: ``forward`` takes the tensor produced by ``CuReTransform`` (``[B, 3, 336, 336]``,
mean=std=0.5 normalized, after the official PNG->JPEG96 / 390-crop / JPEG70 pipeline).

Training: the paper's two-stage mixing-ratio-regression recipe is NOT in the official release, so
``compute_loss`` is plain cross-entropy on the 2-way head (LoRA + subspace + head trainable). It
exists to satisfy AGID's trainer interface; it does not reproduce the paper's training.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from .official_arch import PECore

BASE_REPO_ID = "facebook/PE-Core-L14-336"
BASE_FILENAME = "PE-Core-L14-336.pt"
FEATURE_DIM = 1024
SUBSPACE_DIM = 128
_VISUAL_PREFIX = "visual."


def resolve_base_checkpoint(path=None, repo_id: str = BASE_REPO_ID, filename: str = BASE_FILENAME) -> Path:
    """Local path to the frozen PE Core checkpoint: explicit path > $AGID_PE_CORE_CKPT > HF Hub."""
    path = path or os.environ.get("AGID_PE_CORE_CKPT")
    if path:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PE Core base checkpoint not found: {path}")
        return path
    return Path(hf_hub_download(repo_id, filename))


def load_base_visual_state(path: Path) -> dict:
    """Visual-tower tensors of the PE Core checkpoint, renamed into ``backbone.*``."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return {"backbone." + k[len(_VISUAL_PREFIX):]: v for k, v in state.items() if k.startswith(_VISUAL_PREFIX)}


class AGIDCuReModel(nn.Module):
    """Frozen PE Core backbone + Q/V LoRA + source-response subspace + linear head."""

    def __init__(self, config: DictConfig):
        super().__init__()
        mc = config.ModelConfig
        self.label_smoothing = float(config.TrainerConfig.get("label_smoothing", 0.0))

        self.backbone = PECore()
        self.register_buffer("mu", torch.zeros(FEATURE_DIM))
        self.register_buffer("W", torch.zeros(FEATURE_DIM, SUBSPACE_DIM))
        self.head = nn.Linear(SUBSPACE_DIM, 2)

        if bool(mc.get("load_base_backbone", True)):
            self._load_base(resolve_base_checkpoint(mc.get("base_checkpoint", None),
                                                    str(mc.get("base_repo_id", BASE_REPO_ID)),
                                                    str(mc.get("base_filename", BASE_FILENAME))))
        else:
            self._init_random_backbone()  # architecture-only mode (tests): official params are torch.empty
        self._init_trainable()

        # Everything frozen except the LoRA tensors and the head (the adapter's trainable set).
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name or name.startswith("head.")
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def _load_base(self, path: Path) -> None:
        base = load_base_visual_state(path)
        expected = set(self.backbone.state_dict())
        state = {k[len("backbone."):]: v for k, v in base.items() if k[len("backbone."):] in expected}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        # Only the LoRA tensors (not part of the frozen base) may be missing.
        stray = [k for k in missing if "lora_" not in k]
        if stray or unexpected:
            raise RuntimeError(f"PE Core base checkpoint mismatch: missing={stray[:4]} unexpected={unexpected[:4]}")

    def _init_random_backbone(self) -> None:
        """Deterministic random init of the backbone for architecture-only builds. The vendored
        official module allocates its parameters with ``torch.empty`` (always overwritten by the
        base checkpoint), which would otherwise leave uninitialized memory (NaNs)."""
        for name, param in self.backbone.named_parameters():
            if name.endswith("weight") and param.ndim == 1:      # LayerNorm scale
                nn.init.ones_(param)
            elif name.endswith("bias"):
                nn.init.zeros_(param)
            else:
                nn.init.normal_(param, std=0.02)

    def _init_trainable(self) -> None:
        """Standard LoRA init (A random, B zero) so a fresh model starts as plain PE Core features;
        the released adapter overwrites all of these."""
        for name, param in self.backbone.named_parameters():
            if "lora_A" in name:
                nn.init.kaiming_uniform_(param, a=5 ** 0.5)
            elif "lora_B" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.W)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` [B, 3, 336, 336] -> 2-class logits [B, 2] (fp16 autocast on CUDA, as released)."""
        with torch.amp.autocast("cuda", enabled=images.is_cuda):
            features = self.backbone(images)
        return self.head((features - self.mu) @ self.W)

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        outputs = self.forward(images)
        return {"loss": self.criterion(outputs, labels), "outputs": outputs}
