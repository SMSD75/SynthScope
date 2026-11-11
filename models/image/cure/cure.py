"""CuRe (Learning Continuous Source Responses for Generalizable AI-Generated Image Detection).

Official repo: github.com/manic-cui/CuRe (paper: arXiv 2609.14316). The released model is a frozen
Perception Encoder Core L/14 (336 px) visual tower with Q/V LoRA, a "source-response" subspace
(``mu`` [1024], ``W`` [1024, 128]) and a linear 128 -> 2 head:

    features = PECore(images)                       # mean of the 576 patch tokens, [B, 1024]
    logits   = head((features - mu) @ W)            # [B, 2]; P(fake) = softmax(logits)[:, 1]

``SynthScopeCuReModel`` keeps the official module/key names (``backbone.*``, ``mu``, ``W``, ``head.*``) so
the released 25 MB adapter (``cure_adapter.pt``) loads key-for-key as ``cure.pth``. The frozen base
weights (``PE-Core-L14-336.pt``, Apache-2.0) are fetched from the Hugging Face Hub on first use (or
from ``ModelConfig.base_checkpoint`` / ``$SYNTHSCOPE_PE_CORE_CKPT``), exactly as the official code does.

Input contract: ``forward`` takes the tensor produced by ``CuReTransform`` (``[B, 3, 336, 336]``,
mean=std=0.5 normalized, after the official PNG->JPEG96 / 390-crop / JPEG70 pipeline).

Training: the official release has no training code. SynthScope implements the paper's recipe
(Sec. 3, 4.1-4.2, Appendix A; see README "Training recipe"):

* Stage 1 (``ModelConfig.training_stage: ratio``, run through SynthScope's trainer): a linear ratio
  head ``ratio_head`` (1024 -> 1, sigmoid) on the adapted features of real/generated pixel mixtures
  x_a = (1 - a) x_r + a x_g, trained with BCE against the soft target a; only the LoRA tensors and the
  ratio head are trainable, with their own weight decays (``optimizer_param_groups``). Optional
  gradient checkpointing per transformer block (``grad_checkpointing``) for the per-GPU batch of 64.
* Stages 2 and 3 (``train_stages.py``): the source-response subspace (``mu``, ``W``) from the frozen
  stage-1 encoder (``subspace.py``) and the 128 -> 2 head; ``adapter_state_dict`` exports the result
  with exactly the released adapter's 100 keys.

Without ``training_stage`` ``compute_loss`` is plain cross-entropy on the 2-way head (LoRA + head
trainable); it only satisfies the trainer interface and is not the paper's training.
"""
import os
from pathlib import Path

import _synthscope_env  # noqa: F401  (maps legacy AGID_* variables to SYNTHSCOPE_*)

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.checkpoint import checkpoint
from omegaconf import DictConfig

from .official_arch import PECore

BASE_REPO_ID = "facebook/PE-Core-L14-336"
BASE_FILENAME = "PE-Core-L14-336.pt"
FEATURE_DIM = 1024
SUBSPACE_DIM = 128
_VISUAL_PREFIX = "visual."
# The released adapter's tensors (cure_adapter.pt): 24 layers x (lora_A_q, lora_B_q, lora_A_v, lora_B_v), mu, W, head.
ADAPTER_BUFFERS = ("mu", "W", "head.weight", "head.bias")


def resolve_base_checkpoint(path=None, repo_id: str = BASE_REPO_ID, filename: str = BASE_FILENAME) -> Path:
    """Local path to the frozen PE Core checkpoint: explicit path > $SYNTHSCOPE_PE_CORE_CKPT > HF Hub."""
    path = path or os.environ.get("SYNTHSCOPE_PE_CORE_CKPT")
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


class SynthScopeCuReModel(nn.Module):
    """Frozen PE Core backbone + Q/V LoRA + source-response subspace + linear head."""

    def __init__(self, config: DictConfig):
        super().__init__()
        mc = config.ModelConfig
        self.label_smoothing = float(config.TrainerConfig.get("label_smoothing", 0.0))

        self.backbone = PECore()
        self.register_buffer("mu", torch.zeros(FEATURE_DIM))
        self.register_buffer("W", torch.zeros(FEATURE_DIM, SUBSPACE_DIM))
        self.head = nn.Linear(SUBSPACE_DIM, 2)
        # Stage 1 of the paper's recipe: mixing-ratio regression with a linear ratio head (Appendix A).
        self.training_stage = mc.get("training_stage", None)
        if self.training_stage not in (None, "ratio"):
            raise ValueError(f"CuRe training_stage must be null or 'ratio', got {self.training_stage!r}")
        self.grad_checkpointing = bool(mc.get("grad_checkpointing", False))
        self.lora_weight_decay = float(mc.get("lora_weight_decay", 0.1))
        self.head_weight_decay = float(mc.get("head_weight_decay", 0.01))
        if self.training_stage == "ratio":
            self.ratio_head = nn.Linear(FEATURE_DIM, 1)   # PyTorch default init; discarded after stage 1

        if bool(mc.get("load_base_backbone", True)):
            self._load_base(resolve_base_checkpoint(mc.get("base_checkpoint", None),
                                                    str(mc.get("base_repo_id", BASE_REPO_ID)),
                                                    str(mc.get("base_filename", BASE_FILENAME))))
        else:
            self._init_random_backbone()  # architecture-only mode (tests): official params are torch.empty
        self._init_trainable()

        # Everything frozen except the LoRA tensors and the head (the adapter's trainable set); in stage 1 the ratio
        # head replaces the 2-way head (which is trained later, on the frozen subspace, by train_stages.py).
        head_prefix = "ratio_head." if self.training_stage == "ratio" else "head."
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name or name.startswith(head_prefix)
        # The frozen PE-Core base is loaded above from its own release (``_load_base``), so the released adapter
        # (LoRA + subspace + head only) does not contain it: the engine accepts exactly these keys as missing.
        self.pretrained_prefixes = tuple(n for n, p in self.named_parameters() if not p.requires_grad)
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

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Adapted PE Core features ``[B, 1024]`` (fp16 autocast on CUDA, as ``forward``). With ``grad_checkpointing`` in
        training each transformer block is recomputed in the backward pass (same values, less activation memory)."""
        with torch.amp.autocast("cuda", enabled=images.is_cuda):
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                return self._checkpointed_backbone(images)
            return self.backbone(images)

    def _checkpointed_backbone(self, images: torch.Tensor) -> torch.Tensor:
        """``PECore.forward`` (official_arch.py) step for step, with ``torch.utils.checkpoint`` around every block."""
        bb = self.backbone
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 336, 336):
            raise ValueError("Expected image tensors with shape [N, 3, 336, 336]")
        batch = images.shape[0]
        x = bb.conv1(images).permute(0, 2, 3, 1).reshape(batch, 576, 1024)
        cls = bb.class_embedding.view(1, 1, -1).expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + bb.positional_embedding[None]
        x = bb.ln_pre(x)
        for block in bb.transformer.resblocks:
            x = checkpoint(block, x, use_reentrant=False)
        return bb.ln_post(x)[:, 1:].mean(dim=1).float()

    def optimizer_param_groups(self):
        """Stage 1: AdamW groups with the paper's weight decays (0.1 on the LoRA tensors, 0.01 on the ratio head, all of
        its parameters); used by the trainer instead of its single group. ``None`` otherwise (the trainer's default)."""
        if self.training_stage != "ratio":
            return None
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        return [{"params": [p for n, p in named if "lora_" in n], "weight_decay": self.lora_weight_decay},
                {"params": [p for n, p in named if "lora_" not in n], "weight_decay": self.head_weight_decay}]

    @staticmethod
    def _require_grad_scaler(images: torch.Tensor) -> None:
        """``features`` always runs fp16 autocast on CUDA, so CUDA training needs a gradient scaler; ``accelerate launch
        --mixed_precision fp16`` (exported as ACCELERATE_MIXED_PRECISION, which the trainer's Accelerator reads) provides it.
        Without it the fp16 gradients would underflow silently, so fail loudly instead."""
        if images.is_cuda and torch.is_grad_enabled() and os.environ.get("ACCELERATE_MIXED_PRECISION", "no") != "fp16":
            raise RuntimeError("CuRe stage 1 runs the backbone in fp16 autocast and needs a gradient scaler: launch with "
                               "`accelerate launch --mixed_precision fp16` (experiments/image/cure.sh does)")

    def compute_loss(self, batch):
        images, targets, _folder_ids = batch
        if self.training_stage == "ratio":
            # Eq. 3-4: sigmoid ratio head, BCE with the mixing ratio as the soft target (targets = alpha in [0, 1]).
            self._require_grad_scaler(images)
            logits = self.ratio_head(self.features(images)).squeeze(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            return {"loss": loss, "outputs": torch.sigmoid(logits)}
        outputs = self.forward(images)
        return {"loss": self.criterion(outputs, targets), "outputs": outputs}

    def adapter_state_dict(self) -> dict:
        """The trained parameters in the released adapter's format: exactly its 100 keys (96 LoRA tensors, ``mu``,
        ``W`` [1024, 128], ``head.weight``, ``head.bias``), fp32, so the file loads like ``cure_adapter.pt``."""
        state = self.state_dict()
        out = {k: v.detach().float().cpu().contiguous() for k, v in state.items()
               if "lora_" in k or k in ADAPTER_BUFFERS}
        assert len(out) == 100, f"expected the adapter's 100 tensors, got {len(out)}"
        return out
