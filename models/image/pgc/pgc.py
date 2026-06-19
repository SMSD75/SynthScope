"""PGC integrated into AGID.

Paper: "PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection"
(Zhou et al., 2026). https://github.com/xiaoyu6868/PGC

PGC is a *trained* image detector with two streams fused by a peak-guided calibration:

  * RGB stream      : a LoRA-tuned frozen DINO ViT (e.g. DINOv2-Large) -> CLS token + patch tokens.
  * Residual stream : a small 3-block CNN over a YCbCr quantization residual -> feature map.
  * PGCM            : per-location score heads aggregated with a smooth-max ("peak") logsumexp
                      operator, giving a local calibration bias z_local (paper Eq. 4).
  * Global head     : Linear over L2-normalized [CLS ‖ pooled-residual] -> z_global.
  * Decision        : y = z_global + z_local   (single logit; paper Eq. 5).

This file vendors the official model pieces (LoRA, residual extractor, residual stream, PGCM)
faithfully, with two AGID-framework adaptations:
  1. The 6-channel input (RGB ‖ residual) is built INSIDE the model: the dataset feeds a normal
     3-channel image (ImageNet-normalized via ``normalize_dinov2``), the model de-normalizes to
     [0,1], computes the YCbCr quantization residual (paper Eq. 1), and concatenates. So the
     dataset/transform pipeline stays standard, like every other AGID detector.
  2. The DINO backbone loads by Hugging Face name (auto-download) instead of a local path.

Residual / RGB normalization matches the official transforms: RGB uses ImageNet mean/std (what
DINOv2 expects), the residual uses mean 0 / std 0.5. The residual is resolution-sensitive, so
pair this model with CROP-only transforms (no resize) in the experiment/dataset config.

``compute_loss`` returns ``{"loss": BCEWithLogits(logit, label), "outputs": [-logit, logit]}``
(col 0 = real, col 1 = fake), so ``softmax(outputs)[:, 1]`` is the AGID fake score.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel

logger = logging.getLogger(__name__)

_RESIDUAL_DIM = 256
# LoRA targets for a Hugging Face DINOv2 ViT (attention q/k/v + output dense + MLP fc1/fc2).
_DEFAULT_LORA_TARGETS = [
    "attention.attention.query", "attention.attention.key", "attention.attention.value",
    "attention.output.dense", "mlp.fc1", "mlp.fc2",
]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# RGB -> YCbCr (BT.601) and back; paper Eq. 1.
_M_YUV = torch.tensor([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]], dtype=torch.float32)
_M_YUV_INV = torch.tensor([[1.0, 0.0, 1.402],
                           [1.0, -0.344136, -0.714136],
                           [1.0, 1.772, 0.0]], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA (vendored from the official repo)
# ─────────────────────────────────────────────────────────────────────────────
class _LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=1.0):
        super().__init__()
        self.alpha, self.rank = alpha, rank
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        h = torch.einsum("...d, rd -> ...r", x, self.lora_A)
        h = torch.einsum("...r, or -> ...o", h, self.lora_B)
        return h * (self.alpha / self.rank)


class _LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=1.0, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False
        self.lora = _LoRALayer(original_layer.in_features, original_layer.out_features, rank, alpha)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.original_layer(x) + self.dropout(self.lora(x))


def _get_submodule(model, name):
    cur = model
    for part in name.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _apply_lora(model, rank, alpha, dropout, targets):
    patched = 0
    for name, module in list(model.named_modules()):
        if not any(t in name for t in targets):
            continue
        if isinstance(module, nn.Linear):
            parent = model if "." not in name else _get_submodule(model, name.rsplit(".", 1)[0])
            child = name.rsplit(".", 1)[1] if "." in name else name
            setattr(parent, child, _LoRALinear(module, rank, alpha, dropout))
            patched += 1
    if patched == 0:
        logger.warning("[PGC] LoRA matched 0 layers for targets=%s; backbone fully frozen.", targets)
    return model, patched


def _lora_params(model):
    out, seen = [], set()
    for module in model.modules():
        if isinstance(module, _LoRALinear):
            for p in module.lora.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Residual extractor + stream + PGCM (vendored)
# ─────────────────────────────────────────────────────────────────────────────
class _ResidualStream(nn.Module):
    def __init__(self, out_channels=_RESIDUAL_DIM):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.block3 = nn.Sequential(nn.Conv2d(128, out_channels, 3, 2, 1),
                                    nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, residual):
        return self.block3(self.block2(self.block1(residual)))


class _PeakGuidedCalibrationModule(nn.Module):
    def __init__(self, rgb_dim, residual_dim, tau_rgb=0.5, tau_res=0.5):
        super().__init__()
        self.rgb_score_head = nn.Linear(rgb_dim, 1)
        nn.init.zeros_(self.rgb_score_head.weight); nn.init.zeros_(self.rgb_score_head.bias)
        self.residual_score_head = nn.Conv2d(residual_dim, 1, 1)
        nn.init.zeros_(self.residual_score_head.weight); nn.init.zeros_(self.residual_score_head.bias)
        self.lambda_rgb = nn.Parameter(torch.tensor(1.0))
        self.tau_rgb, self.tau_res = float(tau_rgb), float(tau_res)

    @staticmethod
    def peak_aggregation(scores, tau):
        n = scores.size(1)
        if n == 0:
            return scores.new_zeros((scores.size(0), 1))
        log_n = scores.new_tensor(float(n)).log()
        return (tau * (torch.logsumexp(scores / tau, dim=1) - log_n)).unsqueeze(1)

    def forward(self, f_rgb_tokens, f_residual_map):
        s_rgb = self.rgb_score_head(f_rgb_tokens).squeeze(-1)            # [B, N]
        z_rgb = self.peak_aggregation(s_rgb, self.tau_rgb)               # [B, 1]
        s_res = self.residual_score_head(f_residual_map).flatten(1)      # [B, Hr*Wr]
        z_res = self.peak_aggregation(s_res, self.tau_res)               # [B, 1]
        return z_res + self.lambda_rgb * z_rgb


def _extract_cls_and_tokens(backbone, rgb):
    """Return (cls [B,D], patch_tokens [B,N,D]) from a HF or dinov2-hub backbone."""
    if hasattr(backbone, "forward_features"):
        feats = backbone.forward_features(rgb)
        if isinstance(feats, dict):
            cls = feats.get("x_norm_clstoken")
            patch = feats.get("x_norm_patchtokens")
            if cls is not None and patch is not None:
                return cls, patch
    out = backbone(rgb)
    last = out["last_hidden_state"] if isinstance(out, dict) else getattr(out, "last_hidden_state")
    return last[:, 0], last[:, 1:]


# ─────────────────────────────────────────────────────────────────────────────
# AGID model
# ─────────────────────────────────────────────────────────────────────────────
class AGIDPGCModel(nn.Module):
    """Peak-Guided Calibration AI-generated-image detector (PGC)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        cfg = config.ModelConfig

        backbone_name = str(cfg.get("backbone_name", "facebook/dinov2-large"))
        lora_rank = int(cfg.get("lora_rank", 8))
        lora_alpha = float(cfg.get("lora_alpha", 1.0))
        lora_dropout = float(cfg.get("lora_dropout", 0.1))
        lora_targets = list(cfg.get("lora_targets", _DEFAULT_LORA_TARGETS))
        tau_rgb = float(cfg.get("tau_rgb", 0.5))
        tau_res = float(cfg.get("tau_res", 0.5))

        # --- RGB backbone (DINO via HF) + LoRA ---
        self.backbone = AutoModel.from_pretrained(backbone_name)
        embed_dim = int(self.backbone.config.hidden_size)
        self.backbone, n_lora = _apply_lora(self.backbone, lora_rank, lora_alpha, lora_dropout, lora_targets)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in _lora_params(self.backbone):
            p.requires_grad = True
        logger.info("[PGC] %s (D=%d), LoRA on %d linear layers (rank=%d).",
                    backbone_name, embed_dim, n_lora, lora_rank)

        # --- residual stream + PGCM + global head ---
        self.residual_stream = _ResidualStream(_RESIDUAL_DIM)
        self.pgcm = _PeakGuidedCalibrationModule(embed_dim, _RESIDUAL_DIM, tau_rgb, tau_res)
        self.classifier = nn.Linear(embed_dim + _RESIDUAL_DIM, 1)
        nn.init.normal_(self.classifier.weight, 0.0, 0.02)
        nn.init.zeros_(self.classifier.bias)

        # buffers for residual computation / de-normalization
        self.register_buffer("M_yuv", _M_YUV)
        self.register_buffer("M_yuv_inv", _M_YUV_INV)
        self.register_buffer("imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.criterion = nn.BCEWithLogitsLoss()
        # PGC uses BCE on its own fused logit, so the trainer's CrossEntropy label-smoothing is bypassed;
        # read it here so it still applies (official trains with 0.1). y' = y(1-eps) + eps/2.
        self.label_smoothing = float(cfg.get("label_smoothing", 0.0))

    # ── residual (paper Eq. 1) ───────────────────────────────────────────────
    def _quantization_residual(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: [B,3,H,W] in [0,1] -> residual [B,3,H,W] in [-1,1], then /0.5 normalize.

        ``round(x01*255)`` snaps back to the integer pixel grid the image came from. Inputs
        originate as uint8 (PIL -> ToTensor -> /255), but the engine ImageNet-normalizes the RGB
        and ``forward`` de-normalizes it here — a lossy float round-trip that, for dark/low-residual
        images (e.g. ForenSynths seeingdark), flips ~6% of quantization-residual signs and corrupts
        the residual map. Snapping makes the residual BIT-EXACT to the official PGC pipeline (which
        builds it from the raw [0,1] tensor); verified zero diff in repro/_residual_path_diff.py."""
        xp = torch.round(x01 * 255.0).permute(0, 2, 3, 1)       # [B,H,W,3] on the uint8 grid
        x_yuv = torch.matmul(xp, self.M_yuv.T.to(xp.dtype))
        x_rec = torch.matmul(torch.round(x_yuv), self.M_yuv_inv.T.to(xp.dtype))
        res = torch.clamp((xp - x_rec).permute(0, 3, 1, 2), -1.0, 1.0)
        return res / 0.5                                        # residual std=0.5 normalization

    # ── forward / loss ───────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` [B,3,H,W] ImageNet-normalized -> single logit per sample [B]."""
        # de-normalize the ImageNet-normalized RGB back to [0,1] for the residual extractor
        x01 = (images * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
        residual = self._quantization_residual(x01)

        f_cls, f_tokens = _extract_cls_and_tokens(self.backbone, images)      # [B,D],[B,N,D]
        f_res_map = self.residual_stream(residual)                            # [B,C,Hr,Wr]
        f_res_pooled = torch.flatten(F.adaptive_avg_pool2d(f_res_map, 1), 1)  # [B,C]

        z_local = self.pgcm(f_tokens, f_res_map)                              # [B,1]
        f_global = F.normalize(torch.cat([f_cls, f_res_pooled], dim=1), p=2, dim=1)
        z_global = self.classifier(f_global)                                  # [B,1]
        return (z_global + z_local).squeeze(-1)                               # [B]

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        logits = self.forward(images)                                         # [B]
        target = labels.float()
        if self.label_smoothing > 0:                                          # official recipe: y'=y(1-eps)+eps/2
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = self.criterion(logits, target)
        outputs = torch.stack([-logits, logits], dim=1)                       # [B,2]; col1 = fake
        return {"loss": loss, "outputs": outputs}
