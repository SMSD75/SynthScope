"""Standalone evaluation of the OFFICIAL PGC detector (Peak-Guided Calibration, Zhou et al.).

This script vendors the *official* PGC architecture, the *official* eval
preprocessing, and loads the *official* checkpoints, so its numbers are the
"official-weights" side of an apples-to-apples comparison against the SynthScope
re-implementation (run via ``benchmark/bench_eval.py --model pgc``).

It is deliberately SELF-CONTAINED: it does not import the ``synthscope`` package nor the
``PGC_official`` repo. The architecture/transform below are transcribed from
``/hddstore/ssalehi/AGID/PGC_official`` (= github.com/xiaoyu6868/PGC):
  - models/pgc.py, models/encoder/{rgb_stream,residual_stream,residual_extractor}.py
  - models/pgcm/peak_calibration.py, models/lora/lora.py
  - data/transforms.py (create_eval_transforms: PadCenterCrop -> ToTensor ->
    AppendResidual(YCbCr quant-residual) -> Normalize 6-channel)
  - engine/evaluator.py (P(fake) = sigmoid(z_global + z_local), threshold 0.5)

Discovery (discover/_images/_collect/metrics) is COPIED verbatim from
``benchmark/bench_eval.py`` so the per-generator grouping + UFD real-pairing +
metric definitions are byte-for-byte identical to the SynthScope side.

The DINOv2-large backbone is loaded via HF ``AutoModel.from_pretrained`` from the
local HF cache (export TORCH_HOME / HF_HOME first); no ``pretrained_root`` needed.

CLI::

    python repro/official_evals/pgc_official_eval.py \
        --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
        --select progan --limit 20 --bs 16 --device cpu \
        --ckpt /hddstore/ssalehi/AGID/repro/official_ckpts/pgc/PGC_train_progan_ckpt.pth \
        --tag _probe_pgc_off_v

Which official ckpt goes with which benchmark (per the PGC paper / official README):
  - forensynths  -> PGC_train_progan_ckpt.pth          (ProGAN 4-class)   [Table 5 style]
  - ufd          -> PGC_train_progan_ckpt.pth          (ProGAN 4-class)   [Table 5]
  - pergen (AIGI)-> PGC_train_progan_sdv1_4_ckpt.pth   (ProGAN + SDv1.4)  [Table 4]
  - pool/pergen  -> PGC_train_sdv1_4_ckpt.pth          (SD-v1.4)          [Table 2 GenImage]
The right ckpt is always chosen explicitly via ``--ckpt``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Default caches (can be overridden by the environment before launch).
os.environ.setdefault("TORCH_HOME", "/hddstore/ssalehi/.cache/torch")
os.environ.setdefault("HF_HOME", "/hddstore/ssalehi/.cache/huggingface")

# ==========================================================================
# Constants shared with benchmark/bench_eval.py (kept identical)
# ==========================================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
REAL_DIRNAMES = ("0_real", "real", "reals")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "ai")
SHARED_REAL_DIRS = {"imagenet", "laion", "_real_pool", "real", "0_real", "reals"}
RESULTS_LOG = Path("/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/MASTER_RESULTS.jsonl")
# Ojha UniversalFakeDetect real pairing (verified against the official dataset_paths.py).
UFD_REAL = {"guided": "imagenet"}  # everything else -> laion
UFD_DEFAULT_REAL = "laion"


# ==========================================================================
# Discovery + metrics — COPIED VERBATIM from benchmark/bench_eval.py
# (identical generator grouping incl. UFD real-pairing; do not edit)
# ==========================================================================
def _images(d: Path) -> list[Path]:
    if not d or not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in {e.lower() for e in IMAGE_EXTS})


def _collect(gen_dir: Path, dirnames: tuple[str, ...]) -> list[Path]:
    """All images under any subdir of gen_dir whose name matches one of dirnames (any depth)."""
    out: list[Path] = []
    for name in dirnames:
        for sub in gen_dir.rglob(name):
            if sub.is_dir():
                out += [p for p in sub.rglob("*") if p.suffix.lower() in {e.lower() for e in IMAGE_EXTS}]
    # de-dup while keeping order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return sorted(uniq)


def discover(benchmark: str, root: Path, shared_reals: Path | None, select: set[str] | None):
    """Return [(generator, real_paths, fake_paths)] for the benchmark."""
    shared = _images(shared_reals) if shared_reals else []

    if benchmark == "pool":
        reals = _collect(root, REAL_DIRNAMES) or shared
        fakes = _collect(root, FAKE_DIRNAMES)
        return [("all", reals, fakes)] if reals and fakes else []

    if benchmark == "ufd":
        out = []
        gens = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in SHARED_REAL_DIRS)
        for gen in gens:
            if select and gen not in select:
                continue
            real_src = root / UFD_REAL.get(gen, UFD_DEFAULT_REAL)
            reals = _collect(real_src, REAL_DIRNAMES)
            fakes = _collect(root / gen, FAKE_DIRNAMES)
            if reals and fakes:
                out.append((gen, reals, fakes))
            else:
                print(f"  skip {gen}: reals={len(reals)} fakes={len(fakes)}")
        return out

    if benchmark == "forensynths":
        out = []
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if select and gen_dir.name not in select:
                continue
            reals = _collect(gen_dir, REAL_DIRNAMES) or shared
            fakes = _collect(gen_dir, FAKE_DIRNAMES)
            if reals and fakes:
                out.append((gen_dir.name, reals, fakes))
            else:
                print(f"  skip {gen_dir.name}: reals={len(reals)} fakes={len(fakes)}")
        return out

    # pergen / generic: each top-level dir (excluding shared-real dirs) is a generator
    out = []
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if gen_dir.name in SHARED_REAL_DIRS and gen_dir.name != root.name:
            continue
        if select and gen_dir.name not in select:
            continue
        reals = _collect(gen_dir, REAL_DIRNAMES) or shared
        fakes = _collect(gen_dir, FAKE_DIRNAMES)
        if reals and fakes:
            out.append((gen_dir.name, reals, fakes))
        else:
            print(f"  skip {gen_dir.name}: reals={len(reals)} fakes={len(fakes)}")
    return out


def metrics(fake_probs: torch.Tensor, labels: torch.Tensor) -> dict | None:
    from torchmetrics.functional import auroc, average_precision
    if labels.unique().numel() < 2:
        return None
    ap = float(average_precision(fake_probs, labels, task="binary")) * 100
    au = float(auroc(fake_probs, labels, task="binary")) * 100
    preds = (fake_probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "auroc": round(au, 2),
            "real_acc": round(real_acc, 2), "fake_acc": round(fake_acc, 2),
            "acc": round((real_acc + fake_acc) / 2, 2)}


# ==========================================================================
# OFFICIAL eval preprocessing — transcribed from PGC_official/data/transforms.py
# ==========================================================================
# create_eval_transforms uses MEAN['dino'] / STD['dino'] which are the ImageNet
# stats; the residual block uses RESIDUAL_MEAN/STD.
_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD = [0.229, 0.224, 0.225]
_RESIDUAL_MEAN = [0.0, 0.0, 0.0]
_RESIDUAL_STD = [0.5, 0.5, 0.5]

# RGB -> YCbCr (BT.601), matches PGC_official residual_extractor.py / paper Eq. 1.
_M_YUV = torch.tensor(
    [[0.299, 0.587, 0.114],
     [-0.168736, -0.331264, 0.5],
     [0.5, -0.418688, -0.081312]], dtype=torch.float32)
_M_YUV_INV = torch.tensor(
    [[1.0, 0.0, 1.402],
     [1.0, -0.344136, -0.714136],
     [1.0, 1.772, 0.0]], dtype=torch.float32)


class QuantizationResidualExtractor(nn.Module):
    """YCbCr quantization residual (PGC_official models/encoder/residual_extractor.py)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("M_t", _M_YUV)
        self.register_buffer("M_t_inv", _M_YUV_INV)

    def forward(self, x):
        x_scaled = x * 255.0
        x_permuted = x_scaled.permute(0, 2, 3, 1)  # [B, H, W, 3]
        x_yuv = torch.matmul(x_permuted, self.M_t.T)
        x_quantized = torch.round(x_yuv)
        x_reconstructed = torch.matmul(x_quantized, self.M_t_inv.T)
        residual = x_permuted - x_reconstructed
        residual = residual.permute(0, 3, 1, 2)
        return torch.clamp(residual, -1.0, 1.0)


class PadCenterCrop:
    """Pad to ``size`` if needed, then deterministic center crop (PGC_official)."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = TF.pad(img, padding, fill=0)
        return transforms.CenterCrop(self.size)(img)


class AppendResidual:
    """Append the YCbCr quant-residual as 3 extra channels (PGC_official).

    The residual is computed on the RGB tensor in [0,1] (BEFORE normalization).
    Output: [6, H, W] where channels 0..2 are RGB, channels 3..5 are residual.
    """

    def __init__(self):
        self.residual_extractor = QuantizationResidualExtractor()

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        assert img_tensor.size(0) == 3, f"Expected 3 channels, got {img_tensor.size(0)}"
        residual = self.residual_extractor(img_tensor.unsqueeze(0)).squeeze(0)
        residual = residual.to(dtype=img_tensor.dtype)
        return torch.cat([img_tensor, residual], dim=0)


def create_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Identical to PGC_official create_eval_transforms (MEAN/STD['dino'])."""
    return transforms.Compose([
        PadCenterCrop(image_size),
        transforms.ToTensor(),
        AppendResidual(),
        transforms.Normalize(mean=_DINO_MEAN + _RESIDUAL_MEAN,
                             std=_DINO_STD + _RESIDUAL_STD),
    ])


# ==========================================================================
# OFFICIAL LoRA — transcribed from PGC_official/models/lora/lora.py
# ==========================================================================
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_out = torch.einsum("...d, rd -> ...r", x, self.lora_A)
        lora_out = torch.einsum("...r, or -> ...o", lora_out, self.lora_B)
        return lora_out * (self.alpha / self.rank)


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0, trainable_orig=False, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        if not trainable_orig:
            for param in self.original_layer.parameters():
                param.requires_grad = False
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.dropout(self.lora(x))
        return original_output + lora_output

    def __getattr__(self, name):
        if name == "weight":
            return self.original_layer.weight
        if name == "bias":
            return self.original_layer.bias
        return super().__getattr__(name)


def _get_submodule(model, submodule_name):
    if not submodule_name:
        return model
    current = model
    for part in submodule_name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def apply_lora_to_linear_layers(model, rank=4, alpha=1.0, target_modules=None,
                                trainable_orig=False, dropout=0.1):
    for name, module in model.named_modules():
        if target_modules is not None and not any(t in name for t in target_modules):
            continue
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name
            parent = model if parent_name == "" else _get_submodule(model, parent_name)
            setattr(parent, child_name,
                    LoRALinear(module, rank, alpha, trainable_orig, dropout))
    return model


# ==========================================================================
# OFFICIAL encoders / PGCM / network — transcribed from PGC_official/models/*
# ==========================================================================
_CHANNELS = {"dinov2-base": 768, "dinov2-large": 1024, "dinov2-giant": 1536}
_HF_NAME = {"dinov2-base": "facebook/dinov2-base",
            "dinov2-large": "facebook/dinov2-large",
            "dinov2-giant": "facebook/dinov2-giant"}
_RESIDUAL_DIM = 256
_DEFAULT_LORA_TARGETS = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]


def _extract_cls_and_tokens(backbone, rgb):
    """HF DINOv2 returns last_hidden_state [B, 1+N, D]; CLS=[:,0], patches=[:,1:]."""
    outputs = backbone(rgb)
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None and isinstance(outputs, dict):
        last_hidden = outputs.get("last_hidden_state", None)
    if last_hidden is None:
        raise RuntimeError("DINO backbone output has no 'last_hidden_state'.")
    return last_hidden[:, 0], last_hidden[:, 1:]


class RgbStream(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        if name not in _CHANNELS:
            raise ValueError(f"Unsupported DINO variant '{name}'.")
        from transformers import AutoModel
        self.name = name
        self._embed_dim = _CHANNELS[name]
        self.backbone = AutoModel.from_pretrained(_HF_NAME[name])

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, rgb):
        return _extract_cls_and_tokens(self.backbone, rgb)


class ResidualStream(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, residual):
        return self.block3(self.block2(self.block1(residual)))


class PeakGuidedCalibrationModule(nn.Module):
    def __init__(self, rgb_dim, residual_dim, tau_rgb=0.5, tau_res=0.5):
        super().__init__()
        self.rgb_score_head = nn.Linear(rgb_dim, 1)
        nn.init.zeros_(self.rgb_score_head.weight)
        nn.init.zeros_(self.rgb_score_head.bias)
        self.residual_score_head = nn.Conv2d(residual_dim, 1, kernel_size=1)
        nn.init.zeros_(self.residual_score_head.weight)
        if self.residual_score_head.bias is not None:
            nn.init.zeros_(self.residual_score_head.bias)
        self.lambda_rgb = nn.Parameter(torch.tensor(1.0))
        if tau_rgb <= 0.0 or tau_res <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau_rgb}/{tau_res}")
        self.tau_rgb = float(tau_rgb)
        self.tau_res = float(tau_res)

    @staticmethod
    def peak_aggregation(scores, tau):
        if scores.dim() != 2:
            raise ValueError(f"Expected [B, N] scores, got {list(scores.shape)}")
        n = scores.size(1)
        if n == 0:
            return scores.new_zeros((scores.size(0), 1))
        log_n = scores.new_tensor(float(n)).log()
        z = tau * (torch.logsumexp(scores / tau, dim=1) - log_n)
        return z.unsqueeze(1)

    def forward(self, f_rgb_tokens, f_residual_map):
        s_rgb = self.rgb_score_head(f_rgb_tokens).squeeze(-1)      # [B, N]
        z_rgb = self.peak_aggregation(s_rgb, self.tau_rgb)         # [B, 1]
        s_res = self.residual_score_head(f_residual_map).flatten(1)  # [B, Hr*Wr]
        z_res = self.peak_aggregation(s_res, self.tau_res)        # [B, 1]
        return z_res + self.lambda_rgb * z_rgb


class PGCNetwork(nn.Module):
    def __init__(self, dino_variant, num_classes=1, lora_rank=8, lora_alpha=1.0,
                 lora_dropout=0.1, lora_targets=None, tau_rgb=0.5, tau_res=0.5):
        super().__init__()
        self.rgb_stream = RgbStream(dino_variant)
        self.residual_stream = ResidualStream(out_channels=_RESIDUAL_DIM)
        self.pgcm = PeakGuidedCalibrationModule(
            rgb_dim=self.rgb_stream.embed_dim, residual_dim=_RESIDUAL_DIM,
            tau_rgb=tau_rgb, tau_res=tau_res)
        in_dim = self.rgb_stream.embed_dim + _RESIDUAL_DIM
        self.classifier = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0.0, 0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # LoRA injection: with HF DINOv2 module names, only mlp.fc1/mlp.fc2 match
        # the official targets (attn.qkv/attn.proj match nothing) — exactly the
        # behaviour reflected in the official checkpoints' LoRA keys.
        targets = _DEFAULT_LORA_TARGETS if lora_targets is None else list(lora_targets)
        self.rgb_stream.backbone = apply_lora_to_linear_layers(
            self.rgb_stream.backbone, rank=lora_rank, alpha=lora_alpha,
            dropout=lora_dropout, target_modules=targets, trainable_orig=False)

        self.dino_variant = dino_variant

    def forward(self, x, return_feature=False):
        rgb = x[:, :3, :, :]
        residual = x[:, 3:, :, :]
        f_rgb_cls, f_rgb_tokens = self.rgb_stream(rgb)
        f_residual_map = self.residual_stream(residual)
        f_res_pooled = torch.flatten(F.adaptive_avg_pool2d(f_residual_map, 1), 1)
        z_local = self.pgcm(f_rgb_tokens, f_residual_map)
        f_global = torch.cat([f_rgb_cls, f_res_pooled], dim=1)
        f_global_normed = F.normalize(f_global, p=2, dim=1)
        z_global = self.classifier(f_global_normed)
        y_pred = z_global + z_local
        if return_feature:
            return f_global, y_pred
        return y_pred


# ==========================================================================
# Scoring
# ==========================================================================
@torch.no_grad()
def score(model, transform, paths: List[Path], bs: int, device) -> torch.Tensor:
    """P(fake) per image. Official infer: probs = sigmoid(z_global + z_local)."""
    probs = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                batch.append(transform(Image.open(p).convert("RGB")))
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        logits = model(x).reshape(x.shape[0])
        probs.append(torch.sigmoid(logits).detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Official-weights PGC benchmark eval (standalone).")
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda", help="cuda|cpu")
    ap.add_argument("--ckpt", required=True, help="official PGC checkpoint (.pth)")
    ap.add_argument("--tag", required=True, help="run tag (output filename + log)")
    ap.add_argument("--out", default=None)
    # Architecture knobs (defaults match all 3 official ckpts; expose for completeness).
    ap.add_argument("--dino_variant", default="dinov2-large")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--tau_rgb", type=float, default=0.5)
    ap.add_argument("--tau_res", type=float, default=0.5)
    ap.add_argument("--cropSize", type=int, default=224)
    args = ap.parse_args(argv)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("WARNING: CUDA requested but unavailable; falling back to CPU.")

    # ---- Build + load official model ----
    print(f"Building PGCNetwork ({args.dino_variant}, lora_rank={args.lora_rank}) ...", flush=True)
    model = PGCNetwork(
        dino_variant=args.dino_variant,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        tau_rgb=args.tau_rgb, tau_res=args.tau_res)
    print(f"Loading official checkpoint: {args.ckpt}", flush=True)
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)  # raises on any key/shape mismatch
    model.to(device).eval()

    transform = create_eval_transforms(image_size=args.cropSize)

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== pgc_official_eval {args.tag} | ckpt={Path(args.ckpt).name} "
          f"| {args.benchmark} | {len(gens)} generators | device={device} ===", flush=True)

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs = score(model, transform, reals + fakes, args.bs, device)
        if probs.numel() != labels.numel():
            labels = labels[:probs.numel()]
        m = metrics(probs, labels)
        if m is None:
            print(f"    skip {name}: one class only")
            continue
        per_gen[name] = m
        for k in agg:
            agg[k].append(m[k])
        print(f"    {name}: acc={m['acc']} AP={m['ap']} AUROC={m['auroc']} "
              f"(real_acc={m['real_acc']} fake_acc={m['fake_acc']})", flush=True)

    if not per_gen:
        print("No generator produced valid metrics.")
        return 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2)

    means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "mauroc": mean(agg["auroc"])}
    result = {
        "tag": args.tag,
        "model": "pgc",
        "ckpt": args.ckpt,
        "benchmark": args.benchmark,
        "data_root": str(root),
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "synthscope_version": "official-vendored",
    }
    out = Path(args.out).expanduser() if args.out else (
        RESULTS_LOG.parent / "bench_results" / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOG.open("a") as f:
        f.write(json.dumps({k: result[k] for k in
                            ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                             "n_generators", "means")}) + "\n")

    print(f"\n=== RESULT {args.tag}: mAP={means['map']} mAUROC={means['mauroc']} acc={means['acc']} "
          f"over {len(per_gen)} generators -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
