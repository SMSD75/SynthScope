"""Standalone, FAITHFUL eval of the OFFICIAL AIDE detector (Yan et al., ICLR'25).

This vendors the GENUINE official AIDE network + its real multi-view preprocessing so the
official-weights numbers can sit side by side with the SynthScope re-implementation in the same
``repro/bench_results`` / ``repro/MASTER_RESULTS.jsonl`` schema as ``benchmark/bench_eval.py``.

WHY THE PREVIOUS VERSION WAS WRONG
----------------------------------
The old eval (and the so-called "reference" ``/hddstore/.../repro/aide_eval.py``) ran the SynthScope
``SynthScopeAIDEModel``: a *single* CLIP-normalized 224x224 view whose DCT/SRM preprocessing is done
*inside* the model, with a torchvision ResNet-50 trunk and the official ``model_min.fc`` /
``model_max.fc`` weights DROPPED. That is an explicitly documented approximation (the file even
says "not bit-exact"), so its "official" numbers collapsed to the SynthScope re-impl (64.79 mAP,
real_acc ~chance on adm/biggan/glide).

The REAL official AIDE (https://github.com/shilinyan99/AIDE, ``models/AIDE.py`` + ``data/dct.py``
+ ``data/datasets.py``) is fundamentally a 5-VIEW model:

  preprocessing (TestDataset, vendored verbatim below):
    1. ``ToTensor()`` only (NO resize) -> image stays at native size in [0,1].
    2. ``DCT_base_Rec_Module`` on the native-size tensor scores 32x32 windows over K=6 log-DCT
       bands (weights 2^k), reconstructs each picked window via the level filter + Fold, and
       returns the 2 highest-energy + 2 lowest-energy 32x32 patches.
    3. ``transform_train`` = Resize([256,256]) + Normalize(IMAGENET/DINOv2 mean/std) applied to
       EACH of the 4 patches AND to the full image (the ``tokens`` view).
    4. stack -> ``[5, 3, 256, 256]`` = [x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0].

  model (AIDE_Model, vendored verbatim below):
    - low-frequency branch: HPF (30 SRM kernels, 3->30) then a CUSTOM ResNet-50 (Bottleneck,
      conv1 = 30->64, fc kept as Linear(2048,1000) so the ckpt loads strict). The 2 max patches
      go through ``model_max``, the 2 min patches through ``model_min``; the 4 pooled 2048-d
      vectors are averaged -> x_1 (2048-d).
    - semantic branch: the ``tokens`` view is re-normalized DINOv2->CLIP space via
      ``tokens*(dinov2_std/clip_std) + (dinov2_mean-clip_mean)/clip_std`` and fed to the frozen
      OpenCLIP ``convnext_xxlarge`` trunk -> [B,3072,8,8] -> avgpool -> Linear(3072,256) -> x_0.
    - head: ``Mlp(2048+256, 1024, 2)`` on cat([x_0, x_1]).
    - P(fake) = softmax(out, dim=1)[:, 1].

The official 3.59GB checkpoints bundle the ConvNeXt + ResNet weights, so we build the OpenCLIP
trunk with ``pretrained=None`` (no 4.8GB download) and load all 1025 keys STRICT.

discover()/_images()/_collect()/metrics() are COPIED VERBATIM (not imported) from
benchmark/bench_eval.py so discovery is byte-identical (incl. the Ojha UFD real pairing:
guided->imagenet, else->laion).

CLI (unchanged)::
    --benchmark {forensynths,ufd,pergen,pool} --data <root> [--reals] [--select] [--limit]
    --bs N --device cuda|cpu --ckpt <official AIDE .pth> --tag TAG

Official AIDE ckpts on disk (official_ckpts/aide/AIDE_checkpoints/):
  - progan_train.pth   (ProGAN 4-class)  -> ForenSynths / AIGCDetect / UFD (Table 3 lineage)
  - GenImage_train.pth (GenImage)         -> GenImage-8                     (Table 4 lineage)
  - sd14_train.pth     (SD-v1.4)
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# SynthScope repo (only for the verified-identical SRM kernel list) + caches.
REPO = "/ivi/zfs/s0/original_homes/ssalehi/SynthScope"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("TORCH_HOME", "/hddstore/ssalehi/.cache/torch")
os.environ.setdefault("HF_HOME", "/hddstore/ssalehi/.cache/huggingface")

import open_clip  # noqa: E402
from models.image.aide.srm_kernels import all_normalized_hpf_list  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants (mirror benchmark/bench_eval.py)
# ---------------------------------------------------------------------------
REPRO_DIR = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro")
RESULTS_LOG = REPRO_DIR / "MASTER_RESULTS.jsonl"
BENCH_RESULTS_DIR = REPRO_DIR / "bench_results"
DEFAULT_CKPT = Path("/hddstore/ssalehi/SynthScope/repro/official_ckpts/aide/"
                    "AIDE_checkpoints/progan_train.pth")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
REAL_DIRNAMES = ("0_real", "real", "reals")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "ai")
SHARED_REAL_DIRS = {"imagenet", "laion", "_real_pool", "real", "0_real", "reals"}
# Ojha UniversalFakeDetect real pairing (verified against the official dataset_paths.py).
UFD_REAL = {"guided": "imagenet"}  # everything else -> laion
UFD_DEFAULT_REAL = "laion"


# ===========================================================================
# COPIED VERBATIM from benchmark/bench_eval.py (identical discovery + metrics).
# ===========================================================================
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


# ===========================================================================
# GENUINE OFFICIAL AIDE: data/dct.py (vendored VERBATIM from shilinyan99/AIDE).
# ===========================================================================
def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) *
          np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable
        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)),
                                 requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                requires_grad=False)

    def forward(self, x):
        filt = self.base + norm_sigma(self.learnable) if self.use_learnable else self.base
        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class DCT_base_Rec_Module(nn.Module):
    """x: [C, H, W] -> picks 4 patches each [C, window, window] (window_size=32)."""

    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()
        assert output % window_size == 0
        assert len(level_fliter) > 0
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)

        self._DCT_patch = nn.Parameter(
            torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
        self.fold0 = nn.Fold(output_size=(window_size, window_size),
                             kernel_size=(window_size, window_size), stride=window_size)

        level_f = [Filter(window_size, 0, window_size * 2)]
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([
            Filter(window_size, window_size * 2. / grade_N * i,
                   window_size * 2. / grade_N * (i + 1), norm=True) for i in range(grade_N)])

    def forward(self, x):
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)

        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size)
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)

        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1, 2, 3])
            grade += w * _x
            w *= k

        _, idx = torch.sort(grade)
        max_idx = torch.flip(idx, dims=[0])[:N]
        maxmax_idx = max_idx[0]
        maxmax_idx1 = max_idx[0] if len(max_idx) == 1 else max_idx[1]

        min_idx = idx[:N]
        minmin_idx = idx[0]
        minmin_idx1 = idx[0] if len(min_idx) == 1 else idx[1]

        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx)
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx)
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1)
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1)

        x_minmin = x_minmin.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)

        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)
        return x_minmin, x_maxmax, x_minmin1, x_maxmax1


# ===========================================================================
# GENUINE OFFICIAL AIDE: models/AIDE.py (vendored VERBATIM from shilinyan99/AIDE).
# The SRM kernel list is the SynthScope one (verified bit-identical to the official).
# ===========================================================================
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode="constant")
            all_hpf_list_5x5.append(hpf_item)
        hpf_weight = torch.Tensor(np.array(all_hpf_list_5x5)).view(30, 1, 5, 5).contiguous()
        hpf_weight = nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, input):
        return self.hpf(input)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # (init omitted; weights come from the official checkpoint)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AIDE_Model(nn.Module):
    def __init__(self, convnext_pretrained=None):
        super(AIDE_Model, self).__init__()
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])
        self.fc = Mlp(2048 + 256, 1024, 2)

        # Build the OpenCLIP ConvNeXt-XXL trunk. pretrained=None => no 4.8GB download;
        # the official 3.59GB checkpoint supplies the trunk weights.
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=convnext_pretrained)
        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()
        self.openclip_convnext_xxl.eval()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(nn.Linear(3072, 256))
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: [B, 5, 3, 256, 256] -> [minmin, maxmax, minmin1, maxmax1, tokens]
        b, t, c, h, w = x.shape
        x_minmin = self.hpf(x[:, 0])
        x_maxmax = self.hpf(x[:, 1])
        x_minmin1 = self.hpf(x[:, 2])
        x_maxmax1 = self.hpf(x[:, 3])
        tokens = x[:, 4]

        with torch.no_grad():
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(tokens).view(3, 1, 1)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(tokens).view(3, 1, 1)
            dinov2_mean = torch.tensor([0.485, 0.456, 0.406]).to(tokens).view(3, 1, 1)
            dinov2_std = torch.tensor([0.229, 0.224, 0.225]).to(tokens).view(3, 1, 1)
            local_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std)
            assert local_feats.size()[1:] == (3072, 8, 8), local_feats.size()
            local_feats = self.avgpool(local_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_feats)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4

        return self.fc(torch.cat([x_0, x_1], dim=1))


# ===========================================================================
# Official AIDE TEST preprocessing (vendored VERBATIM from data/datasets.py).
# ===========================================================================
# ToTensor only, then DCT on the native-size tensor.
transform_before_test = transforms.Compose([transforms.ToTensor()])
# Resize to 256 + DINOv2/ImageNet normalize, applied to each view tensor.
transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _Preproc:
    """Replicates AIDE TestDataset.__getitem__ -> [5, 3, 256, 256]."""

    def __init__(self):
        self.dct = DCT_base_Rec_Module()

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        image = transform_before_test(pil_img)  # [3, H, W] in [0,1], native size
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin)
        x_maxmax = transform_train(x_maxmax)
        x_minmin1 = transform_train(x_minmin1)
        x_maxmax1 = transform_train(x_maxmax1)
        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)


# ===========================================================================
# Build + load (strict).
# ===========================================================================
def build_model(ckpt: Path, device: torch.device):
    model = AIDE_Model(convnext_pretrained=None)
    ck = torch.load(str(ckpt), map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load: ckpt has {len(sd)} keys. missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    if missing:
        print(f"  missing(sample): {list(missing)[:10]}", flush=True)
    if unexpected:
        print(f"  unexpected(sample): {list(unexpected)[:10]}", flush=True)
    # The genuine AIDE_Model is key-for-key identical to the official checkpoint.
    assert not missing, f"MISSING: {missing}"
    assert not unexpected, f"UNEXPECTED: {unexpected}"
    return model.to(device).eval()


@torch.no_grad()
def score(model, preproc, device, paths: list[Path], bs: int) -> torch.Tensor:
    probs = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                batch.append(preproc(Image.open(p).convert("RGB")))
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(device)  # [B, 5, 3, 256, 256]
        out = model(x)
        probs.append(out.softmax(dim=1)[:, 1].detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="official AIDE checkpoint .pth")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--tag", required=True, help="run tag (output filename + log line)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        print(f"ckpt not found: {ckpt}")
        return 1

    model = build_model(ckpt, device)
    preproc = _Preproc()

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== aide_official_eval {args.tag} | model=aide (official-vendored) ckpt={ckpt} "
          f"| {args.benchmark} | {len(gens)} generators | device={device} ===", flush=True)

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs = score(model, preproc, device, reals + fakes, args.bs)
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
        "model": "aide",
        "ckpt": str(ckpt),
        "benchmark": args.benchmark,
        "data_root": str(root),
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "synthscope_version": "official-vendored",
    }
    out = Path(args.out).expanduser() if args.out else (BENCH_RESULTS_DIR / f"{args.tag}.json")
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
