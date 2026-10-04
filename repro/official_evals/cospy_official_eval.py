"""Standalone, faithful eval of the OFFICIAL CO-SPY detector (Cheng et al., CVPR'25).

This vendors the official Co-Spy fusion-detector architecture + preprocessing + weight loading
(from github Megum1/Co-Spy detectors/progan/{semantic,artifact,fusion}.py + base.py), so its
official-weights numbers sit side by side with the SynthScope re-implementation in the same
``repro/bench_results`` / ``repro/MASTER_RESULTS.jsonl`` schema as ``benchmark/bench_eval.py``.

The vendored architecture + loading + P(fake) are COPIED from the verified vendored eval
``/hddstore/ssalehi/SynthScope/repro/cospy_eval.py`` (kept faithful). TWO official variants are
supported and AUTO-DETECTED from the semantic fc weight's in-features (768 -> progan/CLIP,
1152 -> sd-v1_4/SigLIP), matching detectors/{progan,sd-v1_4}/* in github Megum1/Co-Spy:

PROGAN variant (CNNDet-trained, semantic fc 768-d):
  - SemanticDetector: transformers.CLIPModel("openai/clip-vit-large-patch14") frozen;
    fc=Linear(768,1); CLIP mean/std normalization.
  - ArtifactDetector: NPR-style truncated ResNet(Bottleneck,[3,4,6,3]); internal artifact =
    x - interp(x,0.5), conv1(artifact*2/3); fc=Linear(512,1); ImageNet mean/std.
  - CoSpyFusionDetector: frozen sem+art; fc=Linear(2,1) over [pred_sem, pred_art];
    test transform = Resize(256) + CenterCrop(224) + ToTensor (per-branch norm inside forward).

SD-V1_4 variant (DRCT-trained, semantic fc 1152-d):
  - SemanticDetector: open_clip "ViT-SO400M-14-SigLIP-384" (webli) frozen; encode_image ->
    1152-d; fc=Linear(1152,1); mean/std=[0.5,0.5,0.5]; resolution 384.
  - ArtifactDetector: VAEReconEncoder over CompVis/stable-diffusion-v1-4 VAE; artifact =
    x - vae_recon(x), scaled x/7*100, then truncated ResNet(Bottleneck,[3,4]) (NO NPR
    interpolate); fc=Linear(512,1); mean/std=[0,0,0]/[1,1,1]; loadSize 256 / crop 224.
  - CoSpyFusionDetector: frozen sem+art; fc=Linear(2,1); outer test transform =
    Resize(384)+CenterCrop(384)+ToTensor; sem branch = Normalize(0.5); art branch =
    Resize(224, antialias=False)+Normalize(0) (downscales the 384 tensor to 224).

  - P(fake) = sigmoid(out); label 1 = fake.

discover()/_images()/_collect()/metrics() are COPIED (not imported) from benchmark/bench_eval.py
so discovery is byte-identical (incl. the Ojha UFD real pairing: guided->imagenet, else->laion).

The ``--ckpt`` here is the triplet DIRECTORY containing semantic_weights.pth, artifact_weights.pth,
and fusion_weights.pth (a directory, NOT a single file).

CLI::
    --benchmark {forensynths,ufd,pergen,pool} --data <root> [--reals] [--select] [--limit]
    --bs N --device cuda|cpu --ckpt <triplet DIR> --tag TAG

Triplet dirs on disk:
  - ProGAN (CNNDet-trained): /hddstore/ssalehi/SynthScope/repro/ckpts/progan/        -> forensynths / ufd
  - SD-v1_4 (DRCT-trained):  /hddstore/ssalehi/SynthScope/repro/official_ckpts/co_spy/sd-v1_4/
                                                                                -> CoSpyBench / GenImage
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from transformers import CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# open_clip + diffusers are only needed by the sd-v1_4 (SigLIP + SD-VAE) variant; imported
# lazily so the progan/CLIP path has no hard dependency on them.

os.environ.setdefault("TORCH_HOME", "/hddstore/ssalehi/.cache/torch")
os.environ.setdefault("HF_HOME", "/hddstore/ssalehi/.cache/huggingface")

# ---------------------------------------------------------------------------
# Paths / constants (mirror benchmark/bench_eval.py)
# ---------------------------------------------------------------------------
REPRO_DIR = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro")
RESULTS_LOG = REPRO_DIR / "MASTER_RESULTS.jsonl"
BENCH_RESULTS_DIR = REPRO_DIR / "bench_results"
DEFAULT_CKPT = Path("/hddstore/ssalehi/SynthScope/repro/ckpts/progan")  # triplet directory

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
# Vendored official Co-Spy architecture (copied from repro/cospy_eval.py).
# ===========================================================================
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
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
    """NPR-style truncated ResNet used as the Co-Spy artifact encoder."""

    def __init__(self, block, layers, num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)  # defined in official code, unused in forward

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def interpolate(self, img, factor):
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='nearest',
                          recompute_scale_factor=True),
            scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)

    def forward(self, x):
        artifact = x - self.interpolate(x, 0.5)
        x = self.relu(self.bn1(self.conv1(artifact * 2.0 / 3.0)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class SemanticDetector(nn.Module):
    def __init__(self, dim_clip=768, num_classes=1):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.requires_grad_(False)
        self.fc = nn.Linear(dim_clip, num_classes)
        self.mean = [0.48145466, 0.45782750, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

    def forward(self, x):
        # transformers 5.x get_image_features returns BaseModelOutputWithPooling whose
        # .pooler_output is the projected 768-d image embedding (== classic 4.x tensor
        # return). Handle both APIs.
        out = self.clip.get_image_features(pixel_values=x)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        return self.fc(feat)

    def load_weights(self, path):
        w = torch.load(path, map_location='cpu')
        self.fc.weight.data = w["fc.weight"]
        self.fc.bias.data = w["fc.bias"]


class ArtifactDetector(nn.Module):
    def __init__(self, dim_artifact=512, num_classes=1):
        super().__init__()
        self.artifact_encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        self.fc = nn.Linear(dim_artifact, num_classes)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x):
        return self.fc(self.artifact_encoder(x))

    def load_weights(self, path):
        w = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(w, strict=False)
        return missing, unexpected


class CoSpyFusionDetector(nn.Module):
    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1):
        super().__init__()
        self.sem = SemanticDetector()
        self.sem.load_weights(semantic_weights_path)
        self.art = ArtifactDetector()
        am, au = self.art.load_weights(artifact_weights_path)
        print(f"artifact load: missing={am}, unexpected={au}", flush=True)
        for p in self.sem.parameters():
            p.requires_grad = False
        for p in self.art.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(2, num_classes)
        self.sem_transform = T.Normalize(self.sem.mean, self.sem.std)
        self.art_transform = T.Normalize(self.art.mean, self.art.std)
        # Official test transform (no normalization here; per-branch in forward).
        self.test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

    def forward(self, x):
        pred_sem = self.sem(self.sem_transform(x))
        pred_art = self.art(self.art_transform(x))
        feat = torch.cat([pred_sem, pred_art], dim=1)
        return self.fc(feat)

    def load_weights(self, path):
        w = torch.load(path, map_location='cpu')
        self.fc.weight.data = w["fc.weight"]
        self.fc.bias.data = w["fc.bias"]


# ===========================================================================
# Vendored official Co-Spy SD-V1_4 (DRCT-trained) variant.
# Faithful to github Megum1/Co-Spy detectors/sd-v1_4/{semantic,artifact,fusion}.py.
# Semantic = open_clip SigLIP-SO400M-384 (1152-d); artifact = SD-v1.4 VAE recon residual.
# ===========================================================================
class SemanticDetectorSD(nn.Module):
    """sd-v1_4 semantic stream: open_clip ViT-SO400M-14-SigLIP-384 (webli), 1152-d."""

    def __init__(self, dim_clip=1152, num_classes=1):
        super().__init__()
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained="webli")
        self.clip.requires_grad_(False)
        self.fc = nn.Linear(dim_clip, num_classes)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def forward(self, x):
        feat = self.clip.encode_image(x)
        return self.fc(feat)

    def load_weights(self, path):
        w = torch.load(path, map_location='cpu')
        self.fc.weight.data = w["fc.weight"]
        self.fc.bias.data = w["fc.bias"]


class VAEReconEncoder(nn.Module):
    """sd-v1_4 artifact encoder: SD-v1.4 VAE reconstruction residual -> truncated ResNet."""

    def __init__(self, vae, block=Bottleneck):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vae = vae

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def reconstruct(self, x):
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mean
            decoded = self.vae.decode(latent).sample
        return decoded

    def forward(self, x):
        x_recon = self.reconstruct(x)
        x = x - x_recon
        x = x / 7. * 100.
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class ArtifactDetectorSD(nn.Module):
    def __init__(self, dim_artifact=512, num_classes=1):
        super().__init__()
        # Official code does StableDiffusionPipeline.from_pretrained(...).vae; we load the VAE
        # subfolder directly (functionally identical, and robust when only the vae/ snapshot is
        # cached offline -- the full pipeline's model_index.json may be absent).
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        vae.requires_grad_(False)
        self.artifact_encoder = VAEReconEncoder(vae)
        self.fc = nn.Linear(dim_artifact, num_classes)
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]
        self.cropSize = 224

    def forward(self, x):
        return self.fc(self.artifact_encoder(x))

    def load_weights(self, path):
        w = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(w, strict=False)
        return missing, unexpected


class CoSpyFusionDetectorSD(nn.Module):
    """sd-v1_4 fusion: SigLIP semantic + SD-VAE artifact; outer transform Resize/Crop 384."""

    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1):
        super().__init__()
        self.sem = SemanticDetectorSD()
        self.sem.load_weights(semantic_weights_path)
        self.art = ArtifactDetectorSD()
        am, au = self.art.load_weights(artifact_weights_path)
        # only the VAE may have benign mismatches; report any non-vae missing/unexpected
        am_n = [k for k in am if not k.startswith("artifact_encoder.vae.")]
        au_n = [k for k in au if not k.startswith("artifact_encoder.vae.")]
        print(f"artifact load: non-vae missing={am_n}, non-vae unexpected={au_n} "
              f"(vae keys handled separately)", flush=True)
        for p in self.sem.parameters():
            p.requires_grad = False
        for p in self.art.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(2, num_classes)
        # sem branch: normalize the 384 tensor only.
        self.sem_transform = T.Normalize(self.sem.mean, self.sem.std)
        # art branch: downscale the 384 tensor to 224, then normalize.
        self.art_transform = T.Compose([
            T.Resize(self.art.cropSize, antialias=False),
            T.Normalize(self.art.mean, self.art.std),
        ])
        # Official test transform: Resize(384) + CenterCrop(384) + ToTensor (no norm here).
        self.test_transform = T.Compose([
            T.Resize(384),
            T.CenterCrop(384),
            T.ToTensor(),
        ])

    def forward(self, x):
        pred_sem = self.sem(self.sem_transform(x))
        pred_art = self.art(self.art_transform(x))
        feat = torch.cat([pred_sem, pred_art], dim=1)
        return self.fc(feat)

    def load_weights(self, path):
        w = torch.load(path, map_location='cpu')
        self.fc.weight.data = w["fc.weight"]
        self.fc.bias.data = w["fc.bias"]


def _semantic_in_features(sem_w: Path) -> int:
    """Read the semantic fc in-features to auto-detect the Co-Spy variant."""
    w = torch.load(sem_w, map_location='cpu')
    return int(w["fc.weight"].shape[1])


def build_model(ckpt_dir: Path, device: torch.device):
    sem_w = ckpt_dir / "semantic_weights.pth"
    art_w = ckpt_dir / "artifact_weights.pth"
    fus_w = ckpt_dir / "fusion_weights.pth"
    for p in (sem_w, art_w, fus_w):
        if not p.is_file():
            raise FileNotFoundError(f"missing triplet weight: {p}")
    dim = _semantic_in_features(sem_w)
    if dim == 768:
        print("co_spy variant: PROGAN (CLIP-L 768-d semantic)", flush=True)
        model = CoSpyFusionDetector(str(sem_w), str(art_w))
    elif dim == 1152:
        print("co_spy variant: SD-V1_4 (SigLIP 1152-d semantic + SD-VAE artifact)", flush=True)
        model = CoSpyFusionDetectorSD(str(sem_w), str(art_w))
    else:
        raise ValueError(
            f"unsupported semantic fc in-features={dim} (expected 768 progan or 1152 sd-v1_4)")
    model.load_weights(str(fus_w))
    print("loaded fusion fc:", {k: tuple(v.shape) for k, v in
                                {"w": model.fc.weight, "b": model.fc.bias}.items()}, flush=True)
    return model.to(device).eval()


@torch.no_grad()
def score(model, transform, device, paths: list[Path], bs: int) -> torch.Tensor:
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
        logits = model(x).squeeze(-1)
        probs.append(torch.sigmoid(logits).detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="triplet DIRECTORY: semantic/artifact/fusion_weights.pth")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--tag", required=True, help="run tag (output filename + log line)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_dir = Path(args.ckpt).expanduser()
    if not ckpt_dir.is_dir():
        print(f"ckpt must be a triplet DIRECTORY (semantic/artifact/fusion): {ckpt_dir}")
        return 1

    model = build_model(ckpt_dir, device)
    transform = model.test_transform

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== cospy_official_eval {args.tag} | model=co_spy (official-vendored) ckpt={ckpt_dir} "
          f"| {args.benchmark} | {len(gens)} generators | device={device} ===", flush=True)

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs = score(model, transform, device, reals + fakes, args.bs)
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
        "model": "co_spy",
        "ckpt": str(ckpt_dir),
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
