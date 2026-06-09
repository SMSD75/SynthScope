"""Standalone, FAITHFUL evaluation of the OFFICIAL CNNDetection detector (Wang et al., CVPR'20).

The AGID ``cnn`` re-implementation does NOT load the official weights faithfully
(in-domain ProGAN AP came out ~72 instead of ~100). This script vendors the EXACT
official ResNet-50 architecture (``networks/resnet.py``) and the EXACT official eval
preprocessing so we can report official-weights numbers side by side with the AGID
re-implementation.

OFFICIAL PIPELINE (verified against github.com/PeterWang512/CNNDetection @ master):
  * Architecture: ``resnet50(num_classes=1)`` — torchvision-style ResNet-50, fc -> 1 logit.
    Weights live under state_dict['model']; keys have NO ``module.`` prefix.
  * Score: ``model(x).sigmoid()`` == P(fake).  (validate.py / demo.py)
  * Eval preprocessing (eval.py sets ``opt.no_resize = True``):
        - NO resize (rz_func = identity, because --no_resize)
        - NO test-time augmentation (blur_prob = jpg_prob = 0)
        - CenterCrop(cropSize=224) by default  [paper Table-2 / "Acc_224/AP_224" protocol]
          OR identity crop if --no_crop          ["recommended No-crop" / "*_nocrop" protocol]
        - ToTensor()
        - Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    i.e. transforms.Compose([CenterCrop(224)?, ToTensor(), Normalize(IMAGENET)])

The discover()/_images()/_collect()/metrics() helpers below are COPIED VERBATIM from
``benchmark/bench_eval.py`` so generator discovery + UFD real-pairing + metrics are
IDENTICAL to the AGID re-implementation runs and the aggregator picks up the output.

CLI:
  python repro/official_evals/cnn_official_eval.py \
      --benchmark {forensynths,ufd,pergen,pool} --data <root> \
      [--reals R] [--select g1,g2] [--limit N] [--no-crop] [--crop-size 224] \
      --bs 32 --device cuda|cpu --ckpt <path> --tag TAG

OUTPUT (identical schema to bench_eval.py):
  repro/bench_results/<tag>.json  +  appended line in repro/MASTER_RESULTS.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Paths / constants — kept in sync with benchmark/bench_eval.py
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
REAL_DIRNAMES = ("0_real", "real", "reals")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "ai")
SHARED_REAL_DIRS = {"imagenet", "laion", "_real_pool", "real", "0_real", "reals"}
# This file lives at repro/official_evals/cnn_official_eval.py -> repro is parents[1].
RESULTS_LOG = Path(__file__).resolve().parents[1] / "MASTER_RESULTS.jsonl"
BENCH_RESULTS_DIR = Path(__file__).resolve().parents[1] / "bench_results"
# Ojha UniversalFakeDetect real pairing (verified against the official dataset_paths.py).
UFD_REAL = {"guided": "imagenet"}  # everything else -> laion
UFD_DEFAULT_REAL = "laion"

# ImageNet normalization used by the official CNNDetection eval.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ===========================================================================
# Vendored OFFICIAL ResNet-50 (networks/resnet.py from PeterWang512/CNNDetection)
# ===========================================================================
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(num_classes=1000):
    """Constructs the official ResNet-50 model (no pretrained download here)."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def load_official_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """Build resnet50(num_classes=1) and load the official checkpoint faithfully."""
    model = resnet50(num_classes=1)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Strip any DataParallel 'module.' prefix just in case.
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)  # strict: every key must match exactly
    model.to(device).eval()
    return model


def build_transform(no_crop: bool, crop_size: int) -> transforms.Compose:
    """EXACT official eval preprocessing (eval.py sets no_resize=True, so no resize).

    Default (paper Table-2 / Acc_224/AP_224 protocol): CenterCrop(crop_size).
    --no-crop (repo's recommended No-crop protocol):    identity (full-resolution image).
    Always: ToTensor() + Normalize(ImageNet mean/std).
    """
    steps = []
    if not no_crop:
        steps.append(transforms.CenterCrop(crop_size))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


# ===========================================================================
# COPIED VERBATIM from benchmark/bench_eval.py (discovery + metrics)
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
# Scoring with the vendored OFFICIAL model + OFFICIAL preprocessing
# ===========================================================================
@torch.no_grad()
def score(model: nn.Module, transform: transforms.Compose, paths: list[Path],
          bs: int, device: torch.device) -> torch.Tensor:
    """sigmoid(logit) == P(fake), exactly like the official validate.py / demo.py.

    NOTE: when --no-crop, images keep their native (varied) sizes, so we score one
    image at a time to avoid stacking mismatched tensors — this matches the official
    repo's recommended ``eval.py --no_crop --batch_size 1`` protocol.
    """
    probs = []
    batch: list[torch.Tensor] = []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        logits = model(x).flatten()
        probs.append(logits.sigmoid().detach().float().cpu())
        batch.clear()

    for p in paths:
        try:
            t = transform(Image.open(p).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            print(f"    bad image {p}: {e}")
            continue
        if batch and batch[-1].shape != t.shape:
            # Variable-sized (no-crop) images can't be stacked together -> flush first.
            flush()
        batch.append(t)
        if len(batch) >= bs:
            flush()
    flush()
    return torch.cat(probs) if probs else torch.empty(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Official CNNDetection standalone eval")
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--ckpt", required=True, help="official ResNet-50 checkpoint (.pth)")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cpu", help="cuda or cpu")
    ap.add_argument("--no-crop", dest="no_crop", action="store_true",
                    help="official 'No-crop' protocol (full-res); default is CenterCrop(224)")
    ap.add_argument("--crop-size", dest="crop_size", type=int, default=224,
                    help="center-crop size for the default 224 protocol")
    ap.add_argument("--tag", required=True, help="run tag (output filename + log key)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: --device cuda requested but CUDA unavailable; falling back to CPU.")

    model = load_official_model(args.ckpt, device)
    transform = build_transform(args.no_crop, args.crop_size)
    proto = "no-crop(full-res)" if args.no_crop else f"center-crop({args.crop_size})"

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== cnn_official_eval {args.tag} | model=cnn(official) ckpt={args.ckpt} "
          f"| {args.benchmark} | {len(gens)} generators | preproc={proto}+ToTensor+ImageNetNorm "
          f"| device={device} ===", flush=True)

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
        "model": "cnn",
        "ckpt": args.ckpt,
        "benchmark": args.benchmark,
        "data_root": str(root),
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "agid_version": "official-vendored",
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
