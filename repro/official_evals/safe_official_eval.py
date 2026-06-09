"""Standalone OFFICIAL SAFE (Li et al., KDD'25) evaluation — official-weights numbers.

This script vendors the EXACT official SAFE architecture + eval preprocessing from
github.com/Ouxiang-Li/SAFE so the in-repo official checkpoint loads faithfully.
The AGID `safe` class does NOT load the official weights correctly (in-domain ProGAN
AP came out ~32, i.e. random), so this is a separate, self-contained reference run.

Faithfully vendored from the official repo:
  * models/resnet.py: a CUSTOM ResNet (conv1 3x3/stride2, NO 7x7 stem, only layer1 +
    layer2 of Bottleneck blocks [3, 4], AdaptiveAvgPool, fc1=Linear(512, 2)). The
    high-frequency DWT transform is INSIDE the model.forward (bior1.3, J=1,
    mode='symmetric'), taking the HH (diagonal) band Yh[0][:, :, 2, :, :] and resizing
    it back to the input size with torchvision transforms.Resize. The head is 2-logit,
    so P(fake) = softmax(logits)[:, 1] (matches official engine_finetune.evaluate()).
  * data/datasets.py Get_Transforms(transform_mode='crop', eval): CenterCrop([256, 256])
    + ToTensor(), NO normalization (scripts/eval.sh: --input_size 256 --transform_mode
    'crop').

The dataset discovery / real-pairing (discover / _images / _collect / metrics) is COPIED
VERBATIM from benchmark/bench_eval.py so the generator grouping + UFD pairing are identical
to the AGID re-implementation runs (apples-to-apples).

Output JSON + MASTER_RESULTS.jsonl line match bench_eval.py exactly, with
model="safe", agid_version="official-vendored".

CLI:
  python repro/official_evals/safe_official_eval.py \
      --benchmark {forensynths,ufd,pergen,pool} --data <root> \
      [--reals DIR] [--select g1,g2] [--limit N] --bs 32 --device cuda|cpu \
      --ckpt <checkpoint-best.pth> --tag TAG
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------------------------------------------------------------------------
# Discovery + metrics: COPIED VERBATIM from benchmark/bench_eval.py (do not import).
# --------------------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
REAL_DIRNAMES = ("0_real", "real", "reals")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "ai")
SHARED_REAL_DIRS = {"imagenet", "laion", "_real_pool", "real", "0_real", "reals"}
RESULTS_LOG = Path("/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/MASTER_RESULTS.jsonl")
# Ojha UniversalFakeDetect real pairing (verified against the official dataset_paths.py).
UFD_REAL = {"guided": "imagenet"}  # everything else -> laion
UFD_DEFAULT_REAL = "laion"


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


# --------------------------------------------------------------------------------------
# OFFICIAL SAFE architecture — vendored VERBATIM from Ouxiang-Li/SAFE models/resnet.py.
# --------------------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
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
    """Official SAFE ResNet (custom: conv1 3x3/s2, only layer1+layer2, fc1=Linear(512,2))."""

    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        self.unfoldSize = 2
        self.unfoldIndex = 0
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

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

    def _preprocess_dwt(self, x, mode="symmetric", wave="bior1.3"):
        from pytorch_wavelets import DWTForward
        DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        Yl, Yh = DWT_filter(x)
        return transforms.Resize([x.shape[-2], x.shape[-1]])(Yh[0][:, :, 2, :, :])

    def forward(self, x):
        x = 1 * self._preprocess_dwt(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def safe_resnet50(num_classes=2):
    """Official SAFE model: ResNet(Bottleneck, [3, 4], num_classes=2)."""
    return ResNet(Bottleneck, [3, 4], num_classes=num_classes)


# Official eval transform (data/datasets.py Get_Transforms, transform_mode='crop', eval):
# CenterCrop([256, 256]) then ToTensor(), NO normalization. input_size=256 from eval.sh.
EVAL_TRANSFORM = transforms.Compose([
    transforms.CenterCrop([256, 256]),
    transforms.ToTensor(),
])


def load_model(ckpt: str, device: torch.device) -> nn.Module:
    model = safe_resnet50(num_classes=2)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck.get("net", ck)))
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load] MISSING keys ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"  [load] UNEXPECTED keys ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")
    model.to(device).eval()
    return model


@torch.no_grad()
def score(model: nn.Module, paths: list[Path], bs: int, device: torch.device) -> torch.Tensor:
    """P(fake) per image = softmax(logits)[:, 1] (official 2-logit head)."""
    probs = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                batch.append(EVAL_TRANSFORM(Image.open(p).convert("RGB")))
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        logits = model(x)
        p_fake = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p_fake.detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Standalone official SAFE eval")
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--ckpt", required=True, help="official checkpoint-best.pth")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tag", required=True, help="run tag (used for the output filename + log)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_model(args.ckpt, device)

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== safe_official_eval {args.tag} | model=safe (official-vendored) ckpt={args.ckpt} "
          f"| {args.benchmark} | {len(gens)} generators | device={device} ===", flush=True)

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        probs = score(model, reals + fakes, args.bs, device)
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
        "model": "safe",
        "ckpt": args.ckpt,
        "benchmark": args.benchmark,
        "data_root": str(root),
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "agid_version": "official-vendored",
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
