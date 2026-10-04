"""Standalone, faithful eval of the OFFICIAL UnivFD / CLIPDetection detector (Ojha et al., CVPR'23).

This bypasses the SynthScope `clip` class (which did not load the official linear head faithfully —
in-domain ProGAN came out random, fake_acc=0) and vendors the official pipeline end to end so the
official-weights numbers can sit side by side with the SynthScope re-implementation in the same
``repro/bench_results`` / ``repro/MASTER_RESULTS.jsonl`` schema as ``benchmark/bench_eval.py``.

Official model (github.com/WisconsinAIVision/UniversalFakeDetect, the "LC" linear-classifier variant):
    OpenAI CLIP ViT-L/14 visual encoder (FROZEN) -> 768-d image feature -> Linear(768, 1) head.
    P(fake) = sigmoid(logit).

Backbone load (matches models/clip_models.py -> `clip.load(name, device="cpu")`):
    clip.load("ViT-L/14", device=<dev>)  using the OpenAI `clip` package (NOT open_clip).
    Weights auto-download / are cached under TORCH_HOME / ~/.cache/clip (ViT-L-14.pt).

Linear head (matches validate.py -> `model.fc.load_state_dict(state_dict)`):
    fc_weights.pth is JUST the Linear head state dict: {weight:[1,768], bias:[1]}.

Preprocessing — replicated VERBATIM from the official validate.py `RealFakeDataset.transform`
(NOT the generic clip.load preprocess). The official TEST transform is:
    CenterCrop(224) -> ToTensor -> Normalize(mean/std = CLIP stats)
There is deliberately NO Resize in the official validation path: ForenSynths/UFD-fake images are
256x256 (center-crop to 224 directly), and UFD ImageNet reals are larger and also center-cropped
without resizing. Reproducing the official numbers requires matching this exactly, so we do.
    CLIP mean = [0.48145466, 0.4578275, 0.40821073]
    CLIP std  = [0.26862954, 0.26130258, 0.27577711]

discover()/_images()/_collect()/metrics() are COPIED (not imported) from benchmark/bench_eval.py so
discovery is byte-identical (incl. the Ojha UFD real pairing: guided->imagenet, everything else->laion).

CLI:
    --benchmark {forensynths,ufd,pergen,pool} --data <root> [--reals] [--select] [--limit]
    --bs 32 --device cuda|cpu --ckpt <path> --tag TAG
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Paths / constants (mirror benchmark/bench_eval.py)
# ---------------------------------------------------------------------------
# Outputs live on the ivi share; the official ckpts live on node-local hddstore.
REPRO_DIR = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro")
RESULTS_LOG = REPRO_DIR / "MASTER_RESULTS.jsonl"
BENCH_RESULTS_DIR = REPRO_DIR / "bench_results"
DEFAULT_CKPT = Path("/hddstore/ssalehi/SynthScope/repro/official_ckpts/clip/fc_weights.pth")

# OpenAI CLIP normalization (validate.py MEAN["clip"], STD["clip"]).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Cache locations so ViT-L/14 weights resolve without re-download.
os.environ.setdefault("TORCH_HOME", "/hddstore/ssalehi/.cache/torch")
os.environ.setdefault("HF_HOME", "/hddstore/ssalehi/.cache/huggingface")

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
# Official UnivFD model (vendored from clip_models.py + validate.py).
# ===========================================================================
class OfficialUnivFD(nn.Module):
    """CLIP:ViT-L/14 frozen visual encoder + Linear(768, 1) head (the LC variant)."""

    def __init__(self, ckpt: Path, device: torch.device):
        super().__init__()
        import clip  # OpenAI clip package (models/clip_models.py uses this, NOT open_clip)
        # clip_models.py: self.model, _ = clip.load(name, device="cpu")
        self.model, _ = clip.load("ViT-L/14", device="cpu")
        self.fc = nn.Linear(768, 1)
        # validate.py: model.fc.load_state_dict(torch.load(ckpt, map_location='cpu'))
        state_dict = torch.load(str(ckpt), map_location="cpu")
        self.fc.load_state_dict(state_dict)
        self.eval()
        # Keep the backbone in fp32 on whatever device (matches CPU/GPU eval; frozen).
        self.model = self.model.float().to(device)
        self.fc = self.fc.float().to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clip_models.py forward: self.fc(self.model.encode_image(x))
        features = self.model.encode_image(x).float()
        return self.fc(features)


def build_transform() -> T.Compose:
    """Replicate validate.py RealFakeDataset.transform for arch='CLIP:...' (stat_from='clip').

    CenterCrop(224) -> ToTensor -> Normalize(CLIP mean/std).  NO Resize (matches official test path).
    """
    return T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


@torch.no_grad()
def score(model: OfficialUnivFD, transform, device, paths: list[Path], bs: int) -> torch.Tensor:
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
        logits = model(x).flatten()
        probs.append(torch.sigmoid(logits).detach().float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True,
                    choices=["forensynths", "ufd", "pergen", "pool"])
    ap.add_argument("--data", required=True, help="benchmark root")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="official Linear-head state dict")
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

    model = OfficialUnivFD(ckpt, device)
    transform = build_transform()

    root = Path(args.data).expanduser()
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root,
                    Path(args.reals).expanduser() if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== clip_official_eval {args.tag} | model=clip (official-vendored) ckpt={ckpt} "
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
        "model": "clip",
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
