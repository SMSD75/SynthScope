"""Verify AGIDCuReModel == the OFFICIAL CuRe model (weights, forward, end-to-end probability).

Needs the official repo (``cure`` package) + albumentations 2.0.8 on PYTHONPATH (the official
preprocessing uses it), the official adapter, and network/HF access (or --base-checkpoint) for the
frozen PE-Core base. Run on a GPU node:

  PYTHONPATH=<albumentations-target>:<CuRe repo> python repro/cure_verify_load.py \\
      --adapter <CuRe>/weights/cure_adapter.pt --images <dir> [--device cuda]

Checks, in order:
 1. engine checkpoint load of the adapter as ``cure.pth`` consumes EVERY adapter key (no unexpected),
 2. the AGID model's full state_dict equals the official model's state_dict, tensor-for-tensor,
 3. logits parity on identical (official-preprocessed) inputs,
 4. end-to-end P(fake) parity: AGID transform + AGID model vs official preprocess + official model.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cure import load_model as official_load_model  # official CuRe repo (on PYTHONPATH)
from cure import preprocess_image as official_preprocess

from config_utils import compose_config
from interface.backend import config as backend_config
from interface.backend.engine import _build_model, _load_checkpoint
from interface.backend.registry import get_spec
from models.image.cure.preprocess import SRC_FORMAT_KEY, CuReTransform

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def agid_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        fmt = source.format
        image = source.convert("RGB")
        image.info[SRC_FORMAT_KEY] = fmt
    return CuReTransform()(image)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--base-checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()
    device = torch.device(args.device)

    spec = get_spec("cure")
    # Stage the official adapter as the registry's checkpoint file (cure.pth) in a temp CKPT dir.
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(args.adapter, tmp / spec.checkpoint)
    backend_config.CHECKPOINTS_DIR = tmp
    overrides = list(spec.overrides)
    if args.base_checkpoint:
        overrides.append(f"ModelConfig.base_checkpoint={args.base_checkpoint}")

    print("[1] AGID model + engine checkpoint load ...")
    agid_model = _build_model(compose_config(overrides))
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=True)
    status = _load_checkpoint(agid_model, spec)
    print("    ", status)
    own = agid_model.state_dict()
    assert all(k in own and own[k].shape == v.shape for k, v in adapter.items()), "adapter key/shape mismatch"
    print(f"     all {len(adapter)} adapter tensors map onto AGID keys with equal shapes")

    print("[2] Official model + full state_dict equality ...")
    official = official_load_model(args.adapter, "cpu", args.base_checkpoint)
    ref = official.state_dict()
    assert set(ref) == set(own), (sorted(set(ref) ^ set(own))[:6])
    bad = [k for k in ref if not torch.equal(ref[k].contiguous().float(), own[k].contiguous().float())]
    assert not bad, f"{len(bad)} tensors differ, e.g. {bad[:4]}"
    print(f"     {len(ref)} tensors identical (backbone + LoRA + mu + W + head)")

    agid_model, official = agid_model.to(device).float().eval(), official.to(device).eval()
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in EXTS)[: args.limit]
    ref_x = torch.stack([official_preprocess(p) for p in paths]).to(device)
    agid_x = torch.stack([agid_tensor(p) for p in paths]).to(device)

    print("[3] Logits parity on identical inputs ...")
    with torch.inference_mode():
        d_logit = (agid_model(ref_x).float() - official(ref_x).float()).abs().max().item()
    print(f"     max |logit diff| = {d_logit:.3e}")

    print("[4] End-to-end P(fake): AGID pipeline vs official pipeline ...")
    with torch.inference_mode():
        p_agid = agid_model(agid_x).float().softmax(1)[:, 1].cpu()
        p_off = official(ref_x).float().softmax(1)[:, 1].cpu()
    d_prob = (p_agid - p_off).abs().max().item()
    flips = int(((p_agid >= 0.5) != (p_off >= 0.5)).sum())
    print(f"     {len(paths)} images | max |P diff| = {d_prob:.3e} | verdict flips = {flips}")
    print(f"     P(fake) range AGID [{p_agid.min():.3f}, {p_agid.max():.3f}]")

    ok = d_logit < 1e-3 and d_prob < 1e-3 and flips == 0
    print("\nRESULT:", "PASS - AGID CuRe reproduces the official model" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
