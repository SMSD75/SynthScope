"""Smoke test for the B-Free integration.

Builds the model from its config (no download: the timm backbone is created without pretrained weights), and checks:
* the state-dict layout, and that the released checkpoint (``$SYNTHSCOPE_CKPT_DIR/bfree.pth``, if present) loads
  strictly through ``registry.remap_bfree``,
* the registry transform keeps the native resolution (ToTensor + ImageNet normalization, nothing else),
* ``forward`` -> [B] finite logits for images smaller than, equal to and larger than the 504 crop (the official
  wrapper tiles a small image's patch grid and takes five crops of a large one); ``compute_loss`` + backward.
"""

from __future__ import annotations

import os
import sys

import torch
from hydra import compose, initialize_config_dir
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from models.image.bfree.bfree import SynthScopeBFreeModel  # noqa: E402
from backend.registry import get_spec, remap_bfree           # noqa: E402


def build() -> SynthScopeBFreeModel:
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["model@ModelConfig=image/bfree"])
    return SynthScopeBFreeModel(cfg)


def test_layout(model) -> None:
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("criterion")}
    assert tuple(sd["patch_embed.proj.weight"].shape) == (768, 3, 14, 14)
    assert tuple(sd["model.head.weight"].shape) == (1, 768) and "model.reg_token" in sd
    ckpt = os.path.join(os.environ.get("SYNTHSCOPE_CKPT_DIR", ""), "bfree.pth")
    if os.path.isfile(ckpt):
        released = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(remap_bfree(released), strict=True)
        print(f"    released checkpoint loads strictly ({len(released)} tensors)")


def test_transform() -> None:
    tf = get_spec("bfree").transform
    for size in [(200, 150), (1100, 700)]:
        x = tf(Image.new("RGB", size, (128, 64, 32)))
        assert tuple(x.shape) == (3, size[1], size[0]) and torch.isfinite(x).all()


def test_forward_and_loss(model) -> None:
    model.eval()
    with torch.no_grad():
        for hw in [(180, 240), (504, 504), (700, 900)]:
            logits = model(torch.randn(2, 3, *hw))
            assert tuple(logits.shape) == (2,) and torch.isfinite(logits).all()
    model.train()
    model.zero_grad()
    model.compute_loss((torch.randn(2, 3, 504, 504), torch.tensor([0, 1]), torch.tensor([0, 0])))["loss"].backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert "model.head.weight" in grads and "patch_embed.proj.weight" in grads


if __name__ == "__main__":
    print("[A] layout ...")
    m = build()
    test_layout(m)
    print("[B] registry transform ...")
    test_transform()
    print("[C] forward + loss ...")
    test_forward_and_loss(m)
    print("B-Free smoke test passed")
