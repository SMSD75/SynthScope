"""Smoke test for the OmniAID integration.

Builds the real model for the three released configurations, so it needs the CLIP ViT-L/14@336 weights
(HF ``openai/clip-vit-large-patch14-336``, downloaded on first use). Checks:
* the state-dict layout matches each release's expert configuration (and the released checkpoint, if
  ``$SYNTHSCOPE_CKPT_DIR/<key>.pth`` exists: same keys and shapes),
* the registry transforms give [3, 336, 336] tensors (v2: one resize; v1-era: 512 then 336),
* ``forward`` -> [B, 2] finite logits; ``compute_loss`` + backward reach the experts, router and head, while the
  frozen CLIP feature extractor and the shared main weights get no gradient.
"""

from __future__ import annotations

import os
import sys

import torch
from hydra import compose, initialize_config_dir
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from models.image.omniaid.omniaid import SynthScopeOmniAIDModel  # noqa: E402
from backend.registry import get_spec                             # noqa: E402

EXPECTED = {"omniaid": (6, 1), "omniaid_v1": (6, 8), "omniaid_genimage": (3, 4)}   # (experts, rank per expert)


def build(key: str) -> SynthScopeOmniAIDModel:
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"model@ModelConfig=image/{key}"])
    return SynthScopeOmniAIDModel(cfg)


def test_layout(key: str, model) -> None:
    experts, rank = EXPECTED[key]
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("criterion")}
    u = sd["encoder_layers.0.self_attn.q_proj.U_experts.0"]
    assert tuple(u.shape) == (1024, rank), u.shape
    assert f"encoder_layers.0.self_attn.q_proj.U_experts.{experts - 1}" in sd
    ckpt = os.path.join(os.environ.get("SYNTHSCOPE_CKPT_DIR", ""), f"{key}.pth")
    if os.path.isfile(ckpt):
        released = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        assert set(released) == set(sd) and all(released[k].shape == sd[k].shape for k in sd)
        print(f"    {key}: released checkpoint matches the layout ({len(sd)} tensors)")


def test_transforms() -> None:
    for key in EXPECTED:
        tf = get_spec(key).transform
        for size in [(200, 250), (1024, 768)]:
            x = tf(Image.new("RGB", size, (128, 64, 32)))
            assert tuple(x.shape) == (3, 336, 336) and torch.isfinite(x).all()


def test_forward_and_loss(model) -> None:
    images = torch.randn(2, 3, 336, 336)
    model.eval()
    with torch.no_grad():
        logits = model(images)
    assert tuple(logits.shape) == (2, 2) and torch.isfinite(logits).all()
    model.train()
    model.zero_grad()
    model.compute_loss((images, torch.tensor([0, 1]), torch.tensor([0, 0])))["loss"].backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert "head.weight" in grads and any("experts" in n for n in grads)
    assert not any(n.startswith("feature_extractor.") or n.endswith("weight_main") for n in grads)


if __name__ == "__main__":
    print("[B] registry transforms ...")
    test_transforms()
    for key in EXPECTED:
        print(f"[A/C] {key}: layout, forward, loss ...")
        m = build(key)
        test_layout(key, m)
        test_forward_and_loss(m)
        del m
    print("OmniAID smoke test passed")
