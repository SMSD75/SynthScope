"""Smoke test for the DDA integration (no weight downloads: the DINOv2 backbone is built without pretrained weights).

Checks, on CPU:
* ``SynthScopeDDAModel`` keeps the official key layout (``base_model.model.*`` with LoRA on attn.qkv/proj and
  mlp.fc1/fc2, ``base_model.fc``), so the released ``DDA_ckpt.pth`` loads key-for-key,
* only the LoRA tensors and the head are trainable (the official trainable set),
* the registry transform is the official one: center-crop 336 (no resize), CLIP normalization,
* ``forward`` -> [B, 1] finite logits; ``compute_loss`` + backward reach the LoRA and head params.
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf
from PIL import Image

# xFormers has no CPU kernels; DINOv2's own switch falls back to standard attention (as interface/backend/config.py does).
os.environ.setdefault("XFORMERS_DISABLED", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interface"))

from models.image.dda.dda import SynthScopeDDAModel  # noqa: E402
from backend.registry import get_spec                 # noqa: E402


def build_test_config() -> OmegaConf:
    return OmegaConf.create({"ModelConfig": {"model_type": "dda", "pretrained_backbone": False},
                             "TrainerConfig": {}})


def test_keys_and_trainable() -> None:
    print("[A] key layout / trainable set ...")
    model = SynthScopeDDAModel(build_test_config())
    keys = set(model.state_dict())
    assert "base_model.fc.weight" in keys and "base_model.model.pos_embed" in keys
    for t in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
        assert f"base_model.model.blocks.0.{t}.lora.lora_A" in keys, t
        assert f"base_model.model.blocks.0.{t}.original_layer.weight" in keys, t
    assert sum(".lora." in k for k in keys) == 24 * 4 * 2       # 24 blocks x 4 layers x (A, B)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora." in n or n.startswith("base_model.fc.") for n in trainable)
    print(f"    {len(keys)} tensors, {len(trainable)} trainable (LoRA + head)")


def test_transform() -> None:
    print("[B] registry transform ...")
    tf = get_spec("dda").transform
    for size in [(200, 250), (500, 375), (1024, 768)]:
        x = tf(Image.new("RGB", size, (128, 64, 32)))
        assert tuple(x.shape) == (3, 336, 336) and torch.isfinite(x).all()


def test_forward_and_loss() -> None:
    print("[C] forward / compute_loss ...")
    model = SynthScopeDDAModel(build_test_config())
    images = torch.randn(2, 3, 336, 336)
    logits = model(images)
    assert tuple(logits.shape) == (2, 1) and torch.isfinite(logits).all()
    out = model.compute_loss((images, torch.tensor([0, 1]), torch.tensor([0, 0])))
    out["loss"].backward()
    assert model.base_model.fc.weight.grad is not None
    assert model.base_model.model.blocks[0].attn.qkv.lora.lora_B.grad is not None


if __name__ == "__main__":
    test_keys_and_trainable()
    test_transform()
    test_forward_and_loss()
    print("DDA smoke test passed")
