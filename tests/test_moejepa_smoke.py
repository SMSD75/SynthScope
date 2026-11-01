"""Smoke test for the MoE-JEPA integration (no weight downloads: a tiny randomly initialized V-JEPA 2 stands in for
facebook/vjepa2-vitl-fpc64-256, saved to a temporary directory and loaded through the model's ``from_pretrained`` path).

Checks, on CPU:
* the module layout is the official ``MoEJEPADeepfakeDetector`` one (``model``, ``pooler``, ``noise_stream``,
  ``noise_proj``, ``ln_semantic``, ``ln_noise``, ``fusion_gate``, ``fc``), with MoE layers in the last blocks; if
  ``$SYNTHSCOPE_CKPT_DIR/moejepa.safetensors`` exists, its keys are exactly the trained part of the full-size model,
* ``pretrained_prefixes`` covers only frozen backbone tensors and shared experts (never routers / routed experts),
* ``forward`` -> [B, 2] whose softmax[:, 1] equals 1 - P(real) of the 3-way head; ``compute_loss`` + backward reach the
  trained parts and not the frozen backbone.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from omegaconf import OmegaConf
from transformers import VJEPA2Config, VJEPA2Model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.moejepa.moejepa import SynthScopeMoEJEPAModel  # noqa: E402

TINY = VJEPA2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=2, mlp_ratio=4.0, crop_size=32,
                    patch_size=16, frames_per_clip=4, tubelet_size=2, pred_hidden_size=32, pred_num_hidden_layers=1,
                    pred_num_attention_heads=2, pred_num_mask_tokens=2)
HEADS = {"model", "pooler", "noise_stream", "noise_proj", "ln_semantic", "ln_noise", "fusion_gate", "fc"}


def build(backbone: str) -> SynthScopeMoEJEPAModel:
    cfg = OmegaConf.create({"ModelConfig": {"model_type": "moejepa", "backbone": backbone, "num_frames": 4,
                                            "moe_layers": 3, "num_experts": 6, "top_k": 2, "router_temperature": 0.5,
                                            "out_classes": 3}, "TrainerConfig": {}})
    return SynthScopeMoEJEPAModel(cfg)


def test_layout(model) -> None:
    print("[A] layout / frozen parts ...")
    keys = list(model.state_dict())
    assert {k.split(".")[0] for k in keys} == HEADS, {k.split(".")[0] for k in keys}
    assert sum(".mlp.router." in k for k in keys) == 2 * 3      # weight + bias in each of the 3 MoE layers
    trained = [k for k in keys if not k.startswith(model.pretrained_prefixes)]
    assert not any(".mlp.shared_expert." in k for k in trained)
    assert all(any(t in k for t in (".mlp.router.", ".mlp.routed_experts.")) for k in trained if k.startswith("model."))
    frozen = [n for n, p in model.named_parameters() if n.startswith(model.pretrained_prefixes)]
    assert frozen and not any(p.requires_grad for n, p in model.named_parameters() if n in set(frozen))
    print(f"    {len(trained)} trained tensors, {len(frozen)} frozen parameters")


def test_checkpoint_keys() -> None:
    ckpt = os.path.join(os.environ.get("SYNTHSCOPE_CKPT_DIR", ""), "moejepa.safetensors")
    if not os.path.isfile(ckpt):
        print("[B] (no moejepa.safetensors in $SYNTHSCOPE_CKPT_DIR; skipped)")
        return
    print("[B] released checkpoint keys == the trained part of the full model ...")
    from safetensors import safe_open
    with safe_open(ckpt, framework="pt") as f:
        released = set(f.keys())
    model = build("facebook/vjepa2-vitl-fpc64-256")
    trained = {k for k in model.state_dict() if not k.startswith(model.pretrained_prefixes)}
    assert released == trained, sorted(released ^ trained)[:8]


def test_forward_and_loss(model) -> None:
    print("[C] forward / compute_loss ...")
    images = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        two, three = model(images), model.forward_3way(images)
    assert two.shape == (2, 2) and three.shape == (2, 3) and torch.isfinite(two).all()
    assert torch.allclose(two.softmax(1)[:, 1], 1 - three.softmax(1)[:, 0], atol=1e-6)
    model.train()
    out = model.compute_loss((images, torch.tensor([0, 1]), torch.tensor([0, 0])))
    out["loss"].backward()
    assert model.fc.weight.grad is not None and model.noise_stream.bayar.weight.grad is not None
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith(model.pretrained_prefixes))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        VJEPA2Model(TINY).save_pretrained(d)
        m = build(d)
        test_layout(m)
        test_forward_and_loss(m)
    test_checkpoint_keys()
    print("MoE-JEPA smoke test passed")
