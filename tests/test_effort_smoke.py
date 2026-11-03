"""Smoke test for the Effort integration.

Builds the model (downloads the CLIP ViT-L/14 weights on first use) and checks:
* the SVD-residual layout of the attention projections (principal rank 1023 frozen, one residual direction trainable)
  and, if ``$SYNTHSCOPE_CKPT_DIR/effort.pth`` exists, that the released checkpoint loads with only the principal SVD
  factors missing (they are recomputed at init, as in the official code);
* the registry transform: the whole image squashed to 224 x 224;
* ``forward`` -> [B, 2] finite logits; ``compute_loss`` + backward reach the residual factors and the head only.
"""

from __future__ import annotations

import os
import sys

import torch
from hydra import compose, initialize_config_dir
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from models.image.effort.effort import SynthScopeEffortModel   # noqa: E402
from backend.registry import get_spec, remap_effort              # noqa: E402


def build() -> SynthScopeEffortModel:
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["model@ModelConfig=image/effort"])
    return SynthScopeEffortModel(cfg)


def test_layout(model) -> None:
    q = model.backbone.encoder.layers[0].self_attn.q_proj
    assert tuple(q.weight_main.shape) == (1024, 1024) and tuple(q.U_residual.shape) == (1024, 1)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.split(".")[-1] in ("S_residual", "U_residual", "V_residual") or n.startswith("head.") for n in trainable)
    ckpt = os.path.join(os.environ.get("SYNTHSCOPE_CKPT_DIR", ""), "effort.pth")
    if os.path.isfile(ckpt):
        missing, unexpected = model.load_state_dict(remap_effort(torch.load(ckpt, map_location="cpu", weights_only=False)), strict=False)
        assert not unexpected and set(missing) == set(model.pretrained_prefixes)
        print(f"    released checkpoint loads ({len(missing)} principal SVD factors recomputed)")


def test_transform() -> None:
    for key in ("effort", "effort_chameleon"):
        tf = get_spec(key).transform
        for size in [(200, 150), (1100, 700)]:
            x = tf(Image.new("RGB", size, (128, 64, 32)))
            assert tuple(x.shape) == (3, 224, 224) and torch.isfinite(x).all()


def test_forward_and_loss(model) -> None:
    images = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(images)
    assert tuple(logits.shape) == (2, 2) and torch.isfinite(logits).all()
    model.train()
    model.zero_grad()
    model.compute_loss((images, torch.tensor([0, 1]), torch.tensor([0, 0])))["loss"].backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert "head.weight" in grads and any(n.endswith("U_residual") for n in grads)
    assert not any(n.endswith("weight_main") for n in grads)


if __name__ == "__main__":
    print("[A] layout ...")
    m = build()
    test_layout(m)
    print("[B] registry transforms ...")
    test_transform()
    print("[C] forward + loss ...")
    test_forward_and_loss(m)
    print("Effort smoke test passed")
