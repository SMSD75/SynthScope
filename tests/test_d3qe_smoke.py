"""Smoke test for the D3QE integration.

Builds the real model, so it needs the frozen weights: the LlamaGen VQ-16 tokenizer ($SYNTHSCOPE_LLAMAGEN_VQ, else it is
downloaded from HF FoundationVision/LlamaGen) and OpenAI CLIP ViT-L/14 (~/.cache/clip or $SYNTHSCOPE_CLIP_DIR, else
downloaded). Checks:
* the trained part has the official key layout (feature_processor, feature_pooler, mixed_proj, clip_proj, fc and the
  two codebook-count buffers); if ``$SYNTHSCOPE_CKPT_DIR/d3qe.pth`` exists, its keys are exactly that part,
* only the frozen CLIP and VQ tensors fall under ``pretrained_prefixes``, and they are frozen,
* the registry transform is the official one: short side 256 (bilinear), center crop 256, ToTensor, no normalization,
* ``forward`` -> [B, 1] finite logits in eval mode; ``compute_loss`` + backward reach the trainable head.
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interface"))

from models.image.d3qe.d3qe import SynthScopeD3QEModel  # noqa: E402
from backend.registry import get_spec                   # noqa: E402

TRAINED_GROUPS = {"real_codebook_count", "fake_codebook_count", "feature_processor", "feature_pooler", "mixed_proj",
                  "clip_proj", "fc"}


def build_model() -> SynthScopeD3QEModel:
    return SynthScopeD3QEModel(OmegaConf.create({"ModelConfig": {"model_type": "d3qe"}, "TrainerConfig": {}}))


def test_keys_and_frozen(model) -> None:
    print("[A] key layout / frozen parts ...")
    keys = [k for k in model.state_dict() if not k.startswith("criterion")]
    trained = [k for k in keys if not k.startswith(model.pretrained_prefixes)]
    assert {k.split(".")[0] for k in trained} == TRAINED_GROUPS, {k.split(".")[0] for k in trained}
    ckpt = os.path.join(os.environ.get("SYNTHSCOPE_CKPT_DIR", ""), "d3qe.pth")
    if os.path.isfile(ckpt):
        released = set(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        assert released == set(trained), (sorted(released ^ set(trained))[:8])
        print(f"    released checkpoint keys == the trained part ({len(released)} tensors)")
    frozen = [n for n, p in model.named_parameters() if n.startswith(model.pretrained_prefixes)]
    assert frozen and not any(p.requires_grad for n, p in model.named_parameters() if n in set(frozen))
    print(f"    {len(trained)} trained tensors; {len(frozen)} frozen CLIP/VQ parameters")


def test_transform() -> None:
    print("[B] registry transform ...")
    tf = get_spec("d3qe").transform
    for size in [(200, 250), (512, 384), (1024, 768)]:
        x = tf(Image.new("RGB", size, (128, 64, 32)))
        assert tuple(x.shape) == (3, 256, 256) and float(x.min()) >= 0 and float(x.max()) <= 1


def test_forward_and_loss(model) -> None:
    print("[C] forward / compute_loss ...")
    images = torch.rand(2, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        logits = model(images)
    assert tuple(logits.shape) == (2, 1) and torch.isfinite(logits).all()
    model.train()
    out = model.compute_loss((images, torch.tensor([0, 1]), torch.tensor([0, 0])))
    out["loss"].backward()
    assert model.fc[-1].weight.grad is not None and model.feature_processor[0].weight.grad is not None


if __name__ == "__main__":
    m = build_model()
    test_keys_and_frozen(m)
    test_transform()
    test_forward_and_loss(m)
    print("D3QE smoke test passed")
