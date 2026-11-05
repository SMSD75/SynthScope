"""Smoke test for the LTF integration (CPU).

Builds the real model, so it needs the pinned CLIP ViT-L/14 (``$SYNTHSCOPE_LTF_BACKBONE``, the OpenAI clip cache
~/.cache/clip/ViT-L-14.pt or ``$SYNTHSCOPE_CLIP_DIR``, else downloaded from HF MuskAI/LTF-v3.3). Checks:
* the tiling policy: tile starts per axis (500 px -> 0, 224, 276; exact multiples add no duplicate; a short axis keeps
  one crop) and the view count, black padding and global view of the registry transform,
* the MIL head has the released key layout; if ``$SYNTHSCOPE_CKPT_DIR/ltf.pth`` exists, it is the pinned file (SHA-256)
  and ``remap_ltf`` keeps exactly those 37 tensors (dropping the 27 disconnected baseline ones), which the engine's
  strict official-weights load accepts,
* only the frozen CLIP falls under ``pretrained_prefixes`` and it is frozen,
* ``forward`` -> [B, 2] finite logits on a synthetic image in eval mode; ``compute_loss`` + backward reach the MIL head
  and not CLIP.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interface"))

from config_utils import compose_config                       # noqa: E402
from models.image.ltf.ltf import CORE_SHA256, SynthScopeLTFModel, sha256   # noqa: E402
from models.image.ltf.preprocess import LTFViews              # noqa: E402
from transformations import CLIP_MEAN, CLIP_STD               # noqa: E402
from backend import config as backend_config                  # noqa: E402
from backend.engine import _load_checkpoint                   # noqa: E402
from backend.registry import get_spec, remap_ltf              # noqa: E402

RELEASED_CORE = 37        # retained MIL / fusion tensors of the released checkpoint
RELEASED_UNUSED = 27      # disconnected baseline tensors the official loader drops


def noise_pil(w: int, h: int, seed: int = 0) -> Image.Image:
    arr = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def build_model() -> SynthScopeLTFModel:
    return SynthScopeLTFModel(compose_config(get_spec("ltf").overrides))


def test_tiling() -> None:
    print("[A] tiling policy / registry transform ...")
    starts = LTFViews.start_positions
    assert starts(500) == [0, 224, 276], starts(500)
    assert starts(448) == [0, 224] and starts(224) == [0] and starts(672) == [0, 224, 448]
    assert starts(200) == [0] and starts(225) == [0, 1]
    tf = get_spec("ltf").transform
    for (w, h), views in {(300, 200): 3, (500, 375): 7, (448, 448): 5, (200, 150): 2, (1300, 900): 31}.items():
        x = tf(noise_pil(w, h))
        assert tuple(x.shape) == (views, 3, 224, 224) and torch.isfinite(x).all(), ((w, h), tuple(x.shape))
    # A 200x150 image: one tile padded with black beyond the image (normalized zeros = -mean/std).
    x = tf(noise_pil(200, 150))
    black = (-torch.tensor(CLIP_MEAN) / torch.tensor(CLIP_STD))[:, None, None]
    assert torch.allclose(x[1, :, 150:, :], black.expand(3, 74, 224))
    assert torch.allclose(x[1, :, :, 200:], black.expand(3, 224, 24))
    assert not torch.allclose(x[0, :, 150:, :], black.expand(3, 74, 224))   # the global view is resized, not padded
    print("    starts(500) = [0, 224, 276]; view counts and padding as the official prepare_image")


def test_keys_and_frozen(model) -> None:
    print("[B] key layout / checkpoint / frozen CLIP ...")
    keys = list(model.state_dict())
    trained = [k for k in keys if not k.startswith(model.pretrained_prefixes)]
    assert all(k.startswith("core.") for k in trained) and len(trained) == RELEASED_CORE, len(trained)
    ckpt = backend_config.CHECKPOINTS_DIR / "ltf.pth"
    if ckpt.is_file():
        released = torch.load(ckpt, map_location="cpu", weights_only=True)
        kept = remap_ltf(released)
        assert sha256(ckpt) == CORE_SHA256, "ltf.pth is not the pinned official weights/epoch_20.pth"
        assert len(released) == RELEASED_CORE + RELEASED_UNUSED, len(released)
        assert set(kept) == set(trained), sorted(set(kept) ^ set(trained))[:8]
        status = _load_checkpoint(model, get_spec("ltf"))
        for k, v in kept.items():
            assert torch.equal(model.state_dict()[k], v), k
        print(f"    {ckpt}: {status}; {len(kept)} retained tensors equal, {len(released) - len(kept)} dropped")
    else:
        print(f"    {ckpt} not present: checkpoint load skipped")
    frozen = [n for n, p in model.named_parameters() if n.startswith(model.pretrained_prefixes)]
    assert frozen and not any(p.requires_grad for n, p in model.named_parameters() if n.startswith("visual."))
    print(f"    {len(trained)} MIL-head tensors; {len(frozen)} frozen CLIP parameters")


def test_forward_and_loss(model) -> None:
    print("[C] forward / compute_loss ...")
    tf = get_spec("ltf").transform
    x = torch.stack([tf(noise_pil(300, 260, seed=s)) for s in (1, 2)])      # [2, 3, 3, 224, 224]
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (2, 2) and torch.isfinite(logits).all(), logits
    p = logits.softmax(1)[:, 1]
    print(f"    P(fake) on noise: {[round(float(v), 4) for v in p]}")
    model.train()
    assert not model.visual.training
    out = model.compute_loss((x, torch.tensor([0, 1]), torch.tensor([0, 0])))
    out["loss"].backward()
    assert model.core.fusion_head[0].weight.grad is not None and model.core.head_level1.weight.grad is not None
    assert model.visual.conv1.weight.grad is None
    model.zero_grad(set_to_none=True)
    model.eval()


if __name__ == "__main__":
    test_tiling()
    m = build_model()
    test_keys_and_frozen(m)
    test_forward_and_loss(m)
    print("LTF smoke test passed")
