"""Smoke test for the CuRe integration (no downloads: the frozen PE-Core base is not fetched).

Checks, on CPU:
* ``CuReTransform``: any PIL size -> finite ``[3, 336, 336]`` in [-1, 1]; the PNG source-format hint
  changes the result (PNG sources get the official JPEG q96 round-trip), non-PNG / no hint does not,
* ``SynthScopeCuReModel`` keeps the official key layout (backbone.*, mu, W [1024,128], head.*) so the
  released adapter loads key-for-key,
* only the LoRA tensors + head are trainable; the rest of the backbone is frozen,
* ``forward`` -> [B, 2] finite logits; ``compute_loss`` + backward reach the LoRA and head params.
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.cure.cure import SynthScopeCuReModel
from models.image.cure.preprocess import SRC_FORMAT_KEY, CuReTransform


def build_test_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "ModelConfig": {"model_type": "cure", "load_base_backbone": False},
            "TrainerConfig": {"label_smoothing": 0.1},
        }
    )


def noise_image(size: tuple[int, int]) -> Image.Image:
    gen = torch.Generator().manual_seed(0)
    arr = torch.randint(0, 256, (size[1], size[0], 3), generator=gen, dtype=torch.uint8)
    return Image.fromarray(arr.numpy(), "RGB")


def test_transform() -> None:
    print("[A] CuReTransform shapes / range / PNG hint ...")
    tf = CuReTransform()
    for size in [(200, 250), (500, 375), (1024, 768), (391, 700)]:
        x = tf(noise_image(size))
        assert x.shape == (3, 336, 336) and x.dtype == torch.float32, (size, x.shape, x.dtype)
        assert torch.isfinite(x).all() and x.min() >= -1.0 and x.max() <= 1.0
    img = noise_image((500, 375))
    plain = tf(img)
    img.info[SRC_FORMAT_KEY] = "JPEG"
    assert torch.equal(plain, tf(img)), "non-PNG hint must not change the result"
    img.info[SRC_FORMAT_KEY] = "PNG"
    assert not torch.equal(plain, tf(img)), "PNG hint must apply the JPEG q96 re-encode"
    print("    shapes + PNG hint OK")


def main() -> None:
    torch.manual_seed(0)
    test_transform()

    print("[1] Building SynthScopeCuReModel (architecture only, no base download) ...")
    model = SynthScopeCuReModel(build_test_config())
    model.eval()

    print("[2] Official key layout ...")
    sd = model.state_dict()
    assert sd["mu"].shape == (1024,) and sd["W"].shape == (1024, 128), (sd["mu"].shape, sd["W"].shape)
    assert sd["head.weight"].shape == (2, 128)
    for key in ("backbone.conv1.weight", "backbone.class_embedding", "backbone.positional_embedding",
                "backbone.transformer.resblocks.23.attn.lora_B_v",
                "backbone.transformer.resblocks.0.attn.in_proj_weight",
                "backbone.transformer.resblocks.0.mlp.c_fc.weight", "backbone.ln_post.weight"):
        assert key in sd, f"missing official key {key}"

    print("[3] Only LoRA + head trainable ...")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n or n.startswith("head.") for n in trainable), trainable[:4]
    assert not any(p.requires_grad for n, p in model.named_parameters() if "lora_" not in n and not n.startswith("head."))
    print(f"    {len(trainable)} trainable tensors (LoRA q/v x 24 layers = 96, head = 2)")
    assert len(trainable) == 24 * 4 + 2

    print("[4] Eval forward -> [B, 2] ...")
    images = torch.randn(2, 3, 336, 336).clamp(-1, 1)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all(), logits
    print(f"    logits = {logits.tolist()}")

    print("[5] compute_loss + backward reaches LoRA and head ...")
    model.train()
    out = model.compute_loss((images, torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long)))
    assert set(out) == {"loss", "outputs"} and out["outputs"].shape == (2, 2) and torch.isfinite(out["loss"])
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    lora_grads = [p.grad for n, p in model.named_parameters() if "lora_B_q" in n]
    assert all(g is not None and torch.isfinite(g).all() for g in lora_grads), "LoRA B must receive gradients"
    print(f"    loss = {out['loss'].item():.4f} | head + LoRA grads OK")

    print("\nAll CuRe smoke checks passed.")


if __name__ == "__main__":
    main()
