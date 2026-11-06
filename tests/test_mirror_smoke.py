"""Smoke test for the MIRROR integration (CPU, no downloads).

Checks:
* the registry transform (``mirror_eval`` + ``to_dtype``): [3, 224, 224] in [0, 1] for small, non-square and large
  images; equal to the official eval steps written out with torchvision; the source-format hint applies the JPEG q96
  "format-aligned" step to PNG / BMP only,
* the peft-free LoRA layer computes base(x) + B(A(x)) * alpha / r (and equals peft's, if peft is installed),
* a tiny-backbone ``SynthScopeMIRRORModel``: official key layout (``base_model.model.*.q_proj.base_layer`` /
  ``lora_A.default`` / ``lora_B.default``), ``forward`` -> [B, 2] finite logits, checkpoints with either ViT block
  prefix (transformers 4 ``dino.layer`` / 5 ``dino.model.layer``) load strictly through ``load_state_dict`` and the
  official ``load_detector_checkpoint`` with identical outputs, ``compute_loss`` + backward reach only LoRA and head,
* the full-size model (built on the meta device) has exactly the keys and shapes of the released checkpoint, if
  ``$SYNTHSCOPE_CKPT_DIR/mirror.pth`` exists (skipped otherwise).
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from backend.registry import get_spec                                       # noqa: E402
from models.image.cure.preprocess import SRC_FORMAT_KEY                     # noqa: E402
from models.image.mirror.mirror import SynthScopeMIRRORModel                # noqa: E402
from models.image.mirror.official_arch import (LoraConfig, LoraLinear,     # noqa: E402
                                               load_detector_checkpoint)

TINY = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}


def noise_image(size: tuple[int, int]) -> Image.Image:
    gen = torch.Generator().manual_seed(size[0] * 7 + size[1])
    arr = torch.randint(0, 256, (size[1], size[0], 3), generator=gen, dtype=torch.uint8)
    return Image.fromarray(arr.numpy(), "RGB")


def official_eval(img: Image.Image, aligned: bool) -> torch.Tensor:
    """The official eval steps (data/datasets.py), written out: short side > 512 -> 512 bicubic, center crop 224,
    JPEG q96 (optimize=True) for lossless sources, ToTensor."""
    w, h = img.size
    if min(w, h) > 512:
        s = 512 / min(w, h)
        img = TF.resize(img, [int(round(h * s)), int(round(w * s))], interpolation=InterpolationMode.BICUBIC,
                        antialias=True)
    img = TF.center_crop(img, [224, 224])
    if aligned:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=96, optimize=True)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return TF.to_tensor(img)


def tiny_model() -> SynthScopeMIRRORModel:
    cfg = OmegaConf.create({"ModelConfig": {"model_type": "mirror", "backbone_config": TINY, "use_amp": True}})
    return SynthScopeMIRRORModel(cfg)


def test_transform() -> None:
    print("[A] registry transform: shapes, official steps, format-aligned JPEG ...")
    tf = get_spec("mirror").transform
    assert list(tf.transform_list) == ["mirror_eval", "to_dtype"], tf
    for size in [(180, 120), (128, 128), (300, 200), (513, 700), (1600, 1066), (300, 1200)]:
        img = noise_image(size)
        for fmt in (None, "JPEG", "WEBP", "PNG", "BMP"):
            if fmt:
                img.info[SRC_FORMAT_KEY] = fmt
            x = tf(img)
            assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1, (size, fmt)
            assert torch.equal(x, official_eval(img, fmt in ("PNG", "BMP"))), (size, fmt)
    small = tf(noise_image((180, 120)))
    assert small[:, :52].abs().sum() == 0 and small[:, :, :22].abs().sum() == 0, "small images must be zero-padded"
    img = noise_image((300, 200))
    img.info[SRC_FORMAT_KEY] = "PNG"
    aligned = tf(img)
    img.info[SRC_FORMAT_KEY] = "JPEG"
    assert not torch.equal(aligned, tf(img)), "the PNG hint must apply the JPEG q96 step"
    print("    6 sizes x 5 format hints equal the official steps")


def test_lora() -> None:
    print("[B] peft-free LoRA ...")
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    layer = LoraLinear(base, LoraConfig(r=8, lora_alpha=16))
    nn.init.normal_(layer.lora_B["default"].weight)
    x = torch.randn(3, 5, 16)
    want = base(x) + layer.lora_B["default"](layer.lora_A["default"](x)) * 2.0
    assert torch.equal(layer(x), want)
    assert set(layer.state_dict()) == {"base_layer.weight", "base_layer.bias", "lora_A.default.weight",
                                       "lora_B.default.weight"}
    if importlib.util.find_spec("peft") is not None:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
        wrapped = get_peft_model(nn.Sequential(nn.Linear(16, 24)), PeftLoraConfig(r=8, lora_alpha=16,
                                                                                    target_modules=["0"]))
        ours = LoraLinear(nn.Linear(16, 24), LoraConfig(r=8, lora_alpha=16))
        ours.load_state_dict({k.replace("base_model.model.0.", ""): v for k, v in wrapped.state_dict().items()})
        nn.init.normal_(ours.lora_B["default"].weight)
        wrapped.base_model.model[0].lora_B["default"].weight.data.copy_(ours.lora_B["default"].weight)
        assert torch.equal(ours(x), wrapped(x)), "differs from peft"
        print("    equals peft")
    else:
        print("    peft not installed: compared with the formula only")


def test_model_and_loader() -> None:
    print("[C] tiny model: key layout, forward, both block prefixes, loss ...")
    torch.manual_seed(0)
    model = tiny_model().eval()
    sd = model.state_dict()
    blocks = "backbone.dino.model.layer." if any(k.startswith("backbone.dino.model.layer.") for k in sd) \
        else "backbone.dino.layer."
    for suffix in ("q_proj.base_layer.weight", "q_proj.lora_A.default.weight", "k_proj.lora_B.default.weight",
                   "v_proj.base_layer.bias"):
        assert f"{blocks}0.base_model.model.attention.{suffix}" in sd, suffix
    assert {"norm.mean", "norm.std", "memory_bank.memory", "detector.head.3.weight"} <= set(sd)
    assert tuple(sd["memory_bank.memory"].shape) == (4096, TINY["hidden_size"])
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (2, 2) and logits.dtype == torch.float32 and torch.isfinite(logits).all()

    other = "backbone.dino.layer." if blocks != "backbone.dino.layer." else "backbone.dino.model.layer."
    for prefix in (blocks, other):
        state = {k.replace(blocks, prefix): v.clone() for k, v in sd.items()}
        for load in ("load_state_dict", "official"):
            restored = tiny_model().eval()
            if load == "official":
                load_detector_checkpoint(restored, {"model": dict(state)})
            else:
                restored.load_state_dict(dict(state), strict=True)
            with torch.no_grad():
                assert torch.equal(restored(images), logits), (prefix, load)

    model.train()
    model.zero_grad()
    out = model.compute_loss((images, torch.tensor([0, 1])))
    assert torch.isfinite(out["loss"]) and out["outputs"].shape == (2, 2)
    out["loss"].backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert any(".lora_A.default." in n for n in grads) and "detector.head.3.weight" in grads
    assert not any(n.startswith("memory_bank.") or ".base_layer." in n or "embeddings" in n for n in grads), grads
    print(f"    {len(sd)} tensors, logits {tuple(logits.shape)}, strict loads with both prefixes, grads on LoRA + head")


def test_released_layout() -> None:
    ckpt = os.path.join(os.environ.get("SYNTHSCOPE_CKPT_DIR", ""), "mirror.pth")
    if not os.path.isfile(ckpt):
        print("[D] released checkpoint layout: skipped ($SYNTHSCOPE_CKPT_DIR/mirror.pth not found)")
        return
    print("[D] full-size model (meta device) vs the released checkpoint ...")
    with torch.device("meta"):
        model = SynthScopeMIRRORModel(OmegaConf.create({"ModelConfig": {"model_type": "mirror"}}))
    ours = {k.replace("backbone.dino.model.layer.", "backbone.dino.layer."): tuple(v.shape)
            for k, v in model.state_dict().items()}
    released = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)["model"]
    assert ours == {k: tuple(v.shape) for k, v in released.items()}
    print(f"    {len(ours)} tensors: same names and shapes")


if __name__ == "__main__":
    test_transform()
    test_lora()
    test_model_and_loader()
    test_released_layout()
    print("MIRROR smoke test passed")
