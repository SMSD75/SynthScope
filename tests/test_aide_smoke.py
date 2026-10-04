"""End-to-end smoke test for the AIDE integration (genuine 5-view arch).

Exercises, without any download (tiny ``convnext_base`` trunk, random ResNet-50s):

* ``AIDEFiveView`` native-resolution preprocessing: arbitrary PIL -> ``[5, 3, 256, 256]``,
* ``DCT_base_Rec_Module`` picks 4 distinct native-res patches,
* ``HPF`` (30 SRM kernels) shape + frozen weights,
* ``SynthScopeAIDEModel`` forward / compute_loss on a ``[B, 5, 3, 256, 256]`` batch + backward,
* the ConvNeXt trunk stays frozen.
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.aide.aide import SynthScopeAIDEModel
from models.image.aide.official_arch import AIDEFiveView, DCT_base_Rec_Module, HPF


def build_test_config() -> OmegaConf:
    # convnext_base (no pretrained) keeps the trunk small; the real model uses convnext_xxlarge.
    return OmegaConf.create(
        {
            "ModelConfig": {
                "model_type": "aide",
                "convnext_model_name": "convnext_base",
                "convnext_pretrained": "",
            },
            "TrainerConfig": {"label_smoothing": 0.1},
        }
    )


def test_hpf_module() -> None:
    print("[A] HPF shape / fixed-weight check ...")
    hpf = HPF()
    assert not hpf.hpf.weight.requires_grad, "HPF weights must be frozen"
    assert hpf.hpf.weight.shape == (30, 3, 5, 5), f"unexpected HPF shape: {hpf.hpf.weight.shape}"
    out = hpf(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 30, 64, 64)
    print(f"    HPF out shape = {tuple(out.shape)}")


def test_dct_selector() -> None:
    print("[B] DCT selector picks distinct native-res patches ...")
    dct = DCT_base_Rec_Module()
    x = torch.rand(3, 137, 211)  # native, arbitrary size, [0,1]
    a, b, c, d = dct(x)
    for name, patch in [("minmin", a), ("maxmax", b), ("minmin1", c), ("maxmax1", d)]:
        assert patch.shape == (3, 32, 32), f"{name} shape {tuple(patch.shape)}"
    assert (a - b).abs().mean() > 1e-6, "min and max picks collapsed"
    print("    distinct picks OK")


def test_five_view() -> None:
    print("[C] AIDEFiveView: arbitrary PIL -> [5,3,256,256] ...")
    tf = AIDEFiveView()
    for size in [(500, 375), (1024, 768), (240, 240)]:
        x = tf(Image.new("RGB", size))
        assert x.shape == (5, 3, 256, 256), f"{size} -> {tuple(x.shape)}"
        assert torch.isfinite(x).all()
    print("    five-view shapes OK")


def main() -> None:
    torch.manual_seed(0)
    test_hpf_module()
    test_dct_selector()
    test_five_view()

    config = build_test_config()
    print("[1] Building SynthScopeAIDEModel (convnext_base, no pretrained, random ResNet-50s) ...")
    model = SynthScopeAIDEModel(config)
    model.eval()

    print("[2] ConvNeXt trunk must be frozen ...")
    trunk = [n for n, p in model.named_parameters()
             if p.requires_grad and "openclip_convnext_xxl" in n]
    assert not trunk, f"convnext leaked into trainable params: {trunk[:3]}"
    for prefix in ("net.model_min", "net.model_max", "net.fc", "net.convnext_proj"):
        assert any(n.startswith(prefix) and p.requires_grad
                   for n, p in model.named_parameters()), f"missing trainable module: {prefix}"

    bsz = 2
    images = torch.rand(bsz, 5, 3, 256, 256)  # the 5-view stack the transform produces
    labels = torch.tensor([0, 1], dtype=torch.long)
    batch = (images, labels, torch.zeros(bsz, dtype=torch.long))

    print("[3] Eval forward(images) ...")
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (bsz, 2), f"expected ({bsz}, 2), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all().item()
    print(f"    logits = {logits.tolist()}")

    print("[4] Train compute_loss + backward ...")
    model.train()
    out = model.compute_loss(batch)
    assert set(out.keys()) == {"loss", "outputs"} and out["outputs"].shape == (bsz, 2)
    assert torch.isfinite(out["loss"]).item()
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"trainable params with no grad: {no_grad[:5]}"
    print(f"    train loss = {out['loss'].item():.4f} | backward OK")

    print("\nAll AIDE smoke checks passed.")


if __name__ == "__main__":
    main()
