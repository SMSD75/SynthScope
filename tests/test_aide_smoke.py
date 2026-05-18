"""End-to-end smoke test for the AIDE integration.

Builds AGIDAIDEModel with a tiny OpenCLIP backbone (``convnext_base`` with no
pretrained weights and randomly-initialised ResNet-50s) so nothing downloads,
then exercises:

* forward / compute_loss on a CLIP-normalized batch,
* parameter freezing (the ConvNeXt trunk must be frozen),
* the DCT patch selector (top/bottom indices must be distinct),
* the HPF / SRM layer (weights frozen, output shape correct),
* a full backward pass through all trainable modules.
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.aide.aide import AGIDAIDEModel, _DCTPatchSelector, _HPF


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _clip_normalize(images_01: torch.Tensor) -> torch.Tensor:
    return (images_01 - _CLIP_MEAN) / _CLIP_STD


def build_test_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "ModelConfig": {
                "model_type": "aide",
                "convnext_model_name": "convnext_base",
                "convnext_pretrained": "",
                "freeze_convnext": True,
                "semantic_dim": 1024,
                "semantic_proj_dim": 64,
                "resnet_pretrained": False,
                "head_hidden": 128,
                "num_classes": 2,
                "dct": {"window_size": 32, "stride": 16, "output_size": 128, "grade_N": 6},
            },
            "TrainerConfig": {"label_smoothing": 0.1},
        }
    )


def test_hpf_module() -> None:
    print("[A] HPF shape / fixed-weight check ...")
    hpf = _HPF()
    assert not hpf.conv.weight.requires_grad, "HPF weights must be frozen"
    assert hpf.conv.weight.shape == (30, 3, 5, 5), f"unexpected HPF shape: {hpf.conv.weight.shape}"
    out = hpf(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 30, 64, 64)
    print(f"    HPF out shape = {tuple(out.shape)}")


def test_dct_selector() -> None:
    print("[B] DCT selector picks distinct patches ...")
    sel = _DCTPatchSelector(window_size=32, stride=16, grade_N=6)
    x = torch.randn(3, 3, 128, 128)
    x_minmin, x_maxmax, x_minmin1, x_maxmax1 = sel(x)
    for name, patch in [
        ("x_minmin", x_minmin),
        ("x_maxmax", x_maxmax),
        ("x_minmin1", x_minmin1),
        ("x_maxmax1", x_maxmax1),
    ]:
        assert patch.shape == (3, 3, 32, 32), f"{name} shape {tuple(patch.shape)}"

    diff_max = (x_minmin - x_maxmax).abs().mean()
    diff_min = (x_minmin - x_minmin1).abs().mean()
    assert diff_max > 1e-6, "min and max picks collapsed onto the same patch"
    assert diff_min > 1e-6, "the two min-frequency picks collapsed onto the same patch"
    print(f"    distinct picks OK (|min - max| = {diff_max:.3f}, |min - min1| = {diff_min:.3f})")


def main() -> None:
    torch.manual_seed(0)

    test_hpf_module()
    test_dct_selector()

    config = build_test_config()

    print("[1] Building AGIDAIDEModel (convnext_base, no pretrained, random ResNet-50) ...")
    model = AGIDAIDEModel(config)
    model.eval()

    print("[2] Checking parameter freezing ...")
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p.numel()) for n, p in model.named_parameters() if not p.requires_grad]
    n_trainable = sum(n for _, n in trainable)
    n_frozen = sum(n for _, n in frozen)
    print(f"    trainable params: {n_trainable:,} | frozen params: {n_frozen:,}")
    for name, _ in trainable:
        assert not name.startswith("convnext."), f"convnext leaked: {name}"
        assert not name.startswith("hpf."), f"hpf leaked: {name}"
    for prefix in ("model_min", "model_max", "semantic_proj", "head"):
        assert any(n.startswith(prefix) for n, _ in trainable), f"missing trainable module: {prefix}"

    print("[3] Building a fake CLIP-normalized batch (B=2, 3, 224, 224) ...")
    bsz = 2
    images = _clip_normalize(torch.rand(bsz, 3, 224, 224))
    labels = torch.tensor([0, 1], dtype=torch.long)
    folder_ids = torch.zeros(bsz, dtype=torch.long)
    batch = (images, labels, folder_ids)

    print("[4] Eval-mode forward(images) ...")
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (bsz, 2), f"expected ({bsz}, 2), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all().item()
    print(f"    logits = {logits.tolist()}")

    print("[5] Eval-mode compute_loss(batch) ...")
    with torch.no_grad():
        out_eval = model.compute_loss(batch)
    assert set(out_eval.keys()) == {"loss", "outputs"}
    assert out_eval["outputs"].shape == (bsz, 2)
    assert torch.isfinite(out_eval["loss"]).item()
    print(f"    eval loss = {out_eval['loss'].item():.4f}")

    print("[6] Train-mode compute_loss + backward ...")
    model.train()
    out_train = model.compute_loss(batch)
    assert torch.isfinite(out_train["loss"]).item()
    model.zero_grad(set_to_none=True)
    out_train["loss"].backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"these trainable params received no grad: {no_grad[:5]}"
    print(f"    train loss = {out_train['loss'].item():.4f} | backward OK")

    print("[7] Odd input resolution to exercise model-side resize ...")
    odd = _clip_normalize(torch.rand(bsz, 3, 199, 251))
    with torch.no_grad():
        odd_logits = model(odd)
    assert odd_logits.shape == (bsz, 2)
    print(f"    odd-shape logits = {odd_logits.tolist()}")

    print("\nAll AIDE smoke checks passed.")


if __name__ == "__main__":
    main()
