"""End-to-end smoke test for the CO-SPY integration.

Builds AGIDCoSpyModel with a tiny OpenCLIP backbone (ViT-B-32/openai) so the
download stays small, then exercises the forward pass, compute_loss, the
two-class output convention, the coefficient-dropout / feature-interpolation
paths, and confirms the parameter freezing layout.
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.co_spy.co_spy import AGIDCoSpyModel


def build_test_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "ModelConfig": {
                "model_type": "co_spy",
                "semantic_model_name": "ViT-B-32",
                "semantic_pretrained": "openai",
                "semantic_image_size": 224,
                "vae_id": "CompVis/stable-diffusion-v1-4",
                "vae_subfolder": "vae",
                "artifact_image_size": 224,
                "artifact_residual_scale": 100.0 / 7.0,
                "coeff_dropout": 0.5,
                "feat_interp_prob": 0.5,
                "feat_interp_alpha": 0.2,
                "freeze_backbone": True,
            },
            "TrainerConfig": {"label_smoothing": 0.1},
        }
    )


def main() -> None:
    torch.manual_seed(0)

    config = build_test_config()

    print("[1] Building AGIDCoSpyModel ...")
    model = AGIDCoSpyModel(config)
    model.eval()

    print(f"    semantic_dim = {model.semantic_dim}, artifact_dim = {model.artifact_dim}")
    assert model.artifact_dim == 512, "Artifact encoder must output 512-d features"
    assert model.semantic_dim > 0

    print("[2] Checking parameter freezing ...")
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p.numel()) for n, p in model.named_parameters() if not p.requires_grad]
    n_trainable = sum(n for _, n in trainable)
    n_frozen = sum(n for _, n in frozen)
    print(f"    trainable params: {n_trainable:,} | frozen params: {n_frozen:,}")
    for name, _ in trainable:
        assert not name.startswith("semantic."), f"semantic backbone leaked: {name}"
        assert not name.startswith("vae."), f"vae leaked: {name}"
    for prefix in ("artifact_encoder", "sem_fc", "art_fc", "fc"):
        assert any(n.startswith(prefix) for n, _ in trainable), (
            f"missing trainable module: {prefix}"
        )

    print("[3] Building a fake batch of 2 raw [0,1] images ...")
    bsz = 2
    images = torch.rand(bsz, 3, 224, 224, dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    folder_ids = torch.zeros(bsz, dtype=torch.long)
    batch = (images, labels, folder_ids)

    print("[4] Eval-mode forward(images) ...")
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (bsz,), f"expected shape ({bsz},), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all().item()
    print(f"    logits = {logits.tolist()}")

    print("[5] Eval-mode compute_loss(batch) ...")
    model.eval()
    with torch.no_grad():
        out_eval = model.compute_loss(batch)
    assert set(out_eval.keys()) == {"loss", "outputs"}
    assert out_eval["outputs"].shape == (bsz, 2), (
        f"expected outputs shape ({bsz}, 2), got {tuple(out_eval['outputs'].shape)}"
    )
    assert torch.isfinite(out_eval["loss"]).item()
    sign_check = out_eval["outputs"][:, 1] + out_eval["outputs"][:, 0]
    assert torch.allclose(sign_check, torch.zeros_like(sign_check), atol=1e-5), (
        "outputs must be [-logit, +logit]"
    )
    print(f"    eval loss = {out_eval['loss'].item():.4f}")
    print(f"    outputs   = {out_eval['outputs'].tolist()}")

    print("[6] Train-mode compute_loss with feature-interp + coeff-dropout ...")
    model.train()
    out_train = model.compute_loss(batch)
    assert out_train["outputs"].shape == (bsz, 2)
    assert torch.isfinite(out_train["loss"]).item()
    print(f"    train loss = {out_train['loss'].item():.4f}")

    print("[7] Backward pass touches all trainable modules ...")
    model.zero_grad(set_to_none=True)
    out_train["loss"].backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"these trainable params received no grad: {no_grad[:5]}"
    print("    backward OK -- all trainable parameters received gradients")

    print("[8] Disabling feature interpolation and coeff dropout ...")
    model.feat_interp_prob = 0.0
    model.coeff_dropout = 0.0
    model.train()
    out_no_aug = model.compute_loss(batch)
    assert torch.isfinite(out_no_aug["loss"]).item()
    print(f"    no-aug train loss = {out_no_aug['loss'].item():.4f}")

    print("[9] Forcing odd input resolution to exercise _resize_if_needed ...")
    odd = torch.rand(bsz, 3, 137, 211, dtype=torch.float32)
    with torch.no_grad():
        odd_logits = model(odd)
    assert odd_logits.shape == (bsz,)
    print(f"    odd-shape logits = {odd_logits.tolist()}")

    print("\nAll CO-SPY smoke checks passed.")


if __name__ == "__main__":
    main()
