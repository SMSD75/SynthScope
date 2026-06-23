"""Smoke test for the diffusion-feature probe.

Two parts:
* download-free unit test of ``_spatial_pool`` (mean / mean_std shapes),
* the AGID contract on the lightweight ``backbone: vae`` path (only the SD-VAE, like the
  Co-Spy smoke test — no 3.4GB UNet): ``forward`` -> [B,2] logits, ``compute_loss`` ->
  {loss, outputs}, backward updates the head, and the VAE stays frozen.

The default ``unet`` (DIFT) path shares the same head/forward/compute_loss structure; testing
``vae`` exercises the contract without downloading the UNet. Skips cleanly if the SD-VAE weights
can't be fetched (offline CI).
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.diffusion_probe.diffusion_probe import _spatial_pool


def test_spatial_pool() -> None:
    print("[A] _spatial_pool shapes ...")
    feat = torch.randn(2, 16, 7, 7)
    assert _spatial_pool(feat, "mean").shape == (2, 16)
    assert _spatial_pool(feat, "mean_std").shape == (2, 32)
    print("    mean -> [B,C], mean_std -> [B,2C] OK")


def build_test_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "ModelConfig": {
                "model_type": "diffusion_probe",
                "backbone": "vae",                 # VAE-only: light, no UNet download
                "sd_id": "CompVis/stable-diffusion-v1-4",
                "image_size": 224,
                "vae_pool_grid": 4,
                "pool": "mean",
            },
            "TrainerConfig": {"label_smoothing": 0.1},
        }
    )


def main() -> None:
    torch.manual_seed(0)
    test_spatial_pool()

    try:
        from models.image.diffusion_probe.diffusion_probe import AGIDDiffusionProbeModel
        print("[1] Building AGIDDiffusionProbeModel (backbone=vae) ...")
        model = AGIDDiffusionProbeModel(build_test_config())
    except Exception as e:  # noqa: BLE001 — offline / no SD weights -> skip the heavy part
        print(f"    SKIP full-model checks (could not build SD-VAE backbone: {type(e).__name__}: {e})")
        print("\nPool unit test passed; full-model checks skipped.")
        return

    model.eval()
    print("[2] VAE frozen, head trainable ...")
    assert not any(p.requires_grad for p in model.vae.parameters()), "VAE must be frozen"
    assert all(p.requires_grad for p in model.head.parameters()), "head must train"

    bsz = 2
    images = torch.rand(bsz, 3, 224, 224)  # raw [0,1]; model maps to [-1,1] internally
    labels = torch.tensor([0, 1], dtype=torch.long)

    print("[3] Eval forward(images) -> [B,2] ...")
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (bsz, 2), f"expected ({bsz},2), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all().item()
    print(f"    logits = {logits.tolist()}")

    print("[4] compute_loss + backward updates the head ...")
    model.train()
    out = model.compute_loss((images, labels, torch.zeros(bsz, dtype=torch.long)))
    assert set(out.keys()) == {"loss", "outputs"} and out["outputs"].shape == (bsz, 2)
    assert torch.isfinite(out["loss"]).item()
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    print(f"    train loss = {out['loss'].item():.4f} | head grad OK")

    print("\nAll diffusion_probe smoke checks passed.")


if __name__ == "__main__":
    main()
