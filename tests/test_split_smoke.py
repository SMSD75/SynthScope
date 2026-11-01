"""Smoke test for the SPLIT integration (no weight downloads: a tiny randomly initialized X-CLIP vision tower stands in
for X-CLIP B/16, saved to a temporary directory and loaded through the model's normal ``from_pretrained`` path).

Checks, on CPU:
* temporal roughness: a straight constant-speed token trajectory gives TTR = 0, a back-and-forth (jittery) one gives
  TTR > 0, matching the paper's definition log2(L1 / L2),
* motion incoherence: a motion field that is the same for every patch gives 0, a varying one gives > 0,
* the registry entry is training-free and uses D3's video preprocessing (16 frames, 224, BGR, ImageNet normalization),
* ``forward`` -> [B, 2] logits whose margin is ``score - threshold``; ``compute_loss`` returns a zero loss that
  backpropagates (the trainer runs, nothing is learned) and the encoder stays frozen in train mode.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from omegaconf import OmegaConf
from transformers import XCLIPVisionConfig, XCLIPVisionModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interface"))

from models.video.split import split as split_module  # noqa: E402
from models.video.split.split import SynthScopeSPLITModel  # noqa: E402
from backend.registry import get_spec                     # noqa: E402

TINY = XCLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         image_size=64, patch_size=16, mit_hidden_size=32, mit_intermediate_size=64,
                         mit_num_hidden_layers=1, mit_num_attention_heads=2, num_frames=4)


def build_model(tmp: str) -> SynthScopeSPLITModel:
    XCLIPVisionModel(TINY).save_pretrained(tmp)
    split_module.ENCODERS["tiny"] = tmp
    cfg = OmegaConf.create({"ModelConfig": {"model_type": "split", "encoder": "tiny", "gamma": 8.0, "threshold": 0.5},
                            "TrainerConfig": {}})
    return SynthScopeSPLITModel(cfg)


def test_temporal_roughness() -> None:
    print("[A] temporal roughness ...")
    t, n, d = 8, 4, 16
    direction = torch.randn(n, d)
    smooth = torch.arange(t, dtype=torch.float32)[None, :, None, None] * direction[None, None]   # [1, T, N, D]
    ttr = SynthScopeSPLITModel.temporal_roughness(smooth)
    assert ttr.shape == (1,) and abs(float(ttr)) < 1e-4, float(ttr)
    jitter = smooth.clone()
    jitter[:, 1::2] += 5 * torch.randn(n, d)          # odd frames jump away and back: long 1-step, short 2-step paths
    assert float(SynthScopeSPLITModel.temporal_roughness(jitter)) > 0.5


def test_motion_incoherence() -> None:
    print("[B] motion incoherence ...")
    t, g, d = 5, 3, 8
    uniform = torch.arange(t, dtype=torch.float32)[None, :, None, None] * torch.randn(d)[None, None, None]
    uniform = uniform.expand(1, t, g * g, d)          # every patch moves by the same vector
    assert float(SynthScopeSPLITModel.motion_incoherence(uniform)) < 1e-5
    assert float(SynthScopeSPLITModel.motion_incoherence(torch.randn(1, t, g * g, d))) > 0.1


def test_registry() -> None:
    print("[C] registry entry ...")
    spec = get_spec("split")
    assert spec.modality == "video" and spec.checkpoint is None and spec.weights == "training_free"
    assert spec.num_frames == 16 and spec.frame_size == 224 and spec.bgr and spec.edge_crop == 0.1


def test_forward_and_loss(model: SynthScopeSPLITModel) -> None:
    print("[D] forward / compute_loss ...")
    frames = torch.randn(2, 6, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        logits, score = model(frames), model.score(frames)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all() and (score >= 0).all()
    assert torch.allclose(logits[:, 1] - logits[:, 0], 2 * (score - model.threshold))
    model.train()
    assert not model.encoder.training and not any(p.requires_grad for p in model.encoder.parameters())
    out = model.compute_loss((frames, torch.tensor([0, 1]), torch.tensor([0, 0])))
    out["loss"].backward()
    assert out["loss"].item() == 0.0 and model._noop.grad is not None


if __name__ == "__main__":
    test_temporal_roughness()
    test_motion_incoherence()
    test_registry()
    with tempfile.TemporaryDirectory() as d:
        test_forward_and_loss(build_model(d))
    print("SPLIT smoke test passed")
