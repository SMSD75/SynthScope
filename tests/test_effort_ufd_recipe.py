"""Checks of the Effort UFD/ProGAN training recipe (conf/*/image/effort_ufd.yaml, experiments/image/effort_ufd.sh) against the
official UniversalFakeDetect_Benchmark code (data/datasets.py, models/trainer.py, script/train_svd.sh); no GPU, no download.

* the train list on a 256x256 uint8 image is a 224 crop of the CLIP-normalised image, possibly mirrored, as the official
  Compose [translate_duplicate(256), RandomCrop(224), RandomHorizontalFlip, ToTensor, Normalize(CLIP)] (all crop offsets and
  both flips occur, no other pixel change, up to float rounding); the eval list is the official validate.py CenterCrop(224) pipeline exactly,
* the trainer config holds the official numbers (lr 2e-4 constant, wd 0, no clipping, seed 0, step budget 24000 / 3000),
* `effort.yaml` (the served model) is untouched: squash eval list, no head init,
* the model builds, the trainable set is the SVD residual factors + the head, nothing else (needs the CLIP weights in the
  HF cache; skipped otherwise),
* ``head_single_logit``: cross-entropy on the logits [0, z] equals ``BCEWithLogitsLoss(z)`` (the official loss), and AdamW
  (weight decay 0 and 0.1) never moves row 0 of the head while row 1 moves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from huggingface_hub import try_to_load_from_cache
from omegaconf import OmegaConf
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from transformations import AIGITransformations, CLIP_MEAN, CLIP_STD   # noqa: E402

CONF = ROOT / "conf"
MODEL = OmegaConf.load(CONF / "model" / "image" / "effort_ufd.yaml")
TRAINER = OmegaConf.load(CONF / "trainer" / "image" / "effort_ufd.yaml")


def test_train_pipeline() -> None:
    print("[A] train list == crop 224 (+ flip) of the CLIP-normalised image ...")
    arr = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    full = T.Compose([T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])(Image.fromarray(arr, "RGB"))
    tf = AIGITransformations(list(MODEL.train_transform_list))
    img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    offsets, flips = set(), set()
    for _ in range(400):
        out = tf(img.clone())
        assert out.shape == (3, 224, 224) and out.dtype == torch.float32
        found = False
        for flip in (False, True):
            m = out.flip(-1) if flip else out        # the flip comes after the crop: mirror back to the crop window
            for di, dj in torch.isclose(full, m[:, 0, 0].view(3, 1, 1), atol=1e-5, rtol=0).all(0).nonzero().tolist():   # noise: the first pixel locates it
                if di + 224 <= 256 and dj + 224 <= 256 and torch.allclose(full[:, di:di + 224, dj:dj + 224], m, atol=1e-5, rtol=0):
                    offsets.add((di, dj)); flips.add(flip); found = True
                    break
            if found:
                break
        assert found, "train output is not a (mirrored) 224 crop of the normalised image"
    assert flips == {False, True}, flips
    assert len(offsets) > 50, len(offsets)
    pil = Image.fromarray(arr, "RGB")
    ev = AIGITransformations(list(MODEL.eval_transform_list))(pil)
    ref = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])(pil)
    assert torch.allclose(ev, ref, atol=1e-6, rtol=0), "eval list != official CenterCrop(224) pipeline"
    print(f"    400 samples: {len(offsets)} distinct crop offsets, both flips; eval == CenterCrop(224) (atol 1e-6)")


def test_recipe_numbers() -> None:
    print("[B] trainer config == script/train_svd.sh + models/trainer.py ...")
    assert TRAINER.lr == 2e-4 and TRAINER.weight_decay == 0.0 and TRAINER.seed == 0
    assert str(TRAINER.warmup.mode).lower() == "off", "a schedule would anneal the official constant lr"
    b = TRAINER.step_budget
    assert (b.unit, b.max_steps, b.save_every, b.grad_accum_steps, b.max_grad_norm) == ("optimizer_step", 24000, 3000, 2, 0.0)
    assert b.max_steps % b.save_every == 0 and 18000 % b.save_every == 0 and "cosine_t_max" not in b
    assert MODEL.head_init_std == 0.02 and MODEL.svd_rank == 1023 and MODEL.model_type == "effort"
    served = OmegaConf.load(CONF / "model" / "image" / "effort.yaml")
    assert list(served.eval_transform_list) == ["effort_cv2_resize_224", "to_dtype", "normalize_clip"]
    assert "head_init_std" not in served and "train_transform_list" not in served
    print("    lr 2e-4 constant, wd 0, no clipping, seed 0, 24000 iterations (2 x 24 accumulated) / ckpt 3000; effort.yaml unchanged")


def test_trainable_set_and_single_logit() -> None:
    print("[C] trainable parameters: SVD residual factors + head; single-logit head == official BCE ...")
    clip = "openai/clip-vit-large-patch14"
    if not isinstance(try_to_load_from_cache(clip, "config.json"), str):
        print(f"    skipped ({clip} not in the HF cache)")
        return
    from models.image.effort.effort import SynthScopeEffortModel
    assert MODEL.head_single_logit is True
    torch.manual_seed(0)
    model = SynthScopeEffortModel(OmegaConf.create({"ModelConfig": MODEL}))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n.startswith("head.") or n.rsplit(".", 1)[-1] in ("S_residual", "U_residual", "V_residual") for n in names), names
    assert sum(n.startswith("head.") for n in names) == 2 and len(names) == 24 * 4 * 3 + 2, len(names)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 96 * (1024 + 1 + 1024) + 1024 * 2 + 2, n
    assert abs(model.head.weight[1].std().item() - 0.02) < 0.002
    assert not model.head.weight[0].any() and model.head.bias[0].item() == 0.0
    print(f"    {len(names)} tensors, {n:,} parameters ({n - 1025:,} can move: row 0 of the head is pinned); head row-1 std {model.head.weight[1].std().item():.4f}")

    feats = torch.randn(64, 1024)
    y = torch.randint(0, 2, (64,))
    logits = model.head(feats)
    assert torch.equal(logits[:, 0], torch.zeros(64)), "logits are not [0, z]"
    ce = nn.CrossEntropyLoss()(logits, y)
    bce = nn.BCEWithLogitsLoss()(logits[:, 1], y.float())
    assert torch.allclose(ce, bce, atol=1e-6, rtol=0), (ce.item(), bce.item())
    ce.backward()                                   # the model's own hooks: row 0 gets no gradient, row 1 does
    assert not model.head.weight.grad[0].any() and model.head.bias.grad[0].item() == 0.0 and model.head.weight.grad[1].abs().max() > 0
    for wd in (0.0, 0.1):
        head = SynthScopeEffortModel.__new__(SynthScopeEffortModel)      # only the head and its hooks matter below
        nn.Module.__init__(head)
        head.head = nn.Linear(1024, 2)
        with torch.no_grad():
            head.head.weight[0].zero_(); head.head.bias[0].zero_()
        for prm in head.head.parameters():
            prm.register_hook(SynthScopeEffortModel._zero_row0)
        before = [prm.detach().clone() for prm in head.head.parameters()]
        opt = torch.optim.AdamW(head.head.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wd)
        for _ in range(5):
            nn.CrossEntropyLoss()(head.head(feats), y).backward()
            opt.step(); opt.zero_grad()
        after = list(head.head.parameters())
        assert all(torch.equal(b0[0], a0[0]) for b0, a0 in zip(before, after)), f"row 0 moved (wd {wd})"
        assert all(not torch.equal(b0[1], a0[1]) for b0, a0 in zip(before, after)), "row 1 did not move"
    print(f"    CE([0, z]) == BCEWithLogits(z) (|diff| {abs(ce.item() - bce.item()):.1e}); AdamW (wd 0 and 0.1): row 0 unchanged over 5 steps, row 1 moves")


if __name__ == "__main__":
    os.chdir(ROOT)
    test_train_pipeline()
    test_recipe_numbers()
    test_trainable_set_and_single_logit()
    print("Effort UFD recipe checks passed")
