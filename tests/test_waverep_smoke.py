"""Smoke test for the WaveRep integration (CPU, no downloads).

Checks:
* the registry transform (``center_crop_504``, ``to_dtype``, ``normalize_imagenet``) is bit-exact with the official
  ``create_transform(504)`` (vendored demo/utils.py: torchvision ``CenterCrop((504, 504))``, ``ToTensor``, ImageNet
  ``Normalize``) on frames smaller than the crop (zero padding), non-square both ways and larger,
* ``media.cv2_frames`` yields exactly the frames of the official ``ReadVideoIteratorCV`` (same count and pixels, with
  and without a limit) on a synthetic clip written with OpenCV,
* ``SynthScopeWaveRepModel``: the official timm model (``vit_base_patch14_reg4_dinov2``, one output) under
  ``model.``; ``forward`` [B, 3, H, W] -> [B] finite logits = the official ``model(x)[:, -1]``; [B, T, ...] -> the
  mean of its frame logits, independent of ``frame_batch``; a flat timm state dict and the Lightning
  ``{"state_dict": {"model.*"}}`` form both load strictly through ``remap_waverep`` with identical outputs;
  ``compute_loss`` (BCE with logits, outputs [-logit, logit]) + backward,
* the ``waverep`` / ``waverep_g4`` specs: official kind, 64 frames, cv2 sampler, sigmoid; the engine's
  ``predict_video_clip`` is sigmoid(mean of the per-frame logits of the first 64 frames) and equals the official
  ``evaluate`` on the official reader + transform (a tiny one-block ViT stands in for the 86M one); ``video_clip``
  stacks the same inputs; the input note; the ``synthscope`` API refuses a still image,
* with ``$SYNTHSCOPE_CKPT_DIR/waverep.pth`` / ``waverep_g4.pth`` (the released weights_dinov2_G{1,4}.ckpt): the engine
  loads them strictly (0 missing / 0 unexpected, no transformers key translation) and every tensor equals the file's
  (skipped otherwise).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

import cv2
import numpy as np
import timm
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from backend import config as backend_config                                     # noqa: E402
from backend import media                                                         # noqa: E402
from backend.engine import Engine, _load_checkpoint                              # noqa: E402
from backend.hf_compat import translate_hf_keys                                  # noqa: E402
from backend.registry import get_spec, input_note, remap_waverep                  # noqa: E402
from models.video.waverep import official_utils as official                       # noqa: E402
from models.video.waverep.waverep import SynthScopeWaveRepModel                   # noqa: E402
from synthscope.api import Detector                                               # noqa: E402

KEYS = ("waverep", "waverep_g4")
# A one-block ViT with the official layout (registers, one output) for fast CPU checks at the real 504 input.
TINY = {"embed_dim": 48, "depth": 1, "num_heads": 2}


def noise_image(size: tuple[int, int]) -> Image.Image:
    gen = torch.Generator().manual_seed(size[0] * 7 + size[1])
    arr = torch.randint(0, 256, (size[1], size[0], 3), generator=gen, dtype=torch.uint8)
    return Image.fromarray(arr.numpy(), "RGB")


def model_with(crop: int = 504, tiny: bool = True, frame_batch: int = 16) -> SynthScopeWaveRepModel:
    cfg = OmegaConf.create({"ModelConfig": {"model_type": "waverep", "crop_size": crop, "frame_batch": frame_batch}})
    model = SynthScopeWaveRepModel(cfg)
    if tiny:                       # the same timm class and key layout, one narrow block
        model.model = timm.create_model("vit_base_patch14_reg4_dinov2", num_classes=1, pretrained=False, img_size=crop,
                                        **TINY)
        # timm starts the head at zero and DINOv2's LayerScale at 1e-5: give the random net an input-dependent output
        torch.nn.init.normal_(model.model.head.weight, std=1.0)
        for block in model.model.blocks:
            block.ls1.gamma.data.fill_(1.0)
            block.ls2.gamma.data.fill_(1.0)
    return model.eval()


def write_clip(path: str, n: int, size=(160, 120)) -> None:
    """A clip of ``n`` frames whose brightness encodes the frame number (OpenCV mp4v)."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, size)
    assert writer.isOpened(), "OpenCV cannot write mp4v"
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), (i * 3) % 256, np.uint8)
        frame[: size[1] // 2] = rng.integers(0, 256, (size[1] // 2, size[0], 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_transform() -> None:
    print("[A] registry transform == the official create_transform(504) ...")
    tf = get_spec("waverep").transform
    assert list(tf.transform_list) == ["center_crop_504", "to_dtype", "normalize_imagenet"]
    assert get_spec("waverep_g4").transform is tf
    golden = official.create_transform(504)
    for size in [(298, 224), (256, 256), (503, 505), (504, 504), (512, 320), (672, 384), (1280, 720), (2160, 544)]:
        img = noise_image(size)
        x = tf(img)
        assert x.shape == (3, 504, 504) and x.dtype == torch.float32, size
        assert torch.equal(x, golden(img)), size                       # bit-exact, incl. the zero padding
    print("    8 sizes equal the official torchvision v1 pipeline")


def test_reader() -> None:
    print("[B] media.cv2_frames == the official ReadVideoIteratorCV ...")
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        write_clip(clip, 70)
        ours = list(media.cv2_frames(clip))
        ref = [f for _, f in official.ReadVideoIteratorCV(clip)]
        assert len(ours) == len(ref) == 70
        assert all(o.mode == "RGB" and np.array_equal(np.asarray(o), np.asarray(r)) for o, r in zip(ours, ref))
        limited = list(media.cv2_frames(clip, limit=64))
        ref64 = [f for _, f in official.ReadVideoIteratorCV(clip, limit=64)]
        assert len(limited) == len(ref64) == 64
        assert all(np.array_equal(np.asarray(o), np.asarray(r)) for o, r in zip(limited, ref64))
        assert len(list(media.cv2_frames(os.path.join(tmp, "missing.mp4")))) == 0
    print("    70 frames bit-exact, limit 64 -> the first 64, a missing file -> no frames")


def test_model_and_loader() -> None:
    print("[C] model: official layout, forward, clips, strict loads, loss ...")
    full = SynthScopeWaveRepModel(OmegaConf.create({"ModelConfig": {"model_type": "waverep"}}))
    assert type(full.model).__name__ == "VisionTransformer" and full.model.num_reg_tokens == 4
    assert full.model.patch_embed.img_size == (504, 504) and full.model.head.out_features == 1
    assert sum(p.numel() for p in full.parameters()) > 80e6
    del full
    torch.manual_seed(0)
    model = model_with(crop=56, tiny=False)              # the full ViT-B/14 at a 4x4 patch grid: quick on CPU
    images = torch.randn(6, 3, 56, 56)
    with torch.no_grad():
        logits = model(images)
        assert torch.equal(logits, model.model(images)[:, -1])
        clip = model(images.view(2, 3, 3, 56, 56))
        model.frame_batch = 2
        clip2 = model(images.view(2, 3, 3, 56, 56))
    assert logits.shape == (6,) and torch.isfinite(logits).all()
    assert clip.shape == (2,) and torch.allclose(clip, logits.view(2, 3).mean(dim=1), atol=1e-6)
    assert torch.allclose(clip2, clip, atol=1e-6)

    flat = {k[len("model."):]: v.clone() for k, v in model.state_dict().items()}
    lightning = {"state_dict": {"model." + k: v for k, v in flat.items()}, "epoch": 3}
    for released in (flat, lightning):
        restored = model_with(crop=56, tiny=False)
        restored.load_state_dict(remap_waverep(released), strict=True)
        with torch.no_grad():
            assert torch.equal(restored(images), logits)

    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    model.train()
    model.zero_grad()
    out = model.compute_loss((images, labels))
    assert torch.isfinite(out["loss"]) and out["outputs"].shape == (6, 2)
    assert torch.equal(out["outputs"][:, 0], -out["outputs"][:, 1])
    out["loss"].backward()
    assert model.model.head.weight.grad is not None and model.model.patch_embed.proj.weight.grad is not None
    print(f"    {len(flat)} tensors, logits {tuple(logits.shape)}, clip mean (frame_batch 16 / 2), flat + Lightning "
          f"strict loads, loss outputs (6, 2) + backward")


def test_specs_and_engine() -> None:
    print("[D] specs, engine video path == official evaluate, input note, API image guard ...")
    for key in KEYS:
        spec = get_spec(key)
        assert spec.modality == "video" and spec.predict_kind == "video" and spec.ckpt_kind == "official_waverep"
        assert (spec.video_sampler, spec.num_frames, spec.video_prob, spec.weights) == ("cv2_frames", 64, "sigmoid",
                                                                                         "official")
        assert spec.overrides == ["model@ModelConfig=video/waverep"] and spec.remap is remap_waverep
    assert (get_spec("waverep").checkpoint, get_spec("waverep_g4").checkpoint) == ("waverep.pth", "waverep_g4.pth")
    note = input_note(get_spec("waverep"))
    assert note["label"] == "The first 64 frames, center 504", note
    assert input_note(dataclasses.replace(get_spec("waverep"), num_frames=0))["label"] == "Every frame, center 504"

    torch.manual_seed(0)
    model = model_with()
    engine = Engine(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        write_clip(clip, 70, size=(640, 480))
        for spec, n in ((get_spec("waverep"), 64), (dataclasses.replace(get_spec("waverep"), key="waverep_all",
                                                                         num_frames=0), 70)):
            engine._cache[spec.key] = model
            reader = official.ReadVideoIteratorCV(clip, transform=official.create_transform(504),
                                                  limit=spec.num_frames or None)
            ref = official.evaluate(model.model, reader, "cpu")     # the official evaluate takes the timm model
            assert len(ref) == n
            logits = engine.video_frame_logits(spec, clip)
            assert logits.shape == (n,)
            assert torch.allclose(logits.double(), torch.tensor(ref["logit"].to_numpy()), atol=1e-5), spec.key
            res = engine.predict_video_clip(spec, clip)
            want = float(torch.sigmoid(torch.tensor(ref["logit"].mean(), dtype=torch.float64)))
            assert res["num_frames"] == n and abs(res["fake_probability"] - round(want, 6)) < 2e-6, (res, want)
            print(f"    {n} frames: official logits mean {ref['logit'].mean():.4f} std {ref['logit'].std():.4f}")
            assert ref["logit"].std() > 0.01                # frames score differently (a non-trivial comparison)
            assert abs(res["logit"] - ref["logit"].mean()) < 1e-5
            stacked = engine.video_clip(spec, clip)
            assert stacked.shape == (n, 3, 504, 504)
            assert torch.equal(stacked[0], get_spec("waverep").transform(next(media.cv2_frames(clip))))
            with torch.no_grad():                    # the generic clip path (model(clip) -> mean) agrees
                p = engine._video_fake_prob(model, spec, stacked)
            assert abs(float(p) - want) < 1e-5
        empty = os.path.join(tmp, "empty.mp4")
        with open(empty, "wb") as fh:
            fh.write(b"not a video" * 100)
        try:
            engine.predict_video_clip(get_spec("waverep"), empty)
        except ValueError as e:
            assert "Could not decode any frames" in str(e), e
        else:
            raise AssertionError("a broken clip was scored")
        open(os.path.join(tmp, "waverep.pth"), "wb").close()
        api = Detector(get_spec("waverep"), checkpoint=os.path.join(tmp, "waverep.pth"), device="cpu")
        try:
            api.predict(noise_image((300, 200)))
        except ValueError as e:
            assert "video-only detector" in str(e), e
        else:
            raise AssertionError("waverep scored a still image")
    print(f"    64 / 70 frame logits == official evaluate, P(fake) {res['fake_probability']:.4f}; note "
          f"'{note['label']}'; broken clip -> error; API image guard")


def test_released_checkpoints() -> None:
    ckpt_dir = os.environ.get("SYNTHSCOPE_CKPT_DIR", "")
    if not all(os.path.isfile(os.path.join(ckpt_dir, get_spec(k).checkpoint)) for k in KEYS):
        print("[E] released checkpoints: skipped ($SYNTHSCOPE_CKPT_DIR/waverep.pth, waverep_g4.pth not found)")
        return
    print("[E] released checkpoints: strict engine load, every tensor equal ...")
    saved = backend_config.CHECKPOINTS_DIR
    backend_config.CHECKPOINTS_DIR = type(saved)(ckpt_dir)
    try:
        for key in KEYS:
            spec = get_spec(key)
            model = SynthScopeWaveRepModel(OmegaConf.create({"ModelConfig": {"model_type": "waverep"}}))
            status = _load_checkpoint(model, spec)
            assert "(missing=0, unexpected=0)" in status and "translated" not in status, status
            released = torch.load(os.path.join(ckpt_dir, spec.checkpoint), map_location="cpu", weights_only=False)
            sd = model.state_dict()
            assert translate_hf_keys(remap_waverep(released), sd)[1] == ""
            assert set(sd) == {"model." + k for k in released}
            assert all(torch.equal(sd["model." + k], v) for k, v in released.items())
            print(f"    {key}: {status}")
    finally:
        backend_config.CHECKPOINTS_DIR = saved


if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    test_transform()
    test_reader()
    test_model_and_loader()
    test_specs_and_engine()
    test_released_checkpoints()
    print("OK")
