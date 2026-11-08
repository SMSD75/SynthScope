"""Smoke test for the VINA integration (CPU, no downloads).

Checks:
* the registry transform (``resize_short_256_bicubic``, ``center_crop_256``, ``to_image``, ``to_dtype``,
  ``normalize_imagenet``):
  [3, 256, 256] for small, non-square and large images, equal to the official eval processing 'default' of
  dataset/dataloader.py written out with torchvision v2 (``Resize(256, interpolation=3)``, ``CenterCrop(256)``,
  ``ToImage``, ``ToDtype(float32, scale=True)``, ImageNet ``Normalize``),
* a tiny-backbone ``SynthScopeVINAModel``: official key layout (``backbone.*`` + ``head.*``), eager attention, fp32,
  ``forward`` -> [B, 1] finite logits, a [B, T, ...] clip -> the mean of its per-frame logits (official eval),
  checkpoints with either ViT block prefix (transformers 4 ``backbone.layer`` / 5 ``backbone.model.layer``), wrapped
  as the official ``{"model_state_dict": ...}`` with ``module.`` names, load strictly through ``remap_vina`` with
  identical outputs, ``compute_loss`` (BCE with logits; outputs [-logit, logit]) + backward, and the trainer's
  ``AIGIMetrics`` (accuracy argmax, AP / AUROC softmax[:, 1]) on those outputs,
* ``vina_video``: the engine turns a clip into sigmoid(mean frame logit); its input notes; it is unavailable without
  ffmpeg; the ``synthscope`` API refuses a still image with the "video-only detector" message,
* ``media.ffmpeg_fps_frames`` on a synthetic clip: the frame count of ffmpeg's fps filter, the central window and the
  last-frame padding of the official test loader (skipped without ffmpeg),
* the full-size model (built on the meta device) has exactly the keys and shapes of the released checkpoint, if
  ``$SYNTHSCOPE_CKPT_DIR/vina.pth`` exists (skipped otherwise).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from omegaconf import OmegaConf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from backend import media                                                   # noqa: E402
from backend.engine import Engine                                           # noqa: E402
from metrics import AIGIMetrics                                             # noqa: E402
from synthscope.api import Detector                                         # noqa: E402
from backend.registry import get_spec, input_note, remap_vina               # noqa: E402
from models.image.vina.vina import SynthScopeVINAModel                      # noqa: E402

TINY = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# The official eval transform (dataset/dataloader.py, processing 'default', height = width = 256).
OFFICIAL_EVAL = v2.Compose([v2.Resize(256, interpolation=3), v2.CenterCrop(256), v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True), v2.Normalize(*IMAGENET)])


def noise_image(size: tuple[int, int]) -> Image.Image:
    gen = torch.Generator().manual_seed(size[0] * 7 + size[1])
    arr = torch.randint(0, 256, (size[1], size[0], 3), generator=gen, dtype=torch.uint8)
    return Image.fromarray(arr.numpy(), "RGB")


def tiny_model() -> SynthScopeVINAModel:
    return SynthScopeVINAModel(OmegaConf.create({"ModelConfig": {"model_type": "vina", "backbone_config": TINY}}))


def test_transform() -> None:
    print("[A] registry transform == the official eval processing ...")
    tf = get_spec("vina").transform
    assert list(tf.transform_list) == ["resize_short_256_bicubic", "center_crop_256", "to_image", "to_dtype",
                                      "normalize_imagenet"]
    assert get_spec("vina_video").transform is tf
    for size in [(180, 120), (128, 128), (256, 256), (300, 200), (513, 700), (1600, 1066), (300, 1200)]:
        img = noise_image(size)
        x = tf(img)
        assert x.shape == (3, 256, 256) and x.dtype == torch.float32, size
        assert torch.equal(x, OFFICIAL_EVAL(img)), size                 # bit-exact
    print("    7 sizes equal the official v2 pipeline")


def test_model_and_loader() -> None:
    print("[B] tiny model: key layout, forward, clips, both block prefixes, loss ...")
    torch.manual_seed(0)
    model = tiny_model().eval()
    assert model.backbone.config._attn_implementation == "eager"
    sd = model.state_dict()
    blocks = "backbone.model.layer." if any(k.startswith("backbone.model.layer.") for k in sd) else "backbone.layer."
    assert {"backbone.embeddings.cls_token", "backbone.norm.weight", "head.weight", "head.bias",
            f"{blocks}0.attention.q_proj.weight", f"{blocks}1.layer_scale2.lambda1"} <= set(sd)
    assert all(v.dtype == torch.float32 for v in sd.values())
    images = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        logits = model(images)
        clip = model(images.view(2, 2, 3, 256, 256))
    assert logits.shape == (4, 1) and torch.isfinite(logits).all()
    assert clip.shape == (2, 1) and torch.allclose(clip, logits.view(2, 2, 1).mean(dim=1), atol=1e-6)

    other = "backbone.layer." if blocks != "backbone.layer." else "backbone.model.layer."
    for prefix in (blocks, other):
        official = {"model_state_dict": {"module." + k.replace(blocks, prefix): v.clone() for k, v in sd.items()}}
        restored = tiny_model().eval()
        restored.load_state_dict(remap_vina(official), strict=True)
        with torch.no_grad():
            assert torch.equal(restored(images), logits), prefix

    labels = torch.tensor([0, 1, 0, 1])
    with torch.no_grad():                  # eval mode: train mode jitters DINOv3's RoPE (pos_embed_rescale)
        out = model.compute_loss((images, labels))
    single = out["outputs"][:, 1]
    assert torch.isfinite(out["loss"]) and out["outputs"].shape == (4, 2)
    assert torch.equal(out["outputs"][:, 0], -single) and torch.allclose(single, logits[:, 0], atol=1e-5)
    # softmax[:, 1] = sigmoid(2 logit): same order (AP / AUROC) and the same 0.5 threshold as sigmoid(logit)
    assert torch.allclose(out["outputs"].softmax(dim=1)[:, 1], torch.sigmoid(2 * single), atol=1e-6)
    assert torch.equal(out["outputs"].argmax(dim=1), (single > 0).long())
    metrics = AIGIMetrics(["real_accuracy", "fake_accuracy", "AP", "AUC"])
    metrics(out["outputs"].detach(), labels, torch.zeros(4, dtype=torch.long))
    values = {k: m.compute().item() for k, m in metrics._get_metrics_dict().items()}
    assert len(values) == 4 and all(0.0 <= v <= 1.0 for v in values.values()), values
    model.train()
    model.zero_grad()
    out = model.compute_loss((images, labels))
    assert torch.isfinite(out["loss"]) and out["outputs"].shape == (4, 2)
    out["loss"].backward()
    assert model.head.weight.grad is not None and model.backbone.embeddings.patch_embeddings.weight.grad is not None
    print(f"    {len(sd)} tensors, logits {tuple(logits.shape)}, clip mean, strict loads with both prefixes, "
          f"loss outputs (4, 2) -> AIGIMetrics {sorted(values)}")


def raise_no_binary() -> str:
    raise RuntimeError("No ffmpeg exe could be found")


def test_video_spec() -> None:
    print("[C] vina_video: sigmoid of the mean frame logit; input note ...")
    spec = get_spec("vina_video")
    assert spec.modality == "video" and spec.video_sampler == "ffmpeg_fps_center" and spec.video_prob == "sigmoid"
    assert (spec.num_frames, spec.fps, spec.checkpoint) == (1, 2.0, "vina.pth")
    torch.manual_seed(0)
    model = tiny_model().eval()
    clip = torch.randn(3, 3, 256, 256)
    p = Engine(device="cpu")._video_fake_prob(model, spec, clip)
    with torch.no_grad():
        want = torch.sigmoid(model(clip).mean())
    assert p.shape == (1,) and torch.allclose(p[0], want, atol=1e-6)
    note = input_note(spec)
    assert note["label"] == "The central frame at 2 fps" and " is scored " in note["explanation"], note
    eight = input_note(dataclasses.replace(spec, num_frames=8))
    assert eight["label"] == "The central 8 frames at 2 fps" and " are scored " in eight["explanation"], eight

    engine = Engine(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "vina.pth"), "wb").close()
        saved = {k: os.environ.get(k) for k in ("SYNTHSCOPE_FFMPEG", "PATH")}
        try:                                       # no ffmpeg anywhere -> unavailable (imageio-ffmpeg: skip this half)
            os.environ["SYNTHSCOPE_FFMPEG"], os.environ["PATH"] = "", tmp
            if media.imageio_ffmpeg is None:
                assert not media.ffmpeg_available()
                assert not engine.is_available(dataclasses.replace(spec, checkpoint=os.path.join(tmp, "vina.pth")))
            os.environ["SYNTHSCOPE_FFMPEG"] = sys.executable          # any existing file: found
            assert engine.is_available(dataclasses.replace(spec, checkpoint=os.path.join(tmp, "vina.pth")))
            os.environ["SYNTHSCOPE_FFMPEG"] = os.path.join(tmp, "no_such_ffmpeg")   # a dangling override: not found
            assert not media.ffmpeg_available()
            assert not engine.is_available(dataclasses.replace(spec, checkpoint=os.path.join(tmp, "vina.pth")))
            os.environ["SYNTHSCOPE_FFMPEG"] = ""
            saved_imageio = media.imageio_ffmpeg                      # imageio-ffmpeg without a binary raises RuntimeError
            media.imageio_ffmpeg = types.SimpleNamespace(get_ffmpeg_exe=raise_no_binary)
            try:
                assert not media.ffmpeg_available()
                assert not engine.is_available(spec) and engine.ffmpeg_missing(spec)
                try:                                                  # the API says ffmpeg, not "download the weights"
                    Detector(spec, device="cpu").predict("clip.mp4")
                except FileNotFoundError as e:
                    assert "needs ffmpeg" in str(e), e
                else:
                    raise AssertionError("vina_video ran without ffmpeg")
            finally:
                media.imageio_ffmpeg = saved_imageio
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        api = Detector(spec, checkpoint=os.path.join(tmp, "vina.pth"), device="cpu")
        try:
            api.predict(noise_image((300, 200)))
        except ValueError as e:
            assert "video-only detector" in str(e), e
        else:
            raise AssertionError("vina_video scored a still image")
    print(f"    P(fake) {p.item():.4f}; notes '{note['label']}' / '{eight['label']}'; ffmpeg gate; API image guard")


def test_ffmpeg_frames() -> None:
    try:
        exe = media.ffmpeg_exe()
    except FileNotFoundError:
        print("[D] ffmpeg_fps_frames: skipped (no ffmpeg; set SYNTHSCOPE_FFMPEG)")
        return
    print("[D] ffmpeg_fps_frames: fps filter, central window, padding ...")
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        # 3.0 s at 10 fps, the frame number drawn in the picture's brightness: fps=2 keeps 6 frames
        subprocess.run([exe, "-loglevel", "quiet", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
                        "-vf", "geq=lum='mod(N*8,256)':cb=128:cr=128", "-pix_fmt", "yuv420p", video], check=True)
        all6 = media.ffmpeg_fps_frames(video, fps=2.0, num_frames=6)
        assert len(all6) == 6 and all(f.size == (320, 240) and f.mode == "RGB" for f in all6)
        means = [float(np.asarray(f.convert("L")).mean()) for f in all6]
        assert means == sorted(means), means                       # time order
        center = media.ffmpeg_fps_frames(video, fps=2.0, num_frames=1)
        assert np.array_equal(center[0], all6[2])                              # (6 - 1) // 2 = 2
        two = media.ffmpeg_fps_frames(video, fps=2.0, num_frames=2)
        assert all(np.array_equal(a, b) for a, b in zip(two, all6[2:4]))
        padded = media.ffmpeg_fps_frames(video, fps=2.0, num_frames=8)
        assert len(padded) == 8 and all(np.array_equal(f, all6[-1]) for f in padded[6:])
        broken = os.path.join(tmp, "broken.mp4")                      # ffmpeg fails, writes nothing: a clear error
        with open(broken, "wb") as fh:
            fh.write(b"not a video" * 100)
        try:
            media.ffmpeg_fps_frames(broken, fps=2.0, num_frames=1)
        except ValueError as e:
            assert "no frames" in str(e), e
        else:
            raise AssertionError("a broken clip produced frames")
    print("    6 frames at 2 fps, center = frame 3, last frame repeated to pad; a broken clip -> 'no frames'")


def test_released_layout() -> None:
    ckpt = os.path.join(os.environ.get("SYNTHSCOPE_CKPT_DIR", ""), "vina.pth")
    if not os.path.isfile(ckpt):
        print("[E] released checkpoint layout: skipped ($SYNTHSCOPE_CKPT_DIR/vina.pth not found)")
        return
    print("[E] full-size model (meta device) vs the released checkpoint ...")
    with torch.device("meta"):
        model = SynthScopeVINAModel(OmegaConf.create({"ModelConfig": {"model_type": "vina"}}))
    ours = {k.replace("backbone.model.layer.", "backbone.layer."): tuple(v.shape)
            for k, v in model.state_dict().items()}
    released = remap_vina(torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True))
    assert ours == {k: tuple(v.shape) for k, v in released.items()}
    print(f"    {len(ours)} tensors: same names and shapes")


if __name__ == "__main__":
    test_transform()
    test_model_and_loader()
    test_video_spec()
    test_ffmpeg_frames()
    test_released_layout()
    print("VINA smoke test passed")
