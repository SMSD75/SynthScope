"""Smoke test for the LTF integration (CPU).

Builds the real model, so it needs the pinned CLIP ViT-L/14 (``$SYNTHSCOPE_LTF_BACKBONE``, the OpenAI clip cache
~/.cache/clip/ViT-L-14.pt or ``$SYNTHSCOPE_CLIP_DIR``, else downloaded from HF MuskAI/LTF-v3.3). Checks:
* the tiling policy: tile starts per axis (500 px -> 0, 224, 276; exact multiples add no duplicate; a short axis keeps
  one crop) and the view count, black padding and global view of the registry transform,
* the MIL head has the released key layout; if ``$SYNTHSCOPE_CKPT_DIR/ltf.pth`` exists, it is the pinned file (SHA-256)
  and ``remap_ltf`` keeps exactly those 37 tensors (dropping the 27 disconnected baseline ones), which the engine's
  strict official-weights load accepts,
* only the frozen CLIP falls under ``pretrained_prefixes`` and it is frozen,
* ``forward`` -> [B, 2] finite logits on a synthetic image in eval mode; ``compute_loss`` + backward reach the MIL head
  and not CLIP; ``compute_loss`` refuses non-view (4-D) input with a clear error (training is not ported),
* the web UI "model input" preview un-normalizes with the CLIP mean / std and wraps many views into a grid,
* the Space: ``fetch_weights.py`` serves ``ltf`` from the authors' repo (not the zoo, still in ``NEVER_REDISTRIBUTE``),
  and ``SYNTHSCOPE_MAX_VIEWS`` refuses an over-limit image before the model loads, without touching normal sizes.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interface"))

from config_utils import compose_config                       # noqa: E402
import models.image.ltf.ltf as ltf_module                     # noqa: E402
from models.image.ltf.ltf import CORE_SHA256, SynthScopeLTFModel, sha256   # noqa: E402
from models.image.ltf.preprocess import LTFViews              # noqa: E402
from transformations import CLIP_MEAN, CLIP_STD               # noqa: E402
from backend import config as backend_config                  # noqa: E402
from backend.engine import Engine, InputTooLarge, _load_checkpoint, bag_views, check_views   # noqa: E402
from backend.registry import get_spec, remap_ltf              # noqa: E402
from synthscope import hub                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

RELEASED_CORE = 37        # retained MIL / fusion tensors of the released checkpoint
RELEASED_UNUSED = 27      # disconnected baseline tensors the official loader drops


def noise_pil(w: int, h: int, seed: int = 0) -> Image.Image:
    arr = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class NoCallTransform:
    """A registry transform stand-in with the same steps (``bag_views`` reads them) that must never be applied."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        raise AssertionError("the transform ran for a refused input")


def refused(fn, *args) -> str:
    """Call ``fn(*args)``, which must raise InputTooLarge; return its message."""
    try:
        fn(*args)
    except InputTooLarge as e:
        return str(e)
    raise AssertionError("an over-limit input was accepted")


def build_model() -> SynthScopeLTFModel:
    return SynthScopeLTFModel(compose_config(get_spec("ltf").overrides))


def test_tiling() -> None:
    print("[A] tiling policy / registry transform ...")
    starts = LTFViews.start_positions
    assert starts(500) == [0, 224, 276], starts(500)
    assert starts(448) == [0, 224] and starts(224) == [0] and starts(672) == [0, 224, 448]
    assert starts(200) == [0] and starts(225) == [0, 1]
    tf = get_spec("ltf").transform
    for (w, h), views in {(300, 200): 3, (500, 375): 7, (448, 448): 5, (200, 150): 2, (1300, 900): 31}.items():
        x = tf(noise_pil(w, h))
        assert tuple(x.shape) == (views, 3, 224, 224) and torch.isfinite(x).all(), ((w, h), tuple(x.shape))
    # A 200x150 image: one tile padded with black beyond the image (normalized zeros = -mean/std).
    x = tf(noise_pil(200, 150))
    black = (-torch.tensor(CLIP_MEAN) / torch.tensor(CLIP_STD))[:, None, None]
    assert torch.allclose(x[1, :, 150:, :], black.expand(3, 74, 224))
    assert torch.allclose(x[1, :, :, 200:], black.expand(3, 224, 24))
    assert not torch.allclose(x[0, :, 150:, :], black.expand(3, 74, 224))   # the global view is resized, not padded
    print("    starts(500) = [0, 224, 276]; view counts and padding as the official prepare_image")


def test_fetch_core() -> None:
    print("[A2] fetch_core: copies the checkpoint as ltf.pth only if the pinned SHA-256 matches ...")
    real_download = ltf_module.hf_hub_download
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.pth"
        bad.write_bytes(b"not the checkpoint")
        calls = []

        def fake_download(repo, filename, revision=None):
            calls.append((repo, filename, revision))
            return str(bad)

        ltf_module.hf_hub_download = fake_download
        try:
            try:
                ltf_module.fetch_core(Path(tmp) / "out")
                raise AssertionError("a file with another SHA-256 was accepted")
            except ValueError:
                pass
            assert calls == [(ltf_module.HUB_REPO, ltf_module.CORE_FILENAME, ltf_module.HUB_REVISION)], calls
            assert not (Path(tmp) / "out" / "ltf.pth").exists()
            ckpt = backend_config.CHECKPOINTS_DIR / "ltf.pth"
            if ckpt.is_file() and sha256(ckpt) == CORE_SHA256:       # the real file: accepted and copied byte for byte
                ltf_module.hf_hub_download = lambda *a, **k: str(ckpt)
                dest = ltf_module.fetch_core(Path(tmp) / "out")
                assert dest.name == "ltf.pth" and sha256(dest) == CORE_SHA256
                print("    wrong bytes refused; the pinned file copied as ltf.pth")
            else:
                print("    wrong bytes refused (no pinned ltf.pth present for the positive case)")
        finally:
            ltf_module.hf_hub_download = real_download


def test_keys_and_frozen(model) -> None:
    print("[B] key layout / checkpoint / frozen CLIP ...")
    keys = list(model.state_dict())
    trained = [k for k in keys if not k.startswith(model.pretrained_prefixes)]
    assert all(k.startswith("core.") for k in trained) and len(trained) == RELEASED_CORE, len(trained)
    ckpt = backend_config.CHECKPOINTS_DIR / "ltf.pth"
    if ckpt.is_file():
        released = torch.load(ckpt, map_location="cpu", weights_only=True)
        kept = remap_ltf(released)
        assert sha256(ckpt) == CORE_SHA256, "ltf.pth is not the pinned official weights/epoch_20.pth"
        assert len(released) == RELEASED_CORE + RELEASED_UNUSED, len(released)
        assert set(kept) == set(trained), sorted(set(kept) ^ set(trained))[:8]
        status = _load_checkpoint(model, get_spec("ltf"))
        for k, v in kept.items():
            assert torch.equal(model.state_dict()[k], v), k
        print(f"    {ckpt}: {status}; {len(kept)} retained tensors equal, {len(released) - len(kept)} dropped")
    else:
        print(f"    {ckpt} not present: checkpoint load skipped")
    frozen = [n for n, p in model.named_parameters() if n.startswith(model.pretrained_prefixes)]
    assert frozen and not any(p.requires_grad for n, p in model.named_parameters() if n.startswith("visual."))
    print(f"    {len(trained)} MIL-head tensors; {len(frozen)} frozen CLIP parameters")


def test_forward_and_loss(model) -> None:
    print("[C] forward / compute_loss ...")
    tf = get_spec("ltf").transform
    x = torch.stack([tf(noise_pil(300, 260, seed=s)) for s in (1, 2)])      # [2, 3, 3, 224, 224]
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (2, 2) and torch.isfinite(logits).all(), logits
    p = logits.softmax(1)[:, 1]
    print(f"    P(fake) on noise: {[round(float(v), 4) for v in p]}")
    model.train()
    assert not model.visual.training
    out = model.compute_loss((x, torch.tensor([0, 1]), torch.tensor([0, 0])))
    out["loss"].backward()
    assert model.core.fusion_head[0].weight.grad is not None and model.core.head_level1.weight.grad is not None
    assert model.visual.conv1.weight.grad is None
    model.zero_grad(set_to_none=True)
    model.eval()
    try:
        model.compute_loss((x[:, 0], torch.tensor([0, 1]), torch.tensor([0, 0])))     # [B, 3, 224, 224]: no views
    except NotImplementedError as e:
        assert "not ported" in str(e), e
    else:
        raise AssertionError("compute_loss accepted 4-D input")


def test_preview() -> None:
    print("[D] web UI model-input preview ...")
    spec, tf = get_spec("ltf"), get_spec("ltf").transform
    gray = Image.new("RGB", (300, 200), (128, 128, 128))
    png = Engine._model_input_data_url(spec, tf(gray).unsqueeze(0))
    img = Image.open(io.BytesIO(base64.b64decode(png.split(",", 1)[1])))
    assert img.size == (3 * 224, 224), img.size                                    # <= 5 views: one row
    assert abs(np.asarray(img)[:, :224].mean() - 128) < 1.5, "global view is not shown with its real colours"
    big = Engine._model_input_data_url(spec, tf(noise_pil(1300, 900)).unsqueeze(0))   # 31 views -> 6 x 6 grid
    img = Image.open(io.BytesIO(base64.b64decode(big.split(",", 1)[1])))
    assert max(img.size) <= 1280 and img.width == img.height, img.size
    print(f"    3 views {(3 * 224, 224)}; 31 views wrapped to {img.size}")


def test_space_fetch() -> None:
    print("[E] Space: ltf fetched from the authors' repo, never from / to the zoo ...")
    path = ROOT / "deploy" / "hf-space" / "fetch_weights.py"
    default = ast.literal_eval(path.read_text().split("DEFAULT = ", 1)[1].split("]", 1)[0] + "]")
    assert "ltf" in default and "ltf" in hub.NEVER_REDISTRIBUTE
    spec_ = importlib.util.spec_from_file_location("fetch_weights", path)
    fetch = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(fetch)
    assert set(fetch.UPSTREAM) == {"ltf"}
    calls = []
    real = ltf_module.fetch_core, ltf_module.resolve_backbone, hub.download
    saved = {k: os.environ.get(k) for k in ("SYNTHSCOPE_HUB_REPO", "SYNTHSCOPE_SPACE_DETECTORS", "SYNTHSCOPE_CKPT_DIR")}
    ltf_module.fetch_core = lambda d: calls.append(("core", Path(d)))
    ltf_module.resolve_backbone = lambda path=None: calls.append(("backbone", path))
    hub.download = lambda key, **kw: calls.append(("zoo", key))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SYNTHSCOPE_CKPT_DIR"] = tmp
            os.environ["SYNTHSCOPE_SPACE_DETECTORS"] = "ltf"
            os.environ.pop("SYNTHSCOPE_HUB_REPO", None)          # the authors' repo needs no SynthScope weights repo
            fetch.main()
            assert calls == [("core", Path(tmp)), ("backbone", None)], calls
            calls.clear()
            os.environ["SYNTHSCOPE_HUB_REPO"] = "test/none"
            Path(tmp, "ltf.pth").write_bytes(b"x")                 # already staged: only the backbone is checked
            Path(tmp, "d3_calibration.json").write_text("{}")      # (keeps main() off the network)
            fetch.main()
            assert calls == [("backbone", None)], calls              # and never hub.download("ltf")
    finally:
        ltf_module.fetch_core, ltf_module.resolve_backbone, hub.download = real
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("    DEFAULT and UPSTREAM hold ltf; fetched via fetch_core + resolve_backbone, no zoo download")


def test_view_count() -> None:
    print("[F0] bag_views(size) == the transform's view count (LTF, PROBE, PROBE-ResNet) ...")
    sizes = [(1, 1), (7, 300), (150, 200), (200, 150), (223, 224), (224, 224), (225, 224), (300, 200), (335, 337),
             (336, 336), (448, 448), (449, 300), (500, 375), (672, 673), (900, 120), (1300, 900), (1920, 1080),
             (2048, 1536)]
    rng = np.random.default_rng(0)
    sizes += [tuple(int(v) for v in rng.integers(1, 1400, 2)) for _ in range(12)]
    checked = 0
    for key in ("ltf", "probe", "probe_resnet"):
        spec = get_spec(key)
        for w, h in sizes:
            n = int(spec.transform(Image.new("RGB", (w, h))).shape[0])
            assert bag_views(spec, (w, h)) == n, (key, (w, h), bag_views(spec, (w, h)), n)
            checked += 1
    assert bag_views(get_spec("aide"), (500, 500)) is None and bag_views(get_spec("spai"), (500, 500)) is None
    print(f"    {checked} (detector, size) pairs match; AIDE / SPAI are not view bags")


def test_view_guard() -> None:
    print("[F] SYNTHSCOPE_MAX_VIEWS guard (from the image size, before any preprocessing) ...")
    spec = get_spec("ltf")
    original = backend_config.MAX_VIEWS, spec.transform
    try:
        backend_config.MAX_VIEWS = 0                                   # default: no limit
        check_views(spec, (1300, 900))
        backend_config.MAX_VIEWS = 31
        check_views(spec, (1300, 900))                                 # 31 views, at the limit: scored as usual
        check_views(get_spec("aide"), (4000, 3000), frames=16)         # fixed-view detectors are never counted
        backend_config.MAX_VIEWS = 30
        msg = refused(check_views, spec, (1300, 900))
        assert "as 31 views" in msg and "at most 30" in msg and "smaller or cropped image" in msg, msg
        msg = refused(check_views, spec, (300, 200), 16)               # video: 3 views x 16 frames = 48
        assert "16 sampled video frames" in msg and "48 views (3 per frame)" in msg and "Video input" in msg, msg
        engine = Engine(device="cpu")

        spec.transform = NoCallTransform(original[1].transforms)      # same steps; running it would fail the test
        refused(engine.predict_image, spec, noise_pil(1300, 900))
        refused(engine.predict_video_frames, spec, [noise_pil(300, 200)] * 16)
        assert not engine._cache                                       # refused before the model is built
    finally:
        backend_config.MAX_VIEWS = original[0]
        spec.transform = original[1]
    print("    no limit by default; 31 views pass at 31, refused at 30 before the transform and the load; "
          "video message names the frames")


def test_availability() -> None:
    print("[G] ltf is available only with ltf.pth AND the pinned backbone (no download at request time) ...")
    spec, engine = get_spec("ltf"), Engine(device="cpu")
    saved = {k: os.environ.get(k) for k in ("SYNTHSCOPE_LTF_BACKBONE", "SYNTHSCOPE_CLIP_DIR")}
    original_dir, original_cache = backend_config.CHECKPOINTS_DIR, ltf_module.try_to_load_from_cache
    try:
        with tempfile.TemporaryDirectory() as tmp:
            backend_config.CHECKPOINTS_DIR = Path(tmp)
            os.environ.pop("SYNTHSCOPE_LTF_BACKBONE", None)
            os.environ["SYNTHSCOPE_CLIP_DIR"] = str(Path(tmp, "noclip"))
            ltf_module.try_to_load_from_cache = lambda *a, **k: None   # not in the Hugging Face cache either
            assert not engine.is_available(spec)                         # nothing
            Path(tmp, "ltf.pth").write_bytes(b"x")
            assert not ltf_module.backbone_present() and not engine.is_available(spec)   # checkpoint only
            ltf_module.try_to_load_from_cache = lambda *a, **k: str(Path(tmp, "ltf.pth"))
            assert ltf_module.backbone_present() and engine.is_available(spec)           # + backbone in the HF cache
            ltf_module.try_to_load_from_cache = lambda *a, **k: None
            os.environ["SYNTHSCOPE_LTF_BACKBONE"] = str(Path(tmp, "ltf.pth"))
            assert engine.is_available(spec)                                             # + explicit backbone file
    finally:
        backend_config.CHECKPOINTS_DIR, ltf_module.try_to_load_from_cache = original_dir, original_cache
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("    unavailable with the checkpoint alone; available once the backbone is cached or configured")


if __name__ == "__main__":
    test_tiling()
    test_fetch_core()
    m = build_model()
    test_keys_and_frozen(m)
    test_forward_and_loss(m)
    test_preview()
    test_space_fetch()
    test_view_count()
    test_view_guard()
    test_availability()
    print("LTF smoke test passed")
