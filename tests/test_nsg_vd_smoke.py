"""Smoke test for the NSG-VD integration (CPU; no downloads).

Checks:
[A] the ``nsgvd_frame_224`` transform == the official ``_preprocess_frame_from_path`` (written out: PIL LANCZOS 224x224,
    ``torch.from_numpy(np.array(img)).float() / 255``, CHW) on JPEG frames of several sizes, bit-exact;
[B] media.ffmpeg_select_frames == the official ``process_video`` command (written out, run through a shell exactly as
    ``os.system`` does) on a synthetic clip, same decoded pixels; media.ffmpeg_frame_count == moviepy's
    ``int(fps * duration)`` when moviepy is importable; a clip shorter than 8 frames and an empty file are a
    VideoDecodeError, a missing or unreadable file an OSError (run with NSG-VD's binary, $SYNTHSCOPE_FFMPEG_NSG_VD or
    else the global choice; skipped when there is none);
[C] SynthScopeNSGVDModel with a small random UNet: NSG features [B, 8, 3, 224, 224] with the end frames x100, score
    [B] finite and per row independent of the batch, ``fake_probability`` = sigmoid((s - 1) / prob_scale); a released
    checkpoint layout (``deep_MMD`` keys) loads strictly through ``remap_nsg_vd``; a mismatch raises in the engine;
    ``compute_loss`` refuses;
[D] with the official code ($NSGVD_OFFICIAL_DIR = an NSG-VD clone, its extra deps importable, e.g. via PYTHONPATH):
    the vendored TALL discriminator, MMD_batch2, score function and NSG (``_get_score_features``) equal the official
    modules on the same inputs and weights (skipped otherwise);
[E] the nsg_vd / nsg_vd_pika / nsg_vd_unbalance specs (official kind, 8 frames, ffmpeg_select_jpeg, mmd, aux files),
    the input note, and the engine's predict_video_clip on a synthetic clip with a small model: P(fake) and the
    reported raw statistic (``logit``) equal the model's, the verdict is the official ``> 1`` (a score of exactly 1 is
    real), the ffmpeg build is recorded and a pin mismatch warned about, $SYNTHSCOPE_FFMPEG_NSG_VD selects the binary
    (other detectors unaffected), and bench_eval_video's cache_meta carries the aux-file and ffmpeg identity (the
    engine part runs when NSG-VD's binary resolves, even with no global ffmpeg);
[F] with $SYNTHSCOPE_CKPT_DIR holding nsg_vd*.pth (the released checkpoints) + 256x256_diffusion_uncond.pt +
    nsg_vd_reference_msrvtt_val100.safetensors: every registered checkpoint loads strictly in the engine (0 missing /
    0 unexpected) and equals the file; the reference set is [100, 8, 3, 224, 224] (skipped otherwise).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import math
import os
import subprocess
import sys
import tempfile
import types
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface"), os.path.join(ROOT, "benchmark")]

from backend import config as backend_config                                     # noqa: E402
from backend import media                                                         # noqa: E402
from backend.engine import Engine, _load_checkpoint, spec_ffmpeg_exe             # noqa: E402
from backend.registry import get_spec, input_note, remap_nsg_vd                   # noqa: E402
from config_utils import compose_config                                           # noqa: E402
from models.video.nsg_vd import official_mmd, official_nsg                        # noqa: E402
from models.video.nsg_vd.nsg_vd import SynthScopeNSGVDModel                       # noqa: E402
from models.video.nsg_vd.official_tall import SingleSwinBlockDiscriminator        # noqa: E402

_bench_spec = importlib.util.spec_from_file_location("bench_eval_video", os.path.join(ROOT, "benchmark", "bench_eval_video.py"))
bench = importlib.util.module_from_spec(_bench_spec)
_bench_spec.loader.exec_module(bench)

KEYS = ("nsg_vd", "nsg_vd_pika", "nsg_vd_unbalance")
# A small UNet with the official layout (6 levels, attention at 32/16/8, learned sigma) for fast CPU checks.
SMALL_UNET = "{model_channels:32,num_res_blocks:1,num_head_channels:16}"


@contextlib.contextmanager
def quiet():
    """The vendored / official Swin prints its layout and warns about torch.meshgrid: silence both (as the model does)."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def small_model(prob_scale: float | None = None) -> SynthScopeNSGVDModel:
    overrides = ["model@ModelConfig=video/nsg_vd", f"+ModelConfig.unet_overrides={SMALL_UNET}"]
    if prob_scale is not None:
        overrides.append(f"ModelConfig.prob_scale={prob_scale}")
    model = SynthScopeNSGVDModel(compose_config(overrides)).eval()
    torch.manual_seed(1)
    torch.nn.init.normal_(model.unet.out[-1].weight, std=0.05)    # guided-diffusion zero-inits the output conv
    for p in model.mmd.net.parameters():                          # a random discriminator with spread-out features
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():                                         # trained-like kernel widths (released checkpoints)
        model.mmd.sigma.fill_(1.0035e8)
        model.mmd.sigma0_u.fill_(5.0)
        model.mmd.ep.fill_(1e-8)
    return model


def nsg_ffmpeg() -> str | None:
    """The ffmpeg binary NSG-VD uses ($SYNTHSCOPE_FFMPEG_NSG_VD, else the global choice), or None when there is none."""
    try:
        return spec_ffmpeg_exe(get_spec("nsg_vd"))
    except (FileNotFoundError, RuntimeError, OSError):
        return None


def write_clip(path: str, n: int, size=(176, 144), fps=10.0) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV cannot write mp4v"
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), (i * 9) % 256, np.uint8)
        frame[: size[1] // 2] = rng.integers(0, 256, (size[1] // 2, size[0], 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_transform() -> None:
    print("[A] nsgvd_frame_224 == the official _preprocess_frame_from_path ...")
    tf = get_spec("nsg_vd").transform
    assert list(tf.transform_list) == ["nsgvd_frame_224"]
    for i, size in enumerate([(320, 240), (224, 224), (1280, 720), (176, 144), (512, 896)]):
        rng = np.random.default_rng(i)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frame1.jpg")
            Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB").save(path)
            frame_img = Image.open(path)                                       # official, written out
            frame_img = frame_img.resize((224, 224), Image.Resampling.LANCZOS)
            golden = (torch.from_numpy(np.array(frame_img)).float() / 255.0).permute(2, 0, 1)
            with Image.open(path) as im:
                im.load()
                x = tf(im.copy())
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and torch.equal(x, golden), size
    print("    5 frame sizes bit-exact")


def test_frames() -> None:
    exe = nsg_ffmpeg()
    if exe is None:
        print("[B] frame extraction: skipped (no ffmpeg for nsg_vd)")
        return
    print(f"[B] media.ffmpeg_select_frames == the official process_video ({exe}) ...")
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        write_clip(clip, 43)
        n = media.ffmpeg_frame_count(clip, exe=exe)
        try:
            from moviepy import VideoFileClip                                  # the official get_video_frame_count
            video = VideoFileClip(clip)
            assert n == int(video.fps * video.duration), (n, video.fps, video.duration)
            video.close()
            how = "== moviepy"
        except ImportError:
            how = "moviepy not installed"
        k = max(1, n // 8)
        out = os.path.join(tmp, "official")
        os.makedirs(out)
        cmd = (f"{exe} -i {clip} -vf select='not(mod(n\\,{k}))',setpts=N/FRAME_RATE/TB "
               f"-vframes 8 {out}/frame%d.jpg > /dev/null 2>&1")              # process_video's command line
        subprocess.run(cmd, shell=True, check=False)
        official = sorted(os.path.join(out, f) for f in os.listdir(out))
        ours = media.ffmpeg_select_frames(clip, 8, exe=exe)
        assert len(official) == 8 and len(ours) == 8
        for path, frame in zip(official, ours):
            with Image.open(path) as im:
                assert np.array_equal(np.array(im), np.array(frame)), path
        short = os.path.join(tmp, "short.mp4")
        write_clip(short, 5)
        try:
            media.ffmpeg_select_frames(short, 8, exe=exe)
            raise AssertionError("a 5-frame clip gave 8 frames")
        except media.VideoDecodeError:
            pass
        # I/O problems are OSErrors (not cached as decode failures); an empty file is a decode failure
        for bad, err in [(os.path.join(tmp, "missing.mp4"), FileNotFoundError)]:
            try:
                media.ffmpeg_select_frames(bad, 8, exe=exe)
                raise AssertionError(f"{bad} scored")
            except media.VideoDecodeError:
                raise AssertionError("a missing file is a decode failure")
            except err:
                pass
        locked = os.path.join(tmp, "locked.mp4")
        write_clip(locked, 20)
        os.chmod(locked, 0)
        if not os.access(locked, os.R_OK):                 # (root can read it anyway)
            try:
                media.ffmpeg_frame_count(locked, exe=exe)
                raise AssertionError("an unreadable file was read")
            except media.VideoDecodeError:
                raise AssertionError("an unreadable file is a decode failure")
            except OSError:
                pass
        empty = os.path.join(tmp, "empty.mp4")
        open(empty, "wb").close()
        try:
            media.ffmpeg_select_frames(empty, 8, exe=exe)
            raise AssertionError("an empty file scored")
        except media.VideoDecodeError:
            pass
    print(f"    frame count {n} ({how}), step {k}: 8 identical frames; a 5-frame clip is refused; missing / "
          "unreadable file -> OSError, empty file -> VideoDecodeError")


def test_model() -> None:
    print("[C] SynthScopeNSGVDModel (small random UNet) ...")
    model = small_model()
    torch.manual_seed(0)
    clips = torch.rand(3, 8, 3, 224, 224)
    x = model.nsg(clips)
    assert x.shape == (3, 8, 3, 224, 224) and torch.isfinite(x).all()
    _, raw = official_nsg.score_features(model.unet, clips.reshape(-1, 3, 224, 224), 3)
    assert torch.equal(x[:, 1:-1], raw[:, 1:-1]) and torch.equal(x[:, 0], raw[:, 0] * 100)
    assert torch.equal(x[:, -1], raw[:, -1] * 100)
    model.set_reference(model.nsg(torch.rand(6, 8, 3, 224, 224)))
    s = model.score(clips)
    assert s.shape == (3,) and torch.isfinite(s).all()
    single = torch.cat([model.score(clips[i:i + 1]) for i in range(3)])
    assert torch.allclose(s, single, rtol=0, atol=1e-5), (s, single)
    p = model.fake_probability(s)
    assert torch.allclose(p, torch.sigmoid((s - 1.0) / model.prob_scale))
    assert float(model.fake_probability(torch.tensor(1.0))) == 0.5
    assert not bool(model.is_fake(torch.tensor(1.0))) and bool(model.is_fake(torch.tensor(1.000001)))   # official > 1
    # the released checkpoint layout: a deep_MMD state dict
    with quiet():
        ref = official_mmd.deep_MMD(SingleSwinBlockDiscriminator(num_features=300))
    ref.set_info(-0.2, 1e-8, 1.0035e8, 5e-7, -0.8)
    sd = {k: v.clone() for k, v in ref.state_dict().items()}
    missing, unexpected = model.load_state_dict(remap_nsg_vd(sd), strict=False)
    assert not unexpected and all(k.startswith("unet.") for k in missing), (missing[:3], unexpected[:3])
    assert model.feature_ref is None                          # reference features recomputed for the new weights
    assert all(torch.equal(model.state_dict()["mmd." + k], v) for k, v in sd.items())
    try:
        model.compute_loss((clips, torch.zeros(3)))
        raise AssertionError("compute_loss trained")
    except NotImplementedError:
        pass
    spec = get_spec("nsg_vd")
    with tempfile.TemporaryDirectory() as tmp:
        bad = dict(sd)
        bad.pop("sigma0_u")
        bad["extra.weight"] = torch.zeros(1)
        torch.save(bad, os.path.join(tmp, spec.checkpoint))
        saved = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = type(saved)(tmp)
        try:
            _load_checkpoint(small_model(), spec)
            raise AssertionError("a mismatched checkpoint loaded")
        except RuntimeError as e:
            assert "mismatch" in str(e)
        finally:
            backend_config.CHECKPOINTS_DIR = saved
    print(f"    scores {[round(float(v), 4) for v in s]} (batch == one by one), strict deep_MMD load, mismatch raises")


def _load_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                  # timm's register_model looks the defining module up
    spec.loader.exec_module(module)
    return module


def test_against_official() -> None:
    off = os.environ.get("NSGVD_OFFICIAL_DIR")
    if not off:
        print("[D] official modules: skipped ($NSGVD_OFFICIAL_DIR not set)")
        return
    print(f"[D] vendored code == the official modules ({off}) ...")
    try:
        tall = _load_file("nsgvd_official_tall", os.path.join(off, "models/tall.py"))
        mmd = _load_file("nsgvd_official_mmd_utils", os.path.join(off, "utils/mmd_utils.py"))
        sys.path.append(off)               # data.*, libs.* (no clash with SynthScope's packages)
        from data.feature_dataset.score_feature_dataset import ScoreFeaturesDataset
        from libs.eps_ad.score_sde import sde_lib
        from libs.eps_ad.score_sde.models import utils as mutils
    except ImportError as e:
        print(f"[D] official modules: skipped (cannot import the official code: {e})")
        return
    torch.manual_seed(0)
    with quiet():
        ours = SingleSwinBlockDiscriminator(num_features=300).eval()
        theirs = tall.SingleSwinBlockDiscriminator(num_features=300).eval()
    theirs.load_state_dict(ours.state_dict(), strict=True)
    x = torch.randn(3, 8, 3, 224, 224)
    with torch.no_grad():
        (va, fa), (vb, fb) = ours(x, out_feature=True), theirs(x, out_feature=True)
    assert torch.equal(fa, fb) and torch.equal(va, vb)
    fea, org = torch.randn(9, 300), torch.randn(9, 1000)
    for smooth in (True, False):
        a = official_mmd.MMD_batch2(fea, 6, org, torch.tensor(1e3), torch.tensor(5.0), torch.tensor(1e-3), is_smooth=smooth)
        b = mmd.MMD_batch2(fea, 6, org, torch.tensor(1e3), torch.tensor(5.0), torch.tensor(1e-3), is_smooth=smooth)
        assert torch.equal(a, b)
    model = small_model()
    model._unet_dtype(torch.device("cpu"))
    frames = torch.rand(16, 3, 224, 224)
    t = torch.full((16,), 0.006)
    official_score = mutils.get_score_fn(sde_lib.VPSDE(beta_min=0.1, beta_max=20, N=1000), model.unet, train=False,
                                         continuous=True)
    assert torch.equal(official_nsg.score_fn(model.unet, 2 * frames - 1, t), official_score(2 * frames - 1, t))
    fake_self = types.SimpleNamespace(diffuse_steps=5, resolution_size=224, input_shape=(224, 224),
                                      score_fn=official_score)
    fake_self._get_score_velocity = types.MethodType(ScoreFeaturesDataset._get_score_velocity, fake_self)
    sa, va = ScoreFeaturesDataset._get_score_features(fake_self, frames, 2)
    sb, vb = official_nsg.score_features(model.unet, frames, 2)
    assert torch.equal(sa, sb) and torch.equal(va, vb)
    print("    TALL discriminator, MMD_batch2 (smooth / not), score_fn and _get_score_features: bit-identical")


def test_specs_and_engine() -> None:
    print("[E] specs + engine.predict_video_clip ...")
    for key in KEYS:
        spec = get_spec(key)
        assert spec.modality == "video" and spec.ckpt_kind == "official_nsg_vd" and spec.remap is remap_nsg_vd
        assert spec.num_frames == 8 and spec.video_sampler == "ffmpeg_select_jpeg" and spec.video_prob == "mmd"
        assert spec.overrides == ["model@ModelConfig=video/nsg_vd"] and spec.weights == "official"
        assert spec.aux_files == {"ModelConfig.diffusion_weights": "256x256_diffusion_uncond.pt",
                                  "ModelConfig.reference_path": "nsg_vd_reference_msrvtt_val100.safetensors"}
        assert spec.transform is get_spec("nsg_vd").transform
    note = input_note(get_spec("nsg_vd"))
    assert "8 evenly spaced frames" in note["label"]
    if nsg_ffmpeg() is None:
        print("    specs ok; engine path skipped (no ffmpeg for nsg_vd)")
        return
    engine = Engine(device="cpu")
    spec = get_spec("nsg_vd")
    model = small_model(prob_scale=0.5)
    model.set_reference(model.nsg(torch.rand(4, 8, 3, 224, 224)))
    engine._cache[spec.key] = model
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        write_clip(clip, 43)
        res = engine.predict_video_clip(spec, clip)
        frames = torch.stack([spec.transform(f) for f in media.ffmpeg_select_frames(clip, 8, exe=spec_ffmpeg_exe(spec))])
        assert torch.equal(engine.video_clip(spec, clip), frames)
        s = float(model.score(frames.unsqueeze(0))[0])
        assert res["num_frames"] == 8 and abs(res["logit"] - s) < 1e-6, (res, s)
        assert abs(res["fake_probability"] - 1 / (1 + math.exp(-(s - 1) / 0.5))) < 1e-6
        assert res["verdict"] == ("fake" if s > 1 else "real")
        assert abs(engine.video_raw_score(spec, clip) - s) < 1e-6
        exe = spec_ffmpeg_exe(spec)
        assert res["ffmpeg"] == {"path": exe, "version": media.ffmpeg_version(exe), "required": "4.2",
                                 "match": media.ffmpeg_version(exe).startswith("4.2.")}
        assert ("warning" in res) == (not res["ffmpeg"]["match"])
        # exactly at the threshold: P(fake) 0.5 and, as the official rule (> 1), "real"
        model.score = lambda x: torch.tensor([1.0])
        at = engine.predict_video_clip(spec, clip)
        del model.score
        assert at["fake_probability"] == 0.5 and at["verdict"] == "real" and at["logit"] == 1.0
        # a pin the binary does not meet is reported and warned about; the per-detector variable picks the binary
        pinned = dataclasses.replace(spec, key="nsg_vd_pin_test", ffmpeg_version="0.1")
        engine._cache[pinned.key] = model
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            other = engine.predict_video_clip(pinned, clip)
        assert not other["ffmpeg"]["match"] and "warning" in other
        assert any(issubclass(w.category, RuntimeWarning) and "pinned 0.1" in str(w.message) for w in caught)
        saved_env = os.environ.get("SYNTHSCOPE_FFMPEG_NSG_VD")
        os.environ["SYNTHSCOPE_FFMPEG_NSG_VD"] = os.path.join(tmp, "no_such_ffmpeg")
        try:
            assert engine.ffmpeg_missing(spec)
            os.environ["SYNTHSCOPE_FFMPEG_NSG_VD"] = exe
            assert spec_ffmpeg_exe(spec) == exe and not engine.ffmpeg_missing(spec)
            # other detectors ignore the NSG-VD variable: they keep the global choice (or have none)
            if media.ffmpeg_available():
                assert spec_ffmpeg_exe(get_spec("vina_video")) == media.ffmpeg_exe()
            else:
                assert engine.ffmpeg_missing(get_spec("vina_video"))
        finally:
            if saved_env is None:
                os.environ.pop("SYNTHSCOPE_FFMPEG_NSG_VD", None)
            else:
                os.environ["SYNTHSCOPE_FFMPEG_NSG_VD"] = saved_env
        # bench_eval_video's cache identity covers the extra files and the ffmpeg build
        ck = os.path.join(tmp, "ck")
        os.makedirs(ck)
        for name in [spec.checkpoint, *spec.aux_files.values()]:
            with open(os.path.join(ck, name), "wb") as f:
                f.write(name.encode())
        saved = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = type(saved)(ck)
        try:
            meta = bench.cache_meta(engine, spec)
        finally:
            backend_config.CHECKPOINTS_DIR = saved
        assert set(meta["aux_sha256"]) == set(spec.aux_files.values())
        assert meta["ffmpeg"]["version"] == media.ffmpeg_version(exe) and len(meta["ffmpeg"]["sha256"]) == 64
    missing = dataclasses.replace(spec, aux_files={"ModelConfig.reference_path": "no_such_file.safetensors"})
    assert not engine.is_available(missing)
    print(f"    3 specs ok; score {s:.4f} -> P(fake) {res['fake_probability']:.4f}; s == 1 -> real; ffmpeg "
          f"{res['ffmpeg']['version']} recorded, pin mismatch warned; cache_meta has aux + ffmpeg; note '{note['label']}'")


def test_released_checkpoints() -> None:
    ckpt_dir = os.environ.get("SYNTHSCOPE_CKPT_DIR", "")
    spec0 = get_spec("nsg_vd")
    files = [get_spec(k).checkpoint for k in KEYS] + list(spec0.aux_files.values())
    if not all(os.path.isfile(os.path.join(ckpt_dir, f)) for f in files):
        print(f"[F] released checkpoints: skipped ($SYNTHSCOPE_CKPT_DIR lacks some of {files})")
        return
    print("[F] released checkpoints + diffusion model + reference: strict engine load ...")
    saved = backend_config.CHECKPOINTS_DIR
    backend_config.CHECKPOINTS_DIR = type(saved)(ckpt_dir)
    try:
        engine = Engine(device="cpu")
        for key in KEYS:
            spec = get_spec(key)
            model = engine._get_model(spec)
            status = engine._status[key]
            assert "(missing=0, unexpected=0)" in status, status
            released = torch.load(os.path.join(ckpt_dir, spec.checkpoint), map_location="cpu", weights_only=True)
            sd = model.state_dict()
            assert {k for k in sd if not k.startswith("unet.")} == {"mmd." + k for k in released}
            assert all(torch.equal(sd["mmd." + k], v) for k, v in released.items())
            assert model.ref_data.shape == (100, 8, 3, 224, 224)
            print(f"    {key}: {status}")
            engine._cache.clear()
    finally:
        backend_config.CHECKPOINTS_DIR = saved


if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    test_transform()
    test_frames()
    test_model()
    test_against_official()
    test_specs_and_engine()
    test_released_checkpoints()
    print("OK")
