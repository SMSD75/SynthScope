"""Checks ``bench_eval --degrade`` and ``--results-log`` (CPU only, no downloads, no model weights).

* every degradation spec of the field-report slice (jpeg95/75/50, down0.5/0.25, blur1/2, social, social2) on synthetic
  images: RGB out, deterministic (two calls give the same pixels), size preserved (jpeg, down, blur) or long side 1080
  (social on a larger image; a smaller one keeps its size), JPEG file size monotone in quality, 4:2:0 chroma subsampling,
  blur / downscale smooth the image, social2 differs from social,
* the source-format hint: "JPEG" after jpeg / social / social2, the file's own format after down / blur,
* malformed specs raise ValueError and ``main`` rejects them with an argparse error (exit 2),
* ``score`` applies the degradation to reals and fakes, before ``--jpeg-fakes`` (fakes only),
* ``main`` with a stubbed detector: without the flags the output JSON and the results-log line have exactly the old keys
  and go to the default log; with them ``degrade`` is added, and the line goes only to ``--results-log``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

import bench_eval                                                    # noqa: E402
from bench_eval import SRC_FORMAT_KEY, parse_degrade                  # noqa: E402

SPECS = ["jpeg95", "jpeg75", "jpeg50", "down0.5", "down0.25", "blur1", "blur2", "social", "social2"]
JPEG_SPECS = {"jpeg95", "jpeg75", "jpeg50", "social", "social2"}
OLD_RESULT_KEYS = ["tag", "model", "ckpt", "model_config", "benchmark", "data_root", "jpeg_fakes", "n_generators", "means",
                   "per_generator", "date", "synthscope_version"]
OLD_LOG_KEYS = ["date", "tag", "model", "ckpt", "benchmark", "data_root", "n_generators", "means"]


def synthetic(size: tuple[int, int], fmt: str = "PNG", seed: int = 0) -> Image.Image:
    """A noisy RGB image carrying ``fmt`` as its source-format hint (what ``bench_eval._open_rgb`` records)."""
    rng = np.random.RandomState(seed)
    image = Image.fromarray(rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")
    image.info[SRC_FORMAT_KEY] = fmt
    return image


def pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image)


def jpeg_bytes(image: Image.Image, q: int) -> int:
    buf = io.BytesIO()
    image.save(buf, format="jpeg", quality=q)
    return len(buf.getvalue())


def roughness(image: Image.Image) -> float:
    """Mean absolute difference between horizontally adjacent pixels (high for noise, low once smoothed)."""
    a = pixels(image).astype(np.float32)
    return float(np.abs(np.diff(a, axis=1)).mean())


def check_degradations() -> None:
    image = synthetic((300, 200))
    for spec in SPECS:
        fn = parse_degrade(spec)
        a, b = fn(image), fn(image)
        assert a.mode == "RGB", spec
        assert a.size == image.size, (spec, a.size)      # 300x200 is below the social long side
        assert np.array_equal(pixels(a), pixels(b)), f"{spec} is not deterministic"
        assert not np.array_equal(pixels(a), pixels(image)), f"{spec} changed nothing"
        want = "JPEG" if spec in JPEG_SPECS else "PNG"
        assert a.info.get(SRC_FORMAT_KEY) == want, (spec, a.info.get(SRC_FORMAT_KEY))
    assert image.info[SRC_FORMAT_KEY] == "PNG" and image.size == (300, 200)      # the input is not modified
    # a JPEG source stays "JPEG" after down / blur too
    for spec in ("down0.5", "blur1"):
        assert parse_degrade(spec)(synthetic((64, 48), "JPEG")).info[SRC_FORMAT_KEY] == "JPEG"

    # JPEG: lower quality -> smaller file and a larger error against the original, default 4:2:0 chroma subsampling
    errors = [float(np.abs(pixels(parse_degrade(f"jpeg{q}")(image)).astype(np.float32) - pixels(image)).mean())
              for q in (95, 75, 50)]
    assert errors[0] < errors[1] < errors[2], errors
    raw = [jpeg_bytes(image, q) for q in (95, 75, 50)]
    assert raw[0] > raw[1] > raw[2], raw
    buf = io.BytesIO()
    image.save(buf, format="jpeg", quality=75)
    assert JpegImagePlugin.get_sampling(Image.open(buf)) == 2                    # 4:2:0
    ref = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    assert np.array_equal(pixels(parse_degrade("jpeg75")(image)), pixels(ref))   # = a plain PIL q75 round trip

    # smoothing degradations
    r0 = roughness(image)
    assert roughness(parse_degrade("blur1")(image)) < r0
    assert roughness(parse_degrade("blur2")(image)) < roughness(parse_degrade("blur1")(image))
    assert roughness(parse_degrade("down0.5")(image)) < r0
    assert roughness(parse_degrade("down0.25")(image)) < roughness(parse_degrade("down0.5")(image))
    assert parse_degrade("down0.5")(synthetic((301, 199))).size == (301, 199)    # odd sizes come back to the original size

    # social: long side 1080 for larger images (landscape and portrait), untouched size for smaller ones
    social = parse_degrade("social")
    assert social(synthetic((2000, 1500))).size == (1080, 810)
    assert social(synthetic((1500, 2000))).size == (810, 1080)
    assert social(synthetic((1081, 1081))).size == (1080, 1080)
    assert social(synthetic((1080, 600))).size == (1080, 600)
    assert parse_degrade("social2")(synthetic((2000, 1500))).size == (1080, 810)
    assert not np.array_equal(pixels(parse_degrade("social2")(image)), pixels(social(image)))
    buf = io.BytesIO()
    social(synthetic((2000, 1500))).save(buf, format="png")
    assert Image.open(io.BytesIO(buf.getvalue())).size == (1080, 810)


def check_bad_specs() -> None:
    for spec in ("", "jpeg", "jpeg0", "jpeg101", "down", "down1", "down0", "down1.5", "blur", "blur0", "social3", "Social",
                 "blur-1", "jpeg75,down0.5", "rot90"):
        try:
            parse_degrade(spec)
        except ValueError:
            continue
        raise AssertionError(f"{spec!r} should be rejected")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            bench_eval.main(["-m", "stub", "--benchmark", "pool", "--data", "/nonexistent", "--tag", "t", "--degrade", "rot90"])
        except SystemExit as e:
            assert e.code == 2, e.code
        else:
            raise AssertionError("main accepted a bad --degrade")
    assert "--degrade" in err.getvalue()


class StubEngine:
    device = torch.device("cpu")

    @staticmethod
    def _fake_prob(model, spec, x):
        return x.flatten(1).mean(1)                       # a deterministic score per image


class StubDetector:
    def __init__(self):
        self.spec = types.SimpleNamespace(transform=lambda im: torch.from_numpy(pixels(im.resize((32, 32))).copy()).float().permute(2, 0, 1) / 255)
        self._engine = types.SimpleNamespace(device=torch.device("cpu"), _fake_prob=StubEngine._fake_prob,
                                             _get_model=lambda spec: None)

    def _ensure_weights(self):
        pass


def make_pool(root: Path) -> None:
    for name, seed in (("real", 0), ("fake", 1)):
        (root / name).mkdir(parents=True)
        for i in range(4):
            synthetic((70 + 5 * i, 50), seed=seed * 100 + i).save(root / name / f"{i}.png")


def check_score_order() -> None:
    """``--degrade`` hits every image first, ``--jpeg-fakes`` the fakes after it; both before the detector's transform."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pool"
        make_pool(root)
        reals, fakes = sorted((root / "real").glob("*.png")), sorted((root / "fake").glob("*.png"))
        paths = reals + fakes
        seen = []

        class Det(StubDetector):
            def __init__(self):
                super().__init__()
                self.spec.transform = lambda im: (seen.append(im.copy()), torch.zeros(3, 8, 8))[1]

        det = Det()
        spec = det.spec
        bench_eval.score(det, None, spec, paths, 4)
        assert all(im.info[SRC_FORMAT_KEY] == "PNG" for im in seen)                      # no flag: untouched
        seen.clear()
        bench_eval.score(det, None, spec, paths, 4, degrade=parse_degrade("blur1"))
        assert len(seen) == 8 and all(im.info[SRC_FORMAT_KEY] == "PNG" for im in seen)   # reals + fakes, hint kept
        seen.clear()
        bench_eval.score(det, None, spec, paths, 4, degrade=parse_degrade("blur1"), jpeg_q=60, jpeg_from=len(reals))
        assert [im.info[SRC_FORMAT_KEY] for im in seen] == ["PNG"] * 4 + ["JPEG"] * 4   # fakes JPEG-ed on top
        expect_fake = bench_eval._jpeg(parse_degrade("blur1")(bench_eval._open_rgb(fakes[0])), 60)
        assert np.array_equal(pixels(seen[4]), pixels(expect_fake))                      # degrade first, then jpeg-fakes
        seen.clear()
        bench_eval.score(det, None, spec, paths, 4, degrade=parse_degrade("jpeg50"))
        assert all(im.info[SRC_FORMAT_KEY] == "JPEG" for im in seen)                     # reals too


def run_main(tmp: Path, name: str, extra: list[str]) -> tuple[dict, Path]:
    out = tmp / f"{name}.json"
    bench_eval.main(["-m", "stub", "--benchmark", "pool", "--data", str(tmp / "pool"), "--tag", name, "--out", str(out)]
                    + extra)
    return json.loads(out.read_text()), out


def check_main() -> None:
    real_load, real_log = bench_eval.synthscope, bench_eval.RESULTS_LOG
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        make_pool(tmp / "pool")
        default_log, custom_log = tmp / "default" / "MASTER.jsonl", tmp / "custom" / "mx.jsonl"
        bench_eval.synthscope = types.SimpleNamespace(load=lambda model, device=None, **kw: StubDetector(), __version__="stub")
        bench_eval.RESULTS_LOG = default_log
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                plain, _ = run_main(tmp, "plain", [])
                assert list(plain) == OLD_RESULT_KEYS, list(plain)                      # nothing added without the flags
                line = json.loads(default_log.read_text().splitlines()[0])
                assert list(line) == OLD_LOG_KEYS, list(line)
                assert not custom_log.exists()

                deg, _ = run_main(tmp, "deg", ["--degrade", "down0.5", "--results-log", str(custom_log)])
                assert deg["degrade"] == "down0.5"
                assert set(deg) == set(OLD_RESULT_KEYS) | {"degrade"}, set(deg)
                assert len(default_log.read_text().splitlines()) == 1                    # the default log is not touched
                line = json.loads(custom_log.read_text().splitlines()[0])
                assert line["tag"] == "deg" and line["degrade"] == "down0.5"
                assert set(line) == set(OLD_LOG_KEYS) | {"degrade"}, set(line)

                only_log, _ = run_main(tmp, "only_log", ["--results-log", str(custom_log)])
                assert list(only_log) == OLD_RESULT_KEYS                                 # --results-log alone adds no key
                assert len(custom_log.read_text().splitlines()) == 2 and len(default_log.read_text().splitlines()) == 1
                assert list(json.loads(custom_log.read_text().splitlines()[1])) == OLD_LOG_KEYS
        finally:
            bench_eval.synthscope, bench_eval.RESULTS_LOG = real_load, real_log


if __name__ == "__main__":
    check_degradations()
    check_bad_specs()
    check_score_order()
    check_main()
    print("bench_eval --degrade / --results-log: all checks passed")
