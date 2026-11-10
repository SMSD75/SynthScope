"""Native-resolution specs are scored one image at a time (CPU, no downloads).

* ``ModelSpec.eval_batch_size`` is 1 for every ``native_res`` spec (LTF, B-Free, SPAI, PROBE, ...) and the requested size
  otherwise,
* ``benchmark/evaluate.py`` ``score`` on images of different sizes: a ``native_res`` spec with a batch size of 8 scores them
  (each forward pass sees one image), a fixed-size spec stacks them in batches of 8.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from benchmark import evaluate                                  # noqa: E402
from backend.registry import MODELS, get_spec                   # noqa: E402

SIZES = [(40, 30), (64, 64), (33, 90), (50, 20), (70, 70)]


class _Engine:
    """Stands in for the inference engine: records the batch sizes it is called with."""
    device = "cpu"

    def __init__(self):
        self.batch_sizes = []

    def _get_model(self, spec):
        return None

    def _fake_prob(self, model, spec, x):
        self.batch_sizes.append(x.shape[0])
        return x.flatten(1).mean(1)


class _Detector:
    def __init__(self, spec):
        self.spec = spec
        self._engine = _Engine()

    def _ensure_weights(self):
        pass


def _score(spec, paths, fixed_size: bool):
    # fixed-size transform: stackable; native transform: keeps the image size (variable-size tensors)
    def transform(img):
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        return t[:, :20, :20] if fixed_size else t
    spec = dataclasses.replace(spec, transform=transform)
    det = _Detector(spec)
    probs = evaluate.score(det, paths, 8)
    return probs, det._engine.batch_sizes


def test_batch_size_rule() -> None:
    native = [k for k, s in MODELS.items() if s.native_res]
    assert {"ltf", "bfree", "spai", "probe", "probe_resnet"} <= set(native), native
    for key, spec in MODELS.items():
        assert spec.eval_batch_size(32) == (1 if spec.native_res else 32), key
    print(f"    eval_batch_size: 1 for the {len(native)} native_res specs, unchanged for the other {len(MODELS) - len(native)}")


def test_score_variable_sizes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (w, h) in enumerate(SIZES):
            path = os.path.join(tmp, f"{i}.png")
            Image.new("RGB", (w, h), (10 * i, 20, 30)).save(path)
            paths.append(path)
        paths = [Path(p) for p in paths]
        native = get_spec("probe")
        assert native.native_res
        probs, batches = _score(native, paths, fixed_size=False)
        assert batches == [1] * len(SIZES) and probs.shape == (len(SIZES),), batches
        fixed = get_spec("clip")
        assert not fixed.native_res
        probs, batches = _score(fixed, paths, fixed_size=True)
        assert batches == [5] and probs.shape == (len(SIZES),), batches
    print("    evaluate.score: variable-size images scored one at a time for probe; clip stacks them in one batch of 5")


if __name__ == "__main__":
    test_batch_size_rule()
    test_score_variable_sizes()
    print("native-res batch test passed")
