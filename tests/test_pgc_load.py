"""Regression test for the served PGC checkpoint load (the official ``rgb_stream.`` key prefix was dropped silently).

Run as a script (``python tests/test_pgc_load.py``), CPU only.

Without weights (always runs):
* ``remap_pgc`` strips the official ``rgb_stream.`` wrapper prefix and leaves SynthScope-trained keys (``backbone.*``) alone;
* the pgc registry spec is a strict official kind with that remap;
* the engine's loader raises on a mismatch for strict kinds and logs a WARNING (no longer silent) for ``synthscope``
  kinds, and stays quiet when the keys match or the model declares them in ``pretrained_prefixes``.

With weights (skipped cleanly when ``$SYNTHSCOPE_CKPT_DIR/pgc.pth`` or the DINOv2-L backbone is not available):
* the pgc checkpoint loads through ``engine._load_checkpoint`` with 0 unexpected / 0 real missing, the whole model state
  matches the checkpoint, and no warning is logged;
* parity on a fixed list of images: probabilities equal those of the UNMODIFIED official repo (stored reference
  JSON file named by the environment variable ``SYNTHSCOPE_PGC_PARITY_REF``: a dict with a ``paths`` list of image files and
  the matching official-repo probabilities, produced by running the official PGC repo on those images; the step is skipped when
  the variable is unset, the file is missing, or the images are not on this machine) with max |dp| < 1e-6.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmark")]

from interface.backend import config, engine  # noqa: E402
from interface.backend.hf_compat import translate_hf_keys  # noqa: E402
from interface.backend.registry import get_spec, remap_pgc  # noqa: E402

_PARITY_ENV = os.environ.get("SYNTHSCOPE_PGC_PARITY_REF")
PARITY_REF = Path(_PARITY_ENV) if _PARITY_ENV else None
PARITY_TOL = 1e-6


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _load_with_capture(model, spec):
    handler = _Capture()
    engine.logger.addHandler(handler)
    try:
        status = engine._load_checkpoint(model, spec)
    finally:
        engine.logger.removeHandler(handler)
    return status, handler.records


class _Tiny(nn.Module):
    def __init__(self, prefixes=()):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        if prefixes:
            self.pretrained_prefixes = prefixes


def test_remap() -> None:
    print("[A] remap_pgc ...")
    out = remap_pgc({"rgb_stream.backbone.a": 1, "rgb_stream.backbone.lora.lora_B": 2, "pgcm.w": 3, "classifier.bias": 4,
                     "backbone.b": 5})
    assert out == {"backbone.a": 1, "backbone.lora.lora_B": 2, "pgcm.w": 3, "classifier.bias": 4, "backbone.b": 5}, out
    trained = {"backbone.x": 1, "classifier.weight": 2}
    assert remap_pgc(trained) == trained, "SynthScope-trained keys must pass through unchanged"
    spec = get_spec("pgc")
    assert spec.ckpt_kind == "official_pgc" and spec.remap is remap_pgc, (spec.ckpt_kind, spec.remap)
    print("    OK")


def test_loader_is_never_silent() -> None:
    print("[B] engine loader: strict kinds raise, other kinds warn on a mismatch ...")
    base = get_spec("pgc")
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.pth"
        bad = Path(tmp) / "bad.pth"
        torch.save({"model": _Tiny().state_dict()}, good)
        torch.save({"model": {"wrap." + k: v for k, v in _Tiny().state_dict().items()}}, bad)

        def spec(path, kind):
            return dataclasses.replace(base, key="tiny", checkpoint=str(path), ckpt_kind=kind, remap=None)

        # matching keys: quiet for both kinds
        for kind in ("synthscope", "official_tiny"):
            status, recs = _load_with_capture(_Tiny(), spec(good, kind))
            assert not recs and "missing=0, unexpected=0" in status, (kind, status, [r.getMessage() for r in recs])
        # wrong prefix: strict kind raises, non-strict kind warns (and names the dropped keys)
        try:
            engine._load_checkpoint(_Tiny(), spec(bad, "official_tiny"))
            raise AssertionError("strict kind must raise on a mismatch")
        except RuntimeError:
            pass
        status, recs = _load_with_capture(_Tiny(), spec(bad, "synthscope"))
        assert len(recs) == 1 and recs[0].levelno == logging.WARNING, [r.getMessage() for r in recs]
        assert "wrap.backbone.weight" in recs[0].getMessage() and "unexpected=4" in status, (status, recs[0].getMessage())
        # keys the model declares as rebuilt in __init__ are expected: no warning
        torch.save({"model": {k: v for k, v in _Tiny().state_dict().items() if k.startswith("head.")}}, good)
        status, recs = _load_with_capture(_Tiny(prefixes=("backbone.",)), spec(good, "synthscope"))
        assert not recs and "missing=0" in status, (status, [r.getMessage() for r in recs])
        status, recs = _load_with_capture(_Tiny(), spec(good, "synthscope"))
        assert len(recs) == 1 and "missing=2" in status, (status, [r.getMessage() for r in recs])
    print("    OK")


def _have_weights() -> str | None:
    ckpt = config.CHECKPOINTS_DIR / get_spec("pgc").checkpoint
    if not ckpt.exists():
        return f"no checkpoint at {ckpt}"
    return None


def test_official_checkpoint_loads_exactly() -> object:
    print("[C] official pgc.pth loads exactly through the engine ...")
    why = _have_weights()
    if why:
        print(f"    SKIP ({why})")
        return None
    from config_utils import compose_config

    spec = get_spec("pgc")
    try:
        model = engine._build_model(compose_config(spec.overrides))
    except Exception as e:  # noqa: BLE001  (backbone weights not downloadable here)
        print(f"    SKIP (cannot build the model: {type(e).__name__}: {e})")
        return None
    status, recs = _load_with_capture(model, spec)
    print("    status:", status)
    assert not recs, [r.getMessage() for r in recs]
    assert "missing=0, unexpected=0" in status, status
    raw = torch.load(str(config.CHECKPOINTS_DIR / spec.checkpoint), map_location="cpu", weights_only=False)
    msd = model.state_dict()
    # the checkpoint holds transformers 5.5.4 DINOv2 names; the loader maps them onto the installed release's names
    sd, _ = translate_hf_keys(spec.remap(raw["model"] if "model" in raw else raw), msd)
    assert set(sd) <= set(msd), sorted(set(sd) - set(msd))[:5]
    assert all(torch.equal(msd[k], v) for k, v in sd.items()), "a loaded tensor differs from the checkpoint"
    lora_b = max(v.abs().max().item() for k, v in msd.items() if k.endswith("lora_B"))
    assert lora_b > 0, "LoRA B is still zero: the trained LoRA was not loaded"
    print(f"    {len(sd)}/{len(msd)} model tensors from the checkpoint (rest = constant buffers), max|lora_B| = {lora_b:.5f}")
    return model


def test_parity(model) -> None:
    print("[D] parity with the unmodified official repo ...")
    if model is None:
        print("    SKIP (no model)")
        return
    if PARITY_REF is None or not PARITY_REF.exists():
        print("    SKIP (set SYNTHSCOPE_PGC_PARITY_REF to the official-repo reference JSON)")
        return
    ref = json.loads(PARITY_REF.read_text())
    paths = [Path(p) for p in ref["paths"]]
    if not all(p.exists() for p in paths):
        print("    SKIP (reference images not available on this machine)")
        return
    import synthscope
    from benchmark.bench_eval import score

    torch.set_num_threads(2)
    det = synthscope.load("pgc", device="cpu")
    spec = det.spec
    model = model.float().eval()
    p, kept = score(det, model, spec, paths, 8, transform=spec.transform)
    assert kept == list(range(len(paths))), kept
    diff = (p - torch.tensor(ref["p_official_repo"])).abs()
    print(f"    {len(paths)} images: max |dp| = {diff.max().item():.3e}, mean |dp| = {diff.mean().item():.3e}, "
          f"decision flips at 0.5 = {int(((p > 0.5) != (torch.tensor(ref['p_official_repo']) > 0.5)).sum())}")
    assert diff.max().item() < PARITY_TOL, f"engine differs from the official repo: max |dp| = {diff.max().item()}"


def main() -> None:
    test_remap()
    test_loader_is_never_silent()
    model = test_official_checkpoint_loads_exactly()
    test_parity(model)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
