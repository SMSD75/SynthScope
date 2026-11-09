"""Tests for the transformers parameter-name compatibility of checkpoint loading (interface/backend/hf_compat.py).

Run as a script (``python tests/test_hf_key_compat.py``), CPU only.

Always runs:
* the rename table maps an older-release state dict onto a newer-release model and the reverse, only when exact;
* anything inexact (shape mismatch, collision, unrelated unknown keys, an already-matching dict) is left untouched;
* ``engine._load_checkpoint`` loads an official-kind checkpoint in either naming, and still raises on a real mismatch;
* the table matches the installed ``transformers``: a tiny ``Dinov2Model`` built from its config, with its keys renamed
  to the other release's names, translates back to exactly the model's keys and loads strictly.

Skipped cleanly when ``$SYNTHSCOPE_CKPT_DIR/pgc.pth`` or the DINOv2-L config is not available:
* every ``backbone.*`` tensor of the real PGC checkpoint resolves onto the installed DINOv2-L (built on the meta device).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path

import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoModel, Dinov2Config, Dinov2Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmark")]

from interface.backend import config, engine  # noqa: E402
from interface.backend.hf_compat import HF_RENAMES, translate_hf_keys  # noqa: E402
from interface.backend.registry import get_spec, remap_pgc  # noqa: E402

OLD_TO_NEW = {
    "attention.attention.query": "attention.q_proj",
    "attention.attention.key": "attention.k_proj",
    "attention.attention.value": "attention.v_proj",
    "attention.output.dense": "attention.o_proj",
}


def _layer(prefix: str, names, dim: int = 4) -> dict:
    """The attention tensors of one DINOv2 layer under ``prefix`` with the given module names (the older release's: the keys of OLD_TO_NEW)."""
    out = {}
    for module in names:
        out[f"{prefix}.{module}.weight"] = torch.randn(dim, dim)
        out[f"{prefix}.{module}.bias"] = torch.randn(dim)
    out[f"{prefix}.layernorm.weight"] = torch.randn(dim)
    return out


def _with_other_names(sd: dict, mapping: dict) -> dict:
    """Rename ``sd`` keys by ``mapping`` (module name fragments), for the test ground truth."""
    out = {}
    for k, v in sd.items():
        for a, b in mapping.items():
            k = k.replace(a, b)
        out[k] = v
    return out


def test_table() -> None:
    print("[A] rename table, both directions ...")
    assert dict((a.strip("."), b.strip(".")) for a, b in HF_RENAMES) == OLD_TO_NEW, HF_RENAMES
    prefix = "backbone.encoder.layer.0"
    old = _layer(prefix, OLD_TO_NEW)
    new = _with_other_names(old, OLD_TO_NEW)
    assert set(old) != set(new) and len(old) == len(new) == 9
    # older checkpoint -> newer model
    out, note = translate_hf_keys(old, new)
    assert set(out) == set(new) and "older -> newer" in note and "8 key" in note, (sorted(out), note)
    for k in old:
        assert out[_with_other_names({k: 0}, OLD_TO_NEW).popitem()[0]] is old[k], k
    # newer checkpoint -> older model
    out, note = translate_hf_keys(new, old)
    assert set(out) == set(old) and "newer -> older" in note, (sorted(out), note)
    for k in old:
        assert torch.equal(out[k], old[k]), k
    # already matching: untouched, no note
    for sd in (old, new):
        out, note = translate_hf_keys(sd, sd)
        assert out is sd and note == "", note
    # LoRA-wrapped projection (PGC with lora_targets on q/v) renames the same way
    wrapped = {f"{prefix}.attention.attention.query.original_layer.weight": torch.zeros(4, 4)}
    out, _ = translate_hf_keys(wrapped, {f"{prefix}.attention.q_proj.original_layer.weight": torch.zeros(4, 4)})
    assert list(out) == [f"{prefix}.attention.q_proj.original_layer.weight"], list(out)
    print("    OK")


def test_inexact_is_untouched() -> None:
    print("[B] inexact renames are left for the strict check ...")
    prefix = "backbone.encoder.layer.0"
    old = _layer(prefix, OLD_TO_NEW)
    new = _with_other_names(old, OLD_TO_NEW)
    # shape mismatch on one tensor
    bad_shape = dict(new)
    bad_shape[f"{prefix}.attention.q_proj.weight"] = torch.zeros(5, 4)
    out, note = translate_hf_keys(old, bad_shape)
    assert out is old and note == "", note
    # collision: the state dict already has the target name
    clash = dict(old)
    clash[f"{prefix}.attention.q_proj.weight"] = torch.zeros(4, 4)
    out, note = translate_hf_keys(clash, dict(new, **{f"{prefix}.attention.q_proj.weight": torch.zeros(4, 4)}))
    assert out is clash and note == "", note
    # the rename lands on no model key (a different layout)
    out, note = translate_hf_keys(old, {"other.weight": torch.zeros(1)})
    assert out is old and note == "", note
    # unrelated unknown keys are not touched
    out, note = translate_hf_keys({"x.y": torch.zeros(1)}, {"a.b": torch.zeros(1)})
    assert set(out) == {"x.y"} and note == "", (out, note)
    print("    OK")


class _TinyAttn(nn.Module):
    """Named like the transformers <= 5.5.4 DINOv2 attention (``attention.attention.{query,key,value}``)."""

    def __init__(self, newer: bool):
        super().__init__()
        self.attention = nn.Module()
        names = ("q_proj", "k_proj", "v_proj", "o_proj") if newer else ("query", "key", "value")
        if newer:
            for n in names:
                setattr(self.attention, n, nn.Linear(4, 4))
        else:
            self.attention.attention = nn.Module()
            for n in names:
                setattr(self.attention.attention, n, nn.Linear(4, 4))
            self.attention.output = nn.Module()
            self.attention.output.dense = nn.Linear(4, 4)


class _Tiny(nn.Module):
    def __init__(self, newer: bool):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer = _TinyAttn(newer)
        self.head = nn.Linear(4, 1)


def test_engine_loader() -> None:
    print("[C] engine._load_checkpoint loads either naming (official kind, strict) ...")
    base = get_spec("pgc")
    with tempfile.TemporaryDirectory() as tmp:
        for ckpt_newer, model_newer in ((False, True), (True, False), (False, False), (True, True)):
            path = Path(tmp) / f"c_{int(ckpt_newer)}_{int(model_newer)}.pth"
            src = _Tiny(ckpt_newer)
            torch.save({"model": src.state_dict()}, path)
            spec = dataclasses.replace(base, key="tiny", checkpoint=str(path), ckpt_kind="official_tiny", remap=remap_pgc)
            dst = _Tiny(model_newer)
            status = engine._load_checkpoint(dst, spec)
            translated = ckpt_newer != model_newer
            assert "missing=0, unexpected=0" in status and ("translated" in status) == translated, status
            # the loaded tensors are the checkpoint's, matched by position in the layer
            s, d = list(src.state_dict().values()), list(dst.state_dict().values())
            assert all(torch.equal(a, b) for a, b in zip(sorted(s, key=lambda t: t.sum().item()),
                                                          sorted(d, key=lambda t: t.sum().item()))), status
        # a genuine mismatch is still raised by the strict check
        bad = Path(tmp) / "bad.pth"
        torch.save({"model": {"wrap." + k: v for k, v in _Tiny(False).state_dict().items()}}, bad)
        spec = dataclasses.replace(base, key="tiny", checkpoint=str(bad), ckpt_kind="official_tiny", remap=None)
        try:
            engine._load_checkpoint(_Tiny(True), spec)
            raise AssertionError("a real mismatch must still raise")
        except RuntimeError:
            pass
    print("    OK")


def test_table_matches_installed_transformers() -> None:
    print("[D] table matches the installed transformers Dinov2Model ...")

    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=28,
                       patch_size=14)
    model = Dinov2Model(cfg)
    msd = model.state_dict()
    uses_old = any(".attention.attention.query." in k for k in msd)
    uses_new = any(".attention.q_proj." in k for k in msd)
    assert uses_old != uses_new, "Dinov2Model uses neither (or both) known attention namings: extend HF_RENAMES"
    other = _with_other_names(msd, OLD_TO_NEW if uses_old else {b: a for a, b in OLD_TO_NEW.items()})
    assert set(other) != set(msd)
    out, note = translate_hf_keys(other, msd)
    assert set(out) == set(msd) and note, note
    assert all(torch.equal(out[k], msd[k]) for k in msd)
    model.load_state_dict(out, strict=True)
    print(f"    OK (transformers {transformers.__version__} names its DINOv2 attention "
          f"{'attention.attention.query/key/value + output.dense' if uses_old else 'q/k/v/o_proj'}; {note})")


def test_real_pgc_checkpoint() -> None:
    print("[E] real pgc.pth backbone keys resolve onto the installed DINOv2-L ...")
    ckpt = config.CHECKPOINTS_DIR / get_spec("pgc").checkpoint
    if not ckpt.exists():
        print(f"    SKIP (no checkpoint at {ckpt})")
        return
    try:
        with torch.device("meta"):
            backbone = AutoModel.from_config(AutoConfig.from_pretrained("facebook/dinov2-large"))
    except Exception as e:  # noqa: BLE001  (config not cached and not downloadable)
        print(f"    SKIP (cannot build DINOv2-L: {type(e).__name__})")
        return
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = remap_pgc(raw["model"] if "model" in raw else raw)
    model_keys = {"backbone." + k: v for k, v in backbone.state_dict().items()}
    out, note = translate_hf_keys(sd, model_keys)
    bb_unknown = [k for k in out if k.startswith("backbone.") and ".lora" not in k and ".original_layer" not in k
                  and k not in model_keys]
    # the LoRA-wrapped fc1/fc2 are not in the plain backbone; every other backbone tensor must have been matched
    assert not bb_unknown, bb_unknown[:5]
    print(f"    OK ({sum(k.startswith('backbone.') for k in out)} backbone tensors; {note or 'no rename needed'})")


def main() -> None:
    test_table()
    test_inexact_is_untouched()
    test_engine_loader()
    test_table_matches_installed_transformers()
    test_real_pgc_checkpoint()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
