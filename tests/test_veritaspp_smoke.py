"""Smoke test for the shared VLM detector kind and Veritas++ (``veritaspp``); CPU, no downloads.

Checks:
[A] the verdict parser = the authors' Veritas v1 ``get_metrics`` answer rule ("real" before "fake", unparsed is None);
    with the official clone ($VERITAS_OFFICIAL_DIR, default SynthScope_shared/vlm_ref/Veritas) the official function
    itself is run on every case (skipped otherwise);
[B] the prompt: the system / user prompts equal ``SYSTEM_PROMPT`` / ``USER_PROMPT`` of the official client (with
    $VERITASPP_OFFICIAL_DIR, default SynthScope_shared/vlm_ref/VeritasPP; skipped otherwise); ``build_prompt`` sends
    [system, user: [image, text]] with ``add_generation_prompt`` and ``enable_thinking=False``; with the released
    processor files ($SYNTHSCOPE_CKPT_DIR/veritaspp) the rendered request equals the gate check's text;
[C] the score math on a stub tokenizer: verdict-token pairs (same spelling, counterpart must be one token), the
    marker search, the verdict step (first verdict token after "<answer>"), logit = l(fake) - l(real) of that pair,
    P(fake) = sigmoid(logit); the "after_marker" fallback and the "none" case (P 0.5, logit 0); the LM-head hook keeps
    the last position's logits at the verdict ids of every forward;
[D] the registry spec (vlm kind, official folder, greedy + official decoding, no pixel cap, one image at a time), the
    input note, the zoo block list, ``_fake_prob`` refusing tensors, the download hint when the folder is missing;
[E] strict loading: ``check_loading_info`` raises on any missing / unexpected / mismatched key; a tiny random
    Qwen3-VL saved with transformers loads through ``SynthScopeVLM`` and raises once a tensor is removed or added;
    with the released processor files: the tiny model scores an image end to end (engine path), the hooked logits
    equal ``generate(output_logits=True)`` exactly and the tokens equal a direct ``generate``;
[F] with the released weights and a GPU: the engine loads all 750 tensors strictly and scores one image with a parsed
    verdict whose sign agrees with the logit (skipped otherwise).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import json
import math
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from interface.backend import config as backend_config                                 # noqa: E402
from interface.backend.engine import Engine                                              # noqa: E402
from interface.backend.registry import get_spec, input_note                              # noqa: E402
from models.image.veritaspp.veritaspp import SYSTEM_PROMPT, USER_PROMPT, VERITASPP   # noqa: E402
from models.vlm.parsers import parse_veritas_v1                                  # noqa: E402
from models.vlm.vlm import (LogitRecorder, SynthScopeVLM, build_prompt, check_loading_info,   # noqa: E402
                            default_pair, locate_verdict, marker_end, verdict_pair_map, verdict_score)
from synthscope import hub                                                       # noqa: E402
from synthscope.api import Detector                                              # noqa: E402

SHARED = Path(ROOT).parent if Path(ROOT).name == "vlm_wt" else Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
VERITAS_DIR = Path(os.environ.get("VERITAS_OFFICIAL_DIR", SHARED / "vlm_ref" / "Veritas"))
VERITASPP_DIR = Path(os.environ.get("VERITASPP_OFFICIAL_DIR", SHARED / "vlm_ref" / "VeritasPP"))
WEIGHTS = backend_config.CHECKPOINTS_DIR / "veritaspp"
PROCESSOR_FILES = ("chat_template.jinja", "preprocessor_config.json", "processor_config.json", "tokenizer.json",
                   "tokenizer_config.json", "generation_config.json")
# The request the 2026-11-16 gate check sent (vlm_gate/out/veritaspp_score.json, "prompt_text").
GATE_PROMPT = ("<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n"
               "<|vision_start|><|image_pad|><|vision_end|>Is this image real or fake?<|im_end|>\n<|im_start|>assistant\n")
PARSER_CASES = {
    "reasoning...\n<answer> real </answer>": "real",
    "reasoning...\n<answer>fake</answer>": "fake",
    "<answer> Fake </answer>": "fake",
    "<answer>\nREAL\n</answer>": "real",
    "<answer> real or fake? fake </answer>": "real",      # 'real' is checked first (official order)
    "<answer> AI-generated </answer>": None,
    "<answer> fake": None,                                 # truncated: no closing tag
    "no answer at all": None,
    "<answer> fake </answer> then <answer> real </answer>": "fake",   # the first answer counts
}
# A tiny Qwen3-VL (random weights) with the released vocabulary and token ids.
TINY_TEXT = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
             "num_key_value_heads": 1, "head_dim": 16, "vocab_size": 151936, "max_position_embeddings": 4096,
             "rope_parameters": {"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_theta": 5000000,
                                 "rope_type": "default"}, "tie_word_embeddings": False, "pad_token_id": 151643,
             "bos_token_id": 151643, "eos_token_id": 151645}
TINY_VISION = {"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 32,
               "deepstack_visual_indexes": [0], "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
               "num_position_embeddings": 2304, "hidden_act": "gelu_pytorch_tanh"}


def official_constants(path: Path) -> dict:
    return {n.targets[0].id: n.value.value for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}


def official_get_metrics(path: Path):
    """The official ``get_metrics`` function, compiled from its source (the module imports openai at the top)."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_metrics")
    namespace = {"re": __import__("re"), "np": np, "precision_score": precision_score, "recall_score": recall_score,
                 "accuracy_score": accuracy_score}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_metrics"]


def test_parser() -> None:
    print("[A] parser = the Veritas v1 rule ...")
    for text, expected in PARSER_CASES.items():
        assert parse_veritas_v1(text) == expected, (text, parse_veritas_v1(text))
    script = VERITAS_DIR / "self_scripts" / "infer" / "cal_metrics.py"
    if not script.exists():
        print(f"    {len(PARSER_CASES)} cases; official cal_metrics.py not found, comparison skipped")
        return
    get_metrics = official_get_metrics(script)
    # Two anchors (parsed real, label 0; parsed fake, label 0) make the official accuracy tell the three outcomes of
    # the case (labelled fake) apart: fake 2/3, real 1/3, unparsed 1/2.
    anchors = [{"result": "<answer> real </answer>", "label": 0}, {"result": "<answer> fake </answer>", "label": 0}]
    outcome = {2 / 3: "fake", 1 / 3: "real", 1 / 2: None}
    for text in PARSER_CASES:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = get_metrics([{"result": text, "label": 1}] + anchors)["acc"]
        official = next(v for k, v in outcome.items() if abs(k - acc) < 1e-3)
        assert official == parse_veritas_v1(text), (text, official)
    print(f"    {len(PARSER_CASES)} cases, identical to the official get_metrics ({script.parent.parent.parent.name})")


class StubProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


def test_prompt() -> None:
    print("[B] official prompt and chat request ...")
    client = VERITASPP_DIR / "self_scripts" / "infer" / "infer_vllm_single.py"
    if client.exists():
        consts = official_constants(client)
        assert consts["SYSTEM_PROMPT"] == SYSTEM_PROMPT and consts["USER_PROMPT"] == USER_PROMPT
        print("    system / user prompts identical to the official client")
    else:
        print("    official client not found, prompt comparison skipped")
    assert VERITASPP.system_prompt == SYSTEM_PROMPT and VERITASPP.user_prompt == USER_PROMPT
    assert "’" in SYSTEM_PROMPT                   # "It’s" with the official curly apostrophe
    stub = StubProcessor()
    assert build_prompt(stub, VERITASPP) == "rendered"
    messages, kwargs = stub.calls[0]
    assert messages == [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}, kwargs
    if all((WEIGHTS / f).exists() for f in PROCESSOR_FILES):
        from transformers import AutoProcessor
        assert build_prompt(AutoProcessor.from_pretrained(str(WEIGHTS)), VERITASPP) == GATE_PROMPT
        print("    rendered request == the gate check's request (released processor)")
    else:
        print("    released processor not found, rendered-request check skipped")


class StubTokenizer:
    """A byte-level-BPE-like vocabulary: 'Ġ' is a leading space, 'ĉ' a tab."""
    VOCAB = {"<": 0, "answer": 1, ">": 2, "Ġreal": 3, "Ġfake": 4, "real": 5, "fake": 6, "Ġ</": 7, "ĠReal": 8,
             "ĠFake": 9, "ĉreal": 10, "x": 11, "ĠAI": 12}

    def get_vocab(self):
        return dict(self.VOCAB)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).replace("Ġ", " ").replace("ĉ", "\t")

    def encode(self, text, add_special_tokens=False):
        return [self.VOCAB[text.replace(" ", "Ġ")]] if text.replace(" ", "Ġ") in self.VOCAB else [11, 11]

    def decode(self, ids):
        inv = {v: k for k, v in self.VOCAB.items()}
        return self.convert_tokens_to_string([inv[i] for i in ids])


def test_score_math() -> None:
    print("[C] verdict step, token pair, logit gap, fallbacks, LM-head hook ...")
    tok = StubTokenizer()
    pairs = VERITASPP.verdict_pairs
    pair_map = verdict_pair_map(tok, pairs)
    assert pair_map == {3: (4, 3), 4: (4, 3), 5: (6, 5), 6: (6, 5), 8: (9, 8), 9: (9, 8)}, pair_map   # no 'ĉfake'
    assert default_pair(tok, pairs) == (4, 3)
    ids = sorted({i for p in pair_map.values() for i in p})
    column = {t: c for c, t in enumerate(ids)}
    gen = [11, 5, 11, 0, 1, 2, 3, 7, 1, 2]                 # "x real x<answer> real </answer>": 'real' before the marker
    pieces = [tok.decode([t]) for t in gen]
    assert marker_end(pieces, "<answer>") == 6 and locate_verdict(gen, pieces, "<answer>", pair_map) == (6, 6)
    torch.manual_seed(0)
    rows = torch.randn(len(gen), len(ids))
    out = verdict_score(gen, pieces, rows, column, "<answer>", pair_map, (4, 3))
    gap = float(rows[6, column[4]] - rows[6, column[3]])
    assert out["score_source"] == "verdict_token" and out["verdict_step"] == 6 and out["pair_ids"] == (4, 3)
    assert math.isclose(out["logit"], gap) and math.isclose(out["fake_probability"], 1 / (1 + math.exp(-gap)))
    # an upper-case verdict uses its own pair
    gen2 = [0, 1, 2, 9]
    out2 = verdict_score(gen2, [tok.decode([t]) for t in gen2], rows[:4], column, "<answer>", pair_map, (4, 3))
    assert out2["pair_ids"] == (9, 8) and math.isclose(out2["logit"], float(rows[3, column[9]] - rows[3, column[8]]))
    # marker but no verdict word: read at the step after the marker with the default pair
    gen3 = [0, 1, 2, 12, 7, 1, 2]
    out3 = verdict_score(gen3, [tok.decode([t]) for t in gen3], rows[:7], column, "<answer>", pair_map, (4, 3))
    assert out3["score_source"] == "after_marker" and out3["verdict_step"] == 3
    assert math.isclose(out3["logit"], float(rows[3, column[4]] - rows[3, column[3]]))
    # no marker (e.g. truncated): P 0.5, logit 0
    out4 = verdict_score([11, 3, 4], [tok.decode([t]) for t in (11, 3, 4)], rows[:3], column, "<answer>", pair_map,
                         (4, 3))
    assert out4["score_source"] == "none" and out4["fake_probability"] == 0.5 and out4["logit"] == 0.0
    # the hook keeps out[:, -1, ids] of every forward
    head = torch.nn.Linear(8, 13)
    with LogitRecorder(head, ids) as rec:
        xs = [torch.randn(1, n, 8) for n in (5, 1, 1)]
        outs = [head(x) for x in xs]
    assert torch.equal(rec.stacked(), torch.cat([o[:, -1, ids] for o in outs]).detach())
    assert not head._forward_hooks                          # removed on exit
    print("    pairs, verdict step, logit / P(fake), fallbacks and hook all as specified")


def test_registry() -> None:
    print("[D] registry spec, input note, zoo block list, engine guards ...")
    spec = get_spec("veritaspp")
    assert (spec.predict_kind, spec.ckpt_kind, spec.checkpoint, spec.weights) == ("vlm", "official_vlm", "veritaspp",
                                                                                  "official")
    assert spec.transform is None and spec.overrides == [] and spec.modality == "image"
    assert spec.vlm is VERITASPP and spec.eval_batch_size(32) == 1
    assert VERITASPP.model_class == "Qwen3VLForConditionalGeneration" and VERITASPP.answer_marker == "<answer>"
    assert VERITASPP.decoding == {"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0}
    assert VERITASPP.official_decoding == {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                           "max_new_tokens": 2048, "repetition_penalty": 1.0}
    assert VERITASPP.max_pixels is None and VERITASPP.chat_template_kwargs == {"enable_thinking": False}
    assert VERITASPP.parser is parse_veritas_v1 and VERITASPP.hf_repo == "EricTan7/VeritasPP"
    assert input_note(spec)["label"].startswith("Native resolution")
    assert "veritaspp" in hub.NEVER_REDISTRIBUTE
    engine = Engine(device="cpu")
    try:
        engine._fake_prob(None, spec, torch.zeros(1, 3, 8, 8))
    except ValueError as e:
        assert "VLM" in str(e)
    else:
        raise AssertionError("_fake_prob accepted a VLM spec")
    with tempfile.TemporaryDirectory() as tmp:
        original = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = Path(tmp)
        try:
            det = Detector(spec, device="cpu", auto_download=True)
            assert not det.available
            try:
                det._ensure_weights()
            except FileNotFoundError as e:
                assert "hf download EricTan7/VeritasPP --revision 22ba6c6b" in str(e), e
            else:
                raise AssertionError("missing folder not reported")
        finally:
            backend_config.CHECKPOINTS_DIR = original
    print("    vlm spec, decoding, note, NEVER_REDISTRIBUTE, _fake_prob guard, download hint")


def tiny_model_dir(tmp: Path, with_processor: bool) -> Path:
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    torch.manual_seed(0)
    config = Qwen3VLConfig(text_config=TINY_TEXT, vision_config=TINY_VISION, image_token_id=151655,
                           video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653,
                           tie_word_embeddings=False)
    model = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(tmp)
    if with_processor:
        for name in PROCESSOR_FILES:
            shutil.copy(WEIGHTS / name, tmp / name)
    return tmp


def test_strict_load() -> None:
    print("[E] strict load (tiny random Qwen3-VL) ...")
    check_loading_info({"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []}, "x")
    for bad in ({"missing_keys": {"lm_head.weight"}}, {"unexpected_keys": {"extra"}},
                {"mismatched_keys": {("a", (1,), (2,))}}, {"error_msgs": ["boom"]}):
        try:
            check_loading_info(bad, "x")
        except RuntimeError as e:
            assert "checkpoint mismatch" in str(e)
        else:
            raise AssertionError(bad)
    with_processor = all((WEIGHTS / f).exists() for f in PROCESSOR_FILES)
    with tempfile.TemporaryDirectory() as tmp:
        folder = tiny_model_dir(Path(tmp) / "veritaspp", with_processor)
        shard = next(folder.glob("*.safetensors"))
        tensors = {k: v.clone() for k, v in load_file(str(shard)).items()}   # not views of the file rewritten below
        n = len(tensors)
        cfg = dataclasses.replace(VERITASPP, decoding={"do_sample": False, "max_new_tokens": 24,
                                                       "repetition_penalty": 1.0})
        if with_processor:
            vlm = SynthScopeVLM(folder, cfg, device="cpu")
            assert len(vlm.hf.state_dict()) == n
            image = Image.fromarray((np.random.RandomState(0).rand(200, 300, 3) * 255).astype("uint8"))
            out = vlm.score(image)
            inputs = vlm.inputs(image)
            with torch.inference_mode(), LogitRecorder(vlm.hf.get_output_embeddings(), vlm.record_ids) as rec:
                ref = vlm.hf.generate(**inputs, **cfg.decoding, output_logits=True, return_dict_in_generate=True)
            assert ref.sequences[0, inputs["input_ids"].shape[1]:].tolist() == out["generated_ids"]
            assert torch.equal(rec.stacked(), torch.stack([row[0, vlm.record_ids] for row in ref.logits]).cpu())
            assert out["input_size"] == [320, 224] and out["new_tokens"] == 24 and out["decoding"] == "greedy"
            # the engine path on the same folder
            original = backend_config.CHECKPOINTS_DIR
            backend_config.CHECKPOINTS_DIR = Path(tmp)
            try:
                spec = dataclasses.replace(get_spec("veritaspp"), vlm=cfg)
                engine = Engine(device="cpu")
                res = engine.predict_image(spec, image)
                assert res["fake_probability"] == round(out["fake_probability"], 6) and res["logit"] == round(out["logit"], 6)
                assert "missing=0, unexpected=0" in res["checkpoint_status"] and res["input_size"] == [320, 224]
            finally:
                backend_config.CHECKPOINTS_DIR = original
            print(f"    tiny model: {n} tensors strict; hooked logits == output_logits; tokens == direct generate; engine")
        else:
            from transformers import Qwen3VLForConditionalGeneration
            _, info = Qwen3VLForConditionalGeneration.from_pretrained(str(folder), output_loading_info=True)
            check_loading_info(info, "tiny")
            print(f"    tiny model: {n} tensors strict (released processor not found: scoring skipped)")
        for name, edit in (("missing", lambda t: t.pop("lm_head.weight")),
                           ("unexpected", lambda t: t.__setitem__("model.extra_tensor", torch.zeros(2)))):
            broken = dict(tensors)
            edit(broken)
            save_file(broken, str(shard), metadata={"format": "pt"})
            try:
                SynthScopeVLM(folder, cfg, device="cpu")
            except RuntimeError as e:
                assert "checkpoint mismatch" in str(e), e
            else:
                raise AssertionError(f"a {name} tensor was accepted")
        print("    a removed or an added tensor raises")


def test_released_weights() -> None:
    if not (WEIGHTS / "model.safetensors.index.json").exists() or not torch.cuda.is_available():
        print("[F] released weights + GPU not available here: skipped")
        return
    print("[F] released weights: strict engine load and one image ...")
    spec = get_spec("veritaspp")
    engine = Engine(device="cuda")
    model = engine._get_model(spec)
    n = len(json.loads((WEIGHTS / "model.safetensors.index.json").read_text())["weight_map"])
    assert n == 750 and len(model.hf.state_dict()) == n, (n, len(model.hf.state_dict()))
    gate = Path("/ssdstore/ssalehi/SynthScope/GenImage_val_full/test/adm/1_fake/0_adm_153.PNG")
    image = Image.open(gate).convert("RGB") if gate.exists() else Image.new("RGB", (256, 256), (90, 120, 200))
    out = engine.vlm_score(spec, image)
    assert out["verdict"] in ("real", "fake") and out["score_source"] == "verdict_token"
    assert (out["logit"] > 0) == (out["verdict"] == "fake")
    print(f"    750 tensors strict; {gate.name if gate.exists() else 'synthetic'}: {out['verdict']} "
          f"(P(fake) {out['fake_probability']:.6f}, {out['new_tokens']} tokens)")


if __name__ == "__main__":
    test_parser()
    test_prompt()
    test_score_math()
    test_registry()
    test_strict_load()
    test_released_weights()
    print("veritaspp smoke test passed")
