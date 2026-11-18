"""Smoke test for the shared VLM detector kind and Veritas++ (``veritaspp``); CPU, no downloads, no released files needed.

Checks:
[A] the verdict parser = the authors' Veritas v1 ``get_metrics`` answer rule ("real" before "fake", unparsed is None);
    with the official clone ($VERITAS_OFFICIAL_DIR, default SynthScope_shared/vlm_ref/Veritas) the official function
    itself is run on every case (skipped otherwise);
[B] the prompt: the system / user prompts equal ``SYSTEM_PROMPT`` / ``USER_PROMPT`` of the official client (with
    $VERITASPP_OFFICIAL_DIR, default SynthScope_shared/vlm_ref/VeritasPP; skipped otherwise); ``VLMFamily.prompt``
    sends [system, user: [image, text]] with ``add_generation_prompt`` and ``enable_thinking=False``, or the raw prompt
    with ``chat_template=False``; with the released processor files ($SYNTHSCOPE_CKPT_DIR/veritaspp) the rendered
    request equals the gate check's text (skipped otherwise);
[C] the score math on a stub tokenizer: verdict-token pairs, the marker search, the verdict step (first verdict token
    after "<answer>", none after "</answer>"; from the start without a marker), logit = l(fake) - l(real) of that pair,
    P(fake) = sigmoid(logit); no verdict token -> P 0.5, logit 0, "unparsed" (ruling U1); the LM-head hook; Veritas++
    has no verdict-offset hook, so the engine's call (offset None) is exactly the marker search;
[D] the image loader = vLLM's (EXIF transpose, transparency on white, RGB) on every branch; the recorded comparison
    with vLLM itself (repro/vlm_image_vllm_parity.json) is all identical; VLM inputs are decoded without an RGB
    conversion (``media.decode_image_raw``, ``Detector._classify``);
[E] the registry spec, the input note, the zoo block list, the engine guards (``_fake_prob``, video refused in the
    engine and the API, the download hint), the web app's 20-item batch cap; /api/predict scores a VLM in a worker
    thread on the raw file and refuses a video;
[F] a tiny random Qwen3-VL with a tiny local processor (no released files): strict load, a removed / added tensor
    raises, an adapter folder without a base raises; ``score`` end to end (forced answer: verdict step, logit ==
    ``output_logits`` gap, parsed verdict; a forced tag-less output is unparsed, nothing read outside the marker),
    hooked logits == ``output_logits``, tokens == a direct ``generate``,
    processed logits == ``output_scores`` under a repetition penalty, two threads give the sequential result (lock),
    the engine's ``predict_image``; with peft importable ($VLM_PYDEPS adds a path): an unmerged LoRA on a local base
    loads strictly, matches a direct peft load, and a tampered adapter raises;
[G] a dummy LLaVA-style family (no ``image_grid_thw``, no chat template, no answer marker, teacher-forced single pass)
    plugs in: P(fake) = sigmoid of a direct forward's gap, input size from the pixel tensor, generate mode without a
    marker runs;
[H] with the released weights and a GPU: the engine loads all 750 tensors strictly and scores one image with a parsed
    verdict whose sign agrees with the logit (skipped otherwise).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, models as tok_models, normalizers, pre_tokenizers, trainers
from transformers import (CLIPImageProcessor, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor,
                          LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerFast, Qwen2VLImageProcessor,
                          Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from interface.backend import app as web_app                                       # noqa: E402
from interface.backend import config as backend_config                             # noqa: E402
from interface.backend import media                                                # noqa: E402
from interface.backend.engine import Engine                                        # noqa: E402
from interface.backend.registry import get_spec, input_note                        # noqa: E402
from models.image.veritaspp.veritaspp import SYSTEM_PROMPT, USER_PROMPT, VERITASPP   # noqa: E402
from models.vlm.image import vllm_rgb                                              # noqa: E402
from models.vlm.parsers import parse_veritas_v1                                    # noqa: E402
from models.vlm.vlm import (QWEN_VL, LogitRecorder, ProcessedLogitRecorder, SynthScopeVLM, VLMConfig,   # noqa: E402
                            VLMFamily, check_loading_info, divergence, locate_verdict, marker_end,
                            token_pieces, verdict_pair_map, verdict_score)
from synthscope import hub                                                         # noqa: E402
from synthscope.api import Detector                                                # noqa: E402

SHARED = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
VERITAS_DIR = Path(os.environ.get("VERITAS_OFFICIAL_DIR", SHARED / "vlm_ref" / "Veritas"))
VERITASPP_DIR = Path(os.environ.get("VERITASPP_OFFICIAL_DIR", SHARED / "vlm_ref" / "VeritasPP"))
PYDEPS = os.environ.get("VLM_PYDEPS", str(SHARED / "vlm_pydeps"))
WEIGHTS = backend_config.CHECKPOINTS_DIR / "veritaspp"
PROCESSOR_FILES = ("chat_template.jinja", "preprocessor_config.json", "processor_config.json", "tokenizer.json",
                   "tokenizer_config.json")
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
# Tiny word-level tokenizer shared by the tiny Qwen3-VL and LLaVA fixtures.
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
           "<|video_pad|>", "<image>", "<unk>"]
WORDS = ("system user assistant You are an expert visual analyst . Please think Is this image real or fake ? Real Fake "
         "REAL FAKE < answer > </ the a it looks Does This is / x y z").split()
QWEN_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content is string %}{{ m.content }}{% else %}"
                 "{% for c in m.content %}{% if c.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
                 "{% else %}{{ c.text }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
TINY_TEXT = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
             "num_key_value_heads": 1, "head_dim": 16, "max_position_embeddings": 4096,
             "rope_parameters": {"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_theta": 5000000,
                                 "rope_type": "default"}, "tie_word_embeddings": False}
TINY_VISION = {"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 32,
               "deepstack_visual_indexes": [0], "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
               "num_position_embeddings": 2304, "hidden_act": "gelu_pytorch_tanh"}


@contextlib.contextmanager
def quiet():
    """Silence transformers' load reports, progress bars and library warnings (stdout / stderr at the fd level)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(os.devnull, "w") as null, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.dup2(null.fileno(), 1)
        os.dup2(null.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


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


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {t: i for i, t in enumerate(SPECIAL + WORDS)}
    backend = Tokenizer(tok_models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.decoder = decoders.Fuse()                     # decode joins tokens without spaces: "<answer>fake</answer>"
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<|endoftext|>",
                                   eos_token="<|im_end|>", additional_special_tokens=SPECIAL[1:8])


def tiny_qwen_folder(folder: Path, seed: int = 0) -> Path:
    """A tiny random Qwen3-VL + a tiny local Qwen3-VL processor, saved with transformers."""
    tok = tiny_tokenizer()
    ids = tok.convert_tokens_to_ids
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(size={"shortest_edge": 65536, "longest_edge": 16777216}, patch_size=16,
                                              merge_size=2, temporal_patch_size=2, image_mean=[0.5] * 3,
                                              image_std=[0.5] * 3),
        tokenizer=tok, video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        chat_template=QWEN_TEMPLATE)
    config = Qwen3VLConfig(
        text_config={**TINY_TEXT, "vocab_size": len(tok), "pad_token_id": ids("<|endoftext|>"),
                     "eos_token_id": ids("<|im_end|>")},
        vision_config=TINY_VISION, image_token_id=ids("<|image_pad|>"), video_token_id=ids("<|video_pad|>"),
        vision_start_token_id=ids("<|vision_start|>"), vision_end_token_id=ids("<|vision_end|>"),
        tie_word_embeddings=False)
    torch.manual_seed(seed)
    with quiet():
        Qwen3VLForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(folder)
        processor.save_pretrained(folder)
    return folder


SP_PIECES = ["<unk>", "<s>", "</s>", "▁", "▁This", "▁is", "▁a", "▁fake", "▁real", "▁Does", "▁the", "▁image", "▁looks",
             "/", "fake", "▁?", "?", ".", "x"]


def sentencepiece_tokenizer() -> PreTrainedTokenizerFast:
    """A synthetic SentencePiece / Llama-style tokenizer (Unigram over "▁"-joined text, no "▁▁" piece) that reproduces
    the LLaVA-1.5 (Llama) tokenizer's splits checked on cn029: " real" -> ["▁", "▁real"], "real" -> ["▁real"],
    "This is a fake" -> ["▁This", "▁is", "▁a", "▁fake"]. (The real tokenizer is Llama 2-licensed, so it is not copied.)"""
    vocab = [(piece, -2.0 if piece == "▁" else -1.0) for piece in SP_PIECES]
    backend = Tokenizer(tok_models.Unigram(vocab, unk_id=0, byte_fallback=False))
    backend.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    backend.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.Fuse(), decoders.Strip(" ", 1, 0)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                   pad_token="<unk>", additional_special_tokens=["<image>"])


def bytelevel_tokenizer() -> PreTrainedTokenizerFast:
    """A small byte-level BPE tokenizer (Qwen / GPT-2 style: "Ġ" is a leading space) trained on a toy corpus."""
    backend = Tokenizer(tok_models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    corpus = ["the image looks real or fake <answer> fake </answer> <answer> real </answer> this is a fake image"] * 50
    backend.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                            special_tokens=["<|endoftext|>"]))
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")


def tiny_llava_folder(folder: Path) -> Path:
    tok = sentencepiece_tokenizer()
    processor = LlavaProcessor(image_processor=CLIPImageProcessor(size={"shortest_edge": 32},
                                                                  crop_size={"height": 32, "width": 32}),
                               tokenizer=tok, patch_size=16, vision_feature_select_strategy="default",
                               num_additional_image_tokens=1)
    config = LlavaConfig(
        vision_config={"model_type": "clip_vision_model", "hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 32, "patch_size": 16},
        text_config={"model_type": "llama", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": len(tok),
                     "eos_token_id": tok.convert_tokens_to_ids("</s>")},
        image_token_id=tok.convert_tokens_to_ids("<image>"), vision_feature_select_strategy="default",
        vision_feature_layer=-2)
    torch.manual_seed(1)
    with quiet():
        LlavaForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(folder)
        processor.save_pretrained(folder)
    return folder


class ForceTokens(LogitsProcessor):
    """Test-only: make generate emit ``script`` (token ids) step by step, then EOS; raw logits are left untouched."""

    def __init__(self, script: list, eos: int, n_prompt: int):
        self.script, self.eos, self.n_prompt = script, eos, n_prompt

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.n_prompt
        target = self.script[step] if step < len(self.script) else self.eos
        forced = torch.full_like(scores, -1e9)
        forced[:, target] = 0.0
        return forced


def rgb_image(w: int = 300, h: int = 200, seed: int = 0) -> Image.Image:
    return Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8"))


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
    print(f"    {len(PARSER_CASES)} cases, identical to the official get_metrics")


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
    assert QWEN_VL.prompt(stub, VERITASPP) == "rendered"
    messages, kwargs = stub.calls[0]
    assert messages == [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}, kwargs
    raw = dataclasses.replace(VERITASPP, chat_template=False, user_prompt="<image>Does the image looks real/fake?")
    assert QWEN_VL.prompt(stub, raw) == "<image>Does the image looks real/fake?" and len(stub.calls) == 1
    if all((WEIGHTS / f).exists() for f in PROCESSOR_FILES):
        with quiet():
            processor = transformers.AutoProcessor.from_pretrained(str(WEIGHTS), local_files_only=True)
        assert QWEN_VL.prompt(processor, VERITASPP) == GATE_PROMPT
        print("    rendered request == the gate check's request (released processor)")
    else:
        print("    released processor not found here, rendered-request check skipped (tiny processor: [F])")


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
    print("[C] verdict step, token pair, logit gap, U1, LM-head hook ...")
    tok = StubTokenizer()
    pairs = VERITASPP.verdict_pairs
    pair_map = verdict_pair_map(tok, pairs)
    assert pair_map == {3: (4, 3), 4: (4, 3), 5: (6, 5), 6: (6, 5), 8: (9, 8), 9: (9, 8)}, pair_map   # no 'ĉfake'
    assert VLMFamily().verdict_sequences(tok, VERITASPP) == ([4], [3]) and divergence([4], [3]) == 0
    assert divergence([7, 4, 1], [7, 3]) == 1
    try:
        divergence([7, 4], [7, 4, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("a prefix pair was accepted")
    ids = sorted({i for p in pair_map.values() for i in p})
    column = {t: c for c, t in enumerate(ids)}
    torch.manual_seed(0)
    rows = torch.randn(12, len(ids))

    def run(gen, marker="<answer>", end="</answer>"):
        return verdict_score(gen, [tok.decode([t]) for t in gen], rows[:len(gen)], column, marker, pair_map, end)

    gen = [11, 5, 11, 0, 1, 2, 3, 7, 1, 2]                 # "x real x<answer> real </answer>": 'real' before the marker
    pieces = [tok.decode([t]) for t in gen]
    assert marker_end(pieces, "<answer>") == 6 and locate_verdict(gen, pieces, "<answer>", pair_map) == (6, 6)
    out = run(gen)
    gap = float(rows[6, column[4]] - rows[6, column[3]])
    assert out["score_source"] == "verdict_token" and out["verdict_step"] == 6 and out["pair_ids"] == (4, 3)
    assert math.isclose(out["logit"], gap) and math.isclose(out["fake_probability"], 1 / (1 + math.exp(-gap)))
    out = run([0, 1, 2, 9])                                # an upper-case verdict uses its own pair
    assert out["pair_ids"] == (9, 8) and math.isclose(out["logit"], float(rows[3, column[9]] - rows[3, column[8]]))
    # no verdict word inside the answer (U1): 0.5 / 0 / unparsed, even if one follows the closing tag
    for gen in ([0, 1, 2, 12, 7, 1, 2], [0, 1, 2, 12, 7, 1, 2, 11, 4], [11, 3, 4]):
        out = run(gen)
        assert (out["score_source"], out["fake_probability"], out["logit"]) == ("unparsed", 0.5, 0.0), gen
    # Veritas++ pairs no verdict-offset hook with its parser (which reads only the <answer> tag): the engine passes
    # offset None, which is exactly the marker search, on every output above
    assert VERITASPP.verdict_offset is None
    for gen in ([11, 5, 11, 0, 1, 2, 3, 7, 1, 2], [0, 1, 2, 9], [0, 1, 2, 12, 7, 1, 2], [0, 1, 2, 12, 7, 1, 2, 11, 4],
                [11, 3, 4], [11, 4]):
        pieces = [tok.decode([t]) for t in gen]
        assert verdict_score(gen, pieces, rows[:len(gen)], column, "<answer>", pair_map, "</answer>", offset=None,
                             text="".join(pieces)) == run(gen), gen
    # without a marker the first verdict word counts (from step 0)
    out = run([11, 4, 3], marker=None, end=None)
    assert out["verdict_step"] == 1 and out["pair_ids"] == (4, 3) and marker_end(["x"], None) == 0
    # token_pieces (incremental decode) == the per-token decode on a byte-level BPE tokenizer (Veritas++ / Qwen): same
    # pieces on ASCII, same marker / verdict positions with non-ASCII bytes split over tokens
    bl = bytelevel_tokenizer()
    for text in ("the image looks real. <answer> fake </answer>", "reasoning x <answer>real</answer> y",
                 "caf\u00e9 \u65e5\u672c \U0001f600 <answer> fake </answer>"):
        gen = bl.encode(text, add_special_tokens=False)
        alone = [bl.decode([t]) for t in gen]
        inc = token_pieces(bl, gen)
        assert len(inc) == len(gen)
        if text.isascii():
            assert inc == alone and "".join(inc) == text, (inc, alone)
        pm = verdict_pair_map(bl, VERITASPP.verdict_pairs)
        assert pm, "no verdict pairs in the toy vocabulary"
        assert marker_end(inc, "<answer>") == marker_end(alone, "<answer>")
        assert locate_verdict(gen, inc, "<answer>", pm, "</answer>") == locate_verdict(gen, alone, "<answer>", pm, "</answer>")
    assert token_pieces(bl, []) == []
    # a SentencePiece / Llama tokenizer drops the leading space of a token decoded alone ("\u2581is" -> "is"), so a
    # multi-word marker is never found in the per-token pieces; the incremental decode keeps it
    sp = sentencepiece_tokenizer()
    gen = sp.encode("This is a fake", add_special_tokens=False)
    alone = [sp.decode([t]) for t in gen]
    assert alone == ["This", "is", "a", "fake"] and marker_end(alone, "This is a") is None   # the old behaviour: lost
    inc = token_pieces(sp, gen)
    assert inc == ["This", " is", " a", " fake"], inc
    sp_pairs = verdict_pair_map(sp, (("fake", "real"),))
    assert marker_end(inc, "This is a") == 3 and locate_verdict(gen, inc, "This is a", sp_pairs) == (3, 3)
    # the hook keeps out[:, -1, ids] of every forward
    head = torch.nn.Linear(8, 13)
    with LogitRecorder(head, ids) as rec:
        outs = [head(torch.randn(1, n, 8)) for n in (5, 1, 1)]
    assert torch.equal(rec.stacked(), torch.cat([o[:, -1, ids] for o in outs]).detach())
    assert not head._forward_hooks                          # removed on exit
    print("    pairs, verdict step (bounded by </answer>), logit / P(fake), U1 and hook as specified")


def test_image_loader() -> None:
    print("[D] image loader = vLLM's (EXIF, transparency on white) ...")
    rng = np.random.RandomState(0)
    rgba = Image.fromarray((rng.rand(5, 7, 4) * 255).astype("uint8"), "RGBA")
    out = vllm_rgb(rgba)
    a, rgb_a = np.asarray(rgba).astype(float), np.asarray(out).astype(float)
    expected = a[..., :3] * a[..., 3:] / 255 + 255 * (1 - a[..., 3:] / 255)    # alpha over white
    assert out.mode == "RGB" and np.abs(rgb_a - expected).max() <= 1.0
    clear = Image.new("RGBA", (4, 4), (10, 20, 30, 0))
    assert vllm_rgb(clear).getpixel((0, 0)) == (255, 255, 255)          # PIL convert("RGB") would give (10, 20, 30)
    assert vllm_rgb(Image.new("LA", (3, 3), (0, 0))).getpixel((1, 1)) == (255, 255, 255)
    pal = Image.new("P", (3, 3), 3)
    pal.info["transparency"] = 3
    assert vllm_rgb(pal).getpixel((0, 0)) == (255, 255, 255)
    exif = Image.Exif()
    exif[0x0112] = 6                                        # rotate 90 degrees clockwise to display
    buf = io.BytesIO()
    rgb_image(53, 37).save(buf, "JPEG", exif=exif.tobytes())
    raw = media.decode_image_raw(buf.getvalue())
    assert raw.size == (53, 37) and vllm_rgb(raw).size == (37, 53)
    assert media.decode_image(buf.getvalue()).size == (53, 37)          # the other detectors: unchanged
    png = io.BytesIO()
    clear.save(png, "PNG")
    assert media.decode_image_raw(png.getvalue()).mode == "RGBA"
    det = Detector(get_spec("veritaspp"), device="cpu")
    assert det._classify(png.getvalue())[1].mode == "RGBA"
    assert Detector(get_spec("clip"), device="cpu")._classify(png.getvalue())[1].mode == "RGB"
    record = Path(ROOT) / "repro" / "vlm_image_vllm_parity.json"
    parity = json.loads(record.read_text())
    assert parity["all_identical"] and len(parity["files"]) == 16
    print(f"    every branch as vLLM; recorded vs vLLM {parity['vllm']}: {len(parity['files'])}/16 files identical; "
          "VLM inputs decoded without RGB conversion")


def test_registry_and_guards() -> None:
    print("[E] registry spec, note, block list, engine / API / web guards ...")
    spec = get_spec("veritaspp")
    assert (spec.predict_kind, spec.ckpt_kind, spec.checkpoint, spec.weights) == ("vlm", "official_vlm", "veritaspp",
                                                                                  "official")
    assert spec.transform is None and spec.overrides == [] and spec.modality == "image"
    assert spec.vlm is VERITASPP and spec.eval_batch_size(32) == 1 and VERITASPP.family is QWEN_VL
    assert VERITASPP.model_class == "Qwen3VLForConditionalGeneration"
    assert (VERITASPP.answer_marker, VERITASPP.answer_end, VERITASPP.verdict_mode, VERITASPP.logits) == \
        ("<answer>", "</answer>", "generate", "raw")
    assert VERITASPP.decoding == {"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0}
    assert VERITASPP.official_decoding == {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                           "max_new_tokens": 2048, "repetition_penalty": 1.0}
    assert VERITASPP.max_pixels is None and VERITASPP.chat_template_kwargs == {"enable_thinking": False}
    assert VERITASPP.parser is parse_veritas_v1 and VERITASPP.image_loader is vllm_rgb and VERITASPP.base_model is None
    assert VERITASPP.hf_repo == "EricTan7/VeritasPP"
    assert input_note(spec)["label"].startswith("Native resolution") and "EXIF" in input_note(spec)["explanation"]
    assert "veritaspp" in hub.NEVER_REDISTRIBUTE
    engine = Engine(device="cpu")
    for call in (lambda: engine._fake_prob(None, spec, torch.zeros(1, 3, 8, 8)),
                 lambda: engine.predict_video_frames(spec, [rgb_image()])):
        try:
            call()
        except ValueError as e:
            assert "VLM" in str(e)
        else:
            raise AssertionError("a VLM spec was accepted")
    assert (web_app.SCORE_BATCH_CAP, web_app.SCORE_BATCH_CAP_VLM) == (200, 20)
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
            Path(tmp, "veritaspp").mkdir()                  # present: the API refuses a video before decoding it
            Path(tmp, "veritaspp", "config.json").write_text("{}")
            try:
                Detector(spec, device="cpu").predict(Path(tmp) / "clip.mp4")
            except ValueError as e:
                assert "still images only" in str(e), e
            else:
                raise AssertionError("a video was accepted")
        finally:
            backend_config.CHECKPOINTS_DIR = original
    test_web_threadpool()
    print("    vlm spec, decoding, note, NEVER_REDISTRIBUTE, _fake_prob / video refused, download hint, cap 20; "
          "/api/predict: worker thread, raw RGBA file, video upload -> 400")


def test_web_threadpool() -> None:
    """/api/predict scores a VLM off the event loop, on the raw file; a video upload is refused."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    seen = {}
    engine = web_app.get_engine()
    original = (engine.is_available, engine.predict_image)

    def fake_predict(spec, pil):
        seen["main_thread"], seen["mode"] = threading.current_thread() is threading.main_thread(), pil.mode
        seen["locked"] = web_app._GPU_LOCK.locked()            # a VLM score holds the GPU lock (finding 8)
        return {"fake_probability": 0.9, "verdict": "fake"}

    engine.is_available, engine.predict_image = (lambda spec: True), fake_predict
    try:
        png = io.BytesIO()
        Image.new("RGBA", (8, 8), (1, 2, 3, 0)).save(png, "PNG")
        client = TestClient(web_app.app)
        r = client.post("/api/predict", data={"model": "veritaspp"}, files={"file": ("x.png", png.getvalue(), "image/png")})
        assert r.status_code == 200, r.text
        assert seen == {"main_thread": False, "mode": "RGBA", "locked": True}, seen
        assert not web_app._GPU_LOCK.locked()                  # and releases it
        # the SOTA-compare path hands a VLM the raw decode too (finding 7), other detectors the RGB one
        modes = {}

        def record(spec, pil):
            modes[spec.key] = modes.get(spec.key, []) + [pil.mode]
            return {"fake_probability": 0.9, "verdict": "fake"}

        engine.predict_image = record
        res = web_app._sota_score(png.getvalue(), png.getvalue(), "veritaspp,clip")
        assert [r.get("detector") for r in res] == ["veritaspp", "clip"] and all("error" not in r for r in res), res
        assert modes == {"veritaspp": ["RGBA", "RGBA"], "clip": ["RGB", "RGB"]}, modes
        r = client.post("/api/predict", data={"model": "veritaspp"}, files={"file": ("v.mp4", b"00000000", "video/mp4")})
        assert r.status_code == 400 and "still images only" in r.json()["detail"], r.text
    finally:
        engine.is_available, engine.predict_image = original


def test_tiny_qwen() -> None:
    print("[F] tiny random Qwen3-VL + tiny local processor: strict load, score, hook, lock, engine, LoRA ...")
    check_loading_info({"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []}, "x")
    for bad in ({"missing_keys": {"lm_head.weight"}}, {"unexpected_keys": {"extra"}},
                {"mismatched_keys": {("a", (1,), (2,))}}, {"error_msgs": ["boom"]}):
        try:
            check_loading_info(bad, "x")
        except RuntimeError as e:
            assert "checkpoint mismatch" in str(e)
        else:
            raise AssertionError(bad)
    cfg = dataclasses.replace(VERITASPP, decoding={"do_sample": False, "max_new_tokens": 16, "repetition_penalty": 1.0})
    image = rgb_image()
    with tempfile.TemporaryDirectory() as tmp:
        folder = tiny_qwen_folder(Path(tmp) / "veritaspp")
        with quiet():
            vlm = SynthScopeVLM(folder, cfg, device="cpu")
        tok = vlm.tokenizer
        n_tensors = len(load_file(str(next(folder.glob("*.safetensors")))))
        assert len(vlm.hf.state_dict()) == n_tensors
        assert vlm.prompt.startswith("<|im_start|>system\n") and vlm.prompt.endswith("<|im_start|>assistant\n")
        # free generation: hooked raw logits == output_logits, same tokens as a direct generate
        with quiet():
            out = vlm.score(image)
            inputs = vlm.inputs(vlm.image(image))
            with torch.inference_mode(), LogitRecorder(vlm.hf.get_output_embeddings(), vlm.record_ids) as rec:
                ref = vlm.hf.generate(**inputs, **cfg.decoding, output_logits=True, return_dict_in_generate=True)
        n_prompt = inputs["input_ids"].shape[1]
        assert ref.sequences[0, n_prompt:].tolist() == out["generated_ids"]
        assert torch.equal(rec.stacked(), torch.stack([row[0, vlm.record_ids] for row in ref.logits]).cpu())
        assert out["input_size"] == [320, 224] and out["decoding"] == "greedy"
        # generated pieces: the incremental decode equals the per-token decode on this (Qwen-style) path
        assert token_pieces(tok, out["generated_ids"]) == [tok.decode([t]) for t in out["generated_ids"]]
        # max_pixels (the QwenVL family caps longest_edge) and the preprocess hook (ADB: fetch_image), tiny processor
        capped = dataclasses.replace(cfg, max_pixels=64 * 64)
        pre_calls = []

        def shrink(im):
            pre_calls.append(im.size)
            return im.resize((100, 100))

        vlm.cfg = capped
        with quiet():
            small = vlm.score(image)
            direct = vlm.processor(text=[vlm.prompt], images=[vlm.image(image)], return_tensors="pt",
                                   images_kwargs={"size": {"shortest_edge": 65536, "longest_edge": 64 * 64}})
        grid = direct["image_grid_thw"][0].tolist()
        assert small["input_size"] == [grid[2] * 16, grid[1] * 16] and small["input_size"] != [320, 224], small["input_size"]
        assert small["input_size"][0] * small["input_size"][1] <= 64 * 64 and small["prompt_tokens"] < out["prompt_tokens"]
        vlm.cfg = dataclasses.replace(cfg, preprocess=shrink)
        assert vlm.image(image).size == (100, 100) and pre_calls == [image.size]      # after the image loader, before the processor
        with quiet():
            pre = vlm.score(image)
            ref_in = vlm.processor(text=[vlm.prompt], images=[image.resize((100, 100))], return_tensors="pt")
        ref_grid = ref_in["image_grid_thw"][0].tolist()
        assert pre["input_size"] == [ref_grid[2] * 16, ref_grid[1] * 16] and pre["prompt_tokens"] == ref_in["input_ids"].shape[1]
        assert pre_calls == [image.size] * 2                  # once in the check above, once in score
        vlm.cfg = cfg
        # a forced answer: verdict step, gap == output_logits, parsed verdict
        script = tok.convert_tokens_to_ids(["x", "<", "answer", ">", "fake", "</", "answer", ">"])
        forcing = ForceTokens(script, tok.convert_tokens_to_ids("<|im_end|>"), n_prompt)
        forced_cfg = dataclasses.replace(cfg, decoding={**cfg.decoding, "logits_processor": LogitsProcessorList([forcing])})
        vlm.cfg = forced_cfg
        with quiet():
            out = vlm.score(image)
            with torch.inference_mode():
                ref = vlm.hf.generate(**inputs, **forced_cfg.decoding, output_logits=True, return_dict_in_generate=True)
        fake_id, real_id = tok.convert_tokens_to_ids(["fake", "real"])
        gap = float(ref.logits[4][0, fake_id] - ref.logits[4][0, real_id])
        assert (out["score_source"], out["verdict_step"], out["verdict"], out["verdict_token"]) == \
            ("verdict_token", 4, "fake", "fake"), out
        assert out["logit"] == gap and out["finished"] and out["new_tokens"] == len(script) + 1
        # two threads at once give the sequential result (the lock serializes the shared LM-head hook)
        results = [None, None]

        def worker(i):
            results[i] = vlm.score(image)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        with quiet():
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert all(r["logit"] == out["logit"] and r["generated_ids"] == out["generated_ids"] for r in results)
        # a tag-less output ("x fake"): the Veritas parser finds no verdict and, with no verdict-offset hook, no word
        # outside the marker is read: unparsed, P 0.5
        script = tok.convert_tokens_to_ids(["x", "fake"])
        tagless = ForceTokens(script, tok.convert_tokens_to_ids("<|im_end|>"), n_prompt)
        vlm.cfg = dataclasses.replace(cfg, decoding={**cfg.decoding, "logits_processor": LogitsProcessorList([tagless])})
        with quiet():
            bare = vlm.score(image)
        assert (bare["score_source"], bare["verdict"], bare["fake_probability"], bare["verdict_step"]) == \
            ("unparsed", None, 0.5, None), bare
        vlm.cfg = forced_cfg
        # processed logits (after a repetition penalty) == output_scores
        rp = dataclasses.replace(cfg, logits="processed", decoding={**cfg.decoding, "repetition_penalty": 1.3})
        with quiet(), torch.inference_mode():
            rec2 = ProcessedLogitRecorder(vlm.record_ids)
            ref = vlm.hf.generate(**inputs, **rp.decoding, logits_processor=LogitsProcessorList([rec2]),
                                  output_scores=True, return_dict_in_generate=True)
        assert torch.equal(rec2.stacked(), torch.stack([row[0, vlm.record_ids] for row in ref.scores]).cpu())
        vlm.cfg = rp
        with quiet():
            assert vlm.score(image)["generated_ids"] == ref.sequences[0, n_prompt:].tolist()
        vlm.cfg = cfg
        # the engine path on the same folder
        original = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = Path(tmp)
        try:
            spec = dataclasses.replace(get_spec("veritaspp"), vlm=cfg)
            with quiet():
                res = Engine(device="cpu").predict_image(spec, image)
                plain = vlm.score(image)
            assert res["fake_probability"] == round(plain["fake_probability"], 6) and res["logit"] == round(plain["logit"], 6)
            assert "missing=0, unexpected=0" in res["checkpoint_status"] and res["input_size"] == [320, 224]
            assert res["verdict"] == (plain["verdict"] or "unparsed")
        finally:
            backend_config.CHECKPOINTS_DIR = original
        # strictness: a removed or an added tensor raises; an adapter folder needs a local base
        shard = next(folder.glob("*.safetensors"))
        tensors = {k: v.clone() for k, v in load_file(str(shard)).items()}   # not views of the file rewritten below
        for name, edit in (("missing", lambda t: t.pop("lm_head.weight")),
                           ("unexpected", lambda t: t.__setitem__("model.extra_tensor", torch.zeros(2)))):
            broken = dict(tensors)
            edit(broken)
            save_file(broken, str(shard), metadata={"format": "pt"})
            try:
                with quiet():
                    SynthScopeVLM(folder, cfg, device="cpu")
            except RuntimeError as e:
                assert "checkpoint mismatch" in str(e), e
            else:
                raise AssertionError(f"a {name} tensor was accepted")
        save_file(tensors, str(shard), metadata={"format": "pt"})
        Path(folder, "adapter_config.json").write_text("{}")
        try:
            SynthScopeVLM(folder, cfg, device="cpu")
        except RuntimeError as e:
            assert "LoRA adapter" in str(e)
        else:
            raise AssertionError("an adapter folder without a base was loaded")
        Path(folder, "adapter_config.json").unlink()
        lora = check_lora(Path(tmp), cfg, image)
    print(f"    {n_tensors} tensors strict; score / hook / tokens / forced verdict / processed logits / lock / engine; "
          f"removed or added tensor raises; adapter needs a base; LoRA: {lora}")


def check_lora(tmp: Path, cfg: VLMConfig, image: Image.Image) -> str:
    """An unmerged LoRA (task_type CAUSAL_LM, as swift saves them) on a local base folder, through the registry /
    engine path: Engine._load_vlm on the adapter folder (adapter_config.json, no config.json, no processor; the base is
    named relative to CHECKPOINTS_DIR and supplies the processor). score() / generate / the LM-head hook run on the
    PeftModel and equal a direct peft load. Needs peft ($VLM_PYDEPS is added to the path when it holds peft)."""
    if PYDEPS and Path(PYDEPS, "peft").exists() and PYDEPS not in sys.path:
        sys.path.append(PYDEPS)
    try:
        import peft
    except ImportError:
        return "skipped (peft not importable)"
    base = tiny_qwen_folder(tmp / "tiny_base", seed=3)
    adapter = tmp / "tiny_adapter"
    with quiet():
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16)
        lora_cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False,
                                   task_type="CAUSAL_LM")
        torch.manual_seed(5)
        peft.get_peft_model(model, lora_cfg).save_pretrained(str(adapter))
    assert (adapter / "adapter_config.json").exists() and not (adapter / "config.json").exists()
    assert not (adapter / "preprocessor_config.json").exists()
    direct = peft.PeftModel.from_pretrained(Qwen3VLForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16),
                                            str(adapter)).eval()
    tok = tiny_tokenizer()
    adapter_cfg = dataclasses.replace(cfg, base_model="tiny_base")
    original = backend_config.CHECKPOINTS_DIR
    backend_config.CHECKPOINTS_DIR = tmp
    try:
        spec = dataclasses.replace(get_spec("veritaspp"), checkpoint="tiny_adapter", vlm=adapter_cfg)
        engine = Engine(device="cpu")
        with quiet():
            vlm = engine._get_model(spec)
        assert isinstance(vlm.hf, peft.PeftModelForCausalLM), type(vlm.hf)
        assert any(n.endswith("lora_A.default") for n, _ in vlm.hf.named_modules())          # unmerged
        assert "missing=0, unexpected=0" in engine._status[spec.key]
        inputs = vlm.inputs(vlm.image(image))
        n_prompt = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            assert torch.equal(vlm.hf(**inputs).logits, direct(**inputs).logits)
        # score() through the engine with a forced answer == a direct peft generate (tokens, gap); hook == output_logits
        script = tok.convert_tokens_to_ids(["y", "<", "answer", ">", "real", "</", "answer", ">"])
        forcing = LogitsProcessorList([ForceTokens(script, tok.convert_tokens_to_ids("<|im_end|>"), n_prompt)])
        vlm.cfg = dataclasses.replace(adapter_cfg, decoding={**adapter_cfg.decoding, "logits_processor": forcing})
        with quiet():
            out = engine.vlm_score(spec, image)
            with torch.inference_mode(), LogitRecorder(direct.get_output_embeddings(), vlm.record_ids) as rec:
                ref = direct.generate(**inputs, **vlm.cfg.decoding, output_logits=True, return_dict_in_generate=True)
        real_id, fake_id = tok.convert_tokens_to_ids(["real", "fake"])
        assert out["generated_ids"] == ref.sequences[0, n_prompt:].tolist()
        assert (out["score_source"], out["verdict_step"], out["verdict"]) == ("verdict_token", 4, "real"), out
        assert out["logit"] == float(ref.logits[4][0, fake_id] - ref.logits[4][0, real_id])
        assert torch.equal(rec.stacked(), torch.stack([row[0, vlm.record_ids] for row in ref.logits]).cpu())
        # free greedy generation on the PeftModel through the engine == direct peft generate
        vlm.cfg = adapter_cfg
        with quiet():
            free = engine.vlm_score(spec, image)
            with torch.inference_mode():
                ref = direct.generate(**inputs, **adapter_cfg.decoding)
        assert free["generated_ids"] == ref[0, n_prompt:].tolist()
        # a tampered adapter (an extra tensor) raises in the engine
        weights = adapter / "adapter_model.safetensors"
        tensors = {k: v.clone() for k, v in load_file(str(weights)).items()}
        first = sorted(tensors)[0]
        tensors[first.replace("lora_A", "lora_Z")] = tensors[first].clone()
        save_file(tensors, str(weights))
        try:
            with quiet():
                Engine(device="cpu")._get_model(spec)
        except RuntimeError as e:
            assert "LoRA adapter mismatch" in str(e), e
        else:
            raise AssertionError("a tampered adapter was accepted")
        # the adapter folder without base_model in the config: refused (no Hub fetch of the base)
        try:
            SynthScopeVLM(adapter, cfg, device="cpu", root=tmp)
        except (RuntimeError, FileNotFoundError):
            pass
        else:
            raise AssertionError("an adapter folder loaded without base_model")
    finally:
        backend_config.CHECKPOINTS_DIR = original
    return (f"CAUSAL_LM adapter via the engine (adapter_config.json only, base + processor from CHECKPOINTS_DIR), "
            f"unmerged, forward / score / generate / hook == direct peft (peft {peft.__version__}), tampered adapter raises")


class LlavaStyleFamily(VLMFamily):
    """A second family for the test: everything from the base class (processor ``text=[...], images=[...]``, input
    size from ``pixel_values``); no ``image_grid_thw``."""


def test_llava_family() -> None:
    print("[G] a dummy LLaVA-style family (SentencePiece tokenizer, no grid / chat template / marker, single pass) ...")
    base = VLMConfig(family=LlavaStyleFamily(), model_class="LlavaForConditionalGeneration", system_prompt="",
                     user_prompt="<image>Does the image looks real/fake ?", parser=parse_veritas_v1, chat_template=False,
                     answer_marker=None, verdict_mode="teacher_forced", teacher_forced_suffix=" This is a",
                     verdict_pairs=(("fake", "real"),), image_loader=vllm_rgb,
                     decoding={"do_sample": False, "max_new_tokens": 6})
    image = rgb_image(50, 40)
    sp = sentencepiece_tokenizer()
    tokens = sp.convert_ids_to_tokens
    assert tokens(sp.encode(" real", add_special_tokens=False)) == ["▁", "▁real"]          # the Llama split
    assert tokens(sp.encode("This is a fake", add_special_tokens=False)) == ["▁This", "▁is", "▁a", "▁fake"]
    with tempfile.TemporaryDirectory() as tmp:
        folder = tiny_llava_folder(Path(tmp) / "fakevlm_like")
        cases = []
        # (config, expected forced prefix): the family default encodes " fake" / " real" -> ["▁", "▁fake"] /
        # ["▁", "▁real"], so "▁" is teacher-forced and the score is read at the next token; explicit token sequences
        # (["▁fake"] / ["▁real"]) are read right after the suffix
        for cfg, prefix in ((base, ["▁"]), (dataclasses.replace(base, verdict_sequences=(("▁fake",), ("▁real",))), [])):
            with quiet():
                vlm = SynthScopeVLM(folder, cfg, device="cpu")
                out = vlm.score(image)
                inputs = vlm.processor(text=[cfg.user_prompt + cfg.teacher_forced_suffix], images=[image],
                                       return_tensors="pt")
                ids = torch.cat([inputs["input_ids"], torch.tensor([sp.convert_tokens_to_ids(prefix)], dtype=torch.long)
                                 .reshape(1, -1)], dim=1)
                with torch.inference_mode():
                    row = vlm.hf(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 pixel_values=inputs["pixel_values"]).logits[0, -1].float()
            f, r = sp.convert_tokens_to_ids(["▁fake", "▁real"])
            assert vlm.prompt == cfg.user_prompt and "image_grid_thw" not in inputs
            assert out["forced_prefix"] == prefix and out["pair_tokens"] == ["▁fake", "▁real"], out
            assert out["score_source"] == "teacher_forced" and out["input_size"] == [32, 32]
            assert math.isclose(out["logit"], float(row[f] - row[r]), abs_tol=1e-4), (out["logit"], float(row[f] - row[r]))
            assert out["verdict"] == ("fake" if out["logit"] > 0 else "real")
            cases.append(out["logit"])
        assert cases[0] != cases[1]                          # the forced "▁" changes the context
        vlm.cfg = dataclasses.replace(base, verdict_mode="generate")
        with quiet():
            gen = vlm.score(image)
        assert gen["new_tokens"] > 0 and gen["score_source"] in ("verdict_token", "unparsed")
        assert gen["rationale"] == gen["response"] and gen["input_size"] == [32, 32]
        assert vlm.pair_map[sp.convert_tokens_to_ids("▁real")] == (f, r)
        # a multi-word marker on a SentencePiece tokenizer (the review's finding 1): the forced answer "This is a fake"
        # is found after "This is a" at step 3 (the per-token decode would lose the spaces and report unparsed)
        n_prompt = vlm.inputs(vlm.image(image))["input_ids"].shape[1]
        script = sp.convert_tokens_to_ids(["▁This", "▁is", "▁a", "▁fake"])
        forcing = ForceTokens(script, sp.convert_tokens_to_ids("</s>"), n_prompt)
        vlm.cfg = dataclasses.replace(base, verdict_mode="generate", answer_marker="This is a",
                                      decoding={"do_sample": False, "max_new_tokens": 8,
                                                "logits_processor": LogitsProcessorList([forcing])})
        with quiet():
            gen = vlm.score(image)
        assert gen["generated_ids"][:4] == script and gen["response"].startswith("This is a fake"), gen["response"]
        assert (gen["score_source"], gen["verdict_step"], gen["verdict_token"]) == ("verdict_token", 3, "▁fake"), gen
    print("    Llama-style split reproduced; teacher-forced score at the first differing token (\"▁\" forced by default, "
          "none with explicit sequences) == direct forward; input size from pixel_values; generate without a marker")


def test_released_weights() -> None:
    if not (WEIGHTS / "model.safetensors.index.json").exists() or not torch.cuda.is_available():
        print("[H] released weights + GPU not available here: skipped")
        return
    print("[H] released weights: strict engine load and one image ...")
    spec = get_spec("veritaspp")
    engine = Engine(device="cuda")
    with quiet():
        model = engine._get_model(spec)
    n = len(json.loads((WEIGHTS / "model.safetensors.index.json").read_text())["weight_map"])
    assert n == 750 and len(model.hf.state_dict()) == n, (n, len(model.hf.state_dict()))
    gate = Path("/ssdstore/ssalehi/SynthScope/GenImage_val_full/test/adm/1_fake/0_adm_153.PNG")
    image = media.decode_image_raw(gate.read_bytes()) if gate.exists() else Image.new("RGB", (256, 256), (90, 120, 200))
    with quiet():
        out = engine.vlm_score(spec, image)
    assert out["verdict"] in ("real", "fake") and out["score_source"] == "verdict_token"
    assert (out["logit"] > 0) == (out["verdict"] == "fake")
    print(f"    750 tensors strict; {gate.name if gate.exists() else 'synthetic'}: {out['verdict']} "
          f"(P(fake) {out['fake_probability']:.6f}, {out['new_tokens']} tokens)")


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    logging.getLogger("DeepSpeed").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Can't initialize NVML")
    test_parser()
    test_prompt()
    test_score_math()
    test_image_loader()
    test_registry_and_guards()
    test_tiny_qwen()
    test_llava_family()
    test_released_weights()
    print("veritaspp smoke test passed")
