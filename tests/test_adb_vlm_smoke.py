"""Smoke test for Align, Don't Blend's VLM branch (``adb_vlm``) on the shared VLM kind; CPU, no downloads, no released
files needed.

Checks:
[A] the verdict parser = the official ``parse_vlm_verdict`` (``detector/core.py``; True -> "fake", False -> "real", a
    ValueError -> None): with the official clone ($ADB_OFFICIAL_DIR, default SynthScope_shared/vlm_ref/Align-Dont-Blend)
    the official function itself is run on every case (skipped otherwise); ``adb_verdict_offset`` points at the verdict
    word exactly when the parser read a tag-less final line;
[B] the prompts equal ``VLM_SYSTEM_PROMPT`` / ``VLM_USER_PROMPT`` of the official file (the user prompt minus its
    "<image>" tag; skipped without the clone); ``VLMFamily.prompt`` sends [system, user: [image, text]]; with the
    released base folder ($SYNTHSCOPE_CKPT_DIR/qwen2.5-vl-7b-instruct) the rendered request equals the gate check's
    (skipped otherwise);
[C] image handling = ms-swift's: ``swift_rgb`` (no EXIF rotation, alpha dropped) and ``swift_qwen_vl_resize`` against
    ms-swift 4.4.2's own ``load_image`` / ``rescale_image`` (compiled from $SWIFT_DIR/swift/template/vision_utils.py) and
    qwen-vl-utils' own ``fetch_image`` with swift's IMAGE_MAX_TOKEN_NUM patch, pixel for pixel, on images below the
    minimum, between, above the 1,003,520-pixel cap and with a 7:1 aspect ratio (skipped where a package is missing);
[D] the score words on a stub byte-level vocabulary: "ĠAI" <-> "Ġauthentic" pairs, an "AI" in the rationale (before
    "<answer>") is not the verdict; ``offset_step``: the token holding the parser's offset (the final line's "AI", not the
    rationale's), None where the text does not match or the token is no verdict word;
[E] the registry spec, the decoding (greedy, 512 tokens, repetition penalty 1.05 = official), the input note, the zoo
    block list, the LICENSES row, the Space allow-list, the download hint naming both the adapter and the base, a missing
    base folder (not available; the engine names it);
[F] a tiny random Qwen2.5-VL (patch 14, merge 2) + a tiny local processor + a tiny LoRA (no released files) through the
    engine (adapter folder ``adb_vlm`` + base folder ``qwen2.5-vl-7b-instruct`` in a temp $SYNTHSCOPE_CKPT_DIR): no
    socket connection while loading (the adapter config names the base by its Hub id, as the release does), unmerged,
    forward == a direct peft load; the preprocess hook (input size, prompt tokens and pixel values == the
    processor on the swift-resized image with ``do_resize=False``); a forced answer: the verdict step after "<answer>",
    logit == the RAW ``output_logits`` gap (repetition penalty 1.05 active; the penalised ``output_scores`` gap differs and
    is not used), parsed verdict; forced TAG-LESS answers (no ``<answer>``, the final line "This is an ... image."): the
    score is read at the final line's verdict word ("verdict_word"), not the rationale's "AI"; a parsed verdict whose word
    is not locatable is "unscored" (P 0.5, logit 0); free greedy tokens == a direct peft generate; the key-rename hook: an adapter saved under
    the pre-4.52 key layout loads and passes the strict check (and fails it without the hook), an unknown key still
    fails, a pre-4.52 vision-tower LoRA (which peft leaves unloaded) fails;
[G] with the released adapter + base and a GPU: the engine loads strictly and scores one image (skipped otherwise).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.util
import io
import math
import json
import os
import re
import socket
import sys
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import TypeVar, Union

import numpy as np
import peft
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, models as tok_models, pre_tokenizers, trainers
from transformers import (LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerFast, Qwen2_5_VLConfig,
                          Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2VLImageProcessor)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from interface.backend import config as backend_config                              # noqa: E402
from interface.backend import media                                                 # noqa: E402
from interface.backend.engine import Engine                                         # noqa: E402
from interface.backend.registry import get_spec, input_note                         # noqa: E402
from models.image.adb_vlm.adb_vlm import (ADB_VLM, GREEDY_DECODING, MAX_PIXELS, SYSTEM_PROMPT, USER_PROMPT,   # noqa: E402
                                          USER_PROMPT_OFFICIAL)
from models.vlm.image import QWEN_VL_MIN_PIXELS, swift_qwen_vl_resize, swift_rgb   # noqa: E402
from models.vlm.parsers import adb_verdict_offset, parse_adb                        # noqa: E402
from models.vlm.vlm import (QWEN_VL, LogitRecorder, ProcessedLogitRecorder, SynthScopeVLM, load_lora_unmerged,   # noqa: E402
                            locate_verdict, offset_step, token_pieces, verdict_pair_map, verdict_score)
from synthscope import hub                                                          # noqa: E402
from synthscope.api import Detector                                                 # noqa: E402

SHARED = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
ADB_DIR = Path(os.environ.get("ADB_OFFICIAL_DIR", SHARED / "vlm_ref" / "Align-Dont-Blend"))
SWIFT_DIR = Path(os.environ.get("SWIFT_DIR", SHARED / "vlm_pydeps" / "swift442"))
BASE_FOLDER = backend_config.CHECKPOINTS_DIR / ADB_VLM.base_model
ADAPTER_FOLDER = backend_config.CHECKPOINTS_DIR / "adb_vlm"
# The request the 2026-11-16 gate check sent (vlm_gate/out/adb_score_unmerged_pil_rep1.05.json, "prompt_text").
GATE_PROMPT = ("<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>"
               "<|vision_end|>\nWhether the image is authentic or AI-generated?<|im_end|>\n<|im_start|>assistant\n")
PARSER_CASES = {
    "<rationale>\nx\n</rationale>\n<answer>\nThis is an AI-generated image.\n</answer>": "fake",
    "<rationale>\nx\n</rationale>\n<answer>\nThis is an authentic image.\n</answer>": "real",
    "<answer>  this IS an   ai-generated image. </answer>": "fake",          # whitespace / case normalised
    "<rationale>x</rationale>\nThis is an authentic image.": "real",       # no tag: the final sentence
    "<rationale>x</rationale>\nThis is an AI-generated image.\n</answer>": "fake",
    "<answer>\nThis is an AI-generated image\n</answer>": None,             # no full stop
    "<answer>This is an authentic image.</answer><answer>This is an authentic image.</answer>": None,   # two tags
    "<answer>\nThis is a fake image.\n</answer>": None,
    "<rationale>\nThe image looks AI-generated.\n</rationale>\n<answer>\nThis is an AI-generated": None,   # truncated
    "no answer at all": None,
}
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
           "<|video_pad|>"]
ANSWER_FAKE = "<rationale>\nThe AI images look smooth.\n</rationale>\n<answer>\nThis is an AI-generated image.\n</answer>"
ANSWER_REAL = "<rationale>\nThe AI hint is absent.\n</rationale>\n<answer>\nThis is an authentic image.\n</answer>"
# about 2% of the released adapter's outputs: no tags, the verdict as a bare final line (the parser accepts it)
TAGLESS_FAKE = "The AI images look smooth.\n\nThis is an AI-generated image."
TAGLESS_REAL = "The AI hint is absent.\n\nThis is an authentic image."
QWEN_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content is string %}{{ m.content }}{% else %}"
                 "{% for c in m.content %}{% if c.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
                 "{% else %}{{ c.text }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]   # ADB's: the LM only


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


class NoNetwork:
    """Context manager: count (and refuse) every socket connection attempt; libraries that swallow the error still
    leave a count."""

    def __enter__(self):
        self.attempts, self._connect = [], socket.socket.connect

        def refuse(sock, address, *args, **kwargs):
            self.attempts.append(address)
            raise OSError("network access refused by the test")

        socket.socket.connect = refuse
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._connect


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


def official_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_swift_image_functions(path: Path) -> dict:
    """ms-swift's ``load_file`` / ``load_image`` / ``rescale_image``, compiled from ``vision_utils.py`` (importing the
    module pulls in all of swift)."""
    tree = ast.parse(path.read_text())
    wanted = {"load_file", "load_image", "rescale_image"}
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {"BytesIO": BytesIO, "Image": Image, "math": math, "os": os, "Union": Union, "_T": TypeVar("_T")}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def adb_tokenizer() -> PreTrainedTokenizerFast:
    """A small byte-level BPE tokenizer (Qwen style: "Ġ" is a leading space) trained on the official prompts and answers,
    so that "ĠAI" and "Ġauthentic" are single tokens as in Qwen2.5's vocabulary."""
    backend = Tokenizer(tok_models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    corpus = [SYSTEM_PROMPT, USER_PROMPT, ANSWER_FAKE, ANSWER_REAL] * 40
    backend.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=700, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                            special_tokens=SPECIAL))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<|endoftext|>", eos_token="<|im_end|>",
                                  additional_special_tokens=SPECIAL[1:])
    vocab = tok.get_vocab()
    assert "ĠAI" in vocab and "Ġauthentic" in vocab, "the toy vocabulary lacks the verdict words"
    return tok


def tiny_qwen25_folder(folder: Path, seed: int = 0) -> Path:
    """A tiny random Qwen2.5-VL (patch 14, merge 2, as the real one) + a tiny local Qwen2.5-VL processor."""
    tok = adb_tokenizer()
    ids = tok.convert_tokens_to_ids
    processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(size={"shortest_edge": 3136, "longest_edge": 12845056}, patch_size=14,
                                              merge_size=2, temporal_patch_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3),
        tokenizer=tok, video_processor=Qwen2VLVideoProcessor(patch_size=14, merge_size=2, temporal_patch_size=2),
        chat_template=QWEN_TEMPLATE)
    config = Qwen2_5_VLConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 1, "vocab_size": len(tok), "max_position_embeddings": 4096,
                     "rope_scaling": {"type": "mrope", "mrope_section": [2, 2, 4]}, "pad_token_id": ids("<|endoftext|>"),
                     "eos_token_id": ids("<|im_end|>"), "tie_word_embeddings": False},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 32,
                       "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "window_size": 56,
                       "fullatt_block_indexes": [1]},
        image_token_id=ids("<|image_pad|>"), video_token_id=ids("<|video_pad|>"),
        vision_start_token_id=ids("<|vision_start|>"), vision_end_token_id=ids("<|vision_end|>"), tie_word_embeddings=False)
    torch.manual_seed(seed)
    with quiet():
        model = Qwen2_5_VLForConditionalGeneration(config).to(torch.bfloat16)
        model.generation_config.repetition_penalty = 1.05       # as the released base's generation_config.json
        model.save_pretrained(folder)
        processor.save_pretrained(folder)
    return folder


def old_layout(key: str) -> str:
    """A current Qwen2.5-VL adapter key in the layout before transformers 4.52 (the inverse of the family's rename)."""
    return (key.replace("base_model.model.model.language_model.", "base_model.model.model.")
               .replace("base_model.model.model.visual.", "base_model.model.visual."))


def rgb_image(w: int = 300, h: int = 200, seed: int = 0) -> Image.Image:
    return Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8"))


def expect_raise(fn, exc, text: str) -> str:
    try:
        fn()
    except exc as e:
        assert text in str(e), e
        return str(e)
    raise AssertionError(f"no {exc.__name__} ({text})")


def test_parser() -> None:
    print("[A] parser = the official parse_vlm_verdict ...")
    for text, expected in PARSER_CASES.items():
        assert parse_adb(text) == expected, (text, parse_adb(text))
        # the verdict-offset hook: set exactly when the parser read a tag-less final line, at its verdict word
        offset = adb_verdict_offset(text)
        if expected is not None and re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL) is None:
            word = "ai-generated" if expected == "fake" else "authentic"
            assert offset is not None and text[offset:].lower().startswith(word), (text, offset)
        else:
            assert offset is None, text
    for text in (TAGLESS_FAKE, TAGLESS_REAL):
        offset = adb_verdict_offset(text)
        assert offset == text.rindex("This is an ") + len("This is an ") and parse_adb(text) in ("fake", "real")
    assert adb_verdict_offset("<answer>\nThis is an authentic image.\n</answer>") is None      # the tag: marker search
    core = ADB_DIR / "detector" / "core.py"
    if not core.exists():
        print(f"    {len(PARSER_CASES)} cases; official detector/core.py not found, comparison skipped")
        return
    official = official_module(core, "adb_official_core")
    for text in PARSER_CASES:
        try:
            want = "fake" if official.parse_vlm_verdict(text) else "real"
        except ValueError:
            want = None
        assert want == parse_adb(text), (text, want)
    print(f"    {len(PARSER_CASES)} cases, identical to the official parse_vlm_verdict (ValueError -> unparsed)")


class StubProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


def test_prompt() -> None:
    print("[B] official prompt and chat request ...")
    core = ADB_DIR / "detector" / "core.py"
    if core.exists():
        consts = {n.targets[0].id: n.value.value for n in ast.parse(core.read_text()).body
                  if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
        assert consts["VLM_SYSTEM_PROMPT"] == SYSTEM_PROMPT and consts["VLM_USER_PROMPT"] == USER_PROMPT_OFFICIAL
        print("    system / user prompts identical to the official detector/core.py")
    else:
        print("    official detector/core.py not found, prompt comparison skipped")
    assert USER_PROMPT_OFFICIAL == "<image>" + USER_PROMPT and USER_PROMPT == "\nWhether the image is authentic or AI-generated?"
    assert ADB_VLM.system_prompt == SYSTEM_PROMPT and ADB_VLM.user_prompt == USER_PROMPT
    assert "“This is an authentic image.”" in SYSTEM_PROMPT and "•" in SYSTEM_PROMPT     # curly quotes, bullets kept
    stub = StubProcessor()
    assert QWEN_VL.prompt(stub, ADB_VLM) == "rendered"
    messages, kwargs = stub.calls[0]
    assert messages == [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}, kwargs
    if (BASE_FOLDER / "tokenizer_config.json").exists():
        with quiet():
            processor = transformers.AutoProcessor.from_pretrained(str(BASE_FOLDER), local_files_only=True)
        assert QWEN_VL.prompt(processor, ADB_VLM) == GATE_PROMPT
        print("    rendered request == the gate check's request (released base processor)")
    else:
        print("    released base processor not found here, rendered-request check skipped (tiny processor: [F])")


def test_image_handling() -> None:
    print("[C] image loading and resize = ms-swift + qwen-vl-utils ...")
    rng = np.random.RandomState(0)
    rgba = Image.fromarray((rng.rand(40, 50, 4) * 255).astype("uint8"), "RGBA")
    assert np.array_equal(np.asarray(swift_rgb(rgba)), np.asarray(rgba)[..., :3])     # alpha dropped, not composited
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    rgb_image(53, 37).save(buf, "JPEG", exif=exif.tobytes())
    assert swift_rgb(media.decode_image_raw(buf.getvalue())).size == (53, 37)       # no EXIF rotation
    files = {}
    for name, img, fmt in (("small_rgb", rgb_image(40, 30, 1), "PNG"),               # below min_pixels: upscaled
                           ("gate_adm_like", rgb_image(256, 256, 2), "PNG"),           # 256 -> 252
                           ("mid_jpeg", rgb_image(504, 364, 3), "JPEG"),
                           ("above_cap", rgb_image(1024, 1024, 4), "PNG"),             # rescale_image, then fetch_image
                           ("wide_above_cap", rgb_image(2800, 400, 5), "JPEG"),        # 7:1
                           ("rgba", Image.fromarray((rng.rand(300, 220, 4) * 255).astype("uint8"), "RGBA"), "PNG"),
                           ("gray", Image.fromarray((rng.rand(90, 120) * 255).astype("uint8"), "L"), "PNG"),
                           ("exif_rot", rgb_image(53, 37, 6), "JPEG")):
        b = io.BytesIO()
        img.save(b, fmt, **({"exif": exif.tobytes()} if name == "exif_rot" else {}))
        files[name] = b.getvalue()
    ours = {n: swift_qwen_vl_resize(swift_rgb(media.decode_image_raw(d)), MAX_PIXELS) for n, d in files.items()}
    for n, im in ours.items():
        w, h = im.size
        assert w % 28 == 0 and h % 28 == 0 and QWEN_VL_MIN_PIXELS <= w * h <= MAX_PIXELS, (n, im.size)
        assert ADB_VLM.preprocess(swift_rgb(media.decode_image_raw(files[n])), **ADB_VLM.preprocess_kwargs).tobytes() == im.tobytes()
    assert ours["gate_adm_like"].size == (252, 252)
    vision_utils = SWIFT_DIR / "swift" / "template" / "vision_utils.py"
    try:
        from qwen_vl_utils import vision_process
    except ImportError:
        vision_process = None
    if not vision_utils.exists() or vision_process is None:
        print("    ms-swift source or qwen-vl-utils not found: official comparison skipped")
        return
    swift = official_swift_image_functions(vision_utils)
    saved = vision_process.IMAGE_MAX_TOKEN_NUM
    vision_process.IMAGE_MAX_TOKEN_NUM = MAX_PIXELS // 28 ** 2       # swift's compat_qwen_vl_utils + patch (MAX_PIXELS env)
    try:
        for n, data in files.items():
            img = swift["rescale_image"](swift["load_image"](data), MAX_PIXELS)     # Template._preprocess_inputs
            official = vision_process.fetch_image({"image": img})                  # Qwen2VLTemplate.replace_tag
            assert official.size == ours[n].size and official.tobytes() == ours[n].tobytes(), (n, official.size, ours[n].size)
    finally:
        vision_process.IMAGE_MAX_TOKEN_NUM = saved
    print(f"    {len(files)} files (RGBA, L, EXIF, below / above the cap, 7:1): pixel-identical to ms-swift 4.4.2 "
          f"load_image + rescale_image and qwen-vl-utils fetch_image; alpha dropped, no EXIF rotation")


class StubTokenizer:
    """A byte-level-BPE-like vocabulary: 'Ġ' is a leading space."""
    VOCAB = {"<": 0, "answer": 1, ">": 2, "ĠAI": 3, "Ġauthentic": 4, "AI": 5, "authentic": 6, "ĠAi": 7, "ĠAuthentic": 8,
             "Ġai": 9, "x": 10, "Ġ</": 11, "-generated": 12, "ĠAIR": 13}

    def get_vocab(self):
        return dict(self.VOCAB)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).replace("Ġ", " ")

    def decode(self, ids):
        inv = {v: k for k, v in self.VOCAB.items()}
        return self.convert_tokens_to_string([inv[i] for i in ids])


def test_score_words() -> None:
    print("[D] verdict words AI / authentic ...")
    tok = StubTokenizer()
    pair_map = verdict_pair_map(tok, ADB_VLM.verdict_pairs)
    assert pair_map == {3: (3, 4), 4: (3, 4), 5: (5, 6), 6: (5, 6), 7: (7, 8), 8: (7, 8), 9: (9, 4)}, pair_map
    assert 13 not in pair_map                                 # "ĠAIR" is not a verdict word
    ids = sorted({i for p in pair_map.values() for i in p})
    column = {t: c for c, t in enumerate(ids)}
    torch.manual_seed(0)
    rows = torch.randn(12, len(ids))
    gen = [3, 12, 10, 0, 1, 2, 10, 3, 12, 11, 1, 2]           # " AI-generated x<answer>x AI-generated </answer>"
    pieces = [tok.decode([t]) for t in gen]
    assert locate_verdict(gen, pieces, "<answer>", pair_map, "</answer>") == (6, 7)   # not the rationale's "AI" (step 0)
    out = verdict_score(gen, pieces, rows[:len(gen)], column, "<answer>", pair_map, "</answer>")
    assert out["verdict_step"] == 7 and out["pair_ids"] == (3, 4)
    assert math.isclose(out["logit"], float(rows[7, column[3]] - rows[7, column[4]]))
    # the parser's word outside <answer> (a tag-less final line): the token that holds its character offset
    gen = [3, 12, 10, 3, 12]                                  # " AI-generated x AI-generated"
    pieces = token_pieces(tok, gen)
    text = "".join(pieces)
    assert offset_step(gen, pieces, text, text.rindex("AI"), pair_map) == 3                 # not the first "AI" (step 0)
    assert offset_step(gen, pieces, text, text.index("x"), pair_map) is None                # no verdict word there
    assert offset_step(gen, pieces, "other" + text[5:], text.rindex("AI"), pair_map) is None   # text differs
    assert offset_step(gen, pieces, text, len(text) + 3, pair_map) is None                  # past the end
    out = verdict_score(gen, pieces, rows[:len(gen)], column, "<answer>", pair_map, "</answer>", offset=text.rindex("AI"),
                        text=text)
    assert (out["score_source"], out["verdict_step"], out["pair_ids"]) == ("verdict_word", 3, (3, 4))
    assert math.isclose(out["logit"], float(rows[3, column[3]] - rows[3, column[4]]))
    out = verdict_score(gen, pieces, rows[:len(gen)], column, "<answer>", pair_map, "</answer>")   # no marker, no offset
    assert (out["score_source"], out["logit"], out["fake_probability"]) == ("unparsed", 0.0, 0.5)
    print("    ĠAI <-> Ġauthentic (and case variants); the verdict is the first one after <answer>, or the token at the "
          "parser's offset")


def test_registry() -> None:
    print("[E] registry spec, decoding, note, block list, licence, Space, download hint ...")
    spec = get_spec("adb_vlm")
    assert (spec.predict_kind, spec.ckpt_kind, spec.checkpoint, spec.weights, spec.modality) == \
        ("vlm", "official_vlm", "adb_vlm", "official", "image")
    assert spec.transform is None and spec.overrides == [] and spec.vlm is ADB_VLM and spec.eval_batch_size(32) == 1
    assert ADB_VLM.family is QWEN_VL and ADB_VLM.model_class == "Qwen2_5_VLForConditionalGeneration"
    assert (ADB_VLM.answer_marker, ADB_VLM.answer_end, ADB_VLM.verdict_mode, ADB_VLM.logits) == \
        ("<answer>", "</answer>", "generate", "raw")
    assert ADB_VLM.decoding == ADB_VLM.official_decoding == GREEDY_DECODING == \
        {"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.05}
    assert ADB_VLM.image_loader is swift_rgb and ADB_VLM.preprocess is swift_qwen_vl_resize and ADB_VLM.parser is parse_adb
    assert ADB_VLM.verdict_offset is adb_verdict_offset
    assert MAX_PIXELS == 1003520 and ADB_VLM.max_pixels is None and ADB_VLM.preprocess_kwargs == {"max_pixels": MAX_PIXELS}
    assert ADB_VLM.processor_kwargs == {"images_kwargs": {"do_resize": False}}
    assert ADB_VLM.base_model == "qwen2.5-vl-7b-instruct" and ADB_VLM.hf_repo == "Gaffeyzz/Align-Dont-Blend-VLM"
    assert ADB_VLM.hf_revision.startswith("49975ccd") and ADB_VLM.base_hf_revision.startswith("cc594898")
    assert "1.0 MP" in input_note(spec)["label"] and "EXIF" in input_note(spec)["explanation"]
    assert "adb_vlm" in hub.NEVER_REDISTRIBUTE and "no licence" in hub.NEVER_REDISTRIBUTE["adb_vlm"]
    row = [line for line in Path(ROOT, "LICENSES.md").read_text().splitlines() if line.startswith("| `adb_vlm` |")]
    assert len(row) == 1 and "no licence" in row[0] and "not in the model zoo" in row[0], row
    fetch = Path(ROOT, "deploy", "hf-space", "fetch_weights.py").read_text()
    default = ast.literal_eval(fetch.split("DEFAULT = ", 1)[1].split("]", 1)[0] + "]")
    assert "adb_vlm" not in default and "veritaspp" not in default
    with tempfile.TemporaryDirectory() as tmp:
        original = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = Path(tmp)
        try:
            det = Detector(spec, device="cpu", auto_download=True)
            assert not det.available
            msg = expect_raise(det._ensure_weights, FileNotFoundError, "hf download Gaffeyzz/Align-Dont-Blend-VLM --revision 49975ccd")
            assert f"hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 " \
                   f"--local-dir {Path(tmp) / 'qwen2.5-vl-7b-instruct'}" in msg, msg
            Path(tmp, "adb_vlm").mkdir()                       # the adapter alone: still not available (no base)
            Path(tmp, "adb_vlm", "adapter_config.json").write_text("{}")
            engine = Engine(device="cpu")
            assert not engine.is_available(spec)
            msg = expect_raise(lambda: engine._get_model(spec), FileNotFoundError, "no base model (config.json) in")
            assert "Qwen/Qwen2.5-VL-7B-Instruct" in msg
            Path(tmp, "qwen2.5-vl-7b-instruct").mkdir()
            assert engine.is_available(spec)
            expect_raise(lambda: Detector(spec, device="cpu").predict(Path(tmp) / "clip.mp4"), ValueError, "still images only")
        finally:
            backend_config.CHECKPOINTS_DIR = original
    print("    vlm spec, official greedy decoding (rep. penalty 1.05), note, NEVER_REDISTRIBUTE, LICENSES 'no licence', "
          "not on the Space, adapter + base download hint, missing base reported")


def test_tiny_lora() -> None:
    print("[F] tiny random Qwen2.5-VL + tiny LoRA through the engine: unmerged, preprocess, raw score, key rename ...")
    image = rgb_image(300, 200)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tiny_qwen25_folder(tmp / "qwen2.5-vl-7b-instruct", seed=3)
        adapter = tmp / "adb_vlm"
        with quiet():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16)
            # ADB's adapter_config: a regex on the language model's projections, r16 / alpha 32 (here r4 / alpha 8)
            lora_cfg = peft.LoraConfig(r=4, lora_alpha=8, init_lora_weights=False, task_type="CAUSAL_LM",
                                       target_modules=r"^(model.language_model.*\.(" + "|".join(LORA_TARGETS) + "))$")
            torch.manual_seed(5)
            peft.get_peft_model(model, lora_cfg).save_pretrained(str(adapter))
            direct = peft.PeftModel.from_pretrained(
                Qwen2_5_VLForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16), str(adapter)).eval()
        # cloned: load_file returns views of the file, which is rewritten below
        saved = {k: v.clone() for k, v in load_file(str(adapter / "adapter_model.safetensors")).items()}
        assert saved and all(k.startswith("base_model.model.model.language_model.layers.") for k in saved), sorted(saved)[:3]
        # the released adapter_config.json names the base by its Hub id: loading must not look it up
        ad_cfg = json.loads((adapter / "adapter_config.json").read_text())
        ad_cfg["base_model_name_or_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"
        (adapter / "adapter_config.json").write_text(json.dumps(ad_cfg))
        cfg = dataclasses.replace(ADB_VLM, decoding={**GREEDY_DECODING, "max_new_tokens": 24})
        spec = dataclasses.replace(get_spec("adb_vlm"), vlm=cfg)
        original = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = tmp
        hub_env = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}   # online mode
        try:
            engine = Engine(device="cpu")
            with quiet(), NoNetwork() as net:
                vlm = engine._get_model(spec)
            assert net.attempts == [], f"the engine load tried the network: {net.attempts}"
            assert isinstance(vlm.hf, peft.PeftModelForCausalLM)
            assert sum(n.endswith("lora_A.default") for n, _ in vlm.hf.named_modules()) == len(saved) // 2   # unmerged
            assert "missing=0, unexpected=0" in engine._status["adb_vlm"] and "LoRA adapter unmerged" in engine._status["adb_vlm"]
            tok = vlm.tokenizer
            assert vlm.prompt.startswith("<|im_start|>system\n" + SYSTEM_PROMPT) and vlm.prompt.endswith(
                "<|vision_start|><|image_pad|><|vision_end|>" + USER_PROMPT + "<|im_end|>\n<|im_start|>assistant\n")
            # the preprocess hook: the swift-resized image goes to the processor unresized (do_resize=False)
            pre = vlm.image(image)
            swift_img = swift_qwen_vl_resize(swift_rgb(image), MAX_PIXELS)
            assert pre.size == swift_img.size == (308, 196) and pre.tobytes() == swift_img.tobytes()
            inputs = vlm.inputs(pre)
            direct_in = vlm.processor(text=[vlm.prompt], images=[swift_img], return_tensors="pt",
                                      images_kwargs={"do_resize": False})
            assert torch.equal(inputs["pixel_values"], direct_in["pixel_values"])
            assert torch.equal(inputs["input_ids"], direct_in["input_ids"])
            default_in = vlm.processor(text=[vlm.prompt], images=[swift_img], return_tensors="pt")
            assert torch.equal(inputs["pixel_values"], default_in["pixel_values"])   # already a fixed point of the resize
            n_prompt = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                assert torch.equal(vlm.hf(**inputs).logits, direct(**inputs).logits)
            # a forced answer: verdict after <answer> (not the rationale's "AI"), RAW gap, parsed verdict
            for answer, verdict in ((ANSWER_FAKE, "fake"), (ANSWER_REAL, "real")):
                script = tok.encode(answer, add_special_tokens=False)
                forcing = LogitsProcessorList([ForceTokens(script, tok.convert_tokens_to_ids("<|im_end|>"), n_prompt)])
                vlm.cfg = dataclasses.replace(cfg, decoding={**cfg.decoding, "max_new_tokens": len(script) + 2,
                                                             "logits_processor": forcing})
                with quiet():
                    out = engine.vlm_score(spec, image)
                    rec_scores = ProcessedLogitRecorder(vlm.record_ids)
                    with torch.inference_mode(), LogitRecorder(direct.get_output_embeddings(), vlm.record_ids) as rec:
                        ref = direct.generate(**inputs, **{**vlm.cfg.decoding, "logits_processor": LogitsProcessorList(
                            [rec_scores, *forcing])}, output_logits=True, output_scores=True, return_dict_in_generate=True)
                gen = ref.sequences[0, n_prompt:].tolist()
                assert out["generated_ids"] == gen and out["response"] == answer, (out["response"], answer)
                pieces = token_pieces(tok, gen)
                marker_step = next(t for t in range(len(gen)) if "<answer>" in "".join(pieces[:t + 1]))
                ai_in_rationale = [t for t in range(marker_step) if gen[t] in vlm.pair_map]
                assert ai_in_rationale, "the rationale should mention AI before the answer"
                step = out["verdict_step"]
                assert step > marker_step and out["verdict"] == verdict and out["score_source"] == "verdict_token"
                assert out["verdict_token"] == ("ĠAI" if verdict == "fake" else "Ġauthentic")
                f, r = tok.convert_tokens_to_ids(["ĠAI", "Ġauthentic"])
                raw_gap = float(ref.logits[step][0, f].float() - ref.logits[step][0, r].float())
                assert out["logit"] == raw_gap, (out["logit"], raw_gap)                 # RAW logits
                assert torch.equal(rec.stacked(), torch.stack([row[0, vlm.record_ids] for row in ref.logits]).cpu())
                # the processed (penalised) scores, before the forcing, differ: the score does not use them
                processed = rec_scores.stacked()[step]
                proc_gap = float(processed[vlm.column[f]] - processed[vlm.column[r]])
                assert proc_gap != raw_gap, "repetition penalty 1.05 should change the gap (AI / authentic are in the prompt)"
            # forced TAG-LESS answers: the parser reads the final line, the score is read at its verdict word; with the
            # offset pointing at a token that is no verdict word, the parsed verdict is "unscored" (P 0.5, logit 0)
            for answer, verdict, hook in ((TAGLESS_FAKE, "fake", ADB_VLM.verdict_offset),
                                          (TAGLESS_REAL, "real", ADB_VLM.verdict_offset),
                                          (TAGLESS_FAKE, "fake", lambda text: 0)):
                script = tok.encode(answer, add_special_tokens=False)
                forcing = LogitsProcessorList([ForceTokens(script, tok.convert_tokens_to_ids("<|im_end|>"), n_prompt)])
                vlm.cfg = dataclasses.replace(cfg, verdict_offset=hook, decoding={
                    **cfg.decoding, "max_new_tokens": len(script) + 2, "logits_processor": forcing})
                with quiet():
                    out = engine.vlm_score(spec, image)
                    with torch.inference_mode():
                        ref = direct.generate(**inputs, **{**vlm.cfg.decoding, "logits_processor": forcing},
                                              output_logits=True, return_dict_in_generate=True)
                gen = ref.sequences[0, n_prompt:].tolist()
                assert out["generated_ids"] == gen and out["response"] == answer and out["verdict"] == verdict
                if hook is ADB_VLM.verdict_offset:
                    word = "ĠAI" if verdict == "fake" else "Ġauthentic"
                    final = max(t for t in range(len(gen)) if tok.convert_ids_to_tokens(gen[t]) == word)
                    assert [tok.convert_ids_to_tokens(gen[t]) for t in range(len(gen))].count("ĠAI") >= 1 + (verdict == "fake")
                    assert (out["score_source"], out["verdict_step"], out["verdict_token"]) == ("verdict_word", final, word), out
                    assert out["logit"] == float(ref.logits[final][0, f].float() - ref.logits[final][0, r].float())
                else:
                    assert (out["score_source"], out["logit"], out["fake_probability"], out["verdict_step"]) == \
                        ("unscored", 0.0, 0.5, None), out
            # free greedy decoding (repetition penalty 1.05): the engine's tokens == a direct peft generate
            vlm.cfg = cfg
            with quiet():
                free = engine.vlm_score(spec, image)
                with torch.inference_mode():
                    ref = direct.generate(**inputs, **cfg.decoding)
            assert free["generated_ids"] == ref[0, n_prompt:].tolist() and free["input_size"] == [308, 196]
            res = engine.predict_image(spec, image)
            assert res["fake_probability"] == round(free["fake_probability"], 6) and res["input_size"] == [308, 196]
            # the key-rename hook: the same adapter saved in the pre-4.52 layout
            old = {old_layout(k): v for k, v in saved.items()}
            assert set(old) != set(saved) and all(QWEN_VL.adapter_key(k) == n for k, n in zip(old, saved))
            assert all(QWEN_VL.adapter_key(k) == k for k in saved)              # current keys unchanged
            save_file(old, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
            with quiet():
                vlm_old = Engine(device="cpu")._get_model(spec)
                with torch.inference_mode():
                    assert torch.equal(vlm_old.hf(**inputs).logits, direct(**inputs).logits)
            fresh = lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16)   # noqa: E731
            with quiet():
                expect_raise(lambda: load_lora_unmerged(fresh(), adapter, rename=None), RuntimeError, "LoRA adapter mismatch")
            # a genuinely unknown key still fails the strict check (with the hook)
            bad = dict(old)
            first = sorted(bad)[0]
            bad[first.replace("lora_A", "lora_Z")] = bad[first].clone()
            save_file(bad, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
            with quiet():
                expect_raise(lambda: Engine(device="cpu")._get_model(spec), RuntimeError, "LoRA adapter mismatch")
            # a pre-4.52 vision-tower LoRA: peft 0.21 does not move "visual." keys, the strict check refuses it
            vis_dir = tmp / "adb_vis"
            with quiet():
                vis_model = peft.get_peft_model(fresh(), peft.LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                                                         target_modules=["q_proj", "qkv"], task_type="CAUSAL_LM"))
                vis_model.save_pretrained(str(vis_dir))
            vis = {k: v.clone() for k, v in load_file(str(vis_dir / "adapter_model.safetensors")).items()}
            assert any(".visual." in k for k in vis)
            save_file({old_layout(k): v for k, v in vis.items()}, str(vis_dir / "adapter_model.safetensors"),
                      metadata={"format": "pt"})
            with quiet():
                expect_raise(lambda: load_lora_unmerged(fresh(), vis_dir, rename=QWEN_VL.adapter_key), RuntimeError,
                             "LoRA adapter mismatch")
            save_file(saved, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
        finally:
            backend_config.CHECKPOINTS_DIR = original
            os.environ.update({k: v for k, v in hub_env.items() if v is not None})
    print(f"    {len(saved)} LoRA tensors unmerged via the engine (no network access, base named by Hub id) == direct peft; preprocess -> 308x196, pixel values == "
          f"processor(do_resize=False); forced answers: verdict after <answer>, logit == RAW output_logits gap (penalised "
          f"gap differs); tag-less answers scored at the final line's word, an unlocatable word unscored; greedy tokens == peft generate; pre-4.52 keys load with the hook, fail without it; unknown key and "
          f"old vision-tower LoRA refused (peft {peft.__version__})")


def test_released_weights() -> None:
    if not ((ADAPTER_FOLDER / "adapter_config.json").exists() and (BASE_FOLDER / "config.json").exists()
            and torch.cuda.is_available()):
        print("[G] released adapter + base + GPU not available here: skipped")
        return
    print("[G] released weights: strict engine load and one image ...")
    spec = get_spec("adb_vlm")
    engine = Engine(device="cuda")
    with quiet():
        model = engine._get_model(spec)
    n = sum(1 for k in load_file(str(ADAPTER_FOLDER / "adapter_model.safetensors")))
    assert n == 392 and sum(m.endswith("lora_A.default") for m, _ in model.hf.named_modules()) == 196
    image = rgb_image(256, 256)
    with quiet():
        out = engine.vlm_score(spec, image)
    assert out["verdict"] in ("real", "fake") and out["score_source"] == "verdict_token"
    assert (out["logit"] > 0) == (out["verdict"] == "fake")
    print(f"    392 adapter tensors strict on the base; synthetic image: {out['verdict']} (P(fake) "
          f"{out['fake_probability']:.6f}, {out['new_tokens']} tokens)")


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    warnings.filterwarnings("ignore", message="Can't initialize NVML")
    test_parser()
    test_prompt()
    test_image_handling()
    test_score_words()
    test_registry()
    test_tiny_lora()
    test_released_weights()
    print("adb_vlm smoke test passed")
