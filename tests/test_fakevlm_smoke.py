"""Smoke test for FakeVLM (``fakevlm``) on the shared VLM kind; CPU, no downloads, no released files needed.

Checks:
[A] the verdict parser = the official rule: with the official clone ($FAKEVLM_OFFICIAL_DIR, default
    SynthScope_shared/vlm_ref/FakeVLM) the official ``scripts/eval.py`` ``validate`` itself is run on every case (a stub
    processor / model hands it the text; its random fallback is recorded and must be where ``parse_fakevlm`` says None;
    skipped otherwise);
[B] the prompt: the raw FakeClue question (equal to every row of the FakeClue test file when a copy is present), no chat
    template; the official decoding (greedy, 256 tokens); with the released processor folder
    ($SYNTHSCOPE_CKPT_DIR/llava-1.5-7b-hf_a272c74_processor) the verdict sequences are "▁▁ This is a" + "▁fake" / "▁real",
    the answer "  This is a fake image." tokenises that way, and the prompt has 576 image tokens and 1,599 positions (4.45.2's layout); with
    the released weights folder, its own tokenizer (add_eos_token=true) is refused (skipped otherwise);
[C] image handling: ``transformers_rgb`` (no EXIF rotation, alpha dropped) and ``clip_resize_center_crop`` + the processor
    with ``do_resize=False, do_center_crop=False`` == transformers 4.45.2's own CLIPImageProcessor (run in a subprocess
    from $TRANSFORMERS_4452, default SynthScope_shared/vlm_pydeps/transformers4452; skipped without it) on RGB, RGBA,
    L, P, small, tall and 8 MP images;
[D] the registry spec, the input note, the zoo block list, the LICENSES row, the Space allow-list, the download hint
    naming the weights AND the processor, a missing processor folder (not available; the engine names it);
[E] a tiny random LLaVA (CLIP ViT patch 16 + Llama) + a tiny local llava-style processor (SentencePiece-like
    tokenizer with "▁▁") through the engine: no socket connection while loading; the processor inputs (the image tokens,
    the left padding to max_length, the pixel values of the 4.45.2-style resize); the generated score (registry mode):
    a forced answer "This is a fake image." scored at its "▁fake" (step 4) == the raw ``output_logits`` gap, "This image
    is fake." at its own "▁fake"; the teacher-forced variant == a direct forward on the prompt + "▁▁ This is a", and
    equal to the generated gap when the answer starts with those tokens (float32); strict load: an extra or a
    missing tensor fails; a tokenizer that appends EOS and a processor without the expansion attributes are refused;
[F] with the released weights + processor and a GPU: strict engine load and one image (skipped otherwise).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, models as tok_models, normalizers, processors
from transformers import (CLIPImageProcessor, LlavaConfig, LlavaForConditionalGeneration, LogitsProcessor,
                          LogitsProcessorList, PreTrainedTokenizerFast)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from interface.backend import config as backend_config                              # noqa: E402
from interface.backend import media                                                 # noqa: E402
from interface.backend.engine import Engine                                         # noqa: E402
from interface.backend.registry import get_spec, input_note                         # noqa: E402
from models.image.fakevlm.fakevlm import (FAKEVLM, FAKEVLM_TEACHER_FORCED, GREEDY_DECODING, MAX_LENGTH, PROCESSOR,   # noqa: E402
                                          USER_PROMPT, VERDICT_SEQUENCES)
from models.vlm.image import clip_resize_center_crop, transformers_rgb              # noqa: E402
from models.vlm.parsers import parse_fakevlm                                        # noqa: E402
from models.vlm.vlm import LLAVA, SynthScopeVLM, VLMConfig                                     # noqa: E402
from synthscope import hub                                                          # noqa: E402
from synthscope.api import Detector                                                 # noqa: E402

SHARED = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
OFFICIAL_DIR = Path(os.environ.get("FAKEVLM_OFFICIAL_DIR", SHARED / "vlm_ref" / "FakeVLM"))
FAKECLUE_TEST = SHARED / "vlm_ref" / "_hfmeta" / "fakeclue_test.json"       # HF lingcco/FakeClue test.json (Apache-2.0)
TRANSFORMERS_4452 = Path(os.environ.get("TRANSFORMERS_4452", SHARED / "vlm_pydeps" / "transformers4452"))
WEIGHTS = backend_config.CHECKPOINTS_DIR / "fakevlm"
PROCESSOR_FOLDER = backend_config.CHECKPOINTS_DIR / PROCESSOR
PARSER_CASES = {
    "  This is a fake image. The fish has an unnatural texture.": "fake",
    "  This is a real image.Ъ": "real",
    "This image is fake. The edges are blurred.": "fake",
    " \"This is a real image. The lighting is natural.": "real",
    "  This is a fake image with unrealistic lighting.": "real",       # official quirk: 'real' in 'unrealistic'
    "  The image shows a cat. It is fake.": "fake",                     # second sentence
    "  The image shows a cat. It looks natural.": None,                 # neither word: official random choice
    "  Nothing here": None,                                             # no second sentence: official random choice
    "  Is it real? Yes, it is fake. Really.": "fake",                   # text after the last '?'
}
TINY_PIECES = ["<unk>", "<s>", "</s>", "<pad>", "<image>", "▁", "▁▁", "▁This", "▁is", "▁a", "▁fake", "▁real", "▁Does", "▁the", "▁image",
               "▁looks", "/", "f", "ake", "?", ".", "x"] + list("abcdefghijklmnopqrstuvwxyzTDI")
TINY_SIZE, TINY_PATCH = 32, 16                              # 2 x 2 patches + CLS: 4 image tokens with "default"


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


class StubTokens:
    """What the official ``validate`` reads from ``model.generate``'s output: a row index into the cases."""

    def __init__(self, index: int):
        self.shape = (1,)
        self.index = index

    def __getitem__(self, i):
        return self.index


def expect_raise(fn, exc, text: str) -> str:
    try:
        fn()
    except exc as e:
        assert text in str(e), (text, str(e))
        return str(e)
    raise AssertionError(f"expected {exc.__name__} containing {text!r}")


def rgb_image(w: int = 300, h: int = 200, seed: int = 0) -> Image.Image:
    return Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8"))


def official_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_verdict(mod, text: str, workdir: Path):
    """``validate`` of the official eval.py on one answer: "real" / "fake", or "random" where it calls random.choice."""
    picks = []
    mod.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(
        decode=lambda ids, skip_special_tokens: "<s> Does the image looks real/fake?" + text))
    mod.random = types.SimpleNamespace(choice=lambda xs: picks.append(xs) or 1)
    model = types.SimpleNamespace(generate=lambda **kw: StubTokens(0))
    inputs = {k: torch.zeros(1, 1, 4) for k in ("input_ids", "attention_mask", "pixel_values")}
    batch = (inputs, [torch.tensor([1])], [("x.png",)], [("fidelity",)])   # label real = 1
    stats = {}
    mod.calculate_results_acc = lambda results: stats.update(results) or {}
    args = types.SimpleNamespace(output_path=str(workdir / "out.json"))
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.validate(args, model, [batch])
    finally:
        os.chdir(cwd)
    if picks:
        return "random"
    return "real" if stats["fidelity"]["right"]["right_real"] == 1 else "fake"


def tiny_tokenizer(add_eos: bool = False) -> PreTrainedTokenizerFast:
    """A synthetic SentencePiece / Llama-style tokenizer (Unigram over "▁"-joined text, with a "▁▁" piece) that splits
    "  This is a fake image." as the LLaVA-1.5 tokenizer does (["▁▁", "▁This", "▁is", "▁a", "▁fake", ...]). (The real
    tokenizer is Llama 2-licensed, so it is not copied.)"""
    vocab = [(piece, -2.0 if piece in ("▁", "f", "ake") else -1.0) for piece in TINY_PIECES]
    backend = Tokenizer(tok_models.Unigram(vocab, unk_id=0, byte_fallback=False))
    backend.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    backend.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.Fuse(), decoders.Strip(" ", 1, 0)])
    if add_eos:
        backend.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                  pad_token="<pad>", additional_special_tokens=["<image>"], padding_side="left")
    return tok


def tiny_folders(root: Path, add_eos: bool = False, seed: int = 1) -> tuple:
    """(weights folder ``fakevlm``, processor folder) of a tiny random LLaVA in float32; the processor is saved without
    the expansion attributes, as the a272c74 processor is."""
    tok = tiny_tokenizer(add_eos)
    image_processor = CLIPImageProcessor(size={"shortest_edge": TINY_SIZE}, crop_size={"height": TINY_SIZE, "width": TINY_SIZE})
    config = LlavaConfig(
        vision_config={"model_type": "clip_vision_model", "hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": TINY_SIZE,
                       "patch_size": TINY_PATCH},
        text_config={"model_type": "llama", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": len(tok),
                     "eos_token_id": tok.convert_tokens_to_ids("</s>"), "bos_token_id": tok.convert_tokens_to_ids("<s>"),
                     "pad_token_id": tok.convert_tokens_to_ids("<pad>")},
        image_token_id=tok.convert_tokens_to_ids("<image>"), vision_feature_select_strategy="default",
        vision_feature_layer=-2)
    weights, proc = root / "fakevlm", root / PROCESSOR
    torch.manual_seed(seed)
    with quiet():
        LlavaForConditionalGeneration(config).save_pretrained(weights)
        # the a272c74 layout: config.json, preprocessor_config.json and the tokenizer files, no processor_config.json
        # (so no patch_size / vision_feature_select_strategy)
        config.save_pretrained(proc)
        image_processor.save_pretrained(proc)
        tok.save_pretrained(proc)
    for name in ("preprocessor_config.json", "tokenizer_config.json"):
        meta = json.loads((proc / name).read_text())
        (proc / name).write_text(json.dumps({**meta, "processor_class": "LlavaProcessor"}, indent=1))
    assert not (proc / "processor_config.json").exists()
    return weights, proc, tok


def tiny_config(**changes) -> VLMConfig:
    return dataclasses.replace(
        FAKEVLM, dtype="float32",
        preprocess_kwargs={"shortest_edge": TINY_SIZE, "crop_size": TINY_SIZE},
        processor_overrides={"patch_size": TINY_PATCH, "vision_feature_select_strategy": "default",
                             "num_additional_image_tokens": 1},
        processor_kwargs={"padding": "max_length", "max_length": 64, "truncation": True,
                          "images_kwargs": {"do_resize": False, "do_center_crop": False}}, **changes)


def test_parser() -> None:
    print("[A] parser vs the official eval.py validate ...")
    for text, want in PARSER_CASES.items():
        assert parse_fakevlm(text) == want, (text, parse_fakevlm(text), want)
    if not (OFFICIAL_DIR / "scripts" / "eval.py").exists():
        print("    official clone not found: the official function was not run (cases checked against the rule)")
        return
    mod = official_module(OFFICIAL_DIR / "scripts" / "eval.py", "fakevlm_official_eval")
    with tempfile.TemporaryDirectory() as tmp:
        for text, want in PARSER_CASES.items():
            got = official_verdict(mod, text, Path(tmp))
            assert got == (want or "random"), (text, got, want)
    print(f"    {len(PARSER_CASES)} answers: parse_fakevlm == the official validate (None where it picks at random)")


def test_prompt() -> None:
    print("[B] prompt, decoding, verdict sequences (released processor if present) ...")
    assert USER_PROMPT == "<image>Does the image looks real/fake?" and FAKEVLM.chat_template is False
    assert FAKEVLM.system_prompt == "" and LLAVA.prompt(None, FAKEVLM) == USER_PROMPT
    assert FAKEVLM.decoding == FAKEVLM.official_decoding == GREEDY_DECODING == {"do_sample": False, "max_new_tokens": 256}
    if FAKECLUE_TEST.exists():
        rows = json.loads(FAKECLUE_TEST.read_text())
        assert len(rows) == 5000 and {r["conversations"][0]["value"] for r in rows} == {USER_PROMPT}
        print("    the prompt equals all 5,000 rows of the FakeClue test file")
    if not (PROCESSOR_FOLDER / "tokenizer.json").exists():
        print("    released processor folder not found: skipped the tokenizer checks")
        return
    with quiet():
        proc = LLAVA.load_processor(WEIGHTS, FAKEVLM, backend_config.CHECKPOINTS_DIR)
    tok = proc.tokenizer
    fake, real = LLAVA.verdict_sequences(tok, FAKEVLM)
    assert fake == [259, 910, 338, 263, 25713] and real == [259, 910, 338, 263, 1855], (fake, real)
    assert tok.encode("  This is a fake image.", add_special_tokens=False)[:5] == fake
    image = FAKEVLM.preprocess(FAKEVLM.image_loader(rgb_image(500, 300)), **FAKEVLM.preprocess_kwargs)
    x = LLAVA.inputs(proc, USER_PROMPT, image, FAKEVLM, "cpu")
    ids = x["input_ids"][0].tolist()
    assert x["input_ids"].shape == (1, 1599) and ids.count(32000) == 576 and ids.count(32001) == 1013 and ids[-1] == 29973
    if (WEIGHTS / "tokenizer.json").exists():               # FakeVLM's own tokenizer appends </s>: refused
        expect_raise(lambda: LLAVA.load_processor(WEIGHTS, dataclasses.replace(FAKEVLM, processor=None),
                                                  backend_config.CHECKPOINTS_DIR), ValueError, "add_eos_token=True")
    print("    verdict sequences ▁▁ This is a + ▁fake / ▁real; 576 image tokens, 1,013 pads, 1,599 positions as 4.45.2; FakeVLM's own "
          "tokenizer refused" if (WEIGHTS / "tokenizer.json").exists() else "")


LEGACY_SCRIPT = """
import sys, numpy as np, transformers
from PIL import Image
from transformers import CLIPImageProcessor
p = CLIPImageProcessor(size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336}, resample=3,
                       image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711])
out = {"version": np.array(transformers.__version__)}
for i, path in enumerate(sys.argv[2:]):
    out[str(i)] = p(Image.open(path), return_tensors="np")["pixel_values"]
np.savez(sys.argv[1], **out)
"""


def test_image_handling() -> None:
    print("[C] image handling vs transformers 4.45.2's CLIPImageProcessor ...")
    rgba = Image.new("RGBA", (40, 30), (10, 200, 30, 0))
    assert transformers_rgb(rgba).getpixel((0, 0)) == (10, 200, 30)              # alpha dropped, not composited
    assert clip_resize_center_crop(rgb_image(500, 300)).size == (336, 336)
    assert clip_resize_center_crop(rgb_image(200, 150)).size == (336, 336)       # upscaled
    expect_raise(lambda: clip_resize_center_crop(rgb_image(50, 50), shortest_edge=20, crop_size=30), ValueError, "crop")
    if not (TRANSFORMERS_4452 / "transformers").exists():
        print(f"    no transformers 4.45.2 at {TRANSFORMERS_4452}: skipped the pixel comparison")
        return
    proc = CLIPImageProcessor(size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336}, resample=3,
                              image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711])
    cases = [("rgb", rgb_image(500, 300, 1)), ("tall", rgb_image(301, 777, 2)), ("small", rgb_image(200, 150, 3)),
             ("square", rgb_image(336, 336, 4)), ("8mp", rgb_image(4000, 2000, 5)),
             ("rgba", Image.fromarray((np.random.RandomState(6).rand(300, 400, 4) * 255).astype("uint8"), "RGBA")),
             ("gray", rgb_image(640, 480, 7).convert("L")), ("palette", rgb_image(250, 260, 8).convert("P"))]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, img in cases:
            paths.append(str(Path(tmp) / f"{name}.png"))
            img.save(paths[-1])
        env = dict(os.environ, PYTHONPATH=str(TRANSFORMERS_4452), HF_HUB_OFFLINE="1")
        subprocess.run([sys.executable, "-c", LEGACY_SCRIPT, str(Path(tmp) / "legacy.npz"), *paths], check=True, env=env,
                       capture_output=True)
        legacy = np.load(Path(tmp) / "legacy.npz")
        assert str(legacy["version"]) == "4.45.2"
        worst_default = 0.0
        for i, ((name, _), path) in enumerate(zip(cases, paths)):
            img = Image.open(path)
            img.load()
            ours = proc(clip_resize_center_crop(transformers_rgb(img)), do_resize=False, do_center_crop=False,
                        return_tensors="np")["pixel_values"]
            diff = float(np.abs(ours - legacy[str(i)]).max())
            assert diff < 1e-6, (name, diff)
            worst_default = max(worst_default, float(np.abs(proc(img, return_tensors="np")["pixel_values"]
                                                        - legacy[str(i)]).max()))
    print(f"    {len(cases)} images (RGB, tall, small, square, 8 MP, RGBA, L, P): pixel values == 4.45.2's to < 1e-6 "
          f"(transformers {transformers.__version__}'s own resize: up to {worst_default:.3f} apart)")


def test_registry() -> None:
    print("[D] registry spec, note, block list, licence, Space, download hint, missing processor ...")
    spec = get_spec("fakevlm")
    assert (spec.predict_kind, spec.ckpt_kind, spec.checkpoint, spec.weights, spec.modality) == \
        ("vlm", "official_vlm", "fakevlm", "official", "image")
    assert spec.transform is None and spec.overrides == [] and spec.vlm is FAKEVLM and spec.eval_batch_size(32) == 1
    assert FAKEVLM.family is LLAVA and FAKEVLM.model_class == "LlavaForConditionalGeneration"
    assert FAKEVLM.verdict_mode == "generate" and FAKEVLM_TEACHER_FORCED.verdict_mode == "teacher_forced"
    assert dataclasses.replace(FAKEVLM, verdict_mode="teacher_forced") == FAKEVLM_TEACHER_FORCED
    assert FAKEVLM.verdict_sequences == VERDICT_SEQUENCES and FAKEVLM.teacher_forced_suffix == ""
    assert (FAKEVLM.answer_marker, FAKEVLM.answer_end, FAKEVLM.logits, FAKEVLM.verdict_offset) == (None, None, "raw", None)
    assert FAKEVLM.image_loader is transformers_rgb and FAKEVLM.preprocess is clip_resize_center_crop
    assert FAKEVLM.preprocess_kwargs == {"shortest_edge": 336, "crop_size": 336} and FAKEVLM.parser is parse_fakevlm
    assert FAKEVLM.processor == PROCESSOR and FAKEVLM.processor_overrides == {
        "patch_size": 14, "vision_feature_select_strategy": "default", "num_additional_image_tokens": 1}
    assert MAX_LENGTH == 1599 and FAKEVLM.processor_kwargs == {"padding": "max_length", "max_length": 1599, "truncation": True,
                                        "images_kwargs": {"do_resize": False, "do_center_crop": False}}
    assert FAKEVLM.hf_repo == "lingcco/fakeVLM" and FAKEVLM.hf_revision.startswith("dce9e88d")
    assert FAKEVLM.processor_hf_repo == "llava-hf/llava-1.5-7b-hf" and FAKEVLM.processor_hf_revision.startswith("a272c74")
    assert "336" in input_note(spec)["label"] and "EXIF" in input_note(spec)["explanation"]
    assert "fakevlm" in hub.NEVER_REDISTRIBUTE and "no licence" in hub.NEVER_REDISTRIBUTE["fakevlm"]
    row = [line for line in Path(ROOT, "LICENSES.md").read_text().splitlines() if line.startswith("| `fakevlm` |")]
    assert len(row) == 1 and "no licence" in row[0] and "Llama 2 Community License" in row[0], row
    fetch = Path(ROOT, "deploy", "hf-space", "fetch_weights.py").read_text()
    assert '"fakevlm"' not in fetch and "'fakevlm'" not in fetch
    with tempfile.TemporaryDirectory() as tmp:
        original = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = Path(tmp)
        try:
            det = Detector(spec, device="cpu", auto_download=True)
            assert not det.available
            msg = expect_raise(det._ensure_weights, FileNotFoundError, "hf download lingcco/fakeVLM --revision dce9e88d")
            assert (f"hf download llava-hf/llava-1.5-7b-hf --revision a272c74b2481d8aff3aa6fc2c4bf891fe57334fb "
                    f"--include '*.json' --include 'tokenizer.model' --local-dir {Path(tmp) / PROCESSOR}") in msg, msg
            Path(tmp, "fakevlm").mkdir()                       # the weights alone: still not available (no processor)
            Path(tmp, "fakevlm", "config.json").write_text("{}")
            engine = Engine(device="cpu")
            assert not engine.is_available(spec)
            msg = expect_raise(lambda: engine._get_model(spec), FileNotFoundError, "no processor (preprocessor_config.json) in")
            assert "llava-hf/llava-1.5-7b-hf" in msg
            Path(tmp, PROCESSOR).mkdir()
            assert engine.is_available(spec)
            expect_raise(lambda: Detector(spec, device="cpu").predict(Path(tmp) / "clip.mp4"), ValueError, "still images only")
        finally:
            backend_config.CHECKPOINTS_DIR = original
    print("    vlm spec (generated default, teacher-forced variant), note, NEVER_REDISTRIBUTE, LICENSES 'no licence' + "
          "Llama 2 base, not on the Space, weights + processor download hint, missing processor reported")


def test_tiny_llava() -> None:
    print("[E] tiny random LLaVA through the engine: offline load, inputs, teacher-forced and generated scores, strict ...")
    image = rgb_image(300, 200)
    cfg = tiny_config()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        weights, proc_dir, tok = tiny_folders(tmp)
        prefix = tok.convert_tokens_to_ids(["▁▁", "▁This", "▁is", "▁a"])
        f, r = tok.convert_tokens_to_ids(["▁fake", "▁real"])
        assert tok.convert_ids_to_tokens(tok.encode("  This is a fake image.", add_special_tokens=False))[:5] == \
            ["▁▁", "▁This", "▁is", "▁a", "▁fake"]
        spec = dataclasses.replace(get_spec("fakevlm"), vlm=cfg)
        original = backend_config.CHECKPOINTS_DIR
        backend_config.CHECKPOINTS_DIR = tmp
        try:
            engine = Engine(device="cpu")
            assert engine.is_available(spec)
            with quiet(), NoNetwork() as net:
                vlm = engine._get_model(spec)
            assert net.attempts == [], f"the engine load tried the network: {net.attempts}"
            assert "missing=0, unexpected=0" in engine._status["fakevlm"] and PROCESSOR in engine._status["fakevlm"]
            # inputs: 4 image tokens, left padding to 64, the 4.45.2-style resize + crop, processor resize / crop off
            x = vlm.inputs(vlm.image(image))
            ids = x["input_ids"][0].tolist()
            image_id, pad = tok.convert_tokens_to_ids(["<image>", "<pad>"])
            assert len(ids) == 64 and ids.count(image_id) == 4 and ids[0] == pad and ids[-1] != pad
            plain = CLIPImageProcessor(size={"shortest_edge": TINY_SIZE}, crop_size={"height": TINY_SIZE, "width": TINY_SIZE})
            want = plain(clip_resize_center_crop(image, TINY_SIZE, TINY_SIZE), do_resize=False, do_center_crop=False,
                         return_tensors="pt")["pixel_values"]
            assert torch.equal(x["pixel_values"], want) and vlm.family.input_size(vlm.processor, x) == [TINY_SIZE, TINY_SIZE]
            # teacher-forced variant: prompt + "▁▁ This is a", next-token gap == a direct forward
            vlm.cfg = dataclasses.replace(cfg, verdict_mode="teacher_forced")
            with quiet():
                tf = engine.vlm_score(spec, image)
            forced = torch.cat([x["input_ids"], torch.tensor([prefix])], 1)
            with torch.inference_mode():
                row = vlm.hf(input_ids=forced, attention_mask=torch.cat([x["attention_mask"], torch.ones(1, 4, dtype=torch.long)], 1),
                             pixel_values=x["pixel_values"]).logits[0, -1].float()
            assert tf["score_source"] == "teacher_forced" and tf["forced_prefix"] == ["▁▁", "▁This", "▁is", "▁a"]
            assert tf["pair_tokens"] == ["▁fake", "▁real"] and math.isclose(tf["logit"], float(row[f] - row[r]), abs_tol=1e-5)
            assert tf["verdict"] == ("fake" if tf["logit"] > 0 else "real") and tf["prompt_tokens"] == 68
            # generated (the registry mode): a forced answer "This is a fake image." -> scored at its "▁fake", step 4
            n_prompt = x["input_ids"].shape[1]
            script = tok.convert_tokens_to_ids(["▁▁", "▁This", "▁is", "▁a", "▁fake", "▁image", "."])
            eos = tok.convert_tokens_to_ids("</s>")
            gen_cfg = dataclasses.replace(cfg, verdict_mode="generate", decoding={
                "do_sample": False, "max_new_tokens": 10,
                "logits_processor": LogitsProcessorList([ForceTokens(script, eos, n_prompt)])})
            vlm.cfg = gen_cfg
            with quiet():
                gen = engine.vlm_score(spec, image)
                direct = vlm.hf.generate(**x, max_new_tokens=10, do_sample=False, output_logits=True,
                                         return_dict_in_generate=True,
                                         logits_processor=LogitsProcessorList([ForceTokens(script, eos, n_prompt)]))
            assert gen["generated_ids"] == script + [eos] and gen["response"] == "  This is a fake image.", gen["response"]
            assert (gen["score_source"], gen["verdict_step"], gen["verdict_token"], gen["verdict"]) == \
                ("verdict_token", 4, "▁fake", "fake"), gen
            raw = float(direct.logits[4][0, f] - direct.logits[4][0, r])
            assert math.isclose(gen["logit"], raw, abs_tol=1e-6), (gen["logit"], raw)
            assert math.isclose(gen["logit"], tf["logit"], abs_tol=1e-4), (gen["logit"], tf["logit"])  # same position
            # a forced "This image is fake." is scored at its own "▁fake" (step 4 by chance), the parser reads "fake"
            other = tok.convert_tokens_to_ids(["▁This", "▁image", "▁is", "▁fake", "."])
            vlm.cfg = dataclasses.replace(gen_cfg, decoding={**gen_cfg.decoding, "logits_processor": LogitsProcessorList(
                [ForceTokens(other, eos, n_prompt)])})
            with quiet():
                alt = engine.vlm_score(spec, image)
            assert (alt["verdict_step"], alt["verdict"]) == (3, "fake") and alt["response"] == "This image is fake."
            vlm.cfg = cfg
        finally:
            backend_config.CHECKPOINTS_DIR = original
        # strict load: an extra tensor or a missing one fails
        state = load_file(str(weights / "model.safetensors"))
        for name, tampered in (("extra", {**state, "model.extra.weight": torch.zeros(2)}),
                               ("missing", {k: v for k, v in state.items() if "multi_modal_projector.linear_2.bias" not in k})):
            bad = tmp / f"bad_{name}"
            bad.mkdir()
            for file in weights.iterdir():
                if file.name != "model.safetensors":
                    (bad / file.name).write_bytes(file.read_bytes())
            save_file(tampered, str(bad / "model.safetensors"), metadata={"format": "pt"})
            with quiet():
                expect_raise(lambda: SynthScopeVLM(bad, cfg, device="cpu", root=tmp), RuntimeError, "checkpoint mismatch")
        # the processor guards: EOS appended to the prompt; no expansion attributes
        eos_root = tmp / "eos"
        eos_root.mkdir()
        tiny_folders(eos_root, add_eos=True)
        with quiet():
            expect_raise(lambda: SynthScopeVLM(eos_root / "fakevlm", cfg, device="cpu", root=eos_root), ValueError,
                         "add_eos_token=True")
            expect_raise(lambda: SynthScopeVLM(weights, dataclasses.replace(cfg, processor_overrides={}), device="cpu",
                                               root=tmp), ValueError, "patch_size")
            # num_additional_image_tokens defaults to 0 on the processor: unset, it is refused at load, not at the forward
            no_extra = {k: v for k, v in cfg.processor_overrides.items() if k != "num_additional_image_tokens"}
            expect_raise(lambda: LLAVA.load_processor(weights, dataclasses.replace(cfg, processor_overrides=no_extra), tmp),
                         ValueError, "num_additional_image_tokens")
        # a processor saved with the three fields (processor_config.json) needs no overrides
        saved_root = tmp / "saved"
        saved_root.mkdir()
        _, saved_proc, _ = tiny_folders(saved_root)
        (saved_proc / "processor_config.json").write_text(json.dumps({
            "processor_class": "LlavaProcessor", "patch_size": TINY_PATCH, "vision_feature_select_strategy": "default",
            "num_additional_image_tokens": 1}))
        with quiet():
            loaded = LLAVA.load_processor(saved_root / "fakevlm", dataclasses.replace(cfg, processor_overrides={}), saved_root)
        assert (loaded.patch_size, loaded.vision_feature_select_strategy, loaded.num_additional_image_tokens) == \
            (TINY_PATCH, "default", 1)
    print("    no network at load; 4 image tokens left-padded to 64, 4.45.2-style pixels; teacher-forced gap == direct "
          "forward; forced 'This is a fake image.' scored at step 4 == raw output_logits gap == teacher-forced gap; "
          "'This image is fake.' at its own ▁fake; extra / missing tensor, EOS tokenizer, unset expansion (incl. "
          "num_additional_image_tokens alone) refused; saved processor_config.json accepted")


def test_released_weights() -> None:
    if not ((WEIGHTS / "model.safetensors.index.json").exists() and (PROCESSOR_FOLDER / "tokenizer.json").exists()
            and torch.cuda.is_available()):
        print("[F] released weights + processor + GPU not available here: skipped")
        return
    print("[F] released weights: strict engine load and one image ...")
    spec = get_spec("fakevlm")
    engine = Engine(device="cuda")
    with quiet(), NoNetwork() as net:
        model = engine._get_model(spec)
    assert net.attempts == [], net.attempts
    n = len(json.loads((WEIGHTS / "model.safetensors.index.json").read_text())["weight_map"])
    assert len(model.hf.state_dict()) == n, (n, len(model.hf.state_dict()))
    gate = Path("/ssdstore/ssalehi/SynthScope/GenImage_val_full/test/adm/1_fake/0_adm_153.PNG")
    image = media.decode_image_raw(gate.read_bytes()) if gate.exists() else rgb_image(256, 256)
    with quiet():
        out = engine.vlm_score(spec, image)
    assert out["verdict"] in ("real", "fake") and out["score_source"] == "verdict_token" and out["new_tokens"] > 4
    print(f"    {n} tensors strict, no network; {gate.name if gate.exists() else 'synthetic'}: {out['verdict']} "
          f"(P(fake) {out['fake_probability']:.6f})")


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    warnings.filterwarnings("ignore", message="Can't initialize NVML")
    test_parser()
    test_prompt()
    test_image_handling()
    test_registry()
    test_tiny_llava()
    test_released_weights()
    print("fakevlm smoke test passed")
