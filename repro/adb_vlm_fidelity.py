"""Align, Don't Blend VLM branch (``adb_vlm``) fidelity check: SynthScope's engine path vs the official ms-swift path and vs
a direct transformers + peft call with the official request, on the same GPU, one after the other.

Three paths on the same images:

* OFFICIAL (ms-swift 4.4.2, ``swift infer --infer_backend pt``): the unmodified official ``predict.py`` (run by the sbatch
  before this script; its output JSONL has the response and the parsed VLM verdict), and the same ``swift infer``
  command once more with ``--logprobs true`` (``swift-logprobs`` below: the command ``predict.py``'s ``run_vlm`` builds,
  captured, plus that flag) for the generated tokens and their log-probabilities;
* REFERENCE (written independently of ``models/vlm``): ``AutoProcessor`` + ``Qwen2_5_VLForConditionalGeneration`` from
  the local base folder, ``peft.PeftModel.from_pretrained`` on the adapter folder (unmerged); the image as ms-swift
  prepares it, by ms-swift's own ``load_image`` / ``rescale_image`` (compiled from ``swift/template/vision_utils.py``) and
  qwen-vl-utils' own ``fetch_image`` (IMAGE_MAX_TOKEN_NUM = 1280, as swift patches it from MAX_PIXELS); the official
  prompts read from ``detector/core.py``; the processor with ``do_resize=False`` (as swift's template calls it);
  ``generate(do_sample=False, max_new_tokens=512, repetition_penalty=1.05, output_logits=True, output_scores=True)``;
  the verdict step is the first AI / authentic token after "<answer>" (before "</answer>"), the gap is that token's and
  its same-spelling counterpart's, RAW (``output_logits``) and PROCESSED (``output_scores``, after the repetition penalty);
  the answer is parsed with the official ``parse_vlm_verdict``;
* ENGINE: ``Engine._get_model`` (strict load) and ``Engine.vlm_score`` on the file decoded as ``synthscope.load(...)``
  decodes it for a VLM detector (``media.decode_image_raw``), with the registry config (raw logits), then once more with
  ``logits="processed"``.

Checks: (a) strict load (every tensor of the base shards equal to the loaded base parameter, the adapter key for key);
(b) identical greedy tokens, text and parsed verdict vs ms-swift; (c) identical tokens and verdict vs the reference, max
|diff| of the raw and of the processed verdict logit gap; (d) throughput and peak memory of the engine path.

Usage (the sbatch runs all three steps on one GPU; SYNTHSCOPE_CKPT_DIR holds ``adb_vlm/`` and ``qwen2.5-vl-7b-instruct/``):
    python repro/adb_vlm_fidelity.py swift-logprobs --inputs IN.txt --out-dir DIR --swift-bin BIN
    python repro/adb_vlm_fidelity.py compare --data ROOT --swift-official OFF.jsonl --swift-logprobs DIR/vlm_result.jsonl \
        --out repro/adb_vlm_fidelity.json
"""
import argparse
import ast
import dataclasses
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from io import BytesIO
from pathlib import Path
from typing import TypeVar, Union

import peft
import torch
import transformers
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import vision_process
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface")]

from interface.backend import config as backend_config               # noqa: E402
from interface.backend import media                                  # noqa: E402
from interface.backend.engine import Engine                          # noqa: E402
from interface.backend.registry import get_spec                      # noqa: E402

SHARED = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
OFFICIAL_DIR = SHARED / "vlm_ref" / "Align-Dont-Blend"               # github.com/GzZning/Align-Dont-Blend @ f4b4794
SWIFT_SRC = SHARED / "vlm_pydeps" / "swift442" / "swift" / "template" / "vision_utils.py"   # ms-swift 4.4.2
# 16 images, real and fake, 256x256 to 8 MP: the gate's six GenImage val images, seven SynthWildX images, two transparent
# (RGBA) WildRF PNGs (swift drops the alpha channel) and one 8 MP WildRF JPEG (shrunk by rescale_image first). The same
# 15 as repro/veritaspp_fidelity.py plus the 8 MP one.
IMAGES = [
    ("GenImage_val_full/test/midjourney/0_real/ILSVRC2012_val_00000008.JPEG", 0),
    ("GenImage_val_full/test/sdv1.4/0_real/ILSVRC2012_val_00000037.JPEG", 0),
    ("GenImage_val_full/test/adm/0_real/ILSVRC2012_val_00000005.JPEG", 0),
    ("GenImage_val_full/test/midjourney/1_fake/0_midjourney_100.png", 1),
    ("GenImage_val_full/test/sdv1.4/1_fake/000_sdv4_00020.png", 1),
    ("GenImage_val_full/test/adm/1_fake/0_adm_153.PNG", 1),
    ("SynthWildX/test/dalle3/0_real/00000_Cb1RvxXW4AAGFd7.jpg", 0),
    ("SynthWildX/test/dalle3/0_real/00001_CwHUi2yWcAEzM1d.jpg", 0),
    ("SynthWildX/test/dalle3/0_real/00002_DKvym1KW4AEet37.jpg", 0),
    ("SynthWildX/test/dalle3/1_fake/00000_F-7R0_TW8AA8IZx.jpg", 1),
    ("SynthWildX/test/firefly/1_fake/00000_F-A4b03W8AAoa-_.jpg", 1),
    ("SynthWildX/test/firefly/1_fake/00002_F-Ai0-DbYAAWm2g.jpg", 1),
    ("SynthWildX/test/midjourney_v5/1_fake/00000_F-0U2WDXkAAaKOI.jpg", 1),
    ("WildRF/test/reddit/0_real/000219_0219.png", 0),       # RGBA
    ("WildRF/test/reddit/0_real/000032_0032.png", 0),       # RGBA
    ("WildRF/test/reddit/0_real/000030_0030.jpg", 0),       # 8.0 MP
]
MAX_PIXELS = 1280 * 28 * 28                                  # predict.py DEFAULT_MAX_PIXELS
MAX_NEW = 512
REPETITION_PENALTY = 1.05                                    # the base generation_config.json (swift leaves it unset)
PAIRS = [("AI", "authentic"), ("Ai", "Authentic"), ("ai", "authentic")]


def official_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def swift_image_functions(path: Path = SWIFT_SRC) -> dict:
    """ms-swift's ``load_file`` / ``load_image`` / ``rescale_image``, compiled from ``vision_utils.py`` unchanged."""
    tree = ast.parse(path.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"load_file", "load_image", "rescale_image"}]
    namespace = {"BytesIO": BytesIO, "Image": Image, "math": math, "os": os, "Union": Union, "_T": TypeVar("_T")}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def official_image(data: bytes, swift: dict, vision_process) -> Image.Image:
    """The image swift hands the processor: ``Template._preprocess_inputs`` (``load_image`` of the file bytes, then
    ``rescale_image`` to --max_pixels), then ``Qwen2VLTemplate.replace_tag`` (``fetch_image({"image": img})`` with
    qwen-vl-utils' IMAGE_MAX_TOKEN_NUM = MAX_PIXELS // 28^2, as ``compat_qwen_vl_utils`` / ``patch_qwen_vl_utils`` set it)."""
    img = swift["rescale_image"](swift["load_image"](data), MAX_PIXELS)
    saved = vision_process.IMAGE_MAX_TOKEN_NUM
    vision_process.IMAGE_MAX_TOKEN_NUM = MAX_PIXELS // 28 ** 2
    try:
        return vision_process.fetch_image({"image": img})
    finally:
        vision_process.IMAGE_MAX_TOKEN_NUM = saved


def base_key(key: str) -> str:
    """A key of the released (transformers 4.4x) Qwen2.5-VL shards under transformers 5's module names (its
    conversion mapping for qwen2_5_vl: ``visual`` -> ``model.visual``, ``model`` -> ``model.language_model``)."""
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("model.") and not key.startswith(("model.language_model.", "model.visual.")):
        return "model.language_model." + key[len("model."):]
    return key


def verdict_ids(tok, word: str) -> list:
    return sorted(i for t, i in tok.get_vocab().items() if tok.convert_tokens_to_string([t]).strip().lower() == word)


def verdict_position(tok, gen: list, fake_ids: list, real_ids: list, marker: str = "<answer>", end: str = "</answer>"):
    """First generated step whose token is a verdict token after ``marker`` and before ``end`` (vlm_gate gate_common)."""
    verdict, start = set(fake_ids) | set(real_ids), None
    for t in range(1, len(gen) + 1):
        if marker in tok.decode(gen[:t], skip_special_tokens=False):
            start = t
            break
    if start is None:
        return None
    for t in range(start, len(gen)):
        if int(gen[t]) in verdict:
            return t
        if end in tok.decode(gen[start:t + 1], skip_special_tokens=False):
            return None
    return None


def paired_ids(tok, tid: int):
    """(fake id, real id): the emitted token and its same-spelling counterpart (vlm_gate gate_common.paired_ids)."""
    s = tok.convert_ids_to_tokens(int(tid))
    for f, r in PAIRS:
        for a, b, fake_first in ((f, r, True), (r, f, False)):
            if a in s:
                other = tok.convert_tokens_to_ids(s.replace(a, b))
                if other is not None and other != tok.unk_token_id:
                    return (int(tid), int(other)) if fake_first else (int(other), int(tid))
    return None


def official_parse(core, response: str) -> str:
    try:
        return "fake" if core.parse_vlm_verdict(response) else "real"
    except ValueError:
        return "unparsed"


def reference(base: Path, adapter: Path, data_root: Path, core) -> tuple:
    swift = swift_image_functions()
    proc = AutoProcessor.from_pretrained(str(base))
    tok = proc.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16, attn_implementation="sdpa",
                                                               device_map="cuda")
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    assert core.VLM_USER_PROMPT.startswith("<image>")
    msgs = [{"role": "system", "content": core.VLM_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": core.VLM_USER_PROMPT[len("<image>"):]}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    fake_ids, real_ids = verdict_ids(tok, "ai"), verdict_ids(tok, "authentic")
    rows = []
    for rel, label in IMAGES:
        img = official_image((data_root / rel).read_bytes(), swift, vision_process)
        inputs = proc(text=[text], images=[img], return_tensors="pt", images_kwargs={"do_resize": False}).to("cuda")
        n = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=REPETITION_PENALTY,
                                 output_logits=True, output_scores=True, return_dict_in_generate=True)
        torch.cuda.synchronize()
        seconds = time.time() - t0
        gen = out.sequences[0, n:].tolist()
        response = tok.decode(gen, skip_special_tokens=True)
        step = verdict_position(tok, gen, fake_ids, real_ids)
        pair = paired_ids(tok, gen[step]) if step is not None else None
        raw = processed = None
        if pair is not None:
            raw = (out.logits[step][0, pair[0]].float() - out.logits[step][0, pair[1]].float()).item()
            processed = (out.scores[step][0, pair[0]].float() - out.scores[step][0, pair[1]].float()).item()
        logprobs = [torch.log_softmax(out.logits[t][0].float(), -1)[g].item() for t, g in enumerate(gen)]
        rows.append({"image": rel, "label_fake": label, "size": list(img.size), "prompt_tokens": n, "generated_ids": gen,
                     "response": response, "parse": official_parse(core, response), "verdict_step": step,
                     "pair_tokens": None if pair is None else tok.convert_ids_to_tokens(list(pair)),
                     "raw_gap": raw, "processed_gap": processed, "raw_logprobs": logprobs, "seconds": round(seconds, 2)})
        print(f"[ref] {rel} {img.size} tokens={len(gen)} parse={rows[-1]['parse']} raw={raw} processed={processed} "
              f"{seconds:.1f}s", flush=True)
        del out
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return text, rows, tok


def tensor_check(vlm, base: Path) -> dict:
    """Every tensor of the base shards vs the loaded base parameter (peft's ``base_model.model.`` prefix and a wrapped
    layer's ``.base_layer`` removed, the shard key under transformers 5's names); the adapter is checked key by key at load
    (``load_lora_unmerged``)."""
    sd = vlm.hf.state_dict()
    loaded = {}
    for k, v in sd.items():
        if "lora_" in k:
            continue
        loaded[k.removeprefix("base_model.model.").replace(".base_layer.", ".")] = v
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    equal, differ = 0, []
    for shard in sorted(set(index.values())):
        with safe_open(str(base / shard), framework="pt", device="cuda") as f:
            for key in f.keys():
                k = base_key(key)
                if k in loaded and torch.equal(loaded[k], f.get_tensor(key)):
                    equal += 1
                else:
                    differ.append(key)
    mapped = {base_key(k) for k in index}
    return {"base_checkpoint_tensors": len(index), "base_model_tensors": len(loaded), "equal": equal,
            "differ_or_absent": differ[:20], "model_only": sorted(set(loaded) - mapped)[:20],
            "lora_tensors_in_model": sum("lora_" in k for k in sd)}


def swift_logprobs(args) -> int:
    """Run the command the official ``predict.py`` ``run_vlm`` builds (captured, not retyped) with ``--logprobs true``."""
    sys.path.insert(0, str(args.official_dir))
    predict = official_module(Path(args.official_dir) / "predict.py", "adb_official_predict")
    core = sys.modules.get("detector.core") or importlib.import_module("detector.core")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    def capture(command, check, env):
        captured.update(command=command, env=env)

    predict.subprocess = types.SimpleNamespace(run=capture)
    records = core.read_inputs(Path(args.inputs).resolve())
    try:
        predict.run_vlm(records, Path(args.adapter).resolve(), args.swift_bin, out_dir, 1, Path(args.base).resolve(),
                        predict.DEFAULT_MAX_PIXELS, "pt")
    except RuntimeError:          # "completed without writing result JSONL": the captured call did not run swift
        pass
    command = captured["command"] + ["--logprobs", "true"]
    (out_dir / "command.json").write_text(json.dumps({"command": command,
                                                      "MAX_PIXELS": captured["env"]["MAX_PIXELS"]}, indent=1) + "\n")
    print("[swift-logprobs]", " ".join(command), flush=True)
    t0 = time.time()
    subprocess.run(command, check=True, env=captured["env"])
    print(f"[swift-logprobs] done in {time.time() - t0:.1f}s", flush=True)
    return 0


def read_swift(official: Path, logprob_file: Path, data_root: Path) -> list:
    """Per image (IMAGES order): the official predict.py row and the logprobs run's row."""
    off = {str(Path(r["image"]).resolve()): r for r in map(json.loads, official.read_text().splitlines()) if r}
    lp = {}
    for line in logprob_file.read_text().splitlines():
        r = json.loads(line)
        img = r["images"][0]
        img = img["path"] if isinstance(img, dict) else img
        lp[str(Path(img).resolve())] = r
    rows = []
    for rel, _ in IMAGES:
        key = str((data_root / rel).resolve())
        o, p = off[key], lp[key]
        content = (p.get("logprobs") or {}).get("content") or []
        rows.append({"response": o["vlm_response"], "vlm_fake": o["vlm_fake"], "logprobs_run_response": p["response"],
                     "tokens": [c["token"] for c in content], "token_bytes": [c["bytes"] for c in content],
                     "logprobs": [c["logprob"] for c in content]})
    return rows


def compare(args) -> int:
    out_path = Path(args.out)
    data_root = Path(args.data)
    spec = get_spec("adb_vlm")
    adapter = backend_config.CHECKPOINTS_DIR / spec.checkpoint
    base = backend_config.CHECKPOINTS_DIR / spec.vlm.base_model
    core = official_module(Path(args.official_dir) / "detector" / "core.py", "adb_official_core")
    result = {"adapter": str(adapter), "base": str(base), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "transformers": transformers.__version__, "peft": peft.__version__,
              "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
              "max_new_tokens": MAX_NEW, "repetition_penalty": REPETITION_PENALTY, "max_pixels": MAX_PIXELS,
              "images": [{"image": rel, "label_fake": lab} for rel, lab in IMAGES]}
    if args.swift_official_seconds:
        result["swift_official_seconds_total"] = args.swift_official_seconds

    def save():
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")

    swift = read_swift(Path(args.swift_official), Path(args.swift_logprobs), data_root)
    result["swift"] = swift
    save()
    result["reference_prompt"], ref, tok = reference(base, adapter, data_root, core)
    result["reference"] = ref
    save()

    # Engine path: strict load, greedy with raw logits (timed: throughput, memory), then the processed-logits score.
    engine = Engine(device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = engine._get_model(spec)
    result["load"] = {"status": engine._status[spec.key], "seconds": round(time.time() - t0, 1),
                      "weights_GB": round(torch.cuda.memory_allocated() / 1e9, 2),
                      "lora_modules": sum(n.endswith("lora_A.default") for n, _ in model.hf.named_modules())}
    result["load"]["tensors"] = tensor_check(model, base)
    result["engine_prompt_equals_reference"] = model.prompt == result["reference_prompt"]
    print("[load]", result["load"], flush=True)
    save()
    greedy = []
    for rel, label in IMAGES:
        pil = media.decode_image_raw((data_root / rel).read_bytes())
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        o = engine.vlm_score(spec, pil)
        torch.cuda.synchronize()
        o["seconds"] = round(time.time() - t0, 2)
        o["peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        o.update(image=rel, label_fake=label, size=list(pil.size))
        greedy.append(o)
        print(f"[engine] {rel} {pil.size}->{o['input_size']} tokens={o['new_tokens']} verdict={o['verdict']} "
              f"logit={o['logit']:.4f} {o['seconds']}s {o['peak_GB']}GB", flush=True)
    result["engine_raw"] = greedy
    save()
    raw_cfg = model.cfg
    model.cfg = dataclasses.replace(raw_cfg, logits="processed")
    processed = []
    try:
        for rel, label in IMAGES:
            o = engine.vlm_score(spec, media.decode_image_raw((data_root / rel).read_bytes()))
            processed.append({k: o[k] for k in ("logit", "fake_probability", "verdict", "verdict_step", "verdict_token",
                                                 "generated_ids", "score_source")} | {"image": rel})
            print(f"[engine processed] {rel} logit={o['logit']:.4f}", flush=True)
    finally:
        model.cfg = raw_cfg
    result["engine_processed"] = processed
    wrapped = engine.predict_image(spec, media.decode_image_raw((data_root / IMAGES[0][0]).read_bytes()))
    result["predict_image_example"] = {k: v for k, v in wrapped.items() if k != "input_note"}
    result["predict_image_matches_vlm_score"] = (wrapped["fake_probability"] == round(greedy[0]["fake_probability"], 6)
                                                 and wrapped["verdict"] == greedy[0]["verdict"])

    # Summary.
    n = len(IMAGES)

    def strings(ids):              # swift's per-token text: tokenizer.decode(token_id), the final <|im_end|> included
        return [tok.decode([i]) for i in ids]

    s_text = [s["response"] == g["response"] for s, g in zip(swift, greedy)]
    s_verdict = [("fake" if s["vlm_fake"] else "real") == g["verdict"] for s, g in zip(swift, greedy)]
    s_runs = [s["response"] == s["logprobs_run_response"] for s in swift]
    s_tokens = [s["tokens"] == strings(g["generated_ids"]) and
                s["token_bytes"] == [list(t.encode("utf8")) for t in strings(g["generated_ids"])]
                for s, g in zip(swift, greedy)]
    lp_diff = [abs(a - b) for s, r in zip(swift, ref) for a, b in zip(s["logprobs"], r["raw_logprobs"])]
    r_tokens = [r["generated_ids"] == g["generated_ids"] for r, g in zip(ref, greedy)]
    r_verdict = [r["parse"] == (g["verdict"] or "unparsed") for r, g in zip(ref, greedy)]
    raw_diff = [abs(r["raw_gap"] - g["logit"]) for r, g in zip(ref, greedy) if r["raw_gap"] is not None]
    proc_diff = [abs(r["processed_gap"] - p["logit"]) for r, p in zip(ref, processed) if r["processed_gap"] is not None]
    summary = {
        "n_images": n,
        "vs_swift_identical_text": f"{sum(s_text)}/{n}",
        "vs_swift_identical_tokens": f"{sum(s_tokens)}/{n}",
        "vs_swift_identical_parsed_verdict": f"{sum(s_verdict)}/{n}",
        "swift_logprobs_run_text_equals_predict_py": f"{sum(s_runs)}/{n}",
        "swift_vs_engine_token_counts": [[len(s["tokens"]), len(g["generated_ids"])] for s, g in zip(swift, greedy)],
        "swift_vs_reference_logprobs_compared": len(lp_diff),
        "max_abs_chosen_token_logprob_diff_swift_vs_reference": max(lp_diff) if lp_diff else None,
        "vs_reference_identical_tokens": f"{sum(r_tokens)}/{n}",
        "vs_reference_identical_parsed_verdict": f"{sum(r_verdict)}/{n}",
        "vs_reference_verdict_steps_equal": all(r["verdict_step"] == g["verdict_step"] for r, g in zip(ref, greedy)),
        "max_abs_raw_logit_gap_diff": max(raw_diff) if raw_diff else None,
        "max_abs_processed_logit_gap_diff": max(proc_diff) if proc_diff else None,
        "processed_run_tokens_equal_raw_run": all(p["generated_ids"] == g["generated_ids"] for p, g in zip(processed, greedy)),
        "raw_vs_processed_gap": [{"image": g["image"], "raw": g["logit"], "processed": p["logit"],
                                  "p_fake_raw": g["fake_probability"], "p_fake_processed": p["fake_probability"]}
                                 for g, p in zip(greedy, processed)],
        "raw_processed_sign_agree": all((g["logit"] > 0) == (p["logit"] > 0) for g, p in zip(greedy, processed)),
        "engine_unparsed_score": sum(g["score_source"] == "unparsed" for g in greedy),
        "engine_correct": sum((g["verdict"] == "fake") == bool(g["label_fake"]) for g in greedy),
        "tensors_all_equal": result["load"]["tensors"]["equal"] == result["load"]["tensors"]["base_checkpoint_tensors"],
    }
    secs = sum(g["seconds"] for g in greedy)
    toks = sum(g["new_tokens"] for g in greedy)
    summary["engine_images_per_s"] = round(n / secs, 4)
    summary["engine_seconds_per_image"] = round(secs / n, 2)
    summary["engine_tokens_per_s"] = round(toks / secs, 1)
    summary["engine_peak_GB_max"] = max(g["peak_GB"] for g in greedy)
    summary["reference_images_per_s"] = round(n / sum(r["seconds"] for r in ref), 4)
    if args.swift_official_seconds:
        summary["swift_official_wall_seconds_incl_load"] = args.swift_official_seconds
    result["summary"] = summary
    save()
    print("[summary]", json.dumps(summary, indent=1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("swift-logprobs")
    a.add_argument("--inputs", required=True, help="the image list predict.py was given")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--swift-bin", required=True)
    a.add_argument("--adapter", default=str(backend_config.CHECKPOINTS_DIR / "adb_vlm"))
    a.add_argument("--base", default=str(backend_config.CHECKPOINTS_DIR / "qwen2.5-vl-7b-instruct"))
    a.add_argument("--official-dir", default=str(OFFICIAL_DIR))
    c = sub.add_parser("compare")
    c.add_argument("--data", required=True, help="root of the IMAGES paths")
    c.add_argument("--swift-official", required=True, help="predict.py's output JSONL")
    c.add_argument("--swift-logprobs", required=True, help="the swift-logprobs run's vlm_result.jsonl")
    c.add_argument("--swift-official-seconds", type=float, default=None)
    c.add_argument("--official-dir", default=str(OFFICIAL_DIR))
    c.add_argument("--out", default=str(ROOT / "repro" / "adb_vlm_fidelity.json"))
    args = ap.parse_args()
    return swift_logprobs(args) if args.cmd == "swift-logprobs" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
