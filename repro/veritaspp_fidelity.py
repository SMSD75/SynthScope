"""Veritas++ (``veritaspp``) fidelity check: SynthScope's engine path vs a direct transformers call with the official
request, on the same GPU, one after the other.

Reference path (written independently of ``models/vlm``): ``AutoProcessor`` + ``Qwen3VLForConditionalGeneration``
loaded from the released folder, the official prompts read from the official client
(``self_scripts/infer/infer_vllm_single.py``, parsed with ``ast``), ``apply_chat_template(..., enable_thinking=False)``,
``generate(do_sample=False, max_new_tokens=2048, repetition_penalty=1.0, output_logits=True)``; the verdict step and
token pair are found with the 2026-11-16 gate-check helpers (``vlm_gate/gate_common.py``) and the answer is parsed
with the gate's copy of the Veritas v1 rule.

Engine path: ``Engine._get_model`` (strict ``from_pretrained``) and ``Engine.vlm_score`` on the image decoded as
``synthscope.load(...).predict(path)`` decodes it (``media.decode_image``).

Checks: (a) strict load (loading info empty; every tensor of the four shards equal to the loaded parameter of the
same name, and no other parameter); (b) identical greedy tokens and parsed verdict; (c) max |diff| of the verdict
logit gap; (d) greedy vs the official sampling (T 0.7, top-p 0.8, top-k 20, seed 0) verdict agreement; throughput and
peak memory of the engine path; a memory probe on large images (the processor's own 16.7 MP cap).

Run on the node that holds the weights, with SYNTHSCOPE_CKPT_DIR set to the folder that contains ``veritaspp/``:
    python repro/veritaspp_fidelity.py --out repro/veritaspp_fidelity.json
"""
import argparse
import ast
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(SHARED / "vlm_gate")]

from gate_common import PAIRS, paired_ids, verdict_position         # noqa: E402  (reference helpers)
from gate_veritaspp import parse_v1, verdict_ids                     # noqa: E402
from interface.backend import config as backend_config               # noqa: E402
from interface.backend import media                                  # noqa: E402
from interface.backend.engine import Engine                          # noqa: E402
from interface.backend.registry import get_spec                      # noqa: E402

OFFICIAL_CLIENT = SHARED / "vlm_ref" / "VeritasPP" / "self_scripts" / "infer" / "infer_vllm_single.py"
DATA = Path("/ssdstore/ssalehi/SynthScope")
# 13 images, real and fake, 256x256 to 2048x2048: the gate's six GenImage val images + seven SynthWildX images.
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
]
# Large WildRF images for the memory probe: 8.0 MP, 16.8 MP (= the processor cap) and 64 MP (resized to the cap).
PROBE = ["WildRF/test/reddit/0_real/000030_0030.jpg", "WildRF/test/reddit/1_fake/000325_0325.jpg",
         "WildRF/test/reddit/0_real/000516_0516.jpg"]
MAX_NEW = 2048


def official_prompts() -> tuple:
    consts = {n.targets[0].id: n.value.value for n in ast.parse(OFFICIAL_CLIENT.read_text()).body
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
    return consts["SYSTEM_PROMPT"], consts["USER_PROMPT"]


def reference(path: Path, images: list) -> tuple:
    proc = AutoProcessor.from_pretrained(str(path))
    tok = proc.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(path), dtype=torch.bfloat16, attn_implementation="sdpa",
                                                            device_map="cuda").eval()
    system, user = official_prompts()
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    fake_ids, real_ids = verdict_ids(tok, "fake"), verdict_ids(tok, "real")
    rows = []
    for rel, label in images:
        img = Image.open(DATA / rel).convert("RGB")
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        n = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.0,
                                 output_logits=True, return_dict_in_generate=True)
        seconds = time.time() - t0
        gen = out.sequences[0, n:].tolist()
        response = tok.decode(gen, skip_special_tokens=True)
        step = verdict_position(tok, gen, fake_ids, real_ids, marker="<answer>")
        pair = paired_ids(tok, gen[step], PAIRS["fake_real"]) if step is not None else None
        gap = None if pair is None else (out.logits[step][0, pair[0]].float() - out.logits[step][0, pair[1]].float()).item()
        rows.append({"image": rel, "label_fake": label, "size": list(img.size), "prompt_tokens": n, "generated_ids": gen,
                     "response": response, "parse": parse_v1(response), "verdict_step": step,
                     "pair_tokens": None if pair is None else tok.convert_ids_to_tokens(list(pair)), "gap": gap,
                     "seconds": round(seconds, 2)})
        print(f"[ref] {rel} {img.size} tokens={len(gen)} parse={rows[-1]['parse']} gap={gap}", flush=True)
        del out
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return text, rows


def tensor_check(model, path: Path) -> dict:
    """Every tensor of the released shards vs the loaded parameter / buffer of the same name."""
    sd = model.hf.state_dict()
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    equal, differ = 0, []
    for shard in sorted(set(index.values())):
        with safe_open(str(path / shard), framework="pt", device="cuda") as f:
            for key in f.keys():
                if key in sd and torch.equal(sd[key], f.get_tensor(key)):
                    equal += 1
                else:
                    differ.append(key)
    return {"checkpoint_tensors": len(index), "model_tensors": len(sd), "equal": equal, "differ_or_absent": differ[:20],
            "model_only": sorted(set(sd) - set(index))[:20]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "repro" / "veritaspp_fidelity.json"))
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()
    out_path = Path(args.out)
    spec = get_spec("veritaspp")
    path = backend_config.CHECKPOINTS_DIR / spec.checkpoint
    result = {"weights": str(path), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "transformers": __import__("transformers").__version__, "max_new_tokens": MAX_NEW}

    def save():
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")

    if not args.skip_reference:
        result["reference_prompt"], result["reference"] = reference(path, IMAGES)
        save()

    # Engine path: strict load, then greedy (timed), official sampling, the predict_image wrapper and the memory probe.
    engine = Engine(device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = engine._get_model(spec)
    result["load"] = {"status": engine._status[spec.key], "seconds": round(time.time() - t0, 1),
                      "weights_GB": round(torch.cuda.memory_allocated() / 1e9, 2)}
    result["load"]["tensors"] = tensor_check(model, path)
    result["engine_prompt_equals_reference"] = model.prompt == result.get("reference_prompt")
    print("[load]", result["load"], flush=True)
    save()

    greedy, sampled = [], []
    for rel, label in IMAGES:
        pil = media.decode_image((DATA / rel).read_bytes())
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        o = engine.vlm_score(spec, pil)
        o["seconds"] = round(time.time() - t0, 2)
        o["peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        o.update(image=rel, label_fake=label, size=list(pil.size))
        greedy.append(o)
        print(f"[engine] {rel} tokens={o['new_tokens']} verdict={o['verdict']} logit={o['logit']:.4f} "
              f"{o['seconds']}s {o['peak_GB']}GB", flush=True)
    result["engine_greedy"] = greedy
    save()
    for rel, label in IMAGES:
        o = engine.vlm_score(spec, media.decode_image((DATA / rel).read_bytes()), decoding="official", seed=0)
        o.update(image=rel, label_fake=label)
        sampled.append(o)
        print(f"[sampled] {rel} tokens={o['new_tokens']} verdict={o['verdict']} logit={o['logit']:.4f}", flush=True)
    result["engine_sampled_T0.7_seed0"] = sampled
    save()
    wrapped = engine.predict_image(spec, media.decode_image((DATA / IMAGES[0][0]).read_bytes()))
    result["predict_image_example"] = {k: v for k, v in wrapped.items() if k != "input_note"}
    result["predict_image_matches_vlm_score"] = (wrapped["fake_probability"] == round(greedy[0]["fake_probability"], 6)
                                                 and wrapped["verdict"] == greedy[0]["verdict"])

    probe = []
    for rel in PROBE:
        pil = media.decode_image((DATA / rel).read_bytes())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            o = engine.vlm_score(spec, pil)
            row = {k: o[k] for k in ("input_size", "prompt_tokens", "new_tokens", "verdict", "logit")}
            row["oom"] = False
        except torch.cuda.OutOfMemoryError as e:
            row = {"oom": True, "error": str(e)[:300]}
        row.update(image=rel, size=list(pil.size), seconds=round(time.time() - t0, 2),
                   peak_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        probe.append(row)
        print("[probe]", row, flush=True)
    result["memory_probe"] = probe

    # Summary.
    ref = result.get("reference", [])
    summary = {"n_images": len(IMAGES)}
    if ref:
        same_tokens = [r["generated_ids"] == g["generated_ids"] for r, g in zip(ref, greedy)]
        same_verdict = [r["parse"] == g["verdict"] for r, g in zip(ref, greedy)]
        diffs = [abs(r["gap"] - g["logit"]) for r, g in zip(ref, greedy) if r["gap"] is not None]
        summary.update(identical_greedy_tokens=f"{sum(same_tokens)}/{len(same_tokens)}",
                       identical_parsed_verdict=f"{sum(same_verdict)}/{len(same_verdict)}",
                       max_abs_logit_gap_diff=max(diffs) if diffs else None,
                       reference_verdict_steps_equal=all(r["verdict_step"] == g["verdict_step"] for r, g in zip(ref, greedy)))
    summary["greedy_correct"] = sum((g["verdict"] == "fake") == bool(g["label_fake"]) for g in greedy)
    summary["sampled_correct"] = sum((s["verdict"] == "fake") == bool(s["label_fake"]) for s in sampled)
    summary["greedy_vs_sampled_verdict_agreement"] = f"{sum(g['verdict'] == s['verdict'] for g, s in zip(greedy, sampled))}/{len(greedy)}"
    secs = sum(g["seconds"] for g in greedy)
    toks = sum(g["new_tokens"] for g in greedy)
    summary["engine_greedy_images_per_s"] = round(len(greedy) / secs, 4)
    summary["engine_greedy_seconds_per_image"] = round(secs / len(greedy), 2)
    summary["engine_greedy_tokens_per_s"] = round(toks / secs, 1)
    summary["engine_peak_GB_max"] = max(g["peak_GB"] for g in greedy)
    summary["probe_peak_GB_max"] = max(p["peak_GB"] for p in probe)
    summary["probe_oom"] = any(p["oom"] for p in probe)
    result["summary"] = summary
    save()
    print("[summary]", json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
