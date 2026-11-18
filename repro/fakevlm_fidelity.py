"""FakeVLM (``fakevlm``) fidelity check: SynthScope's engine vs the official ``scripts/eval.py`` and vs a direct transformers
call, on the same GPU, one after the other.

Paths:

* OFFICIAL (``repro/fakevlm_official.py``, run by the sbatch before this script), each output JSON holding
  ``decode(prompt + answer).split('?')[-1]`` per image:
    - ``eval``: the unmodified ``eval.py`` (left-padded to 1024 tokens, greedy, 256 new tokens) in batches of 4 and of 2
      under the authors' transformers 4.45.2, and in batches of 4 under the SynthScope env's transformers 5.5.4;
    - ``single``: eval.py's own dataset item, model and ``generate`` call one image at a time (what ``validate`` does
      with a batch of one, which its ``.squeeze()`` breaks), under 4.45.2 and under 5.5.4;
* REFERENCE (written independently of ``models/vlm``): ``LlavaForConditionalGeneration`` from the weights folder; the
  inputs of the official processor call under transformers 4.45.2 (``fakevlm_official.py inputs``: ``Image.open``, CLIP
  resize + crop in PIL, the raw prompt, padded to 1024) with the single ``<image>`` token expanded to the 576 tokens
  transformers 5 merges at (and the left padding shortened to keep 1024); one image at a time,
  ``generate(do_sample=False, max_new_tokens=256, output_logits=True)``; the verdict step is the first ``▁fake`` /
  ``▁real`` token, the gap l(▁fake) - l(▁real) there; and a teacher-forced forward on the prompt + the generated tokens
  before that step;
* ENGINE: ``Engine._get_model`` (strict load) and ``Engine.vlm_score`` on the file decoded as ``synthscope.load(...)``
  decodes it for a VLM detector (``media.decode_image_raw``): the registry config ``FAKEVLM`` (the official greedy
  generation), then ``FAKEVLM_TEACHER_FORCED`` (one forward pass on the prompt + "▁▁ This is a").

Checks: (a) strict load (every tensor of the shards equal to the loaded parameter); (b) the engine's processor inputs
== the official 4.45.2 processor's (pixel values to float rounding, token ids after the expansion); (c) greedy text,
first sentence and parsed verdict vs every official run, and the official runs against each other (batch size and
transformers version); (d) identical tokens, verdict step and max |gap diff| vs the reference, for the generated and the
teacher-forced score; (e) the teacher-forced default rule: the generated answer starts with the forced prefix
("▁▁ This is a") and the verdict is the next token, "This is a" text, |P(fake) diff| <= 1e-3; on the 16 images and on an
extended set (``--extended``, 8 images per class folder of GenImage val, SynthWildX and WildRF) where the generation is
cut at 8 tokens (greedy is causal: the first 8 tokens and their logits are those of the full generation); (f)
throughput and peak memory of both engine scorers.

    python repro/fakevlm_fidelity.py --data ROOT --official NAME=OUT.json [--official ...] --official-inputs IN.npz \
        [--extended EXT.txt] --out repro/fakevlm_fidelity.json
"""
import argparse
import dataclasses
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer, LlavaForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface")]

from interface.backend import config as backend_config               # noqa: E402
from interface.backend import media                                  # noqa: E402
from interface.backend.engine import Engine                          # noqa: E402
from interface.backend.registry import get_spec                      # noqa: E402
from models.image.fakevlm.fakevlm import FAKEVLM, FAKEVLM_TEACHER_FORCED   # noqa: E402
from models.vlm.parsers import parse_fakevlm                         # noqa: E402

# 16 images, real and fake, 256x256 to 8 MP: the gate's six GenImage val images, seven SynthWildX images, two transparent
# (RGBA) WildRF PNGs (alpha dropped) and one 8 MP WildRF JPEG; the same 16 as repro/adb_vlm_fidelity.py.
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
MAX_NEW = 256
SHORT_NEW = 8                                                # the extended set: the answer's first 8 tokens
IMAGE_TOKEN, PAD, N_IMAGE_TOKENS = 32000, 32001, 576


def official_test_json(path: Path) -> None:
    """The official test-file format for IMAGES (labels real=1 / fake=0, as FakeClue)."""
    rows = [{"image": rel, "label": 0 if lab else 1, "cate": "fidelity",
             "conversations": [{"from": "human", "value": FAKEVLM.user_prompt}, {"from": "gpt", "value": ""}]}
            for rel, lab in IMAGES]
    path.write_text(json.dumps(rows, indent=1) + "\n")


def expand(ids: np.ndarray) -> torch.Tensor:
    """A 4.45.2 input row (one ``<image>``, left-padded to 1024) with the image token repeated 576 times and the left
    padding shortened so the row stays 1024 long (the layout transformers 5 merges the image features into)."""
    ids = [int(t) for t in ids if t != PAD]
    k = ids.index(IMAGE_TOKEN)
    ids = ids[:k] + [IMAGE_TOKEN] * N_IMAGE_TOKENS + ids[k + 1:]
    return torch.tensor([[PAD] * (1024 - len(ids)) + ids])


def first_verdict(gen: list, fake: int, real: int):
    return next((t for t, g in enumerate(gen) if g in (fake, real)), None)


def reference(model_dir: Path, tok, official_inputs: dict, fake: int, real: int) -> list:
    model = LlavaForConditionalGeneration.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa",
                                                          device_map="cuda").eval()
    rows = []
    for i, (rel, label) in enumerate(IMAGES):
        assert str(official_inputs["images"][i]) == rel
        ids = expand(official_inputs[f"{i}/input_ids"][0]).cuda()
        att = (ids != PAD).long()
        pixels = torch.from_numpy(official_inputs[f"{i}/pixel_values"]).cuda()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=att, pixel_values=pixels, max_new_tokens=MAX_NEW,
                                 do_sample=False, output_logits=True, return_dict_in_generate=True)
        torch.cuda.synchronize()
        seconds = time.time() - t0
        gen = out.sequences[0, ids.shape[1]:].tolist()
        step = first_verdict(gen, fake, real)
        gap = tf_gap = None
        if step is not None:
            gap = (out.logits[step][0, fake].float() - out.logits[step][0, real].float()).item()
            forced = torch.cat([ids, torch.tensor([gen[:step]], device="cuda")], 1)
            with torch.inference_mode():
                row = model(input_ids=forced, attention_mask=(forced != PAD).long(), pixel_values=pixels,
                            logits_to_keep=1).logits[0, -1].float()
            tf_gap = (row[fake] - row[real]).item()
        response = tok.decode(gen, skip_special_tokens=True)
        rows.append({"image": rel, "label_fake": label, "generated_ids": gen, "response": response,
                     "official_string": tok.decode(out.sequences[0].tolist(), skip_special_tokens=True).split("?")[-1],
                     "parse": parse_fakevlm(response), "verdict_step": step, "gap": gap, "teacher_forced_gap": tf_gap,
                     "seconds": round(seconds, 2)})
        print(f"[ref] {rel} step={step} gap={gap} tf_gap={tf_gap} parse={rows[-1]['parse']} {len(gen)} tok "
              f"{seconds:.1f}s", flush=True)
        del out
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def released_key(key: str) -> str:
    """A key of the released (transformers 4.45) LLaVA shards under transformers 5's module names."""
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model."):]
    if key.startswith("language_model.lm_head."):
        return key[len("language_model."):]
    if key.startswith(("vision_tower.", "multi_modal_projector.")):
        return "model." + key
    return key


def tensor_check(vlm, folder: Path) -> dict:
    loaded = vlm.hf.state_dict()
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    equal, differ = 0, []
    for shard in sorted(set(index.values())):
        with safe_open(str(folder / shard), framework="pt", device="cuda") as f:
            for key in f.keys():
                k = released_key(key)
                if k in loaded and torch.equal(loaded[k], f.get_tensor(key)):
                    equal += 1
                else:
                    differ.append(key)
    mapped = {released_key(k) for k in index}
    return {"checkpoint_tensors": len(index), "model_tensors": len(loaded), "equal": equal,
            "differ_or_absent": differ[:20], "model_only": sorted(set(loaded) - mapped)[:20]}


def timed(engine, spec, pil) -> dict:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    o = engine.vlm_score(spec, pil)
    torch.cuda.synchronize()
    o["seconds"] = round(time.time() - t0, 3)
    o["peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return o


def p_of(gap: float) -> float:
    return float(torch.sigmoid(torch.tensor(gap, dtype=torch.float64)))


def first_sentence(text: str) -> str:
    """What the official parser reads first: the text after the last '?', up to the first '.'."""
    return text.split("?")[-1].split(".")[0].strip()


def divergence(a: str, b: str):
    """Index of the first differing character, None if equal."""
    if a == b:
        return None
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="root of the IMAGES (and --extended) paths")
    ap.add_argument("--official", action="append", default=[],
                    help="NAME=PATH: an official output JSON (fakevlm_official.py eval / single), with PATH.acc.json")
    ap.add_argument("--official-inputs", help="fakevlm_official.py inputs (4.45.2) for IMAGES")
    ap.add_argument("--extended", default=None, help="extra image paths (relative to --data), one per line")
    ap.add_argument("--write-test-json", default=None, help="only write the official test JSON for IMAGES and exit")
    ap.add_argument("--out", default=str(ROOT / "repro" / "fakevlm_fidelity.json"))
    args = ap.parse_args()
    if args.write_test_json:
        official_test_json(Path(args.write_test_json))
        return 0
    if not (args.data and args.official and args.official_inputs):
        ap.error("--data, --official and --official-inputs are required")
    out_path, data = Path(args.out), Path(args.data)
    spec = get_spec("fakevlm")
    assert spec.vlm is FAKEVLM and FAKEVLM.verdict_mode == "generate"
    folder = backend_config.CHECKPOINTS_DIR / spec.checkpoint
    proc_dir = backend_config.CHECKPOINTS_DIR / spec.vlm.processor
    result = {"weights": str(folder), "processor": str(proc_dir), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "transformers": transformers.__version__, "max_new_tokens": MAX_NEW,
              "registry_verdict_mode": spec.vlm.verdict_mode, "images": [{"image": r, "label_fake": l} for r, l in IMAGES]}

    def save():
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")

    official = {}
    for item in args.official:
        name, path = item.split("=", 1)
        rows = {Path(r["image_path"]).resolve(): r["output"] for r in json.loads(Path(path).read_text())}
        meta = json.loads(Path(path + ".acc.json").read_text())
        official[name] = {"outputs": [rows[(data / rel).resolve()] for rel, _ in IMAGES], "meta": meta}
    result["official"] = official
    tok = AutoTokenizer.from_pretrained(str(proc_dir), local_files_only=True)
    fake, real = tok.convert_tokens_to_ids(["▁fake", "▁real"])
    inputs = dict(np.load(args.official_inputs))
    result["reference"] = ref = reference(folder, tok, inputs, fake, real)
    save()

    # Engine: strict load, the registry scorer (generated) timed, then the teacher-forced variant.
    engine = Engine(device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = engine._get_model(spec)
    result["load"] = {"status": engine._status[spec.key], "seconds": round(time.time() - t0, 1),
                      "weights_GB": round(torch.cuda.memory_allocated() / 1e9, 2)}
    result["load"]["tensors"] = tensor_check(model, folder)
    print("[load]", result["load"], flush=True)
    checks = []
    for i, (rel, _) in enumerate(IMAGES):                    # (b) the engine's processor inputs vs official 4.45.2
        x = model.inputs(model.image(media.decode_image_raw((data / rel).read_bytes())))
        checks.append({"image": rel, "pixel_max_abs_diff": float(np.abs(x["pixel_values"].float().cpu().numpy()
                                                                        - inputs[f"{i}/pixel_values"]).max()),
                       "input_ids_equal": x["input_ids"].cpu().tolist() == expand(inputs[f"{i}/input_ids"][0]).tolist(),
                       "official_size": list(map(int, inputs[f"{i}/pixel_values"].shape[-2:]))})
    result["inputs_vs_official_4452"] = checks
    save()
    for rel, _ in IMAGES[:2]:                                # warm-up (kernels, allocator), not timed
        engine.vlm_score(spec, media.decode_image_raw((data / rel).read_bytes()))
    gen = []
    for rel, label in IMAGES:
        pil = media.decode_image_raw((data / rel).read_bytes())
        o = timed(engine, spec, pil)
        ids = model.inputs(model.image(pil))["input_ids"][0].tolist()
        o["official_string"] = model.tokenizer.decode(ids + o["generated_ids"], skip_special_tokens=True).split("?")[-1]
        o.update(image=rel, label_fake=label, size=list(pil.size))
        gen.append(o)
        print(f"[engine gen] {rel} step={o['verdict_step']} logit={o['logit']:.4f} verdict={o['verdict']} "
              f"{o['new_tokens']} tok {o['seconds']}s {o['peak_GB']}GB | {o['response'][:60]!r}", flush=True)
    result["engine_generate"] = gen
    model.cfg = FAKEVLM_TEACHER_FORCED
    tf = []
    try:
        for rel, label in IMAGES:
            pil = media.decode_image_raw((data / rel).read_bytes())
            o = timed(engine, spec, pil)
            o.update(image=rel, label_fake=label, size=list(pil.size))
            tf.append(o)
            print(f"[engine tf] {rel} {pil.size} logit={o['logit']:.4f} {o['seconds']}s {o['peak_GB']}GB", flush=True)
    finally:
        model.cfg = FAKEVLM
    result["engine_teacher_forced"] = tf
    save()

    # Extended set: generated (first 8 tokens) vs teacher-forced.
    ext = []
    if args.extended:
        rels = [r for r in Path(args.extended).read_text().split("\n") if r and r not in {i for i, _ in IMAGES}]
        short = dataclasses.replace(FAKEVLM, decoding={"do_sample": False, "max_new_tokens": SHORT_NEW})
        for rel in rels:
            pil = media.decode_image_raw((data / rel).read_bytes())
            try:
                model.cfg = FAKEVLM_TEACHER_FORCED
                t = timed(engine, spec, pil)
                model.cfg = short
                g = timed(engine, spec, pil)
            finally:
                model.cfg = FAKEVLM
            ext.append({"image": rel, "label_fake": int("/1_fake/" in rel), "size": list(pil.size),
                        "generated_ids": g["generated_ids"], "response": g["response"], "verdict_step": g["verdict_step"],
                        "generated_logit": g["logit"], "generated_p": g["fake_probability"], "parse": g["verdict"],
                        "tf_logit": t["logit"], "tf_p": t["fake_probability"], "tf_seconds": t["seconds"],
                        "tf_peak_GB": t["peak_GB"]})
        result["extended"] = ext
        save()

    # Summary.
    n = len(IMAGES)
    prefix_ids = tok.convert_tokens_to_ids(list(FAKEVLM.verdict_sequences[0][:-1]))

    def same_position(ids, step):
        return step == len(prefix_ids) and ids[:len(prefix_ids)] == prefix_ids

    s = {"n_images": n, "registry_scorer": "generate (FAKEVLM)", "forced_prefix_tokens": list(FAKEVLM.verdict_sequences[0][:-1])}
    vs = {}
    for name, run in official.items():
        o, meta = run["outputs"], run["meta"]
        vs[name] = {
            "transformers": meta["transformers"], "batch_size": meta["val_batch_size"],
            "identical_text": f"{sum(a == g['official_string'] for a, g in zip(o, gen))}/{n}",
            "identical_text_stripped": f"{sum(a.strip() == g['response'].strip() for a, g in zip(o, gen))}/{n}",
            "identical_first_sentence": f"{sum(first_sentence(a) == first_sentence(g['response']) for a, g in zip(o, gen))}/{n}",
            "identical_parsed_verdict": f"{sum(parse_fakevlm(a) == g['verdict'] for a, g in zip(o, gen))}/{n}",
            "first_divergence_char": [divergence(g["official_string"], a) for a, g in zip(o, gen)],
            "official_accuracy": meta["accuracy"].get("global_stats"),
            "wall_seconds_incl_load": meta["wall_seconds_incl_load"], "peak_GB": meta["peak_GB"]}
    s["engine_vs_official"] = vs
    names = list(official)
    s["official_vs_official_identical_text"] = {
        f"{a} vs {b}": f"{sum(x == y for x, y in zip(official[a]['outputs'], official[b]['outputs']))}/{n}"
        for k, a in enumerate(names) for b in names[k + 1:]}
    s["inputs_pixel_max_abs_diff_vs_official_4452"] = max(c["pixel_max_abs_diff"] for c in checks)
    s["inputs_ids_equal_vs_official_4452"] = f"{sum(c['input_ids_equal'] for c in checks)}/{n}"
    s["vs_reference_identical_tokens"] = f"{sum(r['generated_ids'] == g['generated_ids'] for r, g in zip(ref, gen))}/{n}"
    s["vs_reference_verdict_steps_equal"] = all(r["verdict_step"] == g["verdict_step"] for r, g in zip(ref, gen))
    s["vs_reference_identical_parsed_verdict"] = f"{sum(r['parse'] == g['verdict'] for r, g in zip(ref, gen))}/{n}"
    s["max_abs_logit_gap_diff_generate_engine_vs_reference"] = max(abs(r["gap"] - g["logit"]) for r, g in zip(ref, gen))
    s["max_abs_logit_gap_diff_teacher_forced_engine_vs_reference"] = \
        max(abs(r["teacher_forced_gap"] - t["logit"]) for r, t in zip(ref, tf))
    s["verdict_steps"] = [g["verdict_step"] for g in gen]
    s["generated_starts_with_forced_prefix_and_verdict_next"] = \
        f"{sum(same_position(g['generated_ids'], g['verdict_step']) for g in gen)}/{n}"
    s["generated_text_starts_with_This_is_a"] = f"{sum(g['response'].strip().startswith('This is a') for g in gen)}/{n}"
    diffs = [abs(g["fake_probability"] - t["fake_probability"]) for g, t in zip(gen, tf)]
    s["max_abs_p_fake_diff_teacher_forced_vs_generated"] = max(diffs)
    s["n_p_fake_diff_over_1e-3"] = sum(d > 1e-3 for d in diffs)
    s["max_abs_logit_diff_teacher_forced_vs_generated"] = max(abs(g["logit"] - t["logit"]) for g, t in zip(gen, tf))
    s["teacher_forced_vs_generated_verdict_agree"] = \
        f"{sum((t['logit'] > 0) == (g['verdict'] == 'fake') for g, t in zip(gen, tf))}/{n}"
    s["reference_teacher_forced_on_generated_prefix_vs_generated_max_abs_p_diff"] = \
        max(abs(p_of(r["gap"]) - p_of(r["teacher_forced_gap"])) for r in ref)
    s["teacher_forced_default_rule_met"] = (s["generated_starts_with_forced_prefix_and_verdict_next"] == f"{n}/{n}"
                                            and s["max_abs_p_fake_diff_teacher_forced_vs_generated"] <= 1e-3
                                            and s["generated_text_starts_with_This_is_a"] == f"{n}/{n}")
    s["engine_correct_generate"] = sum((g["verdict"] == "fake") == bool(g["label_fake"]) for g in gen)
    s["engine_correct_teacher_forced"] = sum((t["logit"] > 0) == bool(t["label_fake"]) for t in tf)
    s["engine_unparsed_generate"] = sum(g["verdict"] is None for g in gen)
    s["engine_score_sources_generate"] = sorted({g["score_source"] for g in gen})
    s["tensors_all_equal"] = result["load"]["tensors"]["equal"] == result["load"]["tensors"]["checkpoint_tensors"]
    s["generate_images_per_s"] = round(n / sum(g["seconds"] for g in gen), 4)
    s["generate_seconds_per_image"] = round(sum(g["seconds"] for g in gen) / n, 2)
    s["generate_tokens_per_s"] = round(sum(g["new_tokens"] for g in gen) / sum(g["seconds"] for g in gen), 1)
    s["generate_new_tokens"] = [g["new_tokens"] for g in gen]
    s["teacher_forced_images_per_s"] = round(n / sum(t["seconds"] for t in tf), 3)
    s["generate_peak_GB_max"] = max(g["peak_GB"] for g in gen)
    s["teacher_forced_peak_GB_max"] = max(t["peak_GB"] for t in tf)
    s["reference_images_per_s"] = round(n / sum(r["seconds"] for r in ref), 4)
    if ext:
        m = len(ext)
        pdiff = [abs(e["generated_p"] - e["tf_p"]) for e in ext]
        s["extended"] = {
            "n_images": m, "generated_tokens": SHORT_NEW,
            "starts_with_forced_prefix_and_verdict_next": f"{sum(same_position(e['generated_ids'], e['verdict_step']) for e in ext)}/{m}",
            "text_starts_with_This_is_a": f"{sum(e['response'].strip().startswith('This is a') for e in ext)}/{m}",
            "other_starts": sorted({repr(e["response"][:12]) + " ids " + str(e["generated_ids"][:5]) for e in ext
                                    if not same_position(e["generated_ids"], e["verdict_step"])}),
            "max_abs_p_fake_diff": max(pdiff), "n_p_diff_over_1e-3": sum(d > 1e-3 for d in pdiff),
            "max_abs_logit_diff": max(abs(e["generated_logit"] - e["tf_logit"]) for e in ext),
            "verdict_agree": f"{sum((e['tf_logit'] > 0) == (e['parse'] == 'fake') for e in ext)}/{m}",
            "teacher_forced_images_per_s": round(m / sum(e["tf_seconds"] for e in ext), 3),
            "teacher_forced_peak_GB_max": max(e["tf_peak_GB"] for e in ext),
            "correct_generated_verdict": sum((e["parse"] == "fake") == bool(e["label_fake"]) for e in ext),
            "correct_teacher_forced": sum((e["tf_logit"] > 0) == bool(e["label_fake"]) for e in ext),
        }
    result["summary"] = s
    save()
    print("[summary]", json.dumps(s, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
