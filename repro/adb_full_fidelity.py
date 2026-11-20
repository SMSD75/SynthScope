"""Align, Don't Blend FULL method (``adb_full``: VLM branch OR the DDA pixel expert) fidelity check: SynthScope's
``Engine.predict_image`` vs the official ``predict.py`` (github.com/GzZning/Align-Dont-Blend @ 5d83e40), unmodified, with
its README's expert command (``python -m detector.expert --checkpoint DDA_ckpt.pth``), on the same GPU.

Images: the 16 of ``repro/adb_vlm_fidelity.py`` plus up to 8 GenImage val images on which the two components disagree
(``select``): a pool of 43 GenImage subset images whose cached ``adb_vlm`` verdict is wrong (fakes the VLM calls real,
reals it calls fake; ``repro/vlm_bench/results/genimage_val100k/adb_vlm.json``) is scored by SynthScope's ``dda``, and the
4 + 4 images with (VLM real, DDA fake) and (VLM fake, DDA real) closest to DDA's threshold are kept, plus the pool image
whose DDA probability is closest to 0.5.

Official runs (the sbatch, before ``compare``):
* ``predict.py --infer-backend pt`` (ms-swift 4.4.2; the model's own dtype, bf16, and attention, as ``adb_vlm``'s
  fidelity check ran it) with ``--expert-command``: per image the final decision, the DDA decision, the VLM decision
  and the VLM response;
* ``python -m detector.expert`` alone on the same list: the expert's probabilities (``predict.py`` keeps only the
  decision);
* ``predict.py`` with its README's full command (``--torch-dtype float32 --attn-impl eager --device-map auto``).

``compare`` scores every image with ``Engine.predict_image(get_spec("adb_full"), ...)`` on the file as
``synthscope.load`` decodes it for this detector (``media.decode_image_raw``), with the registry config (bf16, SDPA),
and once more with the VLM in float32 + eager attention (the README command). Checks, per image, against the run of
the same precision: identical final, DDA and VLM decisions, identical VLM text, the fused score on the right side of 0.5;
and the DDA probability against the official expert's.

Usage (see ``repro/adb_full_fidelity.sbatch``):
    python repro/adb_full_fidelity.py select --data ROOT --out DIR/selection.json --inputs DIR/inputs.txt
    python repro/adb_full_fidelity.py compare --data ROOT --selection DIR/selection.json --official-bf16 A.jsonl \
        --official-fp32 B.jsonl --expert-probs E.jsonl --out repro/adb_full_fidelity.json
"""
import argparse
import dataclasses
import gc
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface")]

from interface.backend import media                                  # noqa: E402
from interface.backend import registry                               # noqa: E402
from interface.backend.engine import Engine                          # noqa: E402
from interface.backend.registry import get_spec                      # noqa: E402
from models.image.adb_full.adb_full import EXPERT_THRESHOLD          # noqa: E402
from repro.adb_vlm_fidelity import IMAGES                            # noqa: E402

SHARED = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
OFFICIAL_DIR = SHARED / "vlm_ref" / "Align-Dont-Blend-20261120"        # github.com/GzZning/Align-Dont-Blend @ 5d83e40
SWIFT_DIR = SHARED / "vlm_pydeps" / "swift442"
GENIMAGE = "GenImage_val_full/test"
POOL_RESULTS = ROOT / "repro" / "vlm_bench" / "results" / "genimage_val100k" / "adb_vlm.json"
POOL_FAKE_GENERATORS = ("adm", "biggan", "vqdm", "glide")              # the VLM's weakest GenImage generators
N_FAKE_PER_GENERATOR = 5                                              # fakes the cached VLM run calls real
N_REAL_PER_GENERATOR = 3                                              # reals the cached VLM run calls fake (8 generators)
N_PER_GROUP = 4


def pool() -> list[dict]:
    """The candidate pool: GenImage subset images on which the cached ``adb_vlm`` verdict is wrong (sorted by key)."""
    fixed = {rel for rel, _ in IMAGES}
    rows = sorted(json.loads(POOL_RESULTS.read_text())["per_image"], key=lambda r: r["key"])
    out, count = [], {}
    for r in rows:
        rel = f"{GENIMAGE}/{r['key']}"
        if rel in fixed:
            continue
        fake_missed = r["label"] == 1 and r["verdict"] == "real" and r["generator"] in POOL_FAKE_GENERATORS
        real_missed = r["label"] == 0 and r["verdict"] == "fake"
        cap = N_FAKE_PER_GENERATOR if fake_missed else N_REAL_PER_GENERATOR if real_missed else 0
        slot = (r["generator"], r["label"])
        if cap and count.get(slot, 0) < cap:
            count[slot] = count.get(slot, 0) + 1
            out.append({"image": rel, "label_fake": r["label"], "cached_vlm_verdict": r["verdict"],
                        "cached_vlm_logit": r["logit"]})
    return out


def select(args) -> int:
    data = Path(args.data)
    engine = Engine(device="cuda")
    spec = get_spec("dda")
    candidates = pool()
    for c in candidates:
        c["dda_fake_probability"] = engine.expert_probability(spec, media.open_rgb(data / c["image"]))
        print(f"[pool] {c['image']} vlm(cached)={c['cached_vlm_verdict']} dda={c['dda_fake_probability']:.6f}", flush=True)
    near = sorted(candidates, key=lambda c: abs(c["dda_fake_probability"] - EXPERT_THRESHOLD))

    def group(vlm_verdict, dda_fake):
        return [c for c in near if c["cached_vlm_verdict"] == vlm_verdict
                and (c["dda_fake_probability"] >= EXPERT_THRESHOLD) == dda_fake][:N_PER_GROUP]
    chosen = group("real", True) + group("fake", False)
    if near and near[0] not in chosen:
        chosen.append(near[0])                                  # the DDA probability closest to the threshold
    chosen.sort(key=lambda c: c["image"])
    Path(args.out).write_text(json.dumps({"pool": candidates, "selected": [c["image"] for c in chosen]}, indent=1))
    images = [rel for rel, _ in IMAGES] + [c["image"] for c in chosen]
    Path(args.inputs).write_text("".join(f"{data / rel}\n" for rel in images))
    print(f"[select] pool {len(candidates)}; selected {len(chosen)}: {[c['image'] for c in chosen]}", flush=True)
    return 0


def read_jsonl(path: Path, key: str = "image") -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(Path(r[key]).resolve()): r for r in rows}


def run_engine(engine: Engine, images: list[tuple[str, int]], data: Path, tag: str) -> list[dict]:
    spec = get_spec("adb_full")
    out = []
    for rel, label in images:
        pil = media.decode_image_raw((data / rel).read_bytes())
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        res = engine.predict_image(spec, pil)
        torch.cuda.synchronize()
        comp = res["components"]
        row = {"image": rel, "label_fake": label, "size": list(pil.size), "mode": pil.mode,
               "verdict": res["verdict"], "fused_score": res["fused_score"], "decided_by": res["decided_by"],
               "vlm_verdict": comp["adb_vlm"]["verdict"], "vlm_fake_probability": comp["adb_vlm"]["fake_probability"],
               "vlm_logit": comp["adb_vlm"]["logit"], "vlm_score_source": comp["adb_vlm"]["score_source"],
               "vlm_new_tokens": comp["adb_vlm"]["new_tokens"], "vlm_input_size": comp["adb_vlm"]["input_size"],
               "dda_verdict": comp["dda"]["verdict"], "dda_fake_probability": comp["dda"]["fake_probability"],
               "response": res["response"], "seconds": round(time.time() - t0, 2),
               "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        out.append(row)
        print(f"[{tag}] {rel} final={row['verdict']} vlm={row['vlm_verdict']} dda={row['dda_verdict']} "
              f"(p {row['dda_fake_probability']:.6f}) score={row['fused_score']:.6f} {row['seconds']}s "
              f"{row['peak_GB']}GB", flush=True)
    return out


def check(ours: list[dict], official: dict, data: Path, expert: dict | None = None) -> dict:
    rows, n = [], len(ours)
    for o in ours:
        a = official[str((data / o["image"]).resolve())]
        row = {"image": o["image"],
               "official": {"fake": a["fake"], "vlm_fake": a["vlm_fake"], "expert_fake": a["expert_fake"]},
               "final_equal": (o["verdict"] == "fake") == a["fake"],
               "vlm_equal": (o["vlm_verdict"] == "fake") == a["vlm_fake"] and o["vlm_verdict"] != "unparsed",
               "dda_equal": (o["dda_verdict"] == "fake") == a["expert_fake"],
               "text_equal": o["response"] == a["vlm_response"],
               "score_side_ok": (o["fused_score"] >= 0.5) == (o["verdict"] == "fake")}
        if expert is not None:
            p = expert[str((data / o["image"]).resolve())]["fake_probability"]
            row["official_dda_fake_probability"] = p
            row["dda_abs_diff"] = abs(p - o["dda_fake_probability"])
        rows.append(row)
    summary = {k: f"{sum(r[k] for r in rows)}/{n}"
               for k in ("final_equal", "dda_equal", "vlm_equal", "text_equal", "score_side_ok")}
    summary["all_decisions_identical"] = all(r["final_equal"] and r["dda_equal"] and r["vlm_equal"] for r in rows)
    if expert is not None:
        summary["max_abs_dda_probability_diff"] = max(r["dda_abs_diff"] for r in rows)
    summary["disagreements_vlm_vs_dda"] = sum(r["official"]["vlm_fake"] != r["official"]["expert_fake"] for r in rows)
    summary["decided_by_expert_only"] = sum(r["official"]["expert_fake"] and not r["official"]["vlm_fake"] for r in rows)
    summary["decided_by_vlm_only"] = sum(r["official"]["vlm_fake"] and not r["official"]["expert_fake"] for r in rows)
    return {"summary": summary, "per_image": rows}


def versions() -> dict:
    """The engine's packages; the official run uses the same interpreter with ms-swift's folder first on its path."""
    def pinned(name):
        dist = next(SWIFT_DIR.glob(f"{name}-*.dist-info"), None)
        return dist.name.removeprefix(f"{name}-").removesuffix(".dist-info") if dist else None
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "peft": importlib.metadata.version("peft"), "pillow": importlib.metadata.version("pillow"),
            "python": sys.version.split()[0], "gpu": torch.cuda.get_device_name(0),
            "official_vlm_process": {"ms_swift": pinned("ms_swift"), "peft": pinned("peft"),
                                     "qwen_vl_utils": pinned("qwen_vl_utils"),
                                     "torch / transformers / pillow": "as the engine (same interpreter)"}}


def compare(args) -> int:
    data = Path(args.data)
    selection = json.loads(Path(args.selection).read_text())
    labels = {c["image"]: c["label_fake"] for c in selection["pool"]}
    images = list(IMAGES) + [(rel, labels[rel]) for rel in selection["selected"]]
    result = {"official_code": "github.com/GzZning/Align-Dont-Blend @ 5d83e40 (predict.py, detector/expert.py, unmodified)",
              "versions": versions(), "n_images": len(images), "images": [rel for rel, _ in images],
              "selection": selection, "expert_threshold": EXPERT_THRESHOLD}
    out_path = Path(args.out)

    def save():
        out_path.write_text(json.dumps(result, indent=1) + "\n")

    official_bf16, official_fp32 = read_jsonl(Path(args.official_bf16)), read_jsonl(Path(args.official_fp32))
    expert = read_jsonl(Path(args.expert_probs))

    engine = Engine(device="cuda")
    ours = run_engine(engine, images, data, "bf16")
    result["engine_bf16"] = ours
    result["status"] = {k: engine._status.get(k) for k in ("adb_vlm", "dda")}
    result["vs_official_bf16"] = check(ours, official_bf16, data, expert)
    save()
    print("[bf16]", json.dumps(result["vs_official_bf16"]["summary"]), flush=True)
    engine.release_gpu()
    del engine
    gc.collect()

    vlm_spec = get_spec("adb_vlm")
    registry.MODELS["adb_vlm"] = dataclasses.replace(
        vlm_spec, vlm=dataclasses.replace(vlm_spec.vlm, dtype="float32", attn_implementation="eager"))
    try:
        engine = Engine(device="cuda")
        ours32 = run_engine(engine, images, data, "fp32")
        result["engine_fp32_eager"] = ours32
        result["vs_official_fp32_eager"] = check(ours32, official_fp32, data, expert)
        engine.release_gpu()
    finally:
        registry.MODELS["adb_vlm"] = vlm_spec
    print("[fp32]", json.dumps(result["vs_official_fp32_eager"]["summary"]), flush=True)

    # The two official runs against each other (bf16 SDPA vs the README's float32 eager).
    keys = [str((data / rel).resolve()) for rel, _ in images]
    result["official_bf16_vs_fp32"] = {
        k: f"{sum(official_bf16[i][f] == official_fp32[i][f] for i in keys)}/{len(keys)}"
        for k, f in (("final_equal", "fake"), ("vlm_equal", "vlm_fake"), ("dda_equal", "expert_fake"),
                     ("text_equal", "vlm_response"))}
    n = len(images)
    result["summary"] = {
        "n_images": n,
        "bf16": result["vs_official_bf16"]["summary"],
        "fp32_eager": result["vs_official_fp32_eager"]["summary"],
        "official_bf16_vs_fp32": result["official_bf16_vs_fp32"],
        "correct_bf16": {"final": sum((o["verdict"] == "fake") == bool(o["label_fake"]) for o in ours),
                         "vlm": sum((o["vlm_verdict"] == "fake") == bool(o["label_fake"]) for o in ours),
                         "dda": sum((o["dda_verdict"] == "fake") == bool(o["label_fake"]) for o in ours)},
        "engine_bf16_seconds_per_image": round(sum(o["seconds"] for o in ours) / n, 2),
        "engine_bf16_peak_GB_max": max(o["peak_GB"] for o in ours),
        "engine_fp32_seconds_per_image": round(sum(o["seconds"] for o in ours32) / n, 2),
        "engine_fp32_peak_GB_max": max(o["peak_GB"] for o in ours32),
    }
    if args.official_seconds:
        result["summary"]["official_wall_seconds_incl_load"] = json.loads(args.official_seconds)
    save()
    print("[summary]", json.dumps(result["summary"], indent=1), flush=True)
    return 0 if result["summary"]["bf16"]["all_decisions_identical"] else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--inputs", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--data", required=True)
    c.add_argument("--selection", required=True)
    c.add_argument("--official-bf16", required=True)
    c.add_argument("--official-fp32", required=True)
    c.add_argument("--expert-probs", required=True)
    c.add_argument("--official-seconds", default="", help='JSON, e.g. {"bf16": 300, "fp32": 900}')
    c.add_argument("--out", default=str(ROOT / "repro" / "adb_full_fidelity.json"))
    args = ap.parse_args()
    return select(args) if args.cmd == "select" else compare(args)


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    sys.exit(main())
