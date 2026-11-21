"""Align, Don't Blend FULL method (``adb_full``) on the paper's benchmarks: fuse per-image ``adb_vlm`` records with per-image
``dda`` records by the official rule (``models/image/adb_full/adb_full.py`` ``fuse_components``).

Inputs, per row (``ROWS``):
* the VLM side: ``bench_eval`` result JSONs of ``adb_vlm`` (``per_image``: key, label, verdict, logit; bf16 greedy unless the
  row says otherwise). The committed subset / WildRF results of ``repro/vlm_bench/results`` (RTX A5000) are read as they
  are: their records are the per-image cache content, traced to the job logs by ``repro/vlm_bench/trace_check.py``. New
  runs are in ``repro/adb_full/vlm`` (``repro/adb_full/bench_a5000.sbatch`` / ``fp32_a6000.sbatch``);
* the expert side: ``bench_eval --model dda --per-image`` on the same files (``repro/adb_full/dda``): key, sha1, label,
  P(fake), with the same ``--jpeg-fakes`` / ``--limit`` / ``--select``.

Checks, per row: the same (generator, key) pairs with the same labels on both sides; the expert's file sha1 equals the VLM
record's (new runs) or the subset manifest's (``repro/vlm_bench/subsets``; committed runs, whose records have no sha1).
P_vlm is the sigmoid of the record's logit, which ``bench_eval`` rounds to 4 decimals. That changes no decision (the VLM
decision is its parsed verdict, the expert's is unrounded) and can affect AUROC / AP only through ties.
Per generator: balanced accuracy of the fused verdict (an unparsed VLM answer with a "real" expert counts as wrong; there
were none), AUROC / AP of the fused score, and the same for each component (VLM: parsed verdict / logit; DDA: P >= 0.5 /
P). Means over the generators, as the paper (and ``bench_eval``) average them.

Output: ``repro/adb_full/results/<row>.json`` (with ``per_image``) and ``repro/adb_full/paper_comparison.json``; one
``=== RESULT adbfull_<row>: mAP=.. mAUROC=.. acc=.. over N generators`` line per row (run it with its output in
``repro/logs/adb_full_fuse.out`` so ``trace_check.py`` can find the lines).

    python repro/adb_full/fuse.py | tee repro/logs/adb_full_fuse.out
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO)]

from models.image.adb_full.adb_full import fuse_components   # noqa: E402

VB = REPO / "repro" / "vlm_bench"
A5000_COMMITTED = "RTX A5000 (IVI, committed adb_vlm run)"
DDA_LOG = "repro/logs/adbfull_bench_297122.out"

# Paper (arXiv 2512.06746 v3), AlignGemini = the full method; balanced accuracy.
PAPER = {
    "WildRF": {"facebook": 92.2, "reddit": 90.9, "twitter": 95.5, "mean": 92.9},
    "AIGI-Bench": {"CommunityAI": 91.9, "SocialRF": 85.6, "mean": 88.8},
    "BFree-Online": {"mean": 97.5},
    "GenImage": {"midjourney": 89.3, "sdv1.4": 89.6, "sdv1.5": 95.6, "adm": 77.2, "glide": 90.3, "wukong": 98.3,
                 "vqdm": 98.1, "biggan": 98.4, "mean": 91.7},
    "AIGCDetectBenchmark": {"ADM": 89.3, "DALLE2": 94.9, "Glide": 90.3, "Midjourney": 95.6, "VQDM": 77.2, "biggan": 90.4,
                            "cyclegan": 74.4, "gaugan": 92.5, "progan": 91.0, "sd_xl": 99.1, "stable_diffusion_v_1_4": 98.3,
                            "stable_diffusion_v_1_5": 98.1, "stargan": 72.9, "stylegan": 86.6, "stylegan2": 89.2,
                            "whichfaceisreal": 52.6, "wukong": 98.4, "mean": 87.7},
}
PAPER_TABLE = {"WildRF": "Table 1", "AIGI-Bench": "Table 1", "BFree-Online": "Table 1", "GenImage": "Table 6 (appendix)",
               "AIGCDetectBenchmark": "Table 8 (appendix)"}

# row -> sources. "vlm" / "dda": lists of JSONs (relative to the repo), joined; "generators": the generators kept
# (None = all); "manifest": the subset manifest whose sha1s the expert's files must match (committed VLM runs); "vlm_logs":
# the job logs whose per-image lines the VLM records trace to (trace_check.py; reused cached records trace to the log
# of the run that scored them). Every dda record traces to repro/logs/adbfull_bench_297122.out.
ROWS = {
    "wildrf_full": {
        "vlm_logs": ["repro/logs/adb_ivi_*.out"],
        "paper": "WildRF", "main": True, "data": "full test split (Facebook, Reddit, Twitter)", "precision": "bf16",
        "gpu": A5000_COMMITTED, "vlm": ["repro/vlm_bench/results/wildrf/adb_vlm__full.json"],
        "dda": ["repro/adb_full/dda/wildrf/dda__full.json"], "generators": None, "manifest": None},
    "wildrf_full_fp32": {
        "vlm_logs": ["repro/logs/adbfull_fp32_297123.out"],
        "paper": "WildRF", "main": False, "data": "full test split", "precision": "float32 + eager attention",
        "gpu": None, "vlm": ["repro/adb_full/vlm/wildrf/adb_vlm__fullfp32.json"],
        "dda": ["repro/adb_full/dda/wildrf/dda__full.json"], "generators": None, "manifest": None},
    "aigibench": {
        "vlm_logs": ["repro/logs/adbfull_bench_297122.out", "repro/logs/adbfull_bench_297122_w2.out"],
        "paper": "AIGI-Bench", "main": True,
        "data": "SocialRF full (3,000 real + 3,000 fake); CommunityAI the first 1,500 real + 1,500 fake (sorted) of 6,000 + 6,000",
        "precision": "bf16", "gpu": None,
        "vlm": ["repro/adb_full/vlm/aigibench25/adb_vlm__limit1500__CommunityAI.json",
                "repro/adb_full/vlm/aigibench25/adb_vlm__full__SocialRF.json"],
        "dda": ["repro/adb_full/dda/aigibench25/dda__limit1500__CommunityAI.json",
                "repro/adb_full/dda/aigibench25/dda__full__SocialRF.json"], "generators": None, "manifest": None},
    "aigibench_subset": {
        "vlm_logs": ["repro/logs/adb_ivi_*.out"],
        "paper": "AIGI-Bench", "main": False, "data": "the vlm_bench subset slice (20 real + 20 fake each)",
        "precision": "bf16", "gpu": A5000_COMMITTED, "vlm": ["repro/vlm_bench/results/aigibench25/adb_vlm.json"],
        "dda": ["repro/adb_full/dda/aigibench25/dda.json"], "generators": ["CommunityAI", "SocialRF"],
        "manifest": "aigibench25"},
    "bfree_online_full": {
        "vlm_logs": ["repro/logs/adbfull_bench_297122_w2.out", "repro/logs/adb_ivi_*.out"],
        "paper": "BFree-Online", "main": True, "data": "all held images (269 real + 538 fake still online of 1,111 listed)",
        "precision": "bf16", "gpu": None, "vlm": ["repro/adb_full/vlm/bfree_online/adb_vlm__full.json"],
        "dda": ["repro/adb_full/dda/bfree_online/dda__full.json"], "generators": None, "manifest": None},
    "bfree_online_subset": {
        "vlm_logs": ["repro/logs/adb_ivi_*.out"],
        "paper": "BFree-Online", "main": False, "data": "the vlm_bench subset (269 real + 500 fake)", "precision": "bf16",
        "gpu": A5000_COMMITTED, "vlm": ["repro/vlm_bench/results/bfree_online/adb_vlm.json"],
        "dda": ["repro/adb_full/dda/bfree_online/dda.json"], "generators": None, "manifest": "bfree_online"},
    "genimage_subset_jpegfakes96": {
        "vlm_logs": ["repro/logs/adb_ivi_*.out"],
        "paper": "GenImage", "main": True, "data": "the vlm_bench subset (63 real + 63 fake per generator), fakes JPEG q96",
        "precision": "bf16", "gpu": A5000_COMMITTED,
        "vlm": ["repro/vlm_bench/results/genimage_val100k/adb_vlm__jpegfakes96.json"],
        "dda": ["repro/adb_full/dda/genimage_val100k/dda__jpegfakes96.json"], "generators": None,
        "manifest": "genimage_val100k"},
    "aigcdetect_subset_jpegfakes96": {
        "vlm_logs": ["repro/logs/adb_ivi_*.out"],
        "paper": "AIGCDetectBenchmark", "main": True,
        "data": "the vlm_bench subset (30 real + 30 fake per generator), fakes JPEG q96 (inferred protocol)",
        "precision": "bf16", "gpu": A5000_COMMITTED,
        "vlm": ["repro/vlm_bench/results/aigcdetect17/adb_vlm__jpegfakes96.json"],
        "dda": ["repro/adb_full/dda/aigcdetect17/dda__jpegfakes96.json"], "generators": None, "manifest": "aigcdetect17"},
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def load(paths: list[str]) -> tuple[list[dict], list[dict]]:
    docs = [json.loads((REPO / p).read_text()) for p in paths]
    return docs, [r for d in docs for r in d["per_image"]]


def pct(xs: list[bool]) -> float:
    return 100.0 * sum(xs) / len(xs)


def gen_metrics(rows: list[dict]) -> dict:
    labels = [r["label"] for r in rows]

    def balanced(field, real_value="real", fake_value="fake"):     # an "unparsed" verdict is wrong for both classes
        return (pct([r[field] == real_value for r in rows if r["label"] == 0]),
                pct([r[field] == fake_value for r in rows if r["label"] == 1]))

    fr, ff = balanced("verdict")
    vr, vf = balanced("vlm_verdict")
    dr, df = balanced("dda_fake", False, True)
    scores = [r["score"] for r in rows]
    return {
        "acc": round((fr + ff) / 2, 2), "real_acc": round(fr, 2), "fake_acc": round(ff, 2),
        "auroc": round(100 * roc_auc_score(labels, scores), 2), "ap": round(100 * average_precision_score(labels, scores), 2),
        "vlm_acc": round((vr + vf) / 2, 2), "vlm_auroc": round(100 * roc_auc_score(labels, [r["vlm_logit"] for r in rows]), 2),
        "dda_acc": round((dr + df) / 2, 2), "dda_auroc": round(100 * roc_auc_score(labels, [r["dda_prob"] for r in rows]), 2),
        "n_real": labels.count(0), "n_fake": labels.count(1),
        "decided_by": {k: sum(r["decided_by"] == k for r in rows) for k in ("vlm", "expert", "both", "neither")},
        "n_unparsed": sum(r["vlm_verdict"] == "unparsed" for r in rows),
    }


def fuse_row(name: str, cfg: dict) -> dict:
    vdocs, vrecs = load(cfg["vlm"])
    ddocs, drecs = load(cfg["dda"])
    keep = set(cfg["generators"]) if cfg["generators"] else None
    vmap = {(r["generator"], r["key"]): r for r in vrecs if keep is None or r["generator"] in keep}
    dmap = {(r["generator"], r["key"]): r for r in drecs if keep is None or r["generator"] in keep}
    if vmap.keys() != dmap.keys():
        raise SystemExit(f"{name}: VLM and DDA records differ ({len(vmap.keys() - dmap.keys())} VLM-only, "
                         f"{len(dmap.keys() - vmap.keys())} DDA-only)")
    manifest = None
    if cfg["manifest"]:
        manifest = json.loads((VB / "subsets" / f"{cfg['manifest']}.json").read_text())["files"]
    n_sha_vlm = n_sha_manifest = 0
    rows = []
    for k in sorted(vmap):
        v, d = vmap[k], dmap[k]
        if v["label"] != d["label"]:
            raise SystemExit(f"{name}: label differs for {k}")
        if "sha1" in v:
            if v["sha1"] != d["sha1"]:
                raise SystemExit(f"{name}: {k}: the expert scored another file than the VLM")
            n_sha_vlm += 1
        elif manifest is not None:
            if manifest[k[1]][0] != d["sha1"]:
                raise SystemExit(f"{name}: {k}: the expert's file differs from the subset manifest")
            n_sha_manifest += 1
        p_vlm = sigmoid(v["logit"])
        fused = fuse_components(p_vlm, v["verdict"], d["prob"])
        rows.append({"generator": k[0], "key": k[1], "label": v["label"], "verdict": fused["verdict"],
                     "score": fused["score"], "decided_by": fused["decided_by"], "vlm_verdict": fused["vlm_verdict"],
                     "vlm_logit": v["logit"], "vlm_new_tokens": v["new_tokens"], "dda_prob": d["prob"],
                     "dda_fake": fused["expert_verdict"] == "fake"})
    gens = sorted({r["generator"] for r in rows})
    per_gen = {g: gen_metrics([r for r in rows if r["generator"] == g]) for g in gens}

    def mean(f):
        return round(sum(m[f] for m in per_gen.values()) / len(per_gen), 2)

    means = {"acc": mean("acc"), "map": mean("ap"), "mauroc": mean("auroc"), "vlm_acc": mean("vlm_acc"),
             "dda_acc": mean("dda_acc"), "vlm_mauroc": mean("vlm_auroc"), "dda_mauroc": mean("dda_auroc")}
    gpus = sorted({d.get("gpu") or cfg["gpu"] or "?" for d in vdocs})
    return {
        "tag": f"adbfull_{name}", "model": "adb_full", "row": name, "paper_benchmark": cfg["paper"], "data": cfg["data"],
        "vlm_precision": cfg["precision"], "vlm_gpu": gpus, "dda_gpu": sorted({d.get("gpu", "?") for d in ddocs}),
        "jpeg_fakes": vdocs[0].get("jpeg_fakes"), "n_images": len(rows), "n_generators": len(per_gen), "means": means,
        "per_generator": per_gen, "sources": {"vlm": cfg["vlm"], "dda": cfg["dda"]},
        "logs": {"vlm": cfg["vlm_logs"], "dda": [DDA_LOG], "fusion": "repro/logs/adb_full_fuse.out"},
        "file_identity": {"sha1_vs_vlm_record": n_sha_vlm, "sha1_vs_subset_manifest": n_sha_manifest,
                          "unchecked": len(rows) - n_sha_vlm - n_sha_manifest},
        "vlm_means": [d["means"] for d in vdocs],
        "metric_note": "acc = balanced accuracy of the official decision (VLM parsed verdict OR DDA P(fake) >= 0.5), mean "
                       "over generators; AUROC / AP from SynthScope's fused score (models/image/adb_full); vlm_* / dda_* "
                       "the components alone on the same images",
        "per_image": rows,
    }


def main() -> int:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    table = {}
    for name, cfg in ROWS.items():
        missing = [p for p in cfg["vlm"] + cfg["dda"] if not (REPO / p).exists()]
        if missing:
            print(f"--- {name}: skipped, missing {missing}", flush=True)
            continue
        res = fuse_row(name, cfg)
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")
        m = res["means"]
        print(f"=== RESULT {res['tag']}: mAP={m['map']} mAUROC={m['mauroc']} acc={m['acc']} over {res['n_generators']} "
              f"generators -> repro/adb_full/results/{name}.json ===", flush=True)
        paper = PAPER[cfg["paper"]]
        table[name] = {
            "paper_benchmark": cfg["paper"], "paper_table": PAPER_TABLE[cfg["paper"]], "main": cfg["main"],
            "paper_acc": paper["mean"], "ours_acc": m["acc"], "delta": round(m["acc"] - paper["mean"], 2),
            "ours_mauroc": m["mauroc"], "vlm_only_acc": m["vlm_acc"], "dda_only_acc": m["dda_acc"],
            "per_generator": {g: {"paper": paper.get(g), "ours": pm["acc"], "vlm": pm["vlm_acc"], "dda": pm["dda_acc"]}
                              for g, pm in res["per_generator"].items()},
            "n_images": res["n_images"], "data": cfg["data"], "vlm_precision": cfg["precision"], "vlm_gpu": res["vlm_gpu"],
            "source": f"repro/adb_full/results/{name}.json", "vlm_logs": cfg["vlm_logs"], "dda_logs": [DDA_LOG]}
        print(f"    {cfg['paper']}: paper {paper['mean']} | ours {m['acc']} (VLM {m['vlm_acc']}, DDA {m['dda_acc']}) "
              f"AUROC {m['mauroc']} | n={res['n_images']} | {cfg['precision']} on {res['vlm_gpu']}", flush=True)
    (HERE / "paper_comparison.json").write_text(json.dumps(table, indent=1) + "\n")
    print(f"wrote repro/adb_full/paper_comparison.json ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
