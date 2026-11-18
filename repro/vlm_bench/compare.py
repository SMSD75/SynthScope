"""Comparison of the VLM detectors (veritaspp, adb_vlm) with the field-report detectors on the identical benchmark subsets.

Reads ``repro/vlm_bench/results/<bench>/<model>[__variant].json`` (bench_eval outputs of run_units.py) and writes
``repro/vlm_bench/comparison.json`` plus a printed summary:

* per subset (clean, and jpeg95 on the probe benchmarks): every VLM's acc (bench_eval's balanced mean over generators, from the
  parsed verdict, unparsed = wrong), pooled acc, mAUROC, mAP, unparsed and unscored counts (unscored: a parsed verdict
  without a verdict token, P(fake) 0.5 in AUROC) and mean tokens, under ``vlms.<model>``; every
  comparator's acc / mAUROC / mAP; the best comparator overall, the best PUBLICLY LICENSED one (not in spec.json's
  ``internal_only``: ruling T2-C2), the median comparator, and each VLM's rank among the comparators plus itself (``rank_acc``,
  ``rank_mauroc``; 1 = best, ties share the better rank) and among the comparators plus every VLM (``rank_all_*``). Best and
  median are over the comparators only. Ties for best (equal rounded value) are broken deterministically: the alphabetically
  first detector key is named and every tied key is listed under ``tied``. ``acc_pooled`` counts each (generator, image) pair, so
  a real pool shared by several generators (ufd8, synthbuster_coco) is counted once per generator; ``acc_pooled_distinct``
  counts each image once;
* ``subset_means``: per VLM, the unweighted mean acc / mAUROC over the clean subsets it has;
* the JPEG95 probe: per detector acc / mAUROC on jpeg95 minus clean, on the same subset;
* greedy vs sampled (the authors' sampling, fixed seed) on the sampled benchmark, for a VLM that has a sampled run;
* the paper rows, per VLM:
  - veritaspp (Tables II / III, Acc.): EvalGen = accuracy on the fakes only, which is OUR INFERENCE (the paper's table does not
    say; the paper's DDA row 97.2 matches DDA's fakes-only EvalGEN accuracy), reported next to the balanced accuracy (fakes +
    MSCOCO reals); RealChain / WildRF = accuracy over all images, each with unparsed answers dropped as in the authors'
    cal_metrics.py (and with them counted wrong, for reference); WildRF from the full-data run when it exists;
  - adb_vlm (arXiv 2512.06746 v3, balanced accuracy): every PRINTED number is the full method (VLM OR the unreleased pixel
    expert), so none is reproducible with the released VLM branch; they are listed as context only, with no delta. The paper
    prints no VLM-only number for the released adapter. Table 4's "Qwen2.5-VL-7B" row is ambiguous (base or trained) and
    Table 5's is the UNTRAINED base: both are listed, never as a reproduction target.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
VLMS = ("veritaspp", "adb_vlm")
PAPER = {  # arXiv 2607.27113 v1, Table II (standard OOD) and Table III (in-the-wild), Acc. (%), VERITAS++ row (not the dagger row)
    # EvalGen: fakes-only accuracy is an inference from DDA's row (97.2), not something the paper states.
    "evalgen": {"paper_acc": 99.5, "table": "II", "metric": "fake_acc"},
    "realchain": {"paper_acc": 82.3, "table": "III", "metric": "acc_pooled"},
    "wildrf": {"paper_acc": 96.4, "table": "III", "metric": "acc_pooled"},
}

# Align, Don't Blend (AlignGemini), arXiv 2512.06746 v3, balanced accuracy (%). "full_method" = the printed AlignGemini row,
# i.e. the VLM branch OR the pixel expert (Qwen2.5-VL-7B + "SD" expert); the expert is not released, so these are context,
# never reproduction targets. The paper prints no VLM-only number for the released adapter. Generator keys are this
# repository's generator folder names.
_AIGCD = ("ADM", "DALLE2", "Glide", "Midjourney", "VQDM", "biggan", "cyclegan", "gaugan", "progan", "sd_xl",
          "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "stargan", "stylegan", "stylegan2", "whichfaceisreal", "wukong")
_GENIMAGE = ("midjourney", "sdv1.4", "sdv1.5", "adm", "glide", "wukong", "vqdm", "biggan")
ADB_PAPER = {
    "genimage_val100k": {
        "paper_benchmark": "GenImage (appendix table 'Comparison on GenImage'; the paper JPEG-compresses the fakes "
                           "'following BFree' to match the reals' format and gives no quality factor)",
        "ours": "jpegfakes96", "ours_note": "subset (63 real + 63 fake per generator), fakes JPEG q96 (bench_eval "
                                            "--jpeg-fakes 96); q96 is our choice, following DDA's published GenImage "
                                            "protocol, not a value stated in this paper",
        "full_method": dict(zip(_GENIMAGE, (89.3, 89.6, 95.6, 77.2, 90.3, 98.3, 98.1, 98.4)), avg=91.7),
    },
    "aigcdetect17": {
        "paper_benchmark": "AIGCDetectBenchmark (appendix table 'Comparison of balanced accuracy on AIGCDetectBenchmark'; "
                           "its caption does not mention JPEG compression)",
        "ours": "jpegfakes96", "ours_note": "subset (30 real + 30 fake per generator), fakes JPEG q96 (bench_eval "
                                            "--jpeg-fakes 96). INFERENCE: the paper states the fake-JPEG step for "
                                            "GenImage only; applying it here, and the q96 quality (which follows DDA's "
                                            "published protocol), are our assumptions. The clean run is in the subset rows",
        "full_method": dict(zip(_AIGCD, (89.3, 94.9, 90.3, 95.6, 77.2, 90.4, 74.4, 92.5, 91.0, 99.1, 98.3, 98.1, 72.9, 86.6,
                                         89.2, 52.6, 98.4)), avg=87.7),
    },
    "wildrf": {
        "paper_benchmark": "WildRF (Table 1, in-the-wild)",
        "ours": "full", "ours_note": "FULL test split (2,503 images), bench_eval balanced accuracy per platform and their "
                                     "mean",
        "full_method": {"facebook": 92.2, "reddit": 90.9, "twitter": 95.5, "avg": 92.9},
        "table4_qwen2.5_vl_7b_ambiguous": 86.3, "table5_qwen2.5_vl_7b_untrained": 83.8,
    },
    "aigibench25": {
        "paper_benchmark": "AIGI-Bench (Table 1: its two in-the-wild subsets CommunityUI = our CommunityAI, and SocialRF)",
        "ours": "clean", "generators": ["CommunityAI", "SocialRF"],
        "ours_note": "SUBSET slice: the CommunityAI and SocialRF generators of the aigibench25 subset only (20 real + 20 fake "
                     "each; the full sets are 12,000 + 6,000 images, not run)",
        "full_method": {"CommunityAI": 91.9, "SocialRF": 85.6, "avg": 88.8},
        "table4_qwen2.5_vl_7b_ambiguous": 80.4, "table5_qwen2.5_vl_7b_untrained": 75.7,
    },
    "cospybench": {
        "paper_benchmark": "CO-SPY-Bench/in-the-wild (Table 1: Civitai, DALL-E 3, instavibe.ai, Lexica, Midjourney-v6)",
        "ours": None, "not_held": "we hold only Co-Spy-Bench's synthetic part (5 caption sets x 22 generators); its "
                                  "in-the-wild part is not downloaded, so there is no comparable number",
        "full_method": {"Civitai": 100.0, "DALL-E 3": 99.6, "instavibe.ai": 75.0, "Lexica": 96.8, "Midjourney-v6": 99.9,
                        "avg": 94.3},
        "table4_qwen2.5_vl_7b_ambiguous": 67.6, "table5_qwen2.5_vl_7b_untrained": 57.4,
    },
    "bfree_online": {
        "paper_benchmark": "BFree-Online (Table 1)",
        "ours": "clean", "ours_note": "subset: 269 real + 500 fake = 769 images (limit 500 per class; the benchmark folder "
                                      "has only 269 reals); acc is balanced, so the imbalance does not weight it",
        "full_method": {"avg": 97.5},
        "table4_qwen2.5_vl_7b_ambiguous": 82.9, "table5_qwen2.5_vl_7b_untrained": 77.6,
    },
}
ADB_PAPER_NOTE = (
    "Align, Don't Blend (AlignGemini) arXiv 2512.06746 v3. The paper's detector is the VLM branch OR a pixel-artifact expert "
    "(DINOv2 + LoRA trained on VAE reconstructions); only the VLM branch is released, so NONE of the paper's tables is "
    "reproducible here. Printed numbers (full_method) are context, not targets. The paper prints no VLM-only number for the "
    "released adapter, so no row gets a delta. Table 4 'Qwen2.5-VL-7B' ('baseline methods under the same training set', "
    "retrained on the same semantic and pixel corpora) is ambiguous between the base and a trained VLM; Table 5 "
    "'Qwen2.5-VL-7B (Untrained)' is the base model without the adapter. Neither is the released adapter. GenImage / "
    "AIGCDetect: ours are also run with the fakes JPEG-compressed at q96 (jpegfakes96): the paper states that step for "
    "GenImage only, and q96 follows DDA's published protocol (an inference, see ours_note). Decoding: greedy (temperature 0, "
    "predict.py), which is the official setting. The paper's AlignGemini GenImage per-generator row appears column-permuted "
    "relative to its AIGCDetect row (the baselines agree across the two tables), so only its 91.7 average is used.")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


ACC_POOLED_NOTE = ("acc_pooled counts every scored (generator, image) pair once: a real pool shared by several generators "
                   "(ufd8, synthbuster_coco; bench_eval's shared-reals layout) is counted once per generator, so reals and "
                   "fakes keep the 1:1 balance of the subset. acc_pooled_distinct counts each image once, so the shared "
                   "reals count once and the fakes dominate (ufd8: 630 distinct images, 504 of them fake); it is not "
                   "comparable with acc. The headline acc (balanced mean over generators) is unaffected by either.")


def n_unscored(per_image: list[dict]) -> int:
    """Records with a parsed verdict but no verdict token (logit 0, P(fake) 0.5 in AUROC / AP); bench_eval's n_unscored,
    counted from the per-image records so that result JSONs written before that field count the same way."""
    return sum(r["verdict"] is not None and r["score_source"] in ("unparsed", "unscored") for r in per_image)


def summary(r: dict) -> dict:
    out = {"acc": r["means"]["acc"], "mauroc": r["means"]["mauroc"], "map": r["means"]["map"]}
    if "vlm" in r:
        v = r["vlm"]
        distinct = pooled(r["per_image"])
        out.update({"acc_pooled": v["acc_pooled"], "acc_pooled_distinct": distinct["acc_unparsed_wrong"],
                    "n_distinct_images": distinct["n"], "acc_pooled_drop": v["acc_pooled_drop"], "n_images": v["n_images"],
                    "n_unparsed": v["n_unparsed"], "n_unscored": n_unscored(r["per_image"]),
                    "n_unfinished": v["n_unfinished"], "mean_tokens": v["mean_tokens"],
                    "mean_seconds": v["mean_seconds_scored"]})
    return out


def pooled(per_image: list[dict], fakes_only: bool = False) -> dict:
    """Accuracy over distinct images (an image shared by several generators, e.g. a real pool, counted once)."""
    seen = {}
    for r in per_image:
        seen.setdefault(r["key"], r)
    rows = [r for r in seen.values() if r["label"] == 1 or not fakes_only]
    ok = [r["verdict"] == ("fake" if r["label"] else "real") for r in rows]
    parsed = [r["verdict"] is not None for r in rows]
    n_drop = sum(parsed)
    return {"n": len(rows), "n_unparsed": len(rows) - n_drop,
            "acc_drop": round(100 * sum(o for o, p in zip(ok, parsed) if p) / n_drop, 2) if n_drop else None,
            "acc_unparsed_wrong": round(100 * sum(ok) / len(rows), 2) if rows else None}


def balance(r: dict) -> dict:
    """Where the VLM's errors are: mean real / fake accuracy over the generators (parsed verdict, unparsed = wrong) and the
    per-generator extremes of the fake accuracy (a generator with a shared real pool still has its own fake accuracy)."""
    pg = r["per_generator"]
    fake = {g: m["fake_acc"] for g, m in pg.items()}
    lo, hi = min(fake, key=lambda g: (fake[g], g)), max(fake, key=lambda g: (fake[g], g))
    return {"mean_real_acc": round(statistics.mean(m["real_acc"] for m in pg.values()), 2),
            "mean_fake_acc": round(statistics.mean(fake.values()), 2),
            "lowest_fake_acc": {"generator": lo, "value": fake[lo]}, "highest_fake_acc": {"generator": hi, "value": fake[hi]},
            "n_generators": len(pg), "n_generators_fake_acc_below_50": sum(v < 50 for v in fake.values()),
            "per_generator_fake_acc": fake}


def best_of(vals: list[tuple[float, str]]) -> dict | None:
    """The best (value, detector) with a deterministic tie break: among equal values the alphabetically first detector key is
    named, and every tied key is listed."""
    if not vals:
        return None
    top = max(v for v, _ in vals)
    tied = sorted(d for v, d in vals if v == top)
    return {"detector": tied[0], "value": top, "tied": tied}


def subset_row(bench: str, variant: str, models: tuple[str, ...], dets: list[str], res: Path,
               internal_only: set[str] = frozenset()) -> dict | None:
    suffix = "" if variant == "clean" else f"__{variant}"
    vlm_r = {m: r for m in models if (r := load(res / bench / f"{m}{suffix}.json")) is not None}
    comps = {d: load(res / bench / f"{d}{suffix}.json") for d in dets}
    comps = {d: dict(summary(r), internal_only=d in internal_only) for d, r in comps.items() if r is not None}
    if not vlm_r and not comps:
        return None
    row = {"benchmark": bench, "variant": variant, "n_comparators": len(comps), "comparators": comps,
           "missing_comparators": sorted(set(dets) - set(comps)), "vlms": {}}
    for m, r in vlm_r.items():
        row["vlms"][m] = dict(summary(r), balance=balance(r), limit=r.get("limit"))
    for metric in ("acc", "mauroc"):
        vals = [(c[metric], d) for d, c in comps.items()]
        public = [(v, d) for v, d in vals if d not in internal_only]
        if vals:
            row[f"best_{metric}"] = best_of(vals)
            row[f"best_public_{metric}"] = best_of(public)
            row[f"median_{metric}"] = round(statistics.median(v for v, _ in vals), 2)
            for m, s in row["vlms"].items():
                others = [v for k, v in row["vlms"].items() if k != m]
                s[f"rank_{metric}"] = 1 + sum(v > s[metric] for v, _ in vals)
                s[f"rank_all_{metric}"] = s[f"rank_{metric}"] + sum(o[metric] > s[metric] for o in others)
    return row


def subset_means(rows: list[dict], models: tuple[str, ...]) -> dict:
    """Per VLM: the unweighted mean acc / mAUROC over the CLEAN subset rows it has, with the image, unparsed and unscored
    counts."""
    means = {}
    for m in models:
        got = [r["vlms"][m] for r in rows if r["variant"] == "clean" and m in r["vlms"]]
        if got:
            means[m] = {"n_subsets": len(got), "acc": round(statistics.mean(s["acc"] for s in got), 2),
                        "mauroc": round(statistics.mean(s["mauroc"] for s in got), 2),
                        "n_images": sum(s["n_images"] for s in got), "n_unparsed": sum(s["n_unparsed"] for s in got),
                        "n_unscored": sum(s["n_unscored"] for s in got)}
    return means


def veritaspp_paper(model: str, res: Path) -> dict:
    paper = {}
    for b, p in PAPER.items():
        full = load(res / b / f"{model}__full.json")
        parts = sorted((res / b).glob(f"{model}__full__*.json"))
        if full is None and parts:       # per-generator parts of the full run: pool their images
            full = {"per_image": [r for p in parts for r in load(p)["per_image"]]}
        sub = load(res / b / f"{model}.json")
        entry = {"paper": p["paper_acc"], "paper_table": p["table"], "metric": p["metric"]}
        if p["metric"] == "fake_acc":
            entry["metric_note"] = ("fakes-only accuracy is our inference (the paper does not say how EvalGEN is scored; its "
                                    "DDA row, 97.2, matches DDA's fakes-only EvalGEN accuracy); balanced_acc is the "
                                    "benchmark's balanced accuracy (fakes + MSCOCO reals) on the same images. Ours is a "
                                    "500-fake subset, not the full set.")
        for name, r in (("full", full), ("subset", sub)):
            if r is None:
                continue
            m = pooled(r["per_image"], fakes_only=p["metric"] == "fake_acc")
            entry[name] = dict(m, delta_vs_paper=None if m["acc_drop"] is None else round(m["acc_drop"] - p["paper_acc"], 2),
                               greedy=True)
            if p["metric"] == "fake_acc" and "means" in r:        # balanced accuracy of the same run (fakes + the reals)
                entry[name]["balanced_acc"] = r["means"]["acc"]
                entry[name]["balanced_delta_vs_paper"] = round(r["means"]["acc"] - p["paper_acc"], 2)
        sampled_r = load(res / b / f"{model}__official.json")
        if sampled_r is not None:
            m = pooled(sampled_r["per_image"], fakes_only=p["metric"] == "fake_acc")
            entry["subset_sampled"] = dict(m, delta_vs_paper=None if m["acc_drop"] is None else
                                           round(m["acc_drop"] - p["paper_acc"], 2))
        paper[b] = entry
    return paper


def adb_paper(model: str, res: Path) -> dict:
    out = {}
    for b, p in ADB_PAPER.items():
        entry = {k: v for k, v in p.items() if k != "ours"}
        variant = p["ours"]
        r = None
        if variant == "full":           # bench_eval's own per-platform values (they trace to the logs' RESULT lines)
            full = load(res / b / f"{model}__full.json")
            runs = [full] if full else [load(q) for q in sorted((res / b).glob(f"{model}__full__*.json"))]
            if runs:
                pg = {g: {k: m[k] for k in ("acc", "real_acc", "fake_acc", "auroc", "n_real", "n_fake", "n_unparsed")}
                      for run in runs for g, m in sorted(run["per_generator"].items())}
                r = {"per_generator": pg, "acc": round(statistics.mean(g["acc"] for g in pg.values()), 2),
                     "n_images": sum(g["n_real"] + g["n_fake"] for g in pg.values()),
                     "n_unparsed": sum(g["n_unparsed"] for g in pg.values()),
                     "source": f"results/{b}/{model}__full.json" if full else
                     [f"results/{b}/{q.name}" for q in sorted((res / b).glob(f"{model}__full__*.json"))]}
                if full:
                    r.update(mauroc=full["means"]["mauroc"], map=full["means"]["map"])
        elif variant is not None:
            suffix = "" if variant == "clean" else f"__{variant}"
            j = load(res / b / f"{model}{suffix}.json")
            if j is not None:
                gens = p.get("generators") or sorted(j["per_generator"])
                pg = {g: {k: j["per_generator"][g][k] for k in ("acc", "real_acc", "fake_acc", "auroc") if k in j["per_generator"][g]}
                      for g in gens}
                r = {"variant": variant, "per_generator": pg, "acc": round(statistics.mean(g["acc"] for g in pg.values()), 2),
                     "n_unparsed": sum(x["verdict"] is None for x in j["per_image"] if x["key"].split("/", 1)[0] in gens),
                     "source": f"results/{b}/{model}{suffix}.json"}
                if not p.get("generators"):
                    r.update(mauroc=j["means"]["mauroc"], map=j["means"]["map"])
        if r is not None:
            entry["ours_vlm_only"] = r
            entry["full_method_note"] = "context only, no delta: the paper's number includes the unreleased pixel expert"
        out[b] = entry
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(VLMS), help="comma list of VLM keys (columns)")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "comparison.json"))
    args = ap.parse_args(argv)
    models = tuple(filter(None, args.models.split(",")))
    spec = json.loads((HERE / "spec.json").read_text())
    res, dets = Path(args.results), spec["detectors"]
    internal = set(spec.get("internal_only", []))
    assert internal <= set(dets), internal - set(dets)
    rows = [r for b in spec["benchmarks"] for r in [subset_row(b["key"], "clean", models, dets, res, internal)] if r]
    rows += [r for b in spec["jpeg95_benchmarks"] for r in [subset_row(b, "jpeg95", models, dets, res, internal)] if r]

    means = subset_means(rows, models)

    probe = {}
    for b in spec["jpeg95_benchmarks"]:
        clean = {r["benchmark"]: r for r in rows if r["variant"] == "clean"}.get(b)
        deg = {r["benchmark"]: r for r in rows if r["variant"] == "jpeg95"}.get(b)
        if not (clean and deg):
            continue
        entry = {}
        for d in list(models) + dets:
            c = clean["vlms"].get(d) if d in models else clean["comparators"].get(d)
            j = deg["vlms"].get(d) if d in models else deg["comparators"].get(d)
            if c and j:
                entry[d] = {"acc_clean": c["acc"], "acc_jpeg95": j["acc"], "d_acc": round(j["acc"] - c["acc"], 2),
                            "mauroc_clean": c["mauroc"], "mauroc_jpeg95": j["mauroc"],
                            "d_mauroc": round(j["mauroc"] - c["mauroc"], 2)}
        others = [v for d, v in entry.items() if d not in models]
        probe[b] = {"per_detector": entry,
                    "median_comparator_d_acc": round(statistics.median(v["d_acc"] for v in others), 2) if others else None,
                    "median_comparator_d_mauroc": round(statistics.median(v["d_mauroc"] for v in others), 2) if others else None}

    sb = spec["sampled_benchmark"]
    sampling = {}
    for m in models:
        greedy, sampled = load(res / sb / f"{m}.json"), load(res / sb / f"{m}__official.json")
        if greedy and sampled:
            g = {r["key"]: r for r in greedy["per_image"]}
            s = {r["key"]: r for r in sampled["per_image"]}
            common = sorted(set(g) & set(s))
            sampling[m] = {"benchmark": sb, "n_images": len(common), "decoding": "official (T 0.7, top-p 0.8, top-k 20), "
                           f"seed {sampled['vlm']['seed']} per image",
                           "verdict_agreement": round(100 * sum(g[k]["verdict"] == s[k]["verdict"] for k in common) / len(common), 2),
                           "greedy": summary(greedy), "sampled": summary(sampled),
                           "d_acc": round(sampled["means"]["acc"] - greedy["means"]["acc"], 2),
                           "d_mauroc": round(sampled["means"]["mauroc"] - greedy["means"]["mauroc"], 2)}

    paper = {}
    if "veritaspp" in models:
        paper["veritaspp"] = veritaspp_paper("veritaspp", res)
    if "adb_vlm" in models:
        paper["adb_vlm"] = adb_paper("adb_vlm", res)

    out = {"models": list(models), "protocol": spec["protocol"], "internal_only": sorted(internal),
           "internal_only_note": spec.get("internal_only_note"),
           "best_tie_break": "equal values (rounded to 2 decimals): the alphabetically first detector key is named, every "
                             "tied key is listed under `tied`",
           "rank_note": "rank_<metric> = 1 + the number of comparators strictly better (the VLM among the comparators plus "
                        "itself); rank_all_<metric> also counts the other VLMs. Best and median are over the comparators.",
           "acc_pooled_note": ACC_POOLED_NOTE,
           "subsets": rows, "subset_means": means, "jpeg95_probe": probe, "greedy_vs_sampled": sampling,
           "paper_comparison": paper,
           "paper_note": {
               "veritaspp": "Veritas++ arXiv 2607.27113 v1, Acc. column. Decoding: the paper's text does not state a sampling "
                            "temperature; T 0.7 is the setting of the authors' official inference script "
                            "(infer_vllm_single.py), with top-p 0.8 / top-k 20 from the released generation_config.json, and "
                            "'sampled' below means that setting. EvalGEN: fakes-only scoring is our inference from the DDA row "
                            "(97.2), not stated in the paper; the balanced accuracy is reported next to it.",
               "adb_vlm": ADB_PAPER_NOTE}}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    for m in models:
        print(f"--- {m}")
        print(f"{'subset':24s} {'acc':>6s} {'AUROC':>6s} {'unp':>4s} {'uns':>4s} {'tok':>5s} | {'best acc (det)':>24s} {'best public':>26s} "
              f"{'med':>6s} | {'best AUROC (det)':>24s} {'med':>6s} | rank acc/AUROC (all)  n")
        for r in rows:
            v = r["vlms"].get(m)
            if v is None:
                continue
            name = r["benchmark"] + ("" if r["variant"] == "clean" else f" {r['variant']}")
            ba, bu, bp = r.get("best_acc", {}), r.get("best_mauroc", {}), r.get("best_public_acc", {})
            print(f"{name:24s} {v['acc']:6.2f} {v['mauroc']:6.2f} {v['n_unparsed']:4d} {v['n_unscored']:4d} {v['mean_tokens']:5.0f} | "
                  f"{ba.get('value', float('nan')):7.2f} ({ba.get('detector', '-'):>14s}) "
                  f"{bp.get('value', float('nan')):7.2f} ({bp.get('detector', '-'):>15s}) "
                  f"{r.get('median_acc', float('nan')):6.2f} | {bu.get('value', float('nan')):7.2f} ({bu.get('detector', '-'):>14s}) "
                  f"{r.get('median_mauroc', float('nan')):6.2f} | {v.get('rank_acc', '-')}/{v.get('rank_mauroc', '-')} "
                  f"({v.get('rank_all_acc', '-')}/{v.get('rank_all_mauroc', '-')}) {r['n_comparators']}")
        if m in means:
            print(f"mean over {means[m]['n_subsets']} clean subsets: acc {means[m]['acc']} AUROC {means[m]['mauroc']}")
        for b, p in probe.items():
            v = p["per_detector"].get(m)
            print(f"JPEG95 {b}: {m} d_acc={v and v['d_acc']} d_AUROC={v and v['d_mauroc']}; comparator median "
                  f"d_acc={p['median_comparator_d_acc']} d_AUROC={p['median_comparator_d_mauroc']}")
        if m in sampling:
            s = sampling[m]
            print(f"greedy vs sampled on {sb}: agreement {s['verdict_agreement']}%, d_acc {s['d_acc']}, d_AUROC {s['d_mauroc']}")
    for b, e in paper.get("veritaspp", {}).items():
        print(f"veritaspp paper {b}: {e['paper']} vs " + ", ".join(
            f"{k} {e[k]['acc_drop']} (n={e[k]['n']}, unparsed {e[k]['n_unparsed']})"
            for k in ("full", "subset", "subset_sampled") if k in e))
    for b, e in paper.get("adb_vlm", {}).items():
        o = e.get("ours_vlm_only")
        print(f"adb_vlm paper {b}: full method (context) {e['full_method']['avg']}"
              + (f", T4 (ambiguous) {e['table4_qwen2.5_vl_7b_ambiguous']}" if "table4_qwen2.5_vl_7b_ambiguous" in e else "")
              + (f", T5 untrained base {e['table5_qwen2.5_vl_7b_untrained']}" if "table5_qwen2.5_vl_7b_untrained" in e else "")
              + (f" | ours VLM-only {o['acc']}" if o else " | ours: " + e.get("not_held", "not run yet")))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
