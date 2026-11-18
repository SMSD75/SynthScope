"""Comparison of a VLM detector (default veritaspp) with the field-report detectors on the identical benchmark subsets.

Reads ``repro/vlm_bench/results/<bench>/<model>[__variant].json`` (bench_eval outputs of run_units.py) and writes
``repro/vlm_bench/comparison.json`` plus a printed summary:

* per subset (clean, and jpeg95 on the probe benchmarks): the VLM's acc (bench_eval's balanced mean over generators, from the
  parsed verdict, unparsed = wrong), pooled acc, mAUROC, mAP, unparsed count and mean tokens; every comparator's acc / mAUROC /
  mAP; the best comparator overall, the best PUBLICLY LICENSED one (not in spec.json's ``internal_only``: ruling T2-C2), the
  median comparator, and the VLM's rank among all scored detectors. Ties for best (equal rounded value) are broken
  deterministically: the alphabetically first detector key is named and every tied key is listed under ``tied``.
  ``acc_pooled`` counts each (generator, image) pair, so a real pool shared by several generators (ufd8, synthbuster_coco) is
  counted once per generator; ``acc_pooled_distinct`` counts each image once;
* the JPEG95 probe: per detector acc / mAUROC on jpeg95 minus clean, on the same subset;
* greedy vs sampled (the authors' inference-script T=0.7 decoding, fixed seed) on the sampled benchmark: verdict agreement, metric deltas;
* the paper rows (Veritas++ Tables II / III, Acc.): EvalGen = accuracy on the fakes only, which is OUR INFERENCE (the paper's
  table does not say; the paper's DDA row 97.2 matches DDA's fakes-only EvalGEN accuracy), reported next to the balanced
  accuracy (fakes + MSCOCO reals); RealChain / WildRF = accuracy over all images, each with unparsed answers dropped as in
  the authors' cal_metrics.py (and with them counted wrong, for reference); WildRF from the full-data run when it exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = {  # arXiv 2607.27113 v1, Table II (standard OOD) and Table III (in-the-wild), Acc. (%), VERITAS++ row (not the dagger row)
    # EvalGen: fakes-only accuracy is an inference from DDA's row (97.2), not something the paper states.
    "evalgen": {"paper_acc": 99.5, "table": "II", "metric": "fake_acc"},
    "realchain": {"paper_acc": 82.3, "table": "III", "metric": "acc_pooled"},
    "wildrf": {"paper_acc": 96.4, "table": "III", "metric": "acc_pooled"},
}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


ACC_POOLED_NOTE = ("acc_pooled counts every scored (generator, image) pair once: a real pool shared by several generators "
                   "(ufd8, synthbuster_coco; bench_eval's shared-reals layout) is counted once per generator, so reals and "
                   "fakes keep the 1:1 balance of the subset. acc_pooled_distinct counts each image once, so the shared "
                   "reals count once and the fakes dominate (ufd8: 630 distinct images, 504 of them fake); it is not "
                   "comparable with acc. The headline acc (balanced mean over generators) is unaffected by either.")


def summary(r: dict) -> dict:
    out = {"acc": r["means"]["acc"], "mauroc": r["means"]["mauroc"], "map": r["means"]["map"]}
    if "vlm" in r:
        v = r["vlm"]
        distinct = pooled(r["per_image"])
        out.update({"acc_pooled": v["acc_pooled"], "acc_pooled_distinct": distinct["acc_unparsed_wrong"],
                    "n_distinct_images": distinct["n"], "acc_pooled_drop": v["acc_pooled_drop"], "n_images": v["n_images"],
                    "n_unparsed": v["n_unparsed"], "n_unfinished": v["n_unfinished"], "mean_tokens": v["mean_tokens"],
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


def subset_row(bench: str, variant: str, model: str, dets: list[str], res: Path, internal_only: set[str] = frozenset()) -> dict | None:
    suffix = "" if variant == "clean" else f"__{variant}"
    vlm = load(res / bench / f"{model}{suffix}.json")
    comps = {d: load(res / bench / f"{d}{suffix}.json") for d in dets}
    comps = {d: dict(summary(r), internal_only=d in internal_only) for d, r in comps.items() if r is not None}
    if vlm is None and not comps:
        return None
    row = {"benchmark": bench, "variant": variant, "n_comparators": len(comps), "comparators": comps,
           "missing_comparators": sorted(set(dets) - set(comps))}
    if vlm is not None:
        row["vlm"] = summary(vlm)
        row["vlm_balance"] = balance(vlm)
        row["limit"] = vlm.get("limit")
    for metric in ("acc", "mauroc"):
        vals = [(c[metric], d) for d, c in comps.items()]
        public = [(v, d) for v, d in vals if d not in internal_only]
        if vals:
            row[f"best_{metric}"] = best_of(vals)
            row[f"best_public_{metric}"] = best_of(public)
            row[f"median_{metric}"] = round(statistics.median(v for v, _ in vals), 2)
            if vlm is not None:
                row[f"vlm_rank_{metric}"] = 1 + sum(v > row["vlm"][metric] for v, _ in vals)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="veritaspp")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "comparison.json"))
    args = ap.parse_args(argv)
    spec = json.loads((HERE / "spec.json").read_text())
    res, dets = Path(args.results), spec["detectors"]
    internal = set(spec.get("internal_only", []))
    assert internal <= set(dets), internal - set(dets)
    rows = [r for b in spec["benchmarks"] for r in [subset_row(b["key"], "clean", args.model, dets, res, internal)] if r]
    rows += [r for b in spec["jpeg95_benchmarks"] for r in [subset_row(b, "jpeg95", args.model, dets, res, internal)] if r]

    probe = {}
    for b in spec["jpeg95_benchmarks"]:
        clean = {r["benchmark"]: r for r in rows if r["variant"] == "clean"}.get(b)
        deg = {r["benchmark"]: r for r in rows if r["variant"] == "jpeg95"}.get(b)
        if not (clean and deg):
            continue
        entry = {}
        for d in [args.model] + dets:
            c = clean["vlm"] if d == args.model and "vlm" in clean else clean["comparators"].get(d)
            j = deg["vlm"] if d == args.model and "vlm" in deg else deg["comparators"].get(d)
            if c and j:
                entry[d] = {"acc_clean": c["acc"], "acc_jpeg95": j["acc"], "d_acc": round(j["acc"] - c["acc"], 2),
                            "mauroc_clean": c["mauroc"], "mauroc_jpeg95": j["mauroc"],
                            "d_mauroc": round(j["mauroc"] - c["mauroc"], 2)}
        others = [v for d, v in entry.items() if d != args.model]
        probe[b] = {"per_detector": entry,
                    "median_comparator_d_acc": round(statistics.median(v["d_acc"] for v in others), 2) if others else None,
                    "median_comparator_d_mauroc": round(statistics.median(v["d_mauroc"] for v in others), 2) if others else None}

    sb = spec["sampled_benchmark"]
    greedy, sampled = load(res / sb / f"{args.model}.json"), load(res / sb / f"{args.model}__official.json")
    sampling = None
    if greedy and sampled:
        g = {r["key"]: r for r in greedy["per_image"]}
        s = {r["key"]: r for r in sampled["per_image"]}
        common = sorted(set(g) & set(s))
        sampling = {"benchmark": sb, "n_images": len(common), "decoding": "official (T 0.7, top-p 0.8, top-k 20), seed "
                    f"{sampled['vlm']['seed']} per image",
                    "verdict_agreement": round(100 * sum(g[k]["verdict"] == s[k]["verdict"] for k in common) / len(common), 2),
                    "greedy": summary(greedy), "sampled": summary(sampled),
                    "d_acc": round(sampled["means"]["acc"] - greedy["means"]["acc"], 2),
                    "d_mauroc": round(sampled["means"]["mauroc"] - greedy["means"]["mauroc"], 2)}

    paper = {}
    for b, p in PAPER.items():
        full = load(res / b / f"{args.model}__full.json")
        parts = sorted((res / b).glob(f"{args.model}__full__*.json"))
        if full is None and parts:       # per-generator parts of the full run: pool their images
            full = {"per_image": [r for p in parts for r in load(p)["per_image"]]}
        sub = load(res / b / f"{args.model}.json")
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
        sampled_r = load(res / b / f"{args.model}__official.json")
        if sampled_r is not None:
            m = pooled(sampled_r["per_image"], fakes_only=p["metric"] == "fake_acc")
            entry["subset_sampled"] = dict(m, delta_vs_paper=None if m["acc_drop"] is None else
                                           round(m["acc_drop"] - p["paper_acc"], 2))
        paper[b] = entry

    out = {"model": args.model, "protocol": spec["protocol"], "internal_only": sorted(internal),
           "internal_only_note": spec.get("internal_only_note"),
           "best_tie_break": "equal values (rounded to 2 decimals): the alphabetically first detector key is named, every "
                             "tied key is listed under `tied`",
           "acc_pooled_note": ACC_POOLED_NOTE,
           "subsets": rows, "jpeg95_probe": probe, "greedy_vs_sampled": sampling, "paper_comparison": paper,
           "paper_note": "Veritas++ arXiv 2607.27113 v1, Acc. column. Decoding: the paper's text does not state a sampling "
                         "temperature; T 0.7 is the setting of the authors' official inference script (infer_vllm_single.py), "
                         "with top-p 0.8 / top-k 20 from the released generation_config.json, and 'sampled' below means that "
                         "setting. EvalGEN: fakes-only scoring is our inference from the DDA row (97.2), not stated in the "
                         "paper; the balanced accuracy is reported next to it."}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'subset':24s} {'VLM acc':>8s} {'AUROC':>6s} {'unp':>4s} {'tok':>5s} | {'best acc (det)':>24s} {'best public':>26s} {'med':>6s} | "
          f"{'best AUROC (det)':>24s} {'med':>6s} | rank acc/AUROC  n")
    for r in rows:
        v = r.get("vlm", {})
        name = r["benchmark"] + ("" if r["variant"] == "clean" else f" {r['variant']}")
        ba, bu, bp = r.get("best_acc", {}), r.get("best_mauroc", {}), r.get("best_public_acc", {})
        print(f"{name:24s} {v.get('acc', float('nan')):8.2f} {v.get('mauroc', float('nan')):6.2f} {v.get('n_unparsed', -1):4d} "
              f"{v.get('mean_tokens', float('nan')):5.0f} | {ba.get('value', float('nan')):7.2f} ({ba.get('detector', '-'):>14s}) "
              f"{bp.get('value', float('nan')):7.2f} ({bp.get('detector', '-'):>15s}) "
              f"{r.get('median_acc', float('nan')):6.2f} | {bu.get('value', float('nan')):7.2f} ({bu.get('detector', '-'):>14s}) "
              f"{r.get('median_mauroc', float('nan')):6.2f} | {r.get('vlm_rank_acc', '-')}/{r.get('vlm_rank_mauroc', '-')} "
              f"{r['n_comparators']}")
    for b, p in probe.items():
        v = p["per_detector"].get(args.model)
        print(f"JPEG95 {b}: VLM d_acc={v and v['d_acc']} d_AUROC={v and v['d_mauroc']}; comparator median "
              f"d_acc={p['median_comparator_d_acc']} d_AUROC={p['median_comparator_d_mauroc']}")
    if sampling:
        print(f"greedy vs sampled on {sb}: agreement {sampling['verdict_agreement']}%, d_acc {sampling['d_acc']}, "
              f"d_AUROC {sampling['d_mauroc']}")
    for b, e in paper.items():
        print(f"paper {b}: {e['paper']} vs " + ", ".join(f"{k} {e[k]['acc_drop']} (n={e[k]['n']}, unparsed {e[k]['n_unparsed']})"
                                                    for k in ("full", "subset", "subset_sampled") if k in e))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
