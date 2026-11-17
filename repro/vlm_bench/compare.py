"""Comparison of a VLM detector (default veritaspp) with the field-report detectors on the identical benchmark subsets.

Reads ``repro/vlm_bench/results/<bench>/<model>[__variant].json`` (bench_eval outputs of run_units.py) and writes
``repro/vlm_bench/comparison.json`` plus a printed summary:

* per subset (clean, and jpeg95 on the probe benchmarks): the VLM's acc (bench_eval's balanced mean over generators, from the
  parsed verdict, unparsed = wrong), pooled acc, mAUROC, mAP, unparsed count and mean tokens; every comparator's acc / mAUROC /
  mAP; the best and median comparator, and the VLM's rank among all scored detectors;
* the JPEG95 probe: per detector acc / mAUROC on jpeg95 minus clean, on the same subset;
* greedy vs sampled (the authors' T=0.7 decoding, fixed seed) on the sampled benchmark: verdict agreement, metric deltas;
* the paper rows (Veritas++ Tables II / III, Acc.): EvalGen = accuracy on the fakes (DDA's protocol, which the paper's
  DDA row 97.2 matches), RealChain / WildRF = accuracy over all images, each with unparsed answers dropped as in the authors'
  cal_metrics.py (and with them counted wrong, for reference); WildRF from the full-data run when it exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = {  # arXiv 2607.27113 v1, Table II (standard OOD) and Table III (in-the-wild), Acc. (%), VERITAS++ row (not the dagger row)
    "evalgen": {"paper_acc": 99.5, "table": "II", "metric": "fake_acc"},
    "realchain": {"paper_acc": 82.3, "table": "III", "metric": "acc_pooled"},
    "wildrf": {"paper_acc": 96.4, "table": "III", "metric": "acc_pooled"},
}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def summary(r: dict) -> dict:
    out = {"acc": r["means"]["acc"], "mauroc": r["means"]["mauroc"], "map": r["means"]["map"]}
    if "vlm" in r:
        v = r["vlm"]
        out.update({"acc_pooled": v["acc_pooled"], "acc_pooled_drop": v["acc_pooled_drop"], "n_images": v["n_images"],
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


def subset_row(bench: str, variant: str, model: str, dets: list[str], res: Path) -> dict | None:
    suffix = "" if variant == "clean" else f"__{variant}"
    vlm = load(res / bench / f"{model}{suffix}.json")
    comps = {d: load(res / bench / f"{d}{suffix}.json") for d in dets}
    comps = {d: summary(r) for d, r in comps.items() if r is not None}
    if vlm is None and not comps:
        return None
    row = {"benchmark": bench, "variant": variant, "n_comparators": len(comps), "comparators": comps,
           "missing_comparators": sorted(set(dets) - set(comps))}
    if vlm is not None:
        row["vlm"] = summary(vlm)
        row["limit"] = vlm.get("limit")
    for metric in ("acc", "mauroc"):
        vals = sorted(((c[metric], d) for d, c in comps.items()), reverse=True)
        if vals:
            row[f"best_{metric}"] = {"detector": vals[0][1], "value": vals[0][0]}
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
    rows = [r for b in spec["benchmarks"] for r in [subset_row(b["key"], "clean", args.model, dets, res)] if r]
    rows += [r for b in spec["jpeg95_benchmarks"] for r in [subset_row(b, "jpeg95", args.model, dets, res)] if r]

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
        for name, r in (("full", full), ("subset", sub)):
            if r is None:
                continue
            m = pooled(r["per_image"], fakes_only=p["metric"] == "fake_acc")
            entry[name] = dict(m, delta_vs_paper=None if m["acc_drop"] is None else round(m["acc_drop"] - p["paper_acc"], 2),
                               greedy=True)
        sampled_r = load(res / b / f"{args.model}__official.json")
        if sampled_r is not None:
            m = pooled(sampled_r["per_image"], fakes_only=p["metric"] == "fake_acc")
            entry["subset_sampled"] = dict(m, delta_vs_paper=None if m["acc_drop"] is None else
                                           round(m["acc_drop"] - p["paper_acc"], 2))
        paper[b] = entry

    out = {"model": args.model, "protocol": spec["protocol"], "subsets": rows, "jpeg95_probe": probe,
           "greedy_vs_sampled": sampling, "paper_comparison": paper,
           "paper_note": "Veritas++ arXiv 2607.27113 v1; the paper samples at T 0.7 (official setting) and reports Acc.; "
                         "EvalGen there is scored on fakes only (its DDA row 97.2 = DDA's fake-only EvalGEN accuracy)."}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'subset':24s} {'VLM acc':>8s} {'AUROC':>6s} {'unp':>4s} {'tok':>5s} | {'best acc (det)':>24s} {'med':>6s} | "
          f"{'best AUROC (det)':>24s} {'med':>6s} | rank acc/AUROC  n")
    for r in rows:
        v = r.get("vlm", {})
        name = r["benchmark"] + ("" if r["variant"] == "clean" else f" {r['variant']}")
        ba, bu = r.get("best_acc", {}), r.get("best_mauroc", {})
        print(f"{name:24s} {v.get('acc', float('nan')):8.2f} {v.get('mauroc', float('nan')):6.2f} {v.get('n_unparsed', -1):4d} "
              f"{v.get('mean_tokens', float('nan')):5.0f} | {ba.get('value', float('nan')):7.2f} ({ba.get('detector', '-'):>14s}) "
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
