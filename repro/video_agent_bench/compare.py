"""Comparison of the agentic video detector WeVisForensics (``wevis``) with SynthScope's video detectors on the identical
benchmark subsets, and the GenBuster-Bench / ++ comparison with the paper.

Reads ``repro/video_agent_bench/results/<bench>/<model>.json`` (bench_eval_video outputs: run_units.py for the agent,
comparators.sh for the detectors) and the subset manifests, and writes ``repro/video_agent_bench/comparison.json`` plus a
printed summary:

* per subset: the agent's acc (bench_eval_video's balanced mean over generators; official = final tool_error / oom videos
  left out of the denominator, the official ``scripts/evaluate.py`` rule, with their counts; strict = counted wrong),
  mAUROC / mAP (ranked by the verdict-token logit gap), the pooled one-row-per-video block (the paper's accuracy, fake
  recall / F1), the unparsed / unscored / tool_error / OOM counts, OOM retries, the GPU of every row and the cost; every
  comparator's acc / mAUROC / mAP / pooled acc; the best comparator, the best publicly licensed one (not in spec.json's
  ``internal_only``) and the median comparator per metric; the agent's rank among the comparators plus itself (``rank_*``;
  1 = best, ties share the better rank: 1 + the number of comparators strictly better). Best and median are over the
  comparators only; ties for best are broken alphabetically and listed under ``tied``;
* ``subset_means``: the unweighted mean over the four held subsets and over all six, for the agent and every comparator;
* ``paper_comparison``: GenBuster-Bench (GB) and GenBuster-Bench++ (GB++) vs the paper's Table 4 / 12 (Acc., fake recall
  R, F1) and, for GB, Table 11's breakdown (ID / OOD per generator / Wild fake detection rates, All). Ours is a
  documented SUBSAMPLE (spec.json ``fraction`` of every folder); the pooled accuracy gets a 95 % Wilson interval and a
  stratified estimate of the full-set accuracy (each folder's rate weighted by its full size);
* ``determinism``: the first L MSR-VTT reals are the same files in all four held subsets (sha1 in the manifests) and were
  scored once per subset: same verdict / logit across the repeats?
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = "wevis"
HELD = ("genvideo", "videophy", "vidprom", "evalcrafter")
# WeVisForensics arXiv 2611.21373, Table 11 (GenBuster-Bench breakdown, WeVisForensics-DTPO row): per-subset forgery
# detection rate (%) for ID, the eight OOD generators, OOD mean and Wild; All = overall accuracy over the 3150 clips.
GB_TABLE11 = {"id": 99.7, "ood": {"sora": 100.0, "pika": 99.0, "gen3": 99.0, "luma": 100.0, "wanx": 100.0, "kling": 99.0,
                                  "jimeng": 100.0, "vidu": 99.3}, "ood_mean": 99.6, "wild": 95.3, "all": 97.7}
PAPER_NOTE = ("WeVisForensics arXiv 2611.21373: Table 4 / Table 12 report accuracy (Acc.), fake recall (R) and F1 with the "
              "generated videos as the positive class, one row per clip (official scripts/evaluate.py: invalid answers "
              "count wrong, tool_error rows are excluded). The paper's run is the official vLLM engine (multi-GPU, its own "
              "kernels); ours is the official transformers backend (demo/wvf_local.py) under SynthScope's runner, greedy, "
              "audio off (has_audio=False for every clip; GenBuster clips are silent), on a documented subsample. "
              "implied_real_acc is derived from Acc. and R and the benchmark's composition (an inference, not printed). "
              "The paper's NSG-VD and ReStraV rows are those detectors RETRAINED by the authors on their training data, "
              "not the public checkpoints scored here.")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def best_of(vals: list[tuple[float, str]]) -> dict | None:
    vals = [(v, d) for v, d in vals if v is not None]
    if not vals:
        return None
    top = max(v for v, _ in vals)
    tied = sorted(d for v, d in vals if v == top)
    return {"detector": tied[0], "value": top, "tied": tied}


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if not n:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def agent_summary(r: dict) -> dict:
    pb, pg = r["means"]["pooled"], r["per_generator"]
    out = {"acc": r["means"]["acc"], "acc_strict": r["means"]["acc_strict"], "mauroc": r["means"]["mauroc"],
           "map": r["means"]["map"], "pooled": pb,
           "mean_real_acc": round(statistics.mean(m["real_acc"] for m in pg.values()), 2),
           "mean_fake_acc": round(statistics.mean(m["fake_acc"] for m in pg.values()), 2),
           "n_generators": len(pg), "limit": r.get("limit"), "ckpt_sha256": r["ckpt_sha256"],
           "agent_retries": r.get("agent_retries"), "failed_other": r.get("failed_other"),
           "failed_to_decode": r.get("failed_to_decode")}
    fake = {g: m["fake_acc"] for g, m in pg.items()}
    out["lowest_fake_acc"] = {"generator": min(fake, key=lambda g: (fake[g], g)), "value": min(fake.values())}
    return out


def comp_summary(r: dict, internal: bool) -> dict:
    out = {"acc": r["means"]["acc"], "mauroc": r["means"]["mauroc"], "map": r["means"]["map"], "internal_only": internal,
           "ckpt_sha256": r.get("ckpt_sha256"), "failed_to_decode": r.get("failed_to_decode"),
           "failed_other": r.get("failed_other")}
    if "pooled" in r["means"]:
        out["acc_pooled"] = r["means"]["pooled"]["acc"]
        out["pooled"] = r["means"]["pooled"]
    return out


def subset_row(b: dict, dets: list[str], res: Path, internal: set) -> dict | None:
    a = load(res / b["key"] / f"{AGENT}.json")
    comps = {d: comp_summary(r, d in internal) for d in dets if (r := load(res / b["key"] / f"{d}.json")) is not None}
    if a is None and not comps:
        return None
    row = {"benchmark": b["key"], "title": b["title"], "protocol": ({"limit": b["limit"]} if "limit" in b
                                                                       else {"fraction": b["fraction"]}),
           "n_comparators": len(comps), "missing_comparators": sorted(set(dets) - set(comps)), "comparators": comps}
    if a is not None:
        row[AGENT] = agent_summary(a)
    metrics = [("acc", "acc"), ("acc_strict", "acc"), ("mauroc", "mauroc"), ("acc_pooled", "acc_pooled")]
    for agent_metric, comp_metric in metrics:
        vals = [(c.get(comp_metric), d) for d, c in comps.items() if c.get(comp_metric) is not None]
        if not vals:
            continue
        if agent_metric == comp_metric:
            row[f"best_{comp_metric}"] = best_of(vals)
            row[f"best_public_{comp_metric}"] = best_of([(v, d) for v, d in vals if d not in internal])
            row[f"median_{comp_metric}"] = round(statistics.median(v for v, _ in vals), 2)
        if a is not None:
            mine = row[AGENT]["pooled"]["acc"] if agent_metric == "acc_pooled" else row[AGENT][agent_metric]
            row[AGENT][f"rank_{agent_metric}"] = 1 + sum(v > mine for v, _ in vals)
            row[AGENT][f"rank_{agent_metric}_of"] = len(vals) + 1
    return row


def subset_means(rows: list[dict], dets: list[str]) -> dict:
    out = {}
    for name, keys in (("held4", HELD), ("all6", tuple(r["benchmark"] for r in rows))):
        sel = [r for r in rows if r["benchmark"] in keys]
        block = {"subsets": [r["benchmark"] for r in sel]}
        if all(AGENT in r for r in sel) and sel:
            block[AGENT] = {k: round(statistics.mean(r[AGENT][k] for r in sel), 2) for k in ("acc", "acc_strict", "mauroc")}
        for d in dets:
            if all(d in r["comparators"] for r in sel) and sel:
                block[d] = {k: round(statistics.mean(r["comparators"][d][k] for r in sel), 2) for k in ("acc", "mauroc")}
        for metric in ("acc", "mauroc"):
            vals = [(block[d][metric], d) for d in dets if d in block]
            if vals:
                block[f"best_{metric}"] = best_of(vals)
                block[f"median_{metric}"] = round(statistics.median(v for v, _ in vals), 2)
                if AGENT in block:
                    mine = block[AGENT][metric]
                    block[AGENT][f"rank_{metric}"] = 1 + sum(v > mine for v, _ in vals)
        out[name] = block
    return out


def per_video(r: dict) -> list[dict]:
    return r.get("per_video") or []


def genbuster_paper(b: dict, res: Path, spec: dict) -> dict | None:
    r = load(res / b["key"] / f"{AGENT}.json")
    p = b["paper"]
    n_real_full = 1000
    n_fake_full = 2150 if b["key"] == "genbuster" else 1000
    entry = {"paper": p, "paper_implied_real_acc": round((p["acc"] * (n_real_full + n_fake_full)
                                                          - p["fake_recall"] * n_fake_full) / n_real_full, 1),
             "full_set": {"n_real": n_real_full, "n_fake": n_fake_full}, "subsample": {"fraction": b["fraction"]}}
    if r is None:
        return entry
    pb = r["means"]["pooled"]
    rows = [x for x in per_video(r) if "err" not in x]
    errs = [x for x in per_video(r) if "err" in x]
    right = sum(x["verdict"] == ("fake" if x["label"] else "real") for x in rows)
    ours = {"n": pb["n"], "n_real": pb["n_real"], "n_fake": pb["n_fake"], "acc": pb["acc"], "acc_strict": pb["acc_strict"],
            "acc_wilson95": wilson(right, len(rows)), "real_acc": pb["real_acc"], "fake_recall": pb["fake_recall"],
            "fake_f1": pb["fake_f1"], "n_tool_error": pb["n_tool_error"], "n_oom": pb["n_oom"],
            "n_unparsed": pb["n_unparsed"], "gpus": pb["gpus"]}
    ours["delta_acc_vs_paper"] = round(pb["acc"] - p["acc"], 2)
    ours["delta_acc_strict_vs_paper"] = round(pb["acc_strict"] - p["acc"], 2)
    ours["paper_acc_inside_wilson95"] = ours["acc_wilson95"][0] <= p["acc"] <= ours["acc_wilson95"][1]
    ours["delta_fake_recall_vs_paper"] = round(pb["fake_recall"] - p["fake_recall"], 2)
    ours["delta_fake_f1_vs_paper"] = round(pb["fake_f1"] - p["fake_f1"], 2)
    # stratified estimate of the full-set accuracy: the real rate weighted by 1000, each folder's fake rate by its size
    full_counts = FULL_FAKES.get(b["key"])
    groups = {}
    for x in rows:
        g = x.get("gen") if x["label"] else "__real__"
        groups.setdefault(g, []).append(x["verdict"] == ("fake" if x["label"] else "real"))
    if full_counts and "__real__" in groups:
        tot = n_real_full * statistics.mean(groups["__real__"]) + sum(
            full_counts[g] * statistics.mean(groups[g]) for g in full_counts if g in groups)
        ours["stratified_full_set_acc_estimate"] = round(100 * tot / (n_real_full + n_fake_full), 2)
    if b["key"] == "genbuster":
        rate = lambda xs: round(100 * statistics.mean(xs), 2) if xs else None            # noqa: E731
        fk = {g: v for g, v in groups.items() if g != "__real__"}
        idv = [o for g, v in fk.items() if g.startswith("id_") for o in v]
        wild = [o for g, v in fk.items() if g.startswith("wild_") for o in v]
        ood = {g[4:]: rate(v) for g, v in sorted(fk.items()) if g.startswith("ood_")}
        ours["table11"] = {"id": rate(idv), "ood": ood, "ood_mean": round(statistics.mean(ood.values()), 2) if ood else None,
                           "wild": rate(wild), "all": pb["acc"], "real_acc": pb["real_acc"],
                           "n": {"id": len(idv), "ood": {g[4:]: len(v) for g, v in sorted(fk.items()) if g.startswith("ood_")},
                                 "wild": len(wild), "real": len(groups["__real__"])}}
        entry["paper_table11"] = GB_TABLE11
    entry["ours_subsample"] = ours
    comps = {}
    for d in spec["detectors"]:
        c = load(res / b["key"] / f"{d}.json")
        if c is not None and "pooled" in c["means"]:
            comps[d] = {k: c["means"]["pooled"][k] for k in ("acc", "real_acc", "fake_recall", "fake_f1")}
    entry["comparators_pooled"] = comps
    vals = [(v["acc"], d) for d, v in comps.items()]
    if vals:
        entry["best_comparator_pooled_acc"] = best_of(vals)
        entry["median_comparator_pooled_acc"] = round(statistics.median(v for v, _ in vals), 2)
        entry["rank_pooled_acc"] = 1 + sum(v > pb["acc"] for v, _ in vals)
    return entry


# GenBuster-Bench / ++ full fake counts per folder (the loader's SOURCE.json, job wevis_gbdl_295596), for the stratified
# full-set estimate.
FULL_FAKES = {
    "genbuster": {"id_cogvideox": 300, "id_easyanimate": 400, "id_hunyuanvideo": 100, "id_ltxvideo": 200, "ood_gen3": 100,
                  "ood_jimeng": 100, "ood_kling": 100, "ood_luma": 100, "ood_pika": 100, "ood_sora": 200, "ood_vidu": 150,
                  "ood_wanx": 150, "wild_gen4.5": 10, "wild_hailuo2.3": 20, "wild_kling2.5_turbo": 15, "wild_kling2.6": 20,
                  "wild_pixverse_v5": 10, "wild_seedance1.5_pro": 20, "wild_seedance2.0": 15, "wild_sora2": 10,
                  "wild_veo3.1": 10, "wild_wan2.5": 10, "wild_wan2.6": 10},
    "genbuster_pp": {"video": 1000},
}


def determinism(res: Path) -> dict:
    """The MSR-VTT reals shared by the held subsets (same sha1), scored once per subset: verdict / logit agreement."""
    by_sha = {}
    for key in HELD:
        r = load(res / key / f"{AGENT}.json")
        man = load(res.parent / "subsets" / f"{key}.json")
        if r is None or man is None:
            continue
        sha = {Path(k).name: man["files"][k][0] for g in man["generators"] for k in g["reals"]}   # one real pool per subset
        for x in per_video(r):
            if x["label"] == 0 and "err" not in x:
                by_sha.setdefault(sha.get(Path(x["k"]).name), []).append((key, x["verdict"], x["logit"], x.get("tool_calls")))
    reps = {s: v for s, v in by_sha.items() if s and len(v) > 1}
    pairs = [(a, b) for v in reps.values() for i, a in enumerate(v) for b in v[i + 1:]]
    return {"n_files_scored_more_than_once": len(reps), "n_pairs": len(pairs),
            "verdict_identical": sum(a[1] == b[1] for a, b in pairs),
            "logit_identical": sum(a[2] == b[2] for a, b in pairs),
            "tool_calls_identical": sum(a[3] == b[3] for a, b in pairs),
            "max_abs_logit_diff": max((abs(a[2] - b[2]) for a, b in pairs), default=None)}


def fmt(v, w=6):
    return f"{v:>{w}.2f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "comparison.json"))
    args = ap.parse_args(argv)
    spec = json.loads((HERE / "spec.json").read_text())
    res = Path(args.results)
    dets, internal = spec["detectors"], set(spec["internal_only"])
    rows = [r for b in spec["benchmarks"] if (r := subset_row(b, dets, res, internal)) is not None]
    out = {"agent": AGENT, "protocol": spec["protocol"], "detectors": dets, "internal_only": sorted(internal),
           "rank_note": "rank_<metric> = 1 + the number of comparators strictly better (the agent among the comparators "
                        "plus itself); acc_strict is ranked against the comparators' acc (they have no tool errors); "
                        "best and median are over the comparators only",
           "acc_note": "acc: bench_eval_video's balanced accuracy (mean of real and fake accuracy per generator, then the "
                       "mean over generators); for the agent from its verdict (unparsed = wrong for both classes), "
                       "official = final tool_error / oom videos excluded with their counts reported, strict = counted "
                       "wrong. pooled: one row per distinct video (the paper's metric)",
           "subsets": rows, "subset_means": subset_means(rows, dets), "determinism": determinism(res),
           "paper_note": PAPER_NOTE,
           "paper_comparison": {b["key"]: genbuster_paper(b, res, spec) for b in spec["benchmarks"] if "paper" in b}}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'subset':14s} {'wevis acc':>9s} {'strict':>6s} {'AUROC':>6s} | {'best cmp acc':>22s} {'median':>6s} "
          f"{'rank':>5s} | {'best cmp AUROC':>22s} {'median':>6s} {'rank':>5s} | te/oom/unp/oomR")
    for r in rows:
        a = r.get(AGENT, {})
        ba, bm = r.get("best_acc") or {}, r.get("best_mauroc") or {}
        pb = a.get("pooled", {})
        print(f"{r['benchmark']:14s} {fmt(a.get('acc'), 9)} {fmt(a.get('acc_strict'))} {fmt(a.get('mauroc'))} | "
              f"{ba.get('detector', '-'):>14s} {fmt(ba.get('value'), 7)} {fmt(r.get('median_acc'))} "
              f"{a.get('rank_acc', '-')!s:>2}/{a.get('rank_acc_of', '-')!s:<2} | {bm.get('detector', '-'):>14s} "
              f"{fmt(bm.get('value'), 7)} {fmt(r.get('median_mauroc'))} {a.get('rank_mauroc', '-')!s:>2}/"
              f"{a.get('rank_mauroc_of', '-')!s:<2} | {pb.get('n_tool_error', '-')}/{pb.get('n_oom', '-')}/"
              f"{pb.get('n_unparsed', '-')}/{pb.get('oom_retries', '-')}")
    for k, e in out["paper_comparison"].items():
        o = (e or {}).get("ours_subsample")
        if o:
            print(f"{k}: paper {e['paper']['acc']} | ours (subsample n={o['n']}) acc {o['acc']} strict {o['acc_strict']} "
                  f"95% {o['acc_wilson95']} stratified {o.get('stratified_full_set_acc_estimate')} | R {o['fake_recall']} "
                  f"(paper {e['paper']['fake_recall']}) F1 {o['fake_f1']} (paper {e['paper']['fake_f1']}) | real acc "
                  f"{o['real_acc']} (paper implied {e['paper_implied_real_acc']})")
    print(f"determinism: {out['determinism']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
