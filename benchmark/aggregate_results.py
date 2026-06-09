"""Aggregate the self-recorded bench_eval results into a master reproduced-results table.

Reads every repro/bench_results/<tag>.json (latest run per tag), prints a compact summary, writes
repro/RESULTS_TABLE.md (model | weights | benchmark | n_gen | acc / mAP / mAUROC), and supports
computing a paper-matching subset mean from a run's per-generator dict:

    python benchmark/aggregate_results.py                       # refresh RESULTS_TABLE.md
    python benchmark/aggregate_results.py --subset fatformer_official_forensynths progan,stylegan,...
    python benchmark/aggregate_results.py --detail npr_official_diffusionforensics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1] / "repro"
RESULTS_DIR = REPRO / "bench_results"


def load_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.stem.startswith(("_validate", "_probe", "_detail")):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        tag = d.get("tag", p.stem)
        if tag not in runs or d.get("date", "") >= runs[tag].get("date", ""):
            runs[tag] = d
    return runs


def weights_of(tag: str) -> str:
    if "_official_" in tag:
        return "official"
    if "_agid_" in tag:
        return "agid"
    return "other"


def subset_mean(run: dict, gens: list[str]) -> dict:
    pg = run.get("per_generator", {})
    found = [g for g in gens if g in pg]
    missing = [g for g in gens if g not in pg]
    if not found:
        return {"error": f"none of {gens} in {list(pg)}"}
    def m(k):
        return round(sum(pg[g][k] for g in found) / len(found), 2)
    return {"n": len(found), "gens": found, "missing": missing,
            "acc": m("acc"), "map": m("ap"), "mauroc": m("auroc")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", nargs=2, metavar=("TAG", "GENS"),
                    help="compute mean over a comma list of generators for one run tag")
    ap.add_argument("--detail", metavar="TAG", help="print per-generator detail for one run tag")
    args = ap.parse_args()
    runs = load_runs()

    if args.subset:
        tag, gens = args.subset
        if tag not in runs:
            print(f"no run '{tag}'. have: {sorted(runs)}"); return 1
        print(json.dumps(subset_mean(runs[tag], gens.split(",")), indent=2)); return 0

    if args.detail:
        if args.detail not in runs:
            print(f"no run '{args.detail}'. have: {sorted(runs)}"); return 1
        r = runs[args.detail]
        print(f"# {args.detail}  ({r['model']}, {r['benchmark']}, {r.get('n_generators')} gens)  means={r['means']}")
        for g, m in sorted(r["per_generator"].items()):
            print(f"  {g:32s} acc={m['acc']:6.2f}  AP={m['ap']:6.2f}  AUROC={m['auroc']:6.2f}  "
                  f"(real={m['real_acc']:.1f} fake={m['fake_acc']:.1f})")
        return 0

    # default: master table
    rows = []
    for tag, r in sorted(runs.items()):
        me = r.get("means", {})
        rows.append((r.get("model", "?"), weights_of(tag), r.get("benchmark", "?"),
                     tag, r.get("n_generators", 0),
                     me.get("acc"), me.get("map"), me.get("mauroc")))
    rows.sort(key=lambda x: (x[0], x[1], x[3]))
    lines = ["# AGID reproduced benchmark results (auto-generated from repro/bench_results/)\n",
             "| model | weights | benchmark | tag | n_gen | acc | mAP | mAUROC |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model, w, bench, tag, n, acc, mp, au in rows:
        lines.append(f"| {model} | {w} | {bench} | `{tag}` | {n} | {acc} | {mp} | {au} |")
    out = REPRO / "RESULTS_TABLE.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
