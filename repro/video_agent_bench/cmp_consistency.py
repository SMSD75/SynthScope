"""Do the comparator scores on the subsets (ivi-cn023, RTX A6000) equal the per-video scores of the merged full-data
cells (repro/bench_results/<det>_<bench>.json, ivi-cn029 RTX A5000 score caches) for the same files?

The subsets keep the full data's relative paths, so bench_eval_video's cache keys (last three path components) are the
same. Per (detector, benchmark): the number of subset videos found in the reference cache, the largest |P(fake)
difference| and |logit difference|, the decision agreement (P >= 0.5, or the engine verdict for NSG-VD), and whether the
two caches' cache_meta (checkpoint sha256, aux files, ffmpeg build, spec) are identical. D3 has no reference cache.

    python repro/video_agent_bench/cmp_consistency.py --ours OURS --ref REF --out repro/video_agent_bench/comparator_consistency.json
    (OURS / REF: <det>/<bench>/{scores.*.jsonl, cache_meta.json})
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(d: Path) -> dict:
    rec = {}
    for f in sorted(d.glob("scores.*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rec[r["k"]] = r
    return rec


def decision(r: dict) -> bool:
    return r["verdict"] == "fake" if "verdict" in r else r["p"] >= 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=str(HERE / "comparator_consistency.json"))
    args = ap.parse_args(argv)
    out = {}
    for d in sorted(p for p in Path(args.ours).glob("*/*") if p.is_dir()):
        det, bench = d.parent.name, d.name
        ref_dir = Path(args.ref) / det / bench
        ours = {k: r for k, r in read(d).items() if "err" not in r}
        if not ref_dir.is_dir():
            out.setdefault(det, {})[bench] = {"n_ours": len(ours), "reference": None}
            continue
        ref = read(ref_dir)
        both = [k for k in ours if k in ref and "err" not in ref[k]]
        dp = [abs(ours[k]["p"] - ref[k]["p"]) for k in both]
        dl = [abs(ours[k]["logit"] - ref[k]["logit"]) for k in both if "logit" in ours[k] and "logit" in ref[k]]
        mo, mr = json.loads((d / "cache_meta.json").read_text()), json.loads((ref_dir / "cache_meta.json").read_text())
        out.setdefault(det, {})[bench] = {
            "n_ours": len(ours), "n_in_reference": len(both),
            "max_abs_p_diff": max(dp, default=None), "max_abs_logit_diff": max(dl, default=None),
            "n_p_identical": sum(x == 0 for x in dp),
            "decision_agreement": sum(decision(ours[k]) == decision(ref[k]) for k in both),
            "cache_meta_identical": mo == mr,
            "cache_meta_diff": sorted(k for k in set(mo) | set(mr) if mo.get(k) != mr.get(k))}
    cells = [(det, b, v) for det, bs in out.items() for b, v in bs.items() if "n_in_reference" in v]
    nsg = [v["max_abs_p_diff"] for det, _, v in cells if det.startswith("nsg_vd")]
    rest = [v["max_abs_p_diff"] for det, _, v in cells if not det.startswith("nsg_vd")]
    out["_summary"] = {
        "n_cells_checked": len(cells),
        "decision_agreement_all": all(v["decision_agreement"] == v["n_in_reference"] == v["n_ours"] for _, _, v in cells),
        "max_abs_p_diff_excluding_nsg_vd": max(rest, default=None), "max_abs_p_diff_nsg_vd": max(nsg, default=None),
        "cache_meta_differences": sorted({k for _, _, v in cells for k in v["cache_meta_diff"]}),
        "not_covered": sorted(f"{det}:{b}" for det, bs in out.items() if not det.startswith("_")
                              for b, v in bs.items() if "n_in_reference" not in v),
        "coverage_note": "Only the four held subsets have a merged full-data reference (the video-matrix caches); the "
                         "GenBuster-Bench / ++ comparator cells have none, so they are not cross-checked. D3 never kept a "
                         "score cache: its cells are covered only by the sha256 identity of its inputs on both nodes "
                         "(d3_calibration.json; X-CLIP from the HF cache), not by per-video scores."}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["_summary"], indent=1))
    for det, bs in out.items():
        if det.startswith("_"):
            continue
        for b, v in bs.items():
            print(f"{det:12s} {b:12s} {json.dumps(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
