"""Every result JSON of repro/vlm_bench/results must trace to its job log: the log's ``=== RESULT <tag>: mAP=.. mAUROC=.. acc=..``
line equals the JSON's means, and (VLM) the per-image log lines equal the JSON's per_image records (verdict, logit, tokens).
A per-image log line belongs to the model of the last ``=== unit ... START`` line before it (its ``--model``), so the logs of
several VLMs can be checked together. ``--models`` limits the check to the result JSONs of those models.

    python repro/vlm_bench/trace_check.py --logs /ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs --glob 'vpp_*.out'
    python repro/vlm_bench/trace_check.py --logs ... --glob 'adb_*.out' --models adb_vlm
    python repro/vlm_bench/trace_check.py --logs ... --glob 'fakevlm_bench_*.out' --models fakevlm
    (default: vpp_*.out,adb_*.out,fakevlm_bench_*.out, every JSON)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = re.compile(r"=== RESULT (\S+): mAP=(\S+) mAUROC=(\S+) acc=(\S+) over")
START = re.compile(r"=== unit (\S+) START .* --model (\S+)")
IMG = re.compile(r"\[vlm:([^\]]+)\]\s+vlm (\S+) label=(\d) verdict=(\S+) logit=(\S+) src=\S+ tok=(\d+)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--glob", default="vpp_*.out,adb_*.out,fakevlm_bench_*.out", help="comma list of log file patterns")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--models", default="", help="comma list: check only the result JSONs of these models (default all)")
    args = ap.parse_args(argv)
    only = set(filter(None, args.models.split(",")))
    logged, imgs = {}, {}
    for log in sorted({p for g in args.glob.split(",") for p in Path(args.logs).glob(g)}):
        model = None
        for line in log.read_text(errors="replace").splitlines():
            if m := START.search(line):
                model = m.group(2)
            elif m := RESULT.search(line):
                logged.setdefault(m.group(1), set()).add((float(m.group(2)), float(m.group(3)), float(m.group(4)), log.name))
            elif (m := IMG.search(line)) and not line.rstrip().endswith("[cached]"):
                unit = m.group(1).split(":")
                variant = unit[1] if len(unit) > 1 and unit[1] != "full" else "clean"
                imgs.setdefault((model, unit[0], variant), {})[m.group(2)] = (m.group(4), round(float(m.group(5)), 4), int(m.group(6)))
    ok = bad = 0
    for f in sorted(Path(args.results).glob("*/*.json")):
        r = json.loads(f.read_text())
        if only and r["model"] not in only:
            continue
        means = (r["means"]["map"], r["means"]["mauroc"], r["means"]["acc"])
        hits = [x for x in logged.get(r["tag"], ()) if x[:3] == means]
        problems = [] if hits else [f"no log RESULT line with {means} (logged: {sorted(logged.get(r['tag'], ()))[:2]})"]
        if "per_image" in r:
            bench, stem = f.parent.name, f.stem
            variant = stem.split("__")[1] if "__" in stem else "clean"
            variant = "clean" if variant == "full" else variant            # full-data runs share the clean cache
            seen = imgs.get((r["model"], bench, variant), {})
            mism = [p["key"] for p in r["per_image"] if p["key"] in seen
                    and seen[p["key"]] != (str(p["verdict"]), round(p["logit"], 4), p["new_tokens"])]
            missing = sum(p["key"] not in seen for p in r["per_image"])
            if mism:
                problems.append(f"{len(mism)} per-image record(s) differ from the log, e.g. {mism[:2]}")
            if missing:
                problems.append(f"{missing} per-image record(s) without a log line")
        if problems:
            bad += 1
            print(f"MISMATCH {f.relative_to(args.results)}: {'; '.join(problems)}")
        else:
            ok += 1
    print(f"trace_check: {ok} result JSONs trace to their logs, {bad} do not")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
