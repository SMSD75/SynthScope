"""Every result JSON of repro/video_agent_bench/results must trace to its job logs:

* the log's ``=== RESULT <tag>: mAP=.. mAP_real=.. mAUROC=.. acc=.. [acc_strict=.. pooled_acc=.. pooled_acc_strict=..
  ...] over`` line equals the JSON's means (and, for the agent, acc_strict, the pooled accuracies and the tool_error /
  oom / unparsed / oom_retries counts);
* for the agent, every per-video record equals the LAST per-video log line of that video under the run's tag
  (``agent[<tag>] <key> verdict=.. logit=.. ... gpu=..`` for a scored video, ``agent[<tag>] <key> FAILED kind=..`` for a
  final tool_error / oom): same verdict, the same logit (exact repr), the same GPU, the same error kind.

    python repro/video_agent_bench/trace_check.py --logs /ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = re.compile(r"=== RESULT (\S+): mAP=(\S+) mAP_real=(\S+) mAUROC=(\S+) acc=(\S+)(.*?) over \d+ gens")
EXTRA = re.compile(r"(\w+)=(\S+)")
AGENT_OK = re.compile(r"agent\[(\S+)\] (\S+) verdict=(\S+) logit=(\S+) src=(\S+) status=(\S+) .* gpu=(.*?) retries=(\S+)")
AGENT_FAIL = re.compile(r"agent\[(\S+)\] (\S+) FAILED kind=(\S+) retries=(\S+) gpu=(.*?): ")


def num(x: str):
    return None if x == "None" else float(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--glob", default="wevis_*.out", help="comma list of log file patterns")
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args(argv)
    logged, videos = {}, {}
    for log in sorted({p for g in args.glob.split(",") for p in Path(args.logs).glob(g)}):
        for line in log.read_text(errors="replace").splitlines():
            if m := RESULT.search(line):
                extra = dict(EXTRA.findall(m.group(6)))
                logged.setdefault(m.group(1), []).append(
                    ((num(m.group(2)), num(m.group(3)), num(m.group(4)), num(m.group(5))), extra, log.name))
            elif m := AGENT_OK.search(line):
                videos[(m.group(1), m.group(2))] = ("ok", m.group(3), float(m.group(4)), m.group(7))
            elif m := AGENT_FAIL.search(line):
                videos[(m.group(1), m.group(2))] = ("err", m.group(3), None, m.group(5))
    ok = bad = n_videos = 0
    for f in sorted(Path(args.results).glob("*/*.json")):
        r = json.loads(f.read_text())
        mm = r["means"]
        want = (mm["map"], mm["map_real"], mm["mauroc"], mm["acc"])
        problems = []
        hits = [x for x in logged.get(r["tag"], []) if x[0] == want]
        if r.get("agent") is not None:
            pb = mm["pooled"]
            exp = {"acc_strict": mm["acc_strict"], "pooled_acc": pb["acc"], "pooled_acc_strict": pb["acc_strict"],
                   "tool_error": pb["n_tool_error"], "oom": pb["n_oom"], "unparsed": pb["n_unparsed"],
                   "oom_retries": pb["oom_retries"]}
            hits = [x for x in hits if all(num(x[1].get(k, "None")) == float(v) for k, v in exp.items())]
            for rec in r["per_video"]:
                n_videos += 1
                seen = videos.get((r["tag"], rec["k"]))
                if seen is None:
                    problems.append(f"no log line for {rec['k']}")
                elif "err" in rec:
                    if seen[:2] != ("err", rec["kind"]) or seen[3] != str(rec.get("gpu")):
                        problems.append(f"{rec['k']}: log {seen} vs JSON error {rec['kind']} / {rec.get('gpu')}")
                elif seen != ("ok", rec["verdict"], float(rec["logit"]), rec["gpu"]):
                    problems.append(f"{rec['k']}: log {seen} vs JSON {(rec['verdict'], rec['logit'], rec['gpu'])}")
        elif "pooled" in mm:
            hits = [x for x in hits if num(x[1].get("pooled_acc", "None")) == mm["pooled"]["acc"]]
        if not hits:
            problems.insert(0, f"no log RESULT line for tag {r['tag']} with {want} "
                               f"(logged: {[x[0] for x in logged.get(r['tag'], [])][:2]})")
        if problems:
            bad += 1
            print(f"MISMATCH {f.relative_to(args.results)}: {'; '.join(problems[:5])}"
                  + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
        else:
            ok += 1
    print(f"trace_check: {ok} result JSONs trace to their logs ({n_videos} agent per-video records checked), {bad} do not")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
