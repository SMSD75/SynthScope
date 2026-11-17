"""Proof that the bench_eval changes of this branch leave every non-VLM detector's scores unchanged.

Runs ``benchmark/bench_eval.py`` from two code trees on the same tiny inputs and compares the output JSONs field by field
(everything except ``date``, ``tag`` and the results-log path; ``per_generator`` and ``means`` must be identical):

    old = the branch before the VLM path (9f8b45c), new = this commit:   clean runs (no new flag)
    ref = fieldreport/harness (2691337, the matrix's bench_eval), new:   --degrade jpeg95 (the ported robustness path)

    python repro/vlm_bench/old_vs_new.py --old-tree OLD --ref-tree REF --new-tree NEW --subsets-root SUB --out OUT.json \\
        --checks npr:wildrf,clip:wildrf,spai:wildrf,cure:evalgen,safe:ufd8 --limit 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(tree: Path, det: str, bench: dict, sub: Path, limit: int, extra: list[str], out: Path, python: str) -> dict:
    data = sub / bench["key"] / "data"
    gen = sorted(p.name for p in data.iterdir() if p.is_dir() and p.name not in {"imagenet", "laion", "real", "0_real"})[0]
    cmd = [python, str(tree / "benchmark" / "bench_eval.py"), "--model", det, "--benchmark", bench["profile"], "--data", str(data),
           "--select", gen, "--limit", str(limit), "--bs", "4", "--tag", "ovn", "--out", str(out)] + extra
    if bench.get("reals"):
        cmd += ["--reals", str(sub / bench["key"] / "reals")]
    log = subprocess.run(cmd, cwd=tree, capture_output=True, text=True)
    if log.returncode != 0 or not out.exists():
        raise RuntimeError(f"{tree.name} {det} {bench['key']} {extra}: rc={log.returncode}\n{log.stdout[-2000:]}\n{log.stderr[-2000:]}")
    return json.loads(out.read_text())


def comparable(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("date", "tag", "data_root")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-tree", required=True)
    ap.add_argument("--ref-tree", required=True)
    ap.add_argument("--new-tree", required=True)
    ap.add_argument("--subsets-root", required=True)
    ap.add_argument("--checks", required=True, help="comma list det:bench")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = json.loads((HERE / "spec.json").read_text())
    benches = {b["key"]: b for b in spec["benchmarks"]}
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for check in args.checks.split(","):
            det, bench = check.split(":")
            for base_name, base, extra in (("old", args.old_tree, []), ("ref", args.ref_tree, ["--degrade", "jpeg95",
                                                                                              "--results-log", f"{tmp}/r.jsonl"])):
                a = run(Path(base), det, benches[bench], Path(args.subsets_root), args.limit, extra,
                        Path(tmp) / f"{base_name}_{det}_{bench}.json", args.python)
                b = run(Path(args.new_tree), det, benches[bench], Path(args.subsets_root), args.limit, extra,
                        Path(tmp) / f"new_{base_name}_{det}_{bench}.json", args.python)
                same = comparable(a) == comparable(b)
                ok &= same
                rows.append({"detector": det, "benchmark": bench, "baseline": base_name, "flags": extra[:2],
                             "identical": same, "baseline_per_generator": a["per_generator"],
                             "new_per_generator": b["per_generator"]})
                print(f"{det:12s} {bench:12s} {base_name} vs new {' '.join(extra[:2]) or '(clean)':16s} "
                      f"{'IDENTICAL' if same else 'DIFFERENT'} {json.dumps(b['per_generator'])}", flush=True)
    Path(args.out).write_text(json.dumps({"old_tree": args.old_tree, "ref_tree": args.ref_tree, "new_tree": args.new_tree,
                                          "limit": args.limit, "all_identical": ok, "checks": rows}, indent=1) + "\n")
    print(f"old_vs_new: {'ALL IDENTICAL' if ok else 'DIFFERENCES FOUND'} ({len(rows)} comparisons) -> {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
