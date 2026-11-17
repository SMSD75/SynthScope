"""Run VLM / comparator bench_eval units on the staged benchmark subsets (repro/vlm_bench/spec.json), resumable, several workers.

A unit is one ``benchmark/bench_eval.py`` subprocess:

    vlm:<bench>[:<variant>]     the VLM (--vlm-model, default veritaspp); variant clean (default) | jpeg95 | official
                                (the authors' sampling, seed --seed) | full (the benchmark's FULL data from --full-data, no
                                --limit; same per-image cache as clean, so the subset images are not scored twice)
    cmp:<det>:<bench>[:jpeg95]  a comparator detector on the same subset (bs 32)

Every unit scores ``<subsets-root>/<bench>/data`` (+ ``reals``) with ``--limit L`` of the spec (a no-op on the subset, recorded in
the JSON) into ``<out-root>/<bench>/<model>[__<variant>].json``; a unit whose JSON exists is skipped. Workers on one machine
share ``--lock-dir``: a unit is run by whoever holds its ``fcntl.flock`` (released by the OS when a worker dies), so a preempted
job's unit is taken by the next worker, which resumes from the VLM per-image cache (``--cache-root/<bench>``). Passes repeat
until a pass runs nothing. The result JSON is written to ``<name>.partial`` first and renamed when bench_eval succeeded.

    python repro/vlm_bench/run_units.py --units vlm:wildrf,vlm:evalgen --subsets-root /local/.../subsets \\
        --cache-root /local/.../cache --out-root /local/.../out --lock-dir /local/.../locks
    python repro/vlm_bench/run_units.py --units all-cmp ...      # every detector of the spec x every benchmark (+ jpeg95)
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BENCH_EVAL = REPO / "benchmark" / "bench_eval.py"
NOISE = re.compile(r"it/s\]|s/it\]|[Ww]arning|^\s*$")
HOST = socket.gethostname().split(".")[0]


def load_spec() -> dict:
    return json.loads((HERE / "spec.json").read_text())


def expand(units: str, spec: dict) -> list[str]:
    out = []
    for u in units.split(","):
        if u == "all-cmp":
            out += [f"cmp:{d}:{b['key']}" for b in spec["benchmarks"] for d in spec["detectors"]]
            out += [f"cmp:{d}:{b}:jpeg95" for b in spec["jpeg95_benchmarks"] for d in spec["detectors"]]
        elif u:
            out.append(u)
    return out


def command(unit: str, spec: dict, args) -> tuple[list[str], Path]:
    parts = unit.split(":")
    kind = parts[0]
    if kind == "vlm":
        model, bench, variant = args.vlm_model, parts[1], (parts[2] if len(parts) > 2 else "clean")
    elif kind == "cmp":
        model, bench, variant = parts[1], parts[2], (parts[3] if len(parts) > 3 else "clean")
    else:
        raise ValueError(f"unit {unit!r}: expected vlm:<bench>[:variant] or cmp:<det>:<bench>[:jpeg95]")
    b = next(x for x in spec["benchmarks"] if x["key"] == bench)
    stem = model + ("" if variant == "clean" else f"__{variant}")
    out = Path(args.out_root) / bench / f"{stem}.json"
    sub = Path(args.subsets_root) / bench
    if variant == "full":
        full = json.loads(Path(args.full_data).read_text())
        data, reals = full[bench], full.get(f"{bench}.reals")
        cmd_limit = []
    else:
        data, reals = sub / "data", (sub / "reals" if b.get("reals") else None)
        cmd_limit = ["--limit", str(b["limit"])]
    cmd = [args.python, str(BENCH_EVAL), "--model", model, "--benchmark", b["profile"], "--data", str(data),
           "--tag", f"vb_{bench}_{stem}", "--out", f"{out}.partial", "--results-log",
           str(Path(args.out_root) / f"RESULTS.{HOST}.{os.getpid()}.jsonl")] + cmd_limit
    if reals:
        cmd += ["--reals", str(reals)]
    if variant == "jpeg95":
        cmd += ["--degrade", "jpeg95"]
    elif variant not in ("clean", "official", "full"):
        raise ValueError(f"unit {unit!r}: unknown variant {variant}")
    if kind == "vlm":
        cmd += ["--vlm-cache", str(Path(args.cache_root) / bench)]
        if variant == "official":
            cmd += ["--vlm-decoding", "official", "--vlm-seed", str(args.seed)]
    else:
        if variant == "official" or variant == "full":
            raise ValueError(f"unit {unit!r}: comparators run clean or jpeg95 only")
        cmd += ["--bs", str(args.bs)]
    return cmd, out


def run_unit(unit: str, spec: dict, args, lock_dir: Path) -> str:
    """'done' (ran OK), 'exists', 'locked', or 'failed'."""
    cmd, out = command(unit, spec, args)
    if out.exists():
        return "exists"
    lock = (lock_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", unit) + ".lock")).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return "locked"
    try:
        if out.exists():                       # finished by another worker between the check and the lock
            return "exists"
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"=== unit {unit} START {time.strftime('%F %T')} on {HOST}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in proc.stdout:
            if NOISE.search(line):
                continue
            tail = (tail + [line.rstrip()])[-30:]
            print(f"[{unit}] {line.rstrip()}", flush=True)
        rc = proc.wait()
        partial = Path(f"{out}.partial")
        if rc == 0 and partial.exists():
            partial.rename(out)
            print(f"=== unit {unit} DONE rc=0 in {(time.time() - t0) / 3600:.2f} h -> {out}", flush=True)
            return "done"
        print(f"=== unit {unit} FAILED rc={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        return "failed"
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", required=True, help="comma list of units (see the module doc) or all-cmp")
    ap.add_argument("--subsets-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lock-dir", required=True)
    ap.add_argument("--cache-root", default=None, help="VLM per-image caches (required for vlm units)")
    ap.add_argument("--full-data", default=None, help="JSON {bench: full data folder} for :full units")
    ap.add_argument("--vlm-model", default="veritaspp")
    ap.add_argument("--seed", type=int, default=0, help="--vlm-seed of :official units")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--exclude-detectors", default="", help="comma list: skip the cmp units of these detectors (failed staging)")
    ap.add_argument("--max-failures", type=int, default=3, help="a unit that failed this often in this process is given up")
    args = ap.parse_args(argv)
    spec = load_spec()
    excluded = set(filter(None, args.exclude_detectors.split(",")))
    units = [u for u in expand(args.units, spec) if not (u.startswith("cmp:") and u.split(":")[1] in excluded)]
    if excluded:
        print(f"=== run_units: comparator units of {sorted(excluded)} skipped (excluded at staging)", flush=True)
    lock_dir = Path(args.lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    failures: dict[str, int] = {}
    while True:
        ran = 0
        for unit in units:
            if failures.get(unit, 0) >= args.max_failures:
                continue
            status = run_unit(unit, spec, args, lock_dir)
            if status == "done":
                ran += 1
            elif status == "failed":
                failures[unit] = failures.get(unit, 0) + 1
                ran += 1
        if not ran:
            break
    left = [u for u in units if not command(u, spec, args)[1].exists()]
    print(f"=== run_units: {len(units) - len(left)}/{len(units)} unit results present; "
          f"failed: {sorted(u for u, n in failures.items() if n)}; not present: {left[:20]}{' ...' if len(left) > 20 else ''}",
          flush=True)
    return 0 if not left else 3


if __name__ == "__main__":
    raise SystemExit(main())
