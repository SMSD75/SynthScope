"""Run the agent's benchmark units on the staged subsets (repro/video_agent_bench/spec.json), resumable, several workers.

Units (``benchmark/bench_eval_video.py`` called IN THIS PROCESS, so the agent's runner, about 4 min to load, stays
resident from one unit to the next):

    wevis:<bench>:<k>of<N>   shard k of N of the subset's distinct videos (``--shard k/N``) into the shared score cache
                             ``<cache-root>/<bench>``; done = marker ``<out-root>/<bench>/.wevis.<k>of<N>.done``
    wevis:<bench>            the aggregation (``--cache-only``), once every shard of <bench> is done:
                             ``<out-root>/<bench>/wevis.json`` (per-generator + pooled metrics, per-video records)

``all`` = every benchmark of the spec (or ``--benchmarks``): its N shards, then its aggregation. Every unit runs
``<subsets-root>/<bench>/<data>`` (+ ``<reals>``) with the spec's ``--limit`` (a no-op on the subset, recorded in the JSON).
Workers on one machine share ``--lock-dir``: a unit is run by whoever holds its ``fcntl.flock`` (released by the OS when a
worker dies), so the shards are shared out dynamically and a dead worker's shard is resumed from the score cache by the
next one. An aggregation whose shards are not all done is skipped. Passes repeat until a pass runs nothing.

    python repro/video_agent_bench/run_units.py --units all --subsets-root S --cache-root C --out-root O --lock-dir L
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOST = socket.gethostname().split(".")[0]


def load_spec() -> dict:
    return json.loads((HERE / "spec.json").read_text())


def load_bench_eval_video():
    sys.path[:0] = [str(REPO), str(REPO / "interface"), str(REPO / "benchmark")]
    spec = importlib.util.spec_from_file_location("bench_eval_video", REPO / "benchmark" / "bench_eval_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expand(units: str, spec: dict, shards: int, only: set) -> list[str]:
    out = []
    for u in units.split(","):
        if u == "all":
            for b in spec["benchmarks"]:
                if not only or b["key"] in only:
                    out += [f"wevis:{b['key']}:{k}of{shards}" for k in range(shards)] + [f"wevis:{b['key']}"]
        elif u:
            out.append(u)
    return out


def bench_argv(b: dict, args) -> list[str]:
    sub = Path(args.subsets_root) / b["key"]
    argv = ["-m", "wevis", "--benchmark", b["profile"], "--data", str(sub / b["data"]), "--tag", f"vab_{b['key']}_wevis",
            "--cache-dir", str(Path(args.cache_root) / b["key"])]
    if b.get("reals"):
        argv += ["--reals", str(sub / b["reals"])]
    if "limit" in b:
        argv += ["--limit", str(b["limit"])]
    return argv


def paths(unit: str, args):
    parts = unit.split(":")
    if parts[0] != "wevis" or len(parts) not in (2, 3):
        raise ValueError(f"unit {unit!r}: expected wevis:<bench>[:<k>of<N>]")
    out = Path(args.out_root) / parts[1]
    if len(parts) == 3:
        return parts, out / f".wevis.{parts[2]}.done"
    return parts, out / "wevis.json"


def shard_markers(bench: str, args) -> list[Path]:
    return [Path(args.out_root) / bench / f".wevis.{k}of{args.shards}.done" for k in range(args.shards)]


def run_unit(unit: str, spec: dict, args, bev) -> str:
    """'done', 'exists', 'locked', 'waiting' (an aggregation whose shards are not all done) or 'failed'."""
    parts, done = paths(unit, args)
    if done.exists():
        return "exists"
    b = next(x for x in spec["benchmarks"] if x["key"] == parts[1])
    if len(parts) == 2 and not all(m.exists() for m in shard_markers(b["key"], args)):
        return "waiting"
    lock = (Path(args.lock_dir) / (re.sub(r"[^A-Za-z0-9_.-]", "_", unit) + ".lock")).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return "locked"
    try:
        if done.exists():
            return "exists"
        done.parent.mkdir(parents=True, exist_ok=True)
        argv = bench_argv(b, args)
        if len(parts) == 3:
            k, n = parts[2].split("of")
            argv += ["--shard", f"{k}/{n}"]
        else:
            partial = done.with_name(done.name + ".partial")
            argv += ["--cache-only", "--out", str(partial), "--results-log",
                     str(Path(args.out_root) / f"RESULTS.{HOST}.{os.getpid()}.jsonl")]
        t0 = time.time()
        print(f"=== unit {unit} START {time.strftime('%F %T')} on {HOST} pid {os.getpid()}: bench_eval_video "
              f"{' '.join(argv)}", flush=True)
        try:
            rc = bev.main(argv)
        except SystemExit as e:                          # bench_eval_video's refusals (cache meta, --cache-only miss)
            rc = e.code if isinstance(e.code, int) else 1
            print(f"=== unit {unit}: SystemExit {e}", flush=True)
        except Exception:  # noqa: BLE001 — one unit failed; the worker goes on with the next
            traceback.print_exc()
            rc = 1
        if rc == 0:
            if len(parts) == 3:
                done.write_text(f"{time.strftime('%F %T')} {HOST} pid {os.getpid()}\n")
            else:
                partial.rename(done)
            print(f"=== unit {unit} DONE rc=0 in {(time.time() - t0) / 3600:.2f} h -> {done}", flush=True)
            return "done"
        print(f"=== unit {unit} FAILED rc={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        return "failed"
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", default="all", help="comma list of units (see the module doc) or all")
    ap.add_argument("--benchmarks", default="", help="with all: only these benchmarks (comma list)")
    ap.add_argument("--shards", type=int, default=4, help="shards per benchmark")
    ap.add_argument("--subsets-root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lock-dir", required=True)
    ap.add_argument("--max-failures", type=int, default=2, help="a unit that failed this often in this process is given up")
    args = ap.parse_args(argv)
    spec = load_spec()
    units = expand(args.units, spec, args.shards, set(filter(None, args.benchmarks.split(","))))
    Path(args.lock_dir).mkdir(parents=True, exist_ok=True)
    bev = load_bench_eval_video()
    failures: dict[str, int] = {}
    while True:
        ran = 0
        for unit in units:
            if failures.get(unit, 0) >= args.max_failures:
                continue
            status = run_unit(unit, spec, args, bev)
            if status == "done":
                ran += 1
            elif status == "failed":
                failures[unit] = failures.get(unit, 0) + 1
                ran += 1
        if not ran:
            break
    left = [u for u in units if not paths(u, args)[1].exists()]
    print(f"=== run_units {HOST} pid {os.getpid()}: {len(units) - len(left)}/{len(units)} units done; "
          f"failed: {sorted(u for u, n in failures.items() if n)}; not done (another worker, or waiting): {left[:20]}",
          flush=True)
    return 0 if not left else 3


if __name__ == "__main__":
    raise SystemExit(main())
