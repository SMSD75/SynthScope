"""Run on cn023 (data host): list the files a `bench_eval --limit N` run would score, per benchmark,
using bench_eval's own discover() so a staged subset is IDENTICAL to the head of the full data.
Writes rsync --files-from lists (paths relative to /hddstore/ssalehi/AGID) + a summary."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/AGID")
spec = importlib.util.spec_from_file_location("be", "/ivi/zfs/s0/original_homes/ssalehi/AGID/benchmark/bench_eval.py")
be = importlib.util.module_from_spec(spec)
spec.loader.exec_module(be)

BASE = Path("/hddstore/ssalehi/AGID")
OUT = Path("/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/cure_ref/lists")
OUT.mkdir(parents=True, exist_ok=True)
JOBS = {  # name -> (bench_eval benchmark mode, root, per-class cap or None=full)
    "ufd": ("ufd", BASE / "universialfakedetection/test", None),
    "aigcdetect": ("pergen", BASE / "AIGCDetectBenchmark/test", int(sys.argv[1]) if len(sys.argv) > 1 else 1000),
    "aigibench": ("pergen", BASE / "AIGIBench/test", int(sys.argv[2]) if len(sys.argv) > 2 else 400),
}
for name, (mode, root, cap) in JOBS.items():
    gens = be.discover(mode, root, None, None)
    files, nbytes = [], 0
    for gen, reals, fakes in gens:
        if cap:
            reals, fakes = reals[:cap], fakes[:cap]
        for p in list(reals) + list(fakes):
            files.append(str(p.relative_to(BASE)))
            nbytes += p.stat().st_size
    (OUT / f"{name}.txt").write_text("\n".join(files) + "\n")
    print(f"{name}: {len(gens)} generators, {len(files)} files, {nbytes/1e9:.2f} GB (cap/class={cap})", flush=True)
