"""Fixed video benchmark subsets for the agentic detector (WeVisForensics, ``wevis``): plan them, list their files for
staging, stage them from local data, verify a staged copy.

Protocol (repro/video_agent_bench/spec.json):
  * held benchmarks (``limit``): ``bench_eval_video --limit L`` with L = ceil(250 / n_generators), i.e. the first L sorted
    reals and the first L sorted fakes of every generator in ``video_layout.discover`` order;
  * GenBuster-Bench / ++ (``fraction``): the first ceil(f * n) sorted clips of every folder (each generator's fakes, and
    the benchmark's one real pool), so the subset keeps the benchmark's real : fake composition for the paper's
    one-row-per-clip accuracy (file names are sha256 hashes, so sorted order carries no meaning).
``plan`` runs that selection on the original data and writes ``subsets/<bench>.json``: per generator the file keys (paths
relative to the benchmark data folder; ``@reals/<rel>`` for a separate real pool) and each file's sha1 and size. A staged
copy mirrors the source paths under ``<dest>/<bench>/`` (``<dest>/<bench>/<data>/...`` and ``<dest>/<bench>/<reals>/...``,
with ``<data>`` / ``<reals>`` the spec's paths relative to the source root), so ``bench_eval_video``'s cache keys (the last
three path components) are those of the full data, and ``verify`` proves the copy gives the same selection with
sha1-identical files. Standard library only (runs in a small ssh session next to the data).

    python repro/video_agent_bench/make_subsets.py plan --site 029 --src-root /ssdstore/ssalehi/SynthScope
    python repro/video_agent_bench/make_subsets.py files genvideo > list.txt     # rsync -aH --files-from list SRC/ DEST/genvideo/
    python repro/video_agent_bench/make_subsets.py stage genbuster --src-root /hddstore/ssalehi/SynthScope --dest DEST
    python repro/video_agent_bench/make_subsets.py verify genvideo DEST/genvideo                       # exit 1 on mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "benchmark"))

from video_layout import discover                                                 # noqa: E402

SPEC = HERE / "spec.json"
SUBSETS = HERE / "subsets"
REALS = "@reals/"


def load_spec() -> dict:
    return json.loads(SPEC.read_text())


def bench_spec(key: str) -> dict:
    return next(b for b in load_spec()["benchmarks"] if b["key"] == key)


def key_of(p: Path, data: Path, reals: Path | None) -> str:
    if reals is not None and p.is_relative_to(reals):
        return REALS + str(p.relative_to(reals))
    return str(p.relative_to(data))


def source_path(key: str, data: Path, reals: Path | None) -> Path:
    return (reals / key[len(REALS):]) if key.startswith(REALS) else data / key


def take(n: int, b: dict, staged: bool) -> int:
    """How many of a sorted list of ``n`` clips the subset keeps: ``limit`` (bench_eval_video --limit), or ceil(f * n) on
    the full data and everything on a staged copy (which holds exactly the subset)."""
    if "limit" in b:
        return min(n, b["limit"])
    return n if staged else math.ceil(b["fraction"] * n)


def selection(b: dict, data: Path, reals: Path | None, staged: bool = False) -> list[dict]:
    """[{name, reals: [key], fakes: [key]}] in discover order."""
    gens = discover(b["profile"], data, reals, None)
    pools = {tuple(p.name for p in r) for _, r, _ in gens}
    if "fraction" in b and len(pools) != 1:
        raise SystemExit(f"{b['key']}: the fraction protocol needs one real pool shared by every generator")
    return [{"name": name, "reals": [key_of(p, data, reals) for p in r[:take(len(r), b, staged)]],
             "fakes": [key_of(p, data, reals) for p in f[:take(len(f), b, staged)]]} for name, r, f in gens]


def sha1_size(path: Path) -> list:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return [h.hexdigest(), path.stat().st_size]


def roots(b: dict, base: Path) -> tuple[Path, Path | None]:
    return base / b["data"], (base / b["reals"] if b.get("reals") else None)


def real_pools(gens: list[dict]) -> set:
    """The distinct real pools, as bench_eval_video tells them apart (``pool_key``: the same file names; the loaders
    hard-link one pool into every generator's 0_real)."""
    return {tuple(Path(k).name for k in g["reals"]) for g in gens}


def n_distinct(gens: list[dict]) -> int:
    """Videos bench_eval_video scores once: each distinct real pool and every fake."""
    return sum(len(p) for p in real_pools(gens)) + sum(len(g["fakes"]) for g in gens)


def plan(args) -> int:
    spec = load_spec()
    SUBSETS.mkdir(exist_ok=True)
    keys = args.benchmarks.split(",") if args.benchmarks else [b["key"] for b in spec["benchmarks"] if b["site"] == args.site]
    rc = 0
    for key in keys:
        b = bench_spec(key)
        data, reals = roots(b, Path(args.src_root))
        if not data.is_dir() or (reals is not None and not reals.is_dir()):
            print(f"{key}: data missing under {args.src_root}", flush=True)
            rc = 1
            continue
        gens = selection(b, data, reals)
        if len(gens) != b["n_generators"]:
            print(f"{key}: {len(gens)} generators found, spec says {b['n_generators']}", flush=True)
            rc = 1
            continue
        files = {}
        for g in gens:
            for k in g["reals"] + g["fakes"]:
                if k not in files:
                    files[k] = sha1_size(source_path(k, data, reals))
        man = {"benchmark": key, "profile": b["profile"], "site": args.site, "source_root": args.src_root,
               "data": b["data"], "reals": b.get("reals"), "n_generators": len(gens),
               **({"limit": b["limit"]} if "limit" in b else {"fraction": b["fraction"]}),
               "n_files": len(files), "n_scored": sum(len(g["reals"]) + len(g["fakes"]) for g in gens),
               "n_distinct": n_distinct(gens), "n_distinct_real": sum(len(p) for p in real_pools(gens)),
               "bytes": sum(s for _, s in files.values()), "generators": gens, "files": files}
        (SUBSETS / f"{key}.json").write_text(json.dumps(man, indent=0) + "\n")
        print(f"{key}: {len(gens)} generators, {man.get('limit', man.get('fraction'))}, {man['n_distinct']} distinct videos "
              f"({man['n_distinct_real']} real), {len(files)} files, {man['bytes'] / 1e9:.2f} GB", flush=True)
    return rc


def manifest(key: str) -> dict:
    return json.loads((SUBSETS / f"{key}.json").read_text())


def rel_to_src(b: dict, k: str) -> str:
    """A manifest key as a path relative to the source root (what ``rsync --files-from`` and ``stage`` take)."""
    return f"{b['reals']}/{k[len(REALS):]}" if k.startswith(REALS) else f"{b['data']}/{k}"


def files(args) -> int:
    b, man = bench_spec(args.benchmark), manifest(args.benchmark)
    for k in man["files"]:
        print(rel_to_src(b, k))
    return 0


def stage(args) -> int:
    """Hard-link (copy across file systems) the subset's files from ``--src-root`` into ``<dest>/<bench>/``."""
    b, man = bench_spec(args.benchmark), manifest(args.benchmark)
    out = Path(args.dest) / args.benchmark
    for k in man["files"]:
        rel = rel_to_src(b, k)
        src, dst = Path(args.src_root) / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    print(f"{args.benchmark}: staged {len(man['files'])} files into {out}", flush=True)
    return 0


def verify(args) -> int:
    b, man = bench_spec(args.benchmark), manifest(args.benchmark)
    data, reals = roots(b, Path(args.dest))
    gens = selection(b, data, reals, staged=True)
    if gens != man["generators"]:
        print(f"{args.benchmark}: the selection on {args.dest} differs from the manifest", flush=True)
        return 1
    bad = [k for k, v in man["files"].items() if sha1_size(source_path(k, data, reals)) != v]
    if bad:
        print(f"{args.benchmark}: {len(bad)} file(s) differ from the manifest, e.g. {bad[:3]}", flush=True)
        return 1
    print(f"{args.benchmark}: verified {args.dest}: {len(gens)} generators, {man['n_distinct']} distinct videos, "
          f"{len(man['files'])} files sha1-identical to the manifest", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--site", required=True, choices=["023", "029"])
    p.add_argument("--src-root", required=True)
    p.add_argument("--benchmarks", default=None, help="comma list (default: every benchmark of --site)")
    p = sub.add_parser("files")
    p.add_argument("benchmark")
    p = sub.add_parser("stage")
    p.add_argument("benchmark")
    p.add_argument("--src-root", required=True)
    p.add_argument("--dest", required=True)
    p = sub.add_parser("verify")
    p.add_argument("benchmark")
    p.add_argument("dest", help="<dest>/<bench>: the staged copy of this benchmark")
    args = ap.parse_args(argv)
    os.umask(0o022)
    return {"plan": plan, "files": files, "stage": stage, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
