"""Fixed benchmark subsets for the VLM (Veritas++) evaluation: plan them, list their files for staging, verify a staged copy.

Protocol (repro/vlm_bench/spec.json): for every held image benchmark of the field-report matrix, ``bench_eval --limit L`` with
L = ceil(500 / n_generators), i.e. the first L sorted reals and the first L sorted fakes of every generator in
``bench_eval.discover`` order. ``plan`` runs that exact selection on the original data and writes
``repro/vlm_bench/subsets/<bench>.json``: every generator's file keys (paths relative to the benchmark root, ``@reals/<rel>``
for a shared real pool) and each file's sha1 and size. A staged copy of just those files with the same relative paths gives
the same selection (``verify`` proves it: ``discover`` + ``--limit`` on the copy == the manifest, and every sha1 matches), so
the VLM and every comparator detector score byte-identical images on any node.

    python repro/vlm_bench/make_subsets.py plan --site 029 --src-root /ssdstore/ssalehi/SynthScope        # on ivi-cn029
    python repro/vlm_bench/make_subsets.py files wildrf data > list.txt      # rsync --files-from list (data | reals)
    python repro/vlm_bench/make_subsets.py verify wildrf /local/.../vpp_subsets/wildrf                       # exit 1 on mismatch

A staged subset lives in ``<dest>/<bench>/data`` (and ``<dest>/<bench>/reals`` for Synthbuster's shared MSCOCO pool).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "interface"), str(REPO / "benchmark")]

from bench_eval import REALS_KEY_PREFIX, discover, image_key       # noqa: E402

SPEC = HERE / "spec.json"
SUBSETS = HERE / "subsets"


def load_spec() -> dict:
    return json.loads(SPEC.read_text())


def bench_spec(key: str) -> dict:
    return next(b for b in load_spec()["benchmarks"] if b["key"] == key)


def selection(b: dict, root: Path, reals: Path | None) -> list[dict]:
    """The bench_eval --limit selection on ``root``: [{name, reals: [key], fakes: [key]}] in discover order."""
    gens = discover(b["profile"], root, reals, None)
    lim = b["limit"]
    return [{"name": name, "reals": [image_key(p, root, reals) for p in r[:lim]],
             "fakes": [image_key(p, root, reals) for p in f[:lim]]} for name, r, f in gens]


def fakes_layout(fakes_root: Path, layout: Path) -> None:
    """``<layout>/<gen>/1_fake`` -> ``<fakes_root>/<gen>/1_fake`` (as benchmark/matrix.py builds it for Synthbuster)."""
    for gen_dir in sorted(p for p in fakes_root.iterdir() if (p / "1_fake").is_dir()):
        (layout / gen_dir.name).mkdir(parents=True, exist_ok=True)
        (layout / gen_dir.name / "1_fake").symlink_to(gen_dir / "1_fake", target_is_directory=True)


def sha1_size(path: Path) -> list:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return [h.hexdigest(), path.stat().st_size]


def source_path(key: str, data: Path, reals: Path | None) -> Path:
    return (reals / key[len(REALS_KEY_PREFIX):]) if key.startswith(REALS_KEY_PREFIX) else data / key


def plan(args) -> int:
    spec = load_spec()
    SUBSETS.mkdir(exist_ok=True)
    keys = args.benchmarks.split(",") if args.benchmarks else [b["key"] for b in spec["benchmarks"] if b["site"] == args.site]
    rc = 0
    for key in keys:
        b = bench_spec(key)
        data = Path(args.src_root) / b["data"]
        reals = Path(args.src_root) / b["reals"] if b.get("reals") else None
        if not data.is_dir() or (reals is not None and not reals.is_dir()):
            print(f"{key}: data missing under {args.src_root}", flush=True)
            rc = 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            root = data
            if b.get("layout") == "fakes_with_shared_reals":
                root = Path(tmp) / "layout"
                fakes_layout(data, root)
            gens = selection(b, root, reals)
        if len(gens) != b["n_generators"]:
            print(f"{key}: {len(gens)} generators found, spec says {b['n_generators']}", flush=True)
            rc = 1
            continue
        files = {}
        for g in gens:
            for k in g["reals"] + g["fakes"]:
                if k not in files:
                    files[k] = sha1_size(source_path(k, data, reals))
        man = {"benchmark": key, "profile": b["profile"], "limit": b["limit"], "n_generators": len(gens),
               "site": args.site, "source_root": str(data), "reals_root": str(reals) if reals else None,
               "layout": b.get("layout"), "n_files": len(files),
               "n_scored": sum(len(g["reals"]) + len(g["fakes"]) for g in gens),
               "bytes": sum(s for _, s in files.values()), "generators": gens, "files": files}
        (SUBSETS / f"{key}.json").write_text(json.dumps(man, indent=0) + "\n")
        print(f"{key}: {len(gens)} generators, L={b['limit']}, {man['n_scored']} scored images, {len(files)} distinct files, "
              f"{man['bytes'] / 1e9:.2f} GB", flush=True)
    return rc


def files(args) -> int:
    man = json.loads((SUBSETS / f"{args.benchmark}.json").read_text())
    reals = args.which == "reals"
    for k in man["files"]:
        if k.startswith(REALS_KEY_PREFIX) == reals:
            print(k[len(REALS_KEY_PREFIX):] if reals else k)
    return 0


def verify(args) -> int:
    man = json.loads((SUBSETS / f"{args.benchmark}.json").read_text())
    b = bench_spec(args.benchmark)
    dest = Path(args.dest)
    data, reals = dest / "data", (dest / "reals" if man["reals_root"] else None)
    gens = selection(b, data, reals)
    if gens != man["generators"]:
        print(f"{args.benchmark}: the selection on {dest} differs from the manifest", flush=True)
        return 1
    bad = [k for k, v in man["files"].items() if sha1_size(source_path(k, data, reals)) != v]
    extra = sum(1 for root in (data, reals) if root for _ in root.rglob("*") if _.is_file()) - len(man["files"])
    if bad:
        print(f"{args.benchmark}: {len(bad)} file(s) differ from the manifest, e.g. {bad[:3]}", flush=True)
        return 1
    print(f"{args.benchmark}: verified {dest}: {len(gens)} generators, {man['n_scored']} scored images, "
          f"{len(man['files'])} files sha1-identical to the manifest" + (f", {extra} extra file(s)" if extra else ""), flush=True)
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
    p.add_argument("which", choices=["data", "reals"])
    p = sub.add_parser("verify")
    p.add_argument("benchmark")
    p.add_argument("dest")
    args = ap.parse_args(argv)
    os.umask(0o022)
    return {"plan": plan, "files": files, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
