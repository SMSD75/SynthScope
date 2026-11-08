"""Cross-evaluation matrix driver: every image detector x every benchmark (x optional degradations).

The cells come from ``repro/matrix/spec.json`` (detectors, benchmarks with their ``bench_eval`` profile and data folder, the
plan's degradations, the existing cells that may be reused). Each cell is one ``benchmark/bench_eval.py`` run in its own
subprocess (a detector's memory never leaks into the next one):

    python benchmark/bench_eval.py --model <det> --bs 32 --tag mx_<det>_<bench>[_<degrade>] --benchmark <profile> --data <dir> \\
        [--reals <dir>] [--degrade <spec>] --out repro/matrix/<bench>/<det>[__<degrade>].json --results-log repro/matrix/MATRIX_RESULTS.jsonl

No ``--limit`` / ``--jpeg-fakes`` / ``--ckpt`` / ``--model-config`` / ``--native-res`` / ``--select``: registry-default checkpoint
(``$SYNTHSCOPE_CKPT_DIR``, or ``--ckpt-dir``), registry eval transform, threshold 0.5, full data. Nothing is appended to
``repro/MASTER_RESULTS.jsonl``.

    # what would run (nothing is executed), for shard 2 of 6, on a node holding the data under /local/data
    python benchmark/matrix.py --dry-run --shard 2/6 --data-root /local/data
    # run the clean cells of two benchmarks for two detectors, then the robustness slice of one
    python benchmark/matrix.py --benchmarks wildrf,evalgen --detectors clip,safe --data-map-file maps/das6.json
    python benchmark/matrix.py --benchmarks wildrf --degrade jpeg75,down0.5 --data-root /hddstore/ssalehi/SynthScope

Cells and sharding
  * ``--degrade`` is a comma list of ``clean`` (the default), ``all`` (clean + every degradation in the spec) and
    ``bench_eval --degrade`` tokens (jpeg75, down0.5, blur2, social, ...). ``--degrade jpeg75`` runs ONLY the jpeg75 cells.
  * The cell list is ordered benchmark-major (spec order), then detector (spec order), then degradation (as listed), and does not
    depend on which files exist. Cells that the spec reuses (clean cells with an existing result in ``reuse``) are not in the list
    unless ``--include-reused``; degraded cells are never reused.
  * ``--shard i/n`` (0 <= i < n) takes cells i, i+n, i+2n, ... of that list, so the n shards partition it. Shards that should
    keep a benchmark's data on one node can filter with ``--benchmarks`` instead.
  * Resume: a cell whose JSON already exists is skipped (the JSON is written under a temporary name and renamed on success, so an
    existing JSON is a complete one). A failed cell is appended, with the error, to ``<out-dir>/FAILED.jsonl`` and the run goes
    on; it is retried by the next run. Two shards may write the same ``MATRIX_RESULTS.jsonl`` / ``FAILED.jsonl``: every line is
    one O_APPEND write.

Data
  * Each benchmark's folder is looked up as ``--data-map NAME=PATH,...`` / ``--data-map-file FILE`` (a JSON ``{name: path}``;
    the command line wins), else ``<data root>/<spec "data">`` with the data root from ``--data-root`` or
    ``$SYNTHSCOPE_MATRIX_DATA``. A benchmark with a shared real pool (Synthbuster fakes + MSCOCO reals) also reads
    ``NAME.reals``; the driver then builds a symlink layout of the fakes under ``--layout-dir`` and passes ``--reals``.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO / "repro" / "matrix" / "spec.json"
DEFAULT_OUT_DIR = REPO / "repro" / "matrix"
BENCH_EVAL = Path(__file__).resolve().parent / "bench_eval.py"
DATA_ROOT_ENV = "SYNTHSCOPE_MATRIX_DATA"
DEGRADE_TOKEN = re.compile(r"social2?|jpeg\d+|down\d+(?:\.\d+)?|blur\d+(?:\.\d+)?")   # the grammar of bench_eval.parse_degrade
TAIL_LINES = 40


@dataclasses.dataclass(frozen=True)
class Cell:
    """One detector on one benchmark, clean (``degrade`` is None) or degraded."""
    detector: str
    benchmark: str
    degrade: str | None = None

    @property
    def stem(self) -> str:
        return self.detector + (f"__{self.degrade}" if self.degrade else "")

    @property
    def name(self) -> str:
        return f"{self.benchmark}/{self.stem}"

    @property
    def tag(self) -> str:
        return f"mx_{self.detector}_{self.benchmark}" + (f"_{self.degrade}" if self.degrade else "")

    def json_path(self, out_dir: Path) -> Path:
        return out_dir / self.benchmark / f"{self.stem}.json"


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    for key in ("detectors", "benchmarks"):
        names = [e["key"] for e in spec[key]]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate {key} keys")
    known = {d["key"] for d in spec["detectors"]}
    benchmarks = {b["key"] for b in spec["benchmarks"]}
    for det, per_bench in spec.get("reuse", {}).items():
        if det not in known or not set(per_bench) <= benchmarks:
            raise ValueError(f"{path}: reuse entry for unknown detector/benchmark: {det}")
    return spec


def split_list(value: str | None, valid: list[str], what: str) -> list[str]:
    """Comma list restricted to ``valid`` (spec order kept, so the cell order never depends on how the user typed it)."""
    if value is None:
        return list(valid)
    wanted = [v for v in value.split(",") if v]
    unknown = [v for v in wanted if v not in valid]
    if unknown:
        raise ValueError(f"unknown {what}: {', '.join(unknown)} (known: {', '.join(valid)})")
    return [v for v in valid if v in wanted]


def parse_degrades(value: str, spec: dict) -> list[str | None]:
    """``--degrade`` list -> [None (clean), 'jpeg75', ...] in the order given (duplicates dropped); ``all`` = clean + the spec's."""
    out: list[str | None] = []
    for token in (t for t in value.split(",") if t):
        if token == "all":
            items: list[str | None] = [None] + list(spec["degradations"])
        elif token == "clean":
            items = [None]
        elif DEGRADE_TOKEN.fullmatch(token):
            items = [token]
        else:
            raise ValueError(f"bad --degrade token {token!r}: clean, all, jpeg<Q>, down<F>, blur<S>, social or social2")
        out += [i for i in items if i not in out]
    if not out:
        raise ValueError("--degrade is empty")
    return out


def is_reused(spec: dict, cell: Cell) -> bool:
    return cell.degrade is None and cell.benchmark in spec.get("reuse", {}).get(cell.detector, {})


def build_cells(spec: dict, detectors: list[str], benchmarks: list[str], degrades: list[str | None],
                include_reused: bool = False) -> list[Cell]:
    """The ordered cell list: benchmark-major, then detector, then degradation (see the module docstring)."""
    cells = [Cell(d, b, g) for b in benchmarks for d in detectors for g in degrades]
    return cells if include_reused else [c for c in cells if not is_reused(spec, c)]


def parse_shard(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)/(\d+)", value)
    if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
        raise ValueError(f"--shard {value!r}: expected i/n with 0 <= i < n")
    return int(m.group(1)), int(m.group(2))


def shard_cells(cells: list[Cell], index: int, count: int) -> list[Cell]:
    return cells[index::count]


def parse_data_map(text: str | None, file: str | None) -> dict[str, str]:
    """``NAME=PATH,...`` over a JSON ``{name: path}`` file (the command line wins)."""
    mapping: dict[str, str] = {}
    if file:
        mapping.update({k: str(v) for k, v in json.loads(Path(file).read_text()).items()})
    for item in (text or "").split(","):
        if item:
            name, sep, path = item.partition("=")
            if not sep or not name or not path:
                raise ValueError(f"--data-map entry {item!r}: expected NAME=PATH")
            mapping[name] = path
    return mapping


def resolve_path(name: str, rel: str | None, data_map: dict[str, str], data_root: str | None) -> Path | None:
    """The folder for a spec entry: ``data_map[name]``, else ``<data root>/<rel>``; None if neither is known."""
    if name in data_map:
        return Path(data_map[name]).expanduser()
    if data_root and rel:
        return Path(data_root).expanduser() / rel
    return None


def ensure_layout(fakes_root: Path, layout: Path) -> None:
    """``<layout>/<generator>/1_fake`` -> ``<fakes_root>/<generator>/1_fake`` (directory symlinks), no ``0_real``: bench_eval's
    pergen profile then pairs every generator with the shared ``--reals`` pool (Synthbuster fakes + MSCOCO reals). Idempotent
    and safe when several shards build it at once."""
    for gen_dir in sorted(p for p in fakes_root.iterdir() if (p / "1_fake").is_dir()):
        target = layout / gen_dir.name / "1_fake"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and Path(os.readlink(target)) == gen_dir / "1_fake":
            continue
        try:
            target.symlink_to(gen_dir / "1_fake", target_is_directory=True)
        except FileExistsError:
            if not target.is_symlink() or Path(os.readlink(target)) != gen_dir / "1_fake":
                raise


@dataclasses.dataclass
class Plan:
    """What one cell needs: its folders (None = unresolved) and the layout to build first, if any."""
    data: Path | None
    reals: Path | None
    layout: Path | None
    fakes_root: Path | None


def plan_cell(spec_bench: dict, data_map: dict[str, str], data_root: str | None, layout_dir: Path) -> Plan:
    key = spec_bench["key"]
    data = resolve_path(key, spec_bench["data"], data_map, data_root)
    if spec_bench.get("layout") != "fakes_with_shared_reals":
        return Plan(data, None, None, None)
    reals = resolve_path(f"{key}.reals", spec_bench["reals"], data_map, data_root)
    layout = layout_dir / key
    return Plan(layout if data and reals else None, reals, layout, data)


def build_command(cell: Cell, spec_bench: dict, plan: Plan, out: Path, results_log: Path, *, python: str, bench_eval: Path,
                  bs: int, device: str | None) -> list[str]:
    cmd = [python, str(bench_eval), "--model", cell.detector, "--bs", str(bs), "--tag", cell.tag,
           "--benchmark", spec_bench["profile"], "--data", str(plan.data)]
    if plan.reals:
        cmd += ["--reals", str(plan.reals)]
    if cell.degrade:
        cmd += ["--degrade", cell.degrade]
    cmd += ["--out", str(out), "--results-log", str(results_log)]
    if device:
        cmd += ["--device", device]
    return cmd


def append_line(path: Path, record: dict) -> None:
    """Append one JSON line with a single O_APPEND write (atomic between processes for lines this short)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(record) + "\n").encode())
    finally:
        os.close(fd)


def run_command(cmd: list[str], env: dict[str, str]) -> tuple[int, list[str]]:
    """Run ``cmd``, echo its output, return (exit code, last lines)."""
    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
    return proc.wait(), list(tail)


def fail(failed_log: Path, cell: Cell, error: str, cmd: list[str] | None = None, rc: int | None = None,
         tail: list[str] | None = None) -> None:
    print(f"!!! FAILED {cell.name}: {error}", flush=True)
    append_line(failed_log, {"time": datetime.datetime.now().isoformat(timespec="seconds"), "cell": cell.name,
                             "detector": cell.detector, "benchmark": cell.benchmark, "degrade": cell.degrade,
                             "rc": rc, "error": error, "cmd": shlex.join(cmd) if cmd else None, "tail": tail or []})


def run_cell(cell: Cell, spec_bench: dict, plan: Plan, out_dir: Path, results_log: Path, failed_log: Path, *,
             python: str, bench_eval: Path, bs: int, device: str | None, env: dict[str, str]) -> bool:
    """Run one cell; True if its JSON now exists."""
    final = cell.json_path(out_dir)
    if plan.data is None:
        fail(failed_log, cell, f"no data folder for {cell.benchmark}: give --data-root, ${DATA_ROOT_ENV} or --data-map "
                               f"{cell.benchmark}=PATH" + (f" and {cell.benchmark}.reals=PATH" if plan.layout else ""))
        return False
    fakes_or_data = plan.fakes_root or plan.data
    for folder in (fakes_or_data, plan.reals):
        if folder is not None and not folder.is_dir():
            fail(failed_log, cell, f"data folder not found: {folder}")
            return False
    if plan.layout:
        ensure_layout(plan.fakes_root, plan.layout)
    partial = final.with_name(final.name + ".partial")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    cmd = build_command(cell, spec_bench, plan, partial, results_log, python=python, bench_eval=bench_eval, bs=bs,
                        device=device)
    print(f"=== cell {cell.name} {datetime.datetime.now():%H:%M:%S}\n    {shlex.join(cmd)}", flush=True)
    rc, tail = run_command(cmd, env)
    if rc != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        last = next((line for line in reversed(tail) if line.strip()), "")
        fail(failed_log, cell, f"bench_eval exited {rc}: {last}", cmd, rc, tail)
        return False
    os.replace(partial, final)
    return True


def cell_status(spec: dict, cell: Cell, out_dir: Path) -> str:
    if cell.json_path(out_dir).is_file():
        return "done"
    return "reused" if is_reused(spec, cell) else "todo"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--spec", default=str(DEFAULT_SPEC), help="matrix spec (default repro/matrix/spec.json)")
    ap.add_argument("--detectors", help="comma list of detector keys (default: all in the spec)")
    ap.add_argument("--benchmarks", help="comma list of benchmark keys (default: all in the spec)")
    ap.add_argument("--degrade", default="clean", help="comma list: clean | all | jpeg<Q> down<F> blur<S> social social2 "
                    "(default clean; `--degrade jpeg75` runs only the jpeg75 cells)")
    ap.add_argument("--shard", help="i/n: run cells i, i+n, ... of the cell list (0 <= i < n)")
    ap.add_argument("--data-root", default=os.environ.get(DATA_ROOT_ENV),
                    help=f"folder holding every benchmark under its spec path (default ${DATA_ROOT_ENV})")
    ap.add_argument("--data-map", help="NAME=PATH,...: a benchmark's folder (NAME.reals: its shared real pool)")
    ap.add_argument("--data-map-file", help="JSON {name: path}, same keys as --data-map")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="results folder (default repro/matrix)")
    ap.add_argument("--layout-dir", help="where symlink layouts are built (default <out-dir>/_layouts)")
    ap.add_argument("--results-log", help="bench_eval summary lines (default <out-dir>/MATRIX_RESULTS.jsonl)")
    ap.add_argument("--ckpt-dir", help="SYNTHSCOPE_CKPT_DIR for the runs (default: the environment's)")
    ap.add_argument("--include-reused", action="store_true", help="also run the clean cells the spec reuses")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--python", default=sys.executable, help="interpreter for bench_eval (default: this one)")
    ap.add_argument("--bench-eval", default=str(BENCH_EVAL), help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help="print the cells and commands, run nothing")
    args = ap.parse_args(argv)

    try:
        spec = load_spec(Path(args.spec))
        detectors = split_list(args.detectors, [d["key"] for d in spec["detectors"]], "detector")
        benchmarks = split_list(args.benchmarks, [b["key"] for b in spec["benchmarks"]], "benchmark")
        degrades = parse_degrades(args.degrade, spec)
        shard = parse_shard(args.shard) if args.shard else (0, 1)
        data_map = parse_data_map(args.data_map, args.data_map_file)
    except (ValueError, OSError, KeyError) as e:
        ap.error(str(e))

    out_dir = Path(args.out_dir)
    layout_dir = Path(args.layout_dir) if args.layout_dir else out_dir / "_layouts"
    results_log = Path(args.results_log) if args.results_log else out_dir / "MATRIX_RESULTS.jsonl"
    failed_log = out_dir / "FAILED.jsonl"
    by_key = {b["key"]: b for b in spec["benchmarks"]}
    all_cells = build_cells(spec, detectors, benchmarks, degrades, args.include_reused)
    mine = [(i, c) for i, c in enumerate(all_cells) if i % shard[1] == shard[0]]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if args.ckpt_dir:
        env["SYNTHSCOPE_CKPT_DIR"] = args.ckpt_dir

    print(f"[matrix] spec={args.spec} cells={len(all_cells)} shard={shard[0]}/{shard[1]} -> {len(mine)} cells "
          f"({len(detectors)} detectors x {len(benchmarks)} benchmarks x {len(degrades)} degradations"
          f"{'' if args.include_reused else ', reused clean cells left out'}) out={out_dir}", flush=True)
    counts: collections.Counter[str] = collections.Counter()
    for index, cell in mine:
        spec_bench = by_key[cell.benchmark]
        plan = plan_cell(spec_bench, data_map, args.data_root, layout_dir)
        status = cell_status(spec, cell, out_dir)
        if args.dry_run:
            if plan.data is None:
                shown = f"UNRESOLVED (no data folder for {cell.benchmark})"
            else:
                cmd = build_command(cell, spec_bench, plan, cell.json_path(out_dir), results_log, python=args.python,
                                    bench_eval=Path(args.bench_eval), bs=args.bs, device=args.device)
                shown = shlex.join(cmd) + (f"   [layout {plan.layout} <- {plan.fakes_root}]" if plan.layout else "")
            print(f"#{index:<4d} {status:6s} {cell.name}  {shown}")
            counts[status] += 1
            continue
        if cell.json_path(out_dir).is_file():
            print(f"--- skip {cell.name}: result exists", flush=True)
            counts["skipped"] += 1
            continue
        ok = run_cell(cell, spec_bench, plan, out_dir, results_log, failed_log, python=args.python,
                      bench_eval=Path(args.bench_eval), bs=args.bs, device=args.device, env=env)
        counts["ok" if ok else "failed"] += 1
    print(f"[matrix] {'dry run' if args.dry_run else 'finished'}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
          flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
