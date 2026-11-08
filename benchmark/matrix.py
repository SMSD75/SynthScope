"""Cross-evaluation matrix driver: every image detector x every benchmark (x optional degradations).

The cells come from ``repro/matrix/spec.json`` (detectors with an estimated throughput, benchmarks with their ``bench_eval``
profile and data folder, the plan's degradations, the existing cells that may be reused). The work is split into UNITS:

  * a cell is one unit, or, for the slow detectors (``"split": true`` in the spec), for the benchmarks flagged ``"split": true``
    (AIGIBench, Co-Spy-Bench), a few units that each score a chunk of the benchmark's generators (``bench_eval --select g1,g2,...``).
    Generators are listed with ``bench_eval.discover`` (so the machine that plans must be able to read the data) and packed in order
    into chunks of about ``--chunk-hours`` (estimate: images / the detector's ``imgs_per_s``); a generator bigger than that is a chunk
    of its own. A split cell with one chunk is just a whole-cell unit.
  * a unit scores into ``<out-root>/<bench>/<det>[__<degrade>].json`` (whole cell) or
    ``<out-root>/<bench>/_parts/<det>[__<degrade>]/<chunk>.json`` (chunk file names carry a hash of their generator list, so a
    change of ``--chunk-hours`` or of the data never mixes old and new parts). When the last part of a cell lands, the parts are
    merged into the cell JSON (the union of ``per_generator`` must equal the listing; the unweighted means are recomputed exactly as
    bench_eval does); ``--merge`` does only that, for parts brought together onto one filesystem.
  * units are ordered largest estimated cost first, ties by (benchmark, detector, degradation, chunk) in spec order.

Each unit is one ``benchmark/bench_eval.py`` subprocess (a detector's memory never leaks into the next one):

    python benchmark/bench_eval.py --model <det> --bs 32 --tag mx_<det>_<bench>[_<degrade>][__<chunk>] --benchmark <profile> \\
        --data <dir> [--reals <dir>] [--select g1,g2] [--degrade <spec>] --out <result>.partial.<writer> \\
        --results-log <out-root>/MATRIX_RESULTS.<host>.<pid>.jsonl

No ``--limit`` / ``--jpeg-fakes`` / ``--ckpt`` / ``--model-config`` / ``--native-res``: registry-default checkpoint
(``$SYNTHSCOPE_CKPT_DIR``, or ``--ckpt-dir``), registry eval transform, threshold 0.5, full data. Nothing is appended to
``repro/MASTER_RESULTS.jsonl``.

    python benchmark/matrix.py --dry-run --data-map-file repro/matrix/datamaps/cn029.json     # units, cost, commands; runs nothing
    python benchmark/matrix.py --worker --benchmarks wildrf,evalgen --data-map-file repro/matrix/datamaps/cn029.json
    python benchmark/matrix.py --worker --degrade jpeg75,down0.5 --benchmarks wildrf --data-root /hddstore/ssalehi/SynthScope
    python benchmark/matrix.py --merge --out-root /var/scratch/ssalehid/mx --data-root ...    # merge parts that are all there

Running several workers
  * Claiming: before running a unit a worker claims it by creating ``<claim-dir>/<unit>`` with O_CREAT|O_EXCL (default
    ``<out-root>/_claims``); a heartbeat thread touches the claim while the unit runs, and a claim not touched for
    ``--stale-minutes`` (30) is taken over, so a killed worker's unit is picked up again. ``--worker`` repeats passes over the unit list
    until a pass claims nothing; without it one pass runs. Claims need a filesystem with working O_EXCL (local disk, NFSv3+, Lustre).
    Clusters that do not share a filesystem must not share claims: give each its own ``--claim-dir`` and ``--out-root`` and split the
    work between them with ``--benchmarks`` (data locality); bring the result files together and run ``--merge``.
  * ``--shard i/n`` (0 <= i < n) takes units i, i+n, ... of the cost-sorted list: a static split on top of (or instead of) claiming.
  * No log is shared between writers: appends from several machines to one NFS file are not atomic. The results log and the failure
    log are per writer (``MATRIX_RESULTS.<host>.<pid>.jsonl``, ``FAILED.<host>.<pid>.jsonl``; ``--results-log`` overrides the first),
    and a result is renamed into place only when complete, so an existing JSON is a complete one. ``matrix_table.py`` reads every
    writer's files and takes the latest record per cell.
  * Resume: a unit whose result exists is skipped; a failed unit is logged with its error and retried by the next run (not again in
    the same process). The worker aborts (exit 2) after ``--max-same-failures`` (5) consecutive failures with the same error signature.
  * Before a result is kept its ``n_generators`` (and, for a chunk, its generator names) must equal the expected list; otherwise
    the unit is logged as failed and the file removed.

Cells and degradations
  * ``--degrade`` is a comma list of ``clean`` (the default), ``all`` (clean + every degradation in the spec) and ``bench_eval --degrade``
    tokens (jpeg75, down0.5, blur2, social, ...). ``--degrade jpeg75`` runs ONLY the jpeg75 cells.
  * Cells that the spec reuses (clean cells with an existing result in ``reuse``) are not run unless ``--include-reused``; degraded
    cells are never reused.

Data
  * Each benchmark's folder is looked up as ``--data-map NAME=PATH,...`` / ``--data-map-file FILE`` (a JSON ``{name: path}``, keys starting
    with ``_`` are comments; the command line wins), else ``<data root>/<spec "data">`` with the data root from ``--data-root`` or
    ``$SYNTHSCOPE_MATRIX_DATA``. A benchmark with a shared real pool (Synthbuster fakes + MSCOCO reals) also reads ``NAME.reals``; the
    driver then builds a symlink layout of the fakes under ``--layout-dir`` (a folder named by a hash of the two data paths, so nodes with
    different data roots never collide) and passes ``--reals``. Planning builds the layout, dry run included.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import datetime
import hashlib
import io
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO / "repro" / "matrix" / "spec.json"
DEFAULT_OUT_ROOT = REPO / "repro" / "matrix"
BENCH_EVAL = Path(__file__).resolve().parent / "bench_eval.py"
DATA_ROOT_ENV = "SYNTHSCOPE_MATRIX_DATA"
DEGRADE_TOKEN = re.compile(r"social2?|jpeg\d+|down\d+(?:\.\d+)?|blur\d+(?:\.\d+)?")   # the grammar of bench_eval.parse_degrade
TAIL_LINES = 40
DEFAULT_IMGS_PER_S = 22.0           # when the spec gives a detector no estimate
DEFAULT_CHUNK_HOURS = 2.0
WRITER = f"{socket.gethostname().split('.')[0]}.{os.getpid()}"


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

    def json_path(self, out_root: Path) -> Path:
        return out_root / self.benchmark / f"{self.stem}.json"

    def parts_dir(self, out_root: Path) -> Path:
        return out_root / self.benchmark / "_parts" / self.stem


@dataclasses.dataclass(frozen=True)
class Unit:
    """What one bench_eval run scores: a whole cell (``chunk`` None, ``generators`` None) or a chunk of its generators."""
    cell: Cell
    chunk: str | None = None            # part file stem, e.g. "c01-3fa9c2"
    index: int = 0                      # position of the chunk within the cell
    generators: tuple[str, ...] | None = None
    n_expected: int | None = None       # generators the result must contain (None: unknown)
    cost: float = 0.0                   # estimated seconds
    profile: str = "pergen"
    data: Path | None = None
    reals: Path | None = None
    error: str | None = None            # planning problem (no data folder ...): the unit fails when it is reached

    @property
    def key(self) -> str:
        return self.cell.name + (f"#{self.chunk}" if self.chunk else "")

    @property
    def claim_name(self) -> str:
        return self.key.replace("/", ".").replace("#", ".")

    @property
    def tag(self) -> str:
        return self.cell.tag + (f"__{self.chunk}" if self.chunk else "")

    def result_path(self, out_root: Path) -> Path:
        if self.chunk:
            return self.cell.parts_dir(out_root) / f"{self.chunk}.json"
        return self.cell.json_path(out_root)


class CircuitOpen(RuntimeError):
    """Too many consecutive failures with the same error signature."""


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
    """Comma list restricted to ``valid`` (spec order kept, so the order never depends on how the user typed it)."""
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
    """The cells in canonical order: benchmark-major, then detector, then degradation (as listed)."""
    cells = [Cell(d, b, g) for b in benchmarks for d in detectors for g in degrades]
    return cells if include_reused else [c for c in cells if not is_reused(spec, c)]


def parse_shard(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)/(\d+)", value)
    if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
        raise ValueError(f"--shard {value!r}: expected i/n with 0 <= i < n")
    return int(m.group(1)), int(m.group(2))


def shard_items(items: list, index: int, count: int) -> list:
    return items[index::count]


def parse_data_map(text: str | None, file: str | None) -> dict[str, str]:
    """``NAME=PATH,...`` over a JSON ``{name: path}`` file (keys starting with ``_`` are comments; the command line wins)."""
    mapping: dict[str, str] = {}
    if file:
        mapping.update({k: str(v) for k, v in json.loads(Path(file).read_text()).items() if not k.startswith("_")})
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
    and safe when several workers build it at once."""
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
    """A benchmark's folders on this machine (None = unresolved) and the layout to build first, if any."""
    data: Path | None
    reals: Path | None
    layout: Path | None
    fakes_root: Path | None


def plan_benchmark(spec_bench: dict, data_map: dict[str, str], data_root: str | None, layout_dir: Path) -> Plan:
    key = spec_bench["key"]
    data = resolve_path(key, spec_bench["data"], data_map, data_root)
    if spec_bench.get("layout") != "fakes_with_shared_reals":
        return Plan(data, None, None, None)
    reals = resolve_path(f"{key}.reals", spec_bench["reals"], data_map, data_root)
    if not (data and reals):
        return Plan(None, reals, None, data)
    digest = hashlib.sha1(f"{data}|{reals}".encode()).hexdigest()[:8]       # nodes with other data roots get other layouts
    layout = layout_dir / f"{key}-{digest}"
    return Plan(layout, reals, layout, data)


def plan_problem(plan: Plan, spec_bench: dict) -> str | None:
    """Why a benchmark cannot be scored here, or None."""
    key = spec_bench["key"]
    if plan.data is None:
        return (f"no data folder for {key}: give --data-root, ${DATA_ROOT_ENV} or --data-map {key}=PATH"
                + (f" and {key}.reals=PATH" if spec_bench.get("layout") else ""))
    for folder in (plan.fakes_root or plan.data, plan.reals):
        if folder is not None and not folder.is_dir():
            return f"data folder not found: {folder}"
    return None


# ---- planning: generators, chunks, units ----------------------------------------------------------------------------------

def list_generators(profile: str, root: Path, reals: Path | None) -> list[tuple[str, int]]:
    """``[(generator, images scored)]`` as ``bench_eval.discover`` finds them (reals + fakes; shared reals count per generator)."""
    for path in (str(REPO), str(REPO / "interface"), str(Path(__file__).resolve().parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import bench_eval                      # heavy (torch, registry): only when a split cell needs its generator list
    with contextlib.redirect_stdout(io.StringIO()):
        found = bench_eval.discover(profile, root, reals, None)
    return [(name, len(real) + len(fake)) for name, real, fake in found]


def make_chunks(listing: list[tuple[str, int]], imgs_per_s: float, target_s: float,
                max_generators: int | None = None) -> list[list[tuple[str, int]]]:
    """Pack the generators, in order, into chunks of about ``target_s`` estimated seconds (a bigger generator is a chunk of its own,
    and a chunk never has more than ``max_generators`` generators)."""
    chunks: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    seconds = 0.0
    for name, count in listing:
        cost = count / imgs_per_s
        if current and (seconds + cost > target_s or (max_generators and len(current) >= max_generators)):
            chunks.append(current)
            current, seconds = [], 0.0
        current.append((name, count))
        seconds += cost
    if current:
        chunks.append(current)
    return chunks


def chunk_name(index: int, generators: list[str]) -> str:
    return f"c{index:02d}-" + hashlib.sha1("\n".join(generators).encode()).hexdigest()[:6]


def plan_units(spec: dict, cells: list[Cell], data_map: dict[str, str], data_root: str | None, layout_dir: Path,
               chunk_seconds: float, list_fn=list_generators) -> tuple[list[Unit], dict[Cell, list[Unit]]]:
    """``(units sorted largest estimated cost first, chunk units of each split cell in chunk order)``. Planning problems become units
    with ``error`` set instead of exceptions, so one missing benchmark does not stop the rest."""
    detectors = {d["key"]: d for d in spec["detectors"]}
    benchmarks = {b["key"]: b for b in spec["benchmarks"]}
    order = {c: i for i, c in enumerate(cells)}
    plans: dict[str, Plan] = {}
    problems: dict[str, str | None] = {}
    listings: dict[str, list[tuple[str, int]] | str] = {}
    units: list[Unit] = []
    chunked: dict[Cell, list[Unit]] = {}
    for cell in cells:
        sb, sd = benchmarks[cell.benchmark], detectors[cell.detector]
        ips = float(sd.get("imgs_per_s", DEFAULT_IMGS_PER_S))
        if cell.benchmark not in plans:
            plans[cell.benchmark] = plan_benchmark(sb, data_map, data_root, layout_dir)
            problems[cell.benchmark] = plan_problem(plans[cell.benchmark], sb)
            if problems[cell.benchmark] is None and plans[cell.benchmark].layout:
                ensure_layout(plans[cell.benchmark].fakes_root, plans[cell.benchmark].layout)
        plan, problem = plans[cell.benchmark], problems[cell.benchmark]
        whole_cost = (sb.get("images") or 0) / ips
        base = dict(cell=cell, profile=sb["profile"], data=plan.data, reals=plan.reals)
        if problem:
            units.append(Unit(**base, cost=whole_cost, error=problem))
            continue
        if not (sd.get("split") or sb.get("split")):
            units.append(Unit(**base, n_expected=sb["n_generators"], cost=whole_cost))
            continue
        if cell.benchmark not in listings:
            try:
                listings[cell.benchmark] = list_fn(sb["profile"], plan.data, plan.reals) or "no generator with reals and fakes"
            except Exception as e:                                                  # noqa: BLE001
                listings[cell.benchmark] = f"cannot list the generators of {plan.data}: {e}"
        listing = listings[cell.benchmark]
        if isinstance(listing, str):
            units.append(Unit(**base, cost=whole_cost, error=listing))
            continue
        chunks = make_chunks(listing, ips, chunk_seconds, sb.get("max_chunk_generators"))
        if len(chunks) == 1:
            units.append(Unit(**base, n_expected=len(listing), cost=sum(n for _, n in listing) / ips))
            continue
        parts = [Unit(**base, chunk=chunk_name(i, [g for g, _ in chunk]), index=i, generators=tuple(g for g, _ in chunk),
                      n_expected=len(chunk), cost=sum(n for _, n in chunk) / ips) for i, chunk in enumerate(chunks)]
        chunked[cell] = parts
        units += parts
    units.sort(key=lambda u: (-u.cost, order[u.cell], u.index))
    return units, chunked


# ---- claims ---------------------------------------------------------------------------------------------------------------

def _write_claim(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps({"writer": WRITER, "time": datetime.datetime.now().isoformat(timespec="seconds")}).encode())
    finally:
        os.close(fd)
    return True


def try_claim(claim_dir: Path, name: str, stale_s: float) -> Path | None:
    """Claim ``name`` by creating its file with O_CREAT|O_EXCL; None if another worker holds a live claim. A claim whose mtime is
    older than ``stale_s`` is taken over (serialised by a short-lived ``.takeover`` file so two workers cannot both take it)."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    path = claim_dir / name
    if _write_claim(path):
        return path
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return path if _write_claim(path) else None
    if age < stale_s:
        return None
    lock = claim_dir / f"{name}.takeover"
    try:
        if time.time() - lock.stat().st_mtime > 60:                  # a crashed takeover: clear its lock
            lock.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    if not _write_claim(lock):
        return None
    try:
        try:
            if time.time() - path.stat().st_mtime < stale_s:           # somebody refreshed or re-claimed it meanwhile
                return None
        except FileNotFoundError:
            pass
        path.unlink(missing_ok=True)
        return path if _write_claim(path) else None
    finally:
        lock.unlink(missing_ok=True)


class Heartbeat:
    """Touch a claim file every ``interval`` seconds while the unit runs; remove it on exit."""

    def __init__(self, path: Path, interval: float):
        self.path, self.interval = path, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        while not self.stop.wait(self.interval):
            with contextlib.suppress(OSError):
                os.utime(self.path)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.path.unlink(missing_ok=True)


# ---- merging --------------------------------------------------------------------------------------------------------------

def mean2(values: list[float]) -> float:
    return round(sum(values) / len(values), 2)           # as bench_eval.main


def merge_results(parts: list[dict], expected: list[str], tag: str, chunks: list[str]) -> dict:
    """One cell result from its chunk results: ``per_generator`` is the union (in ``expected`` order, which must be exactly the union),
    the means are recomputed from it the way ``bench_eval`` does (round(sum / n, 2) of the per-generator values, which are already
    rounded). Metadata comes from the first part; ``date`` is the latest. ValueError if the parts do not fit together."""
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate generators in the expected list")
    union: dict[str, dict] = {}
    for part in parts:
        for name, metrics in part["per_generator"].items():
            if name in union:
                raise ValueError(f"generator {name} is in two parts")
            union[name] = metrics
    if set(union) != set(expected):
        raise ValueError(f"parts cover {len(union)} generators, the listing has {len(expected)}: missing "
                         f"{sorted(set(expected) - set(union))[:5]}, extra {sorted(set(union) - set(expected))[:5]}")
    first = parts[0]
    for part in parts[1:]:
        for key in ("model", "ckpt", "model_config", "benchmark", "jpeg_fakes", "degrade"):
            if part.get(key) != first.get(key):
                raise ValueError(f"parts disagree on {key}: {first.get(key)!r} vs {part.get(key)!r}")
    per_gen = {name: union[name] for name in expected}
    values = list(per_gen.values())
    merged = {
        "tag": tag, "model": first["model"], "ckpt": first.get("ckpt"), "model_config": first.get("model_config"),
        "benchmark": first.get("benchmark"), "data_root": first.get("data_root"), "jpeg_fakes": first.get("jpeg_fakes"),
        "n_generators": len(per_gen),
        "means": {"acc": mean2([m["acc"] for m in values]), "map": mean2([m["ap"] for m in values]),
                  "mauroc": mean2([m["auroc"] for m in values])},
        "per_generator": per_gen,
        "date": max(part.get("date", "") for part in parts),
        "synthscope_version": first.get("synthscope_version"),
    }
    if "degrade" in first:
        merged["degrade"] = first["degrade"]
    merged["merged_parts"] = chunks
    return merged


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{WRITER}")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def merge_cell(cell: Cell, chunk_units: list[Unit], out_root: Path) -> bool:
    """Merge the cell's parts into its JSON if they are all there and the JSON is not. True if it wrote the JSON; ValueError if the
    parts do not fit."""
    final = cell.json_path(out_root)
    paths = [u.result_path(out_root) for u in chunk_units]
    if final.is_file() or not all(p.is_file() for p in paths):
        return False
    parts = [json.loads(p.read_text()) for p in paths]
    expected = [g for u in chunk_units for g in (u.generators or ())]
    write_json_atomic(final, merge_results(parts, expected, cell.tag, [u.chunk for u in chunk_units]))
    return True


# ---- running --------------------------------------------------------------------------------------------------------------

def build_command(unit: Unit, out: Path, results_log: Path, *, python: str, bench_eval: Path, bs: int,
                  device: str | None) -> list[str]:
    cmd = [python, str(bench_eval), "--model", unit.cell.detector, "--bs", str(bs), "--tag", unit.tag,
           "--benchmark", unit.profile, "--data", str(unit.data)]
    if unit.reals:
        cmd += ["--reals", str(unit.reals)]
    if unit.generators:
        cmd += ["--select", ",".join(unit.generators)]
    if unit.cell.degrade:
        cmd += ["--degrade", unit.cell.degrade]
    cmd += ["--out", str(out), "--results-log", str(results_log)]
    if device:
        cmd += ["--device", device]
    return cmd


def append_line(path: Path, record: dict) -> None:
    """Append one JSON line. Only the process that owns ``path`` writes it (per-writer files): appends from several machines to one
    NFS file are not atomic, so nothing in this module shares a log between writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_command(cmd: list[str], env: dict[str, str]) -> tuple[int, list[str]]:
    """Run ``cmd``, echo its output, return (exit code, last lines)."""
    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
    return proc.wait(), list(tail)


def error_signature(error: str) -> str:
    """The failure with numbers and paths blanked, to tell 'the same failure again' from a different one."""
    return re.sub(r"\d+", "#", re.sub(r"/\S+", "<path>", error))[:120]


@dataclasses.dataclass
class Runner:
    out_root: Path
    claim_dir: Path
    results_log: Path
    failed_log: Path
    python: str
    bench_eval: Path
    bs: int
    device: str | None
    env: dict[str, str]
    chunked: dict[Cell, list[Unit]]
    stale_s: float = 1800.0
    heartbeat_s: float = 60.0
    max_same_failures: int = 5
    failed_here: set = dataclasses.field(default_factory=set)
    last_signature: str = ""
    same_failures: int = 0

    def fail(self, unit: Unit, error: str, cmd: list[str] | None = None, rc: int | None = None,
             tail: list[str] | None = None) -> None:
        print(f"!!! FAILED {unit.key}: {error}", flush=True)
        append_line(self.failed_log, {
            "time": datetime.datetime.now().isoformat(timespec="seconds"), "writer": WRITER, "cell": unit.key,
            "detector": unit.cell.detector, "benchmark": unit.cell.benchmark, "degrade": unit.cell.degrade, "chunk": unit.chunk,
            "rc": rc, "error": error, "cmd": shlex.join(cmd) if cmd else None, "tail": tail or []})
        self.failed_here.add(unit.key)
        signature = error_signature(error)
        self.same_failures = self.same_failures + 1 if signature == self.last_signature else 1
        self.last_signature = signature
        if self.same_failures >= self.max_same_failures:
            raise CircuitOpen(f"{self.same_failures} consecutive failures with the same error: {error}")

    def succeeded(self) -> None:
        self.same_failures, self.last_signature = 0, ""

    def done(self, unit: Unit) -> bool:
        return unit.cell.json_path(self.out_root).is_file() or unit.result_path(self.out_root).is_file()

    def merge(self, cell: Cell) -> None:
        units = self.chunked.get(cell)
        merge_unit = Unit(cell, chunk="merge")
        if not units or merge_unit.key in self.failed_here:
            return
        try:
            if merge_cell(cell, units, self.out_root):
                print(f"=== merged {len(units)} parts -> {cell.json_path(self.out_root)}", flush=True)
        except (ValueError, KeyError, OSError) as e:
            self.fail(merge_unit, f"merge failed: {e}")

    def run_unit(self, unit: Unit) -> bool:
        """Run one claimed unit; True if its result now exists. Any exception other than an open circuit breaker is a logged failure."""
        try:
            return self._run_unit(unit)
        except CircuitOpen:
            raise
        except Exception as e:                                                       # noqa: BLE001
            self.fail(unit, f"{type(e).__name__}: {e}")
            return False

    def _run_unit(self, unit: Unit) -> bool:
        if unit.error:
            self.fail(unit, unit.error)
            return False
        final = unit.result_path(self.out_root)
        partial = final.with_name(final.name + f".partial.{WRITER}")
        final.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        cmd = build_command(unit, partial, self.results_log, python=self.python, bench_eval=self.bench_eval, bs=self.bs,
                            device=self.device)
        print(f"=== unit {unit.key} (~{unit.cost / 3600:.1f} h) {datetime.datetime.now():%H:%M:%S}\n    {shlex.join(cmd)}", flush=True)
        rc, tail = run_command(cmd, self.env)
        if rc != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            last = next((line for line in reversed(tail) if line.strip()), "")
            self.fail(unit, f"bench_eval exited {rc}: {last}", cmd, rc, tail)
            return False
        problem = self.check_result(unit, partial)
        if problem:
            partial.unlink(missing_ok=True)
            self.fail(unit, problem, cmd, rc, tail)
            return False
        os.replace(partial, final)
        self.succeeded()
        if unit.chunk:
            self.merge(unit.cell)
        return True

    @staticmethod
    def check_result(unit: Unit, path: Path) -> str | None:
        """Why a finished result must not be kept, or None."""
        try:
            data = json.loads(path.read_text())
            per_gen = data["per_generator"]
        except (OSError, ValueError, KeyError) as e:
            return f"unreadable result: {e}"
        if unit.n_expected is not None and data.get("n_generators") != unit.n_expected:
            return f"scored {data.get('n_generators')} generators, expected {unit.n_expected}"
        if unit.generators is not None and set(per_gen) != set(unit.generators):
            return f"scored generators {sorted(per_gen)[:5]}..., expected the chunk {list(unit.generators)[:5]}..."
        return None

    def run_pass(self, units: list[Unit]) -> int:
        """One pass: claim and run what is neither done, nor held by another worker, nor failed here. Returns the number claimed."""
        claimed = 0
        for unit in units:
            if unit.key in self.failed_here:
                continue
            if self.done(unit):
                print(f"--- skip {unit.key}: result exists", flush=True)
                continue
            claim = try_claim(self.claim_dir, unit.claim_name, self.stale_s)
            if claim is None:
                print(f"--- skip {unit.key}: claimed by another worker", flush=True)
                continue
            claimed += 1
            with Heartbeat(claim, self.heartbeat_s):
                if self.done(unit):                          # finished by someone between our check and the claim
                    continue
                self.run_unit(unit)
        for cell in self.chunked:
            self.merge(cell)
        return claimed


def unit_status(unit: Unit, spec: dict, out_root: Path) -> str:
    if unit.cell.json_path(out_root).is_file() or unit.result_path(out_root).is_file():
        return "done"
    return "reused" if is_reused(spec, unit.cell) else "todo"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--spec", default=str(DEFAULT_SPEC), help="matrix spec (default repro/matrix/spec.json)")
    ap.add_argument("--detectors", help="comma list of detector keys (default: all in the spec)")
    ap.add_argument("--benchmarks", help="comma list of benchmark keys (default: all in the spec)")
    ap.add_argument("--degrade", default="clean", help="comma list: clean | all | jpeg<Q> down<F> blur<S> social social2 "
                    "(default clean; `--degrade jpeg75` runs only the jpeg75 cells)")
    ap.add_argument("--shard", help="i/n: take units i, i+n, ... of the cost-sorted unit list (0 <= i < n)")
    ap.add_argument("--worker", action="store_true", help="repeat passes over the unit list, claiming units, until a pass claims nothing")
    ap.add_argument("--merge", action="store_true", help="only merge the cells whose parts are all there (no unit is run)")
    ap.add_argument("--data-root", default=os.environ.get(DATA_ROOT_ENV),
                    help=f"folder holding every benchmark under its spec path (default ${DATA_ROOT_ENV})")
    ap.add_argument("--data-map", help="NAME=PATH,...: a benchmark's folder (NAME.reals: its shared real pool)")
    ap.add_argument("--data-map-file", help="JSON {name: path}, same keys as --data-map (see repro/matrix/datamaps/)")
    ap.add_argument("--out-root", "--out-dir", dest="out_root", default=str(DEFAULT_OUT_ROOT),
                    help="results folder (default repro/matrix)")
    ap.add_argument("--claim-dir", help="claim files of the workers sharing this folder (default <out-root>/_claims)")
    ap.add_argument("--layout-dir", help="where symlink layouts are built (default <out-root>/_layouts)")
    ap.add_argument("--results-log", help="bench_eval summary lines (default <out-root>/MATRIX_RESULTS.<host>.<pid>.jsonl, one per writer)")
    ap.add_argument("--ckpt-dir", help="SYNTHSCOPE_CKPT_DIR for the runs (default: the environment's)")
    ap.add_argument("--chunk-hours", type=float, help="target estimated GPU hours per chunk (default: the spec's, else 2)")
    ap.add_argument("--stale-minutes", type=float, default=30.0, help="a claim not touched for this long is taken over")
    ap.add_argument("--heartbeat-seconds", type=float, default=60.0)
    ap.add_argument("--max-same-failures", type=int, default=5, help="abort after this many consecutive failures with the same error")
    ap.add_argument("--include-reused", action="store_true", help="also run the clean cells the spec reuses")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--python", default=sys.executable, help="interpreter for bench_eval (default: this one)")
    ap.add_argument("--bench-eval", default=str(BENCH_EVAL), help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help="print the units and commands, run nothing")
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

    out_root = Path(args.out_root)
    layout_dir = Path(args.layout_dir) if args.layout_dir else out_root / "_layouts"
    claim_dir = Path(args.claim_dir) if args.claim_dir else out_root / "_claims"
    results_log = Path(args.results_log) if args.results_log else out_root / f"MATRIX_RESULTS.{WRITER}.jsonl"
    failed_log = out_root / f"FAILED.{WRITER}.jsonl"
    chunk_hours = args.chunk_hours or float(spec.get("chunk_hours", DEFAULT_CHUNK_HOURS))
    cells = build_cells(spec, detectors, benchmarks, degrades, args.include_reused)
    units, chunked = plan_units(spec, cells, data_map, args.data_root, layout_dir, chunk_hours * 3600)
    mine = [(i, u) for i, u in enumerate(units) if i % shard[1] == shard[0]]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if args.ckpt_dir:
        env["SYNTHSCOPE_CKPT_DIR"] = args.ckpt_dir

    print(f"[matrix] spec={args.spec} cells={len(cells)} units={len(units)} shard={shard[0]}/{shard[1]} -> {len(mine)} units "
          f"({len(detectors)} detectors x {len(benchmarks)} benchmarks x {len(degrades)} degradations"
          f"{'' if args.include_reused else ', reused clean cells left out'}; chunks of ~{chunk_hours:g} h) out={out_root}", flush=True)
    counts: collections.Counter[str] = collections.Counter()
    runner = Runner(out_root, claim_dir, results_log, failed_log, args.python, Path(args.bench_eval), args.bs, args.device, env,
                    chunked, stale_s=args.stale_minutes * 60, heartbeat_s=args.heartbeat_seconds,
                    max_same_failures=args.max_same_failures)
    if args.dry_run:
        for index, unit in mine:
            status = unit_status(unit, spec, out_root)
            if unit.error:
                shown = f"UNRESOLVED ({unit.error})"
            else:
                shown = shlex.join(build_command(unit, unit.result_path(out_root), results_log, python=args.python,
                                                 bench_eval=Path(args.bench_eval), bs=args.bs, device=args.device))
            print(f"#{index:<4d} {status:6s} ~{unit.cost / 3600:5.1f} h  {unit.key}  {shown}")
            counts[status] += 1
        print("[matrix] dry run: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
        return 0
    try:
        if args.merge:
            for cell in chunked:
                runner.merge(cell)
        else:
            while True:
                claimed = runner.run_pass([u for _, u in mine])
                if not args.worker or not claimed:
                    break
    except CircuitOpen as e:
        print(f"[matrix] ABORT (circuit breaker): {e}", flush=True)
        return 2
    print(f"[matrix] finished: failed={len(runner.failed_here)}", flush=True)
    return 1 if runner.failed_here else 0


if __name__ == "__main__":
    raise SystemExit(main())
