"""Build ``repro/matrix/MATRIX.md`` from the matrix cell JSONs (``benchmark/matrix.py``).

    python benchmark/matrix_table.py                       # repro/matrix/spec.json + repro/matrix/<bench>/<det>[__<degrade>].json
    python benchmark/matrix_table.py --out-dir /tmp/mx --output /tmp/mx/MATRIX.md

Tables (detector x benchmark, unweighted mean over a benchmark's generators as ``bench_eval`` reports it): mAUROC, mAP and
accuracy at the fixed 0.5 threshold, a per-detector mean over the benchmarks it has (with their count), then, if degraded cells
exist, the robustness change versus the clean cell (percentage points). A cell comes from ``<out-dir>/<bench>/<det>.json`` (a
fresh matrix run) or, failing that, from the ``reuse`` map of the spec (an existing ``bench_results`` JSON, marked with a dagger);
degraded cells are never reused. Every number is read from one of those JSONs. Missing cells are shown as ``-``. Failures are read
from every writer's ``FAILED*.jsonl`` (latest record per unit); cells with parts but no merged JSON are listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matrix import DEFAULT_OUT_ROOT, REPO, Cell, load_spec     # noqa: E402

METRICS = (("mauroc", "mAUROC (%)"), ("map", "mAP (%)"), ("acc", "Accuracy at threshold 0.5 (%)"))
REUSED, INTERNAL, MISMATCH = "†", "‡", "!"


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_cell(spec: dict, cell: Cell, out_dir: Path, repo: Path) -> dict | None:
    """``{"means", "n_generators", "reused", "path"}`` for a cell, or None. A fresh result beats a reused one."""
    path = cell.json_path(out_dir)
    reused = False
    if not path.is_file() and cell.degrade is None:
        rel = spec.get("reuse", {}).get(cell.detector, {}).get(cell.benchmark)
        if rel:
            path, reused = repo / rel, True
    data = load_json(path)
    if not data or "means" not in data:
        return None
    return {"means": data["means"], "n_generators": data.get("n_generators"), "reused": reused, "path": str(path)}


def failed_cells(out_dir: Path) -> dict[str, dict]:
    """The latest record per unit over every writer's ``FAILED*.jsonl`` (``FAILED.<host>.<pid>.jsonl``, or an older ``FAILED.jsonl``)."""
    out: dict[str, dict] = {}
    for path in sorted(out_dir.glob("FAILED*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
                name = record["cell"]
            except (ValueError, KeyError, TypeError):
                continue
            if name not in out or record.get("time", "") >= out[name].get("time", ""):
                out[name] = record
    return out


def still_failed(record: dict, out_dir: Path) -> bool:
    """A failure counts while neither the cell's JSON nor (for a chunk) its part exists."""
    cell = Cell(record["detector"], record["benchmark"], record.get("degrade"))
    if cell.json_path(out_dir).is_file():
        return False
    chunk = record.get("chunk")
    return not (chunk and chunk != "merge" and (cell.parts_dir(out_dir) / f"{chunk}.json").is_file())


def unmerged_cells(out_dir: Path) -> dict[str, int]:
    """Cells with part files but no cell JSON -> number of parts landed."""
    out: dict[str, int] = {}
    for parts in sorted(out_dir.glob("*/_parts/*")):
        if parts.is_dir() and not (parts.parent.parent / f"{parts.name}.json").is_file():
            out[f"{parts.parent.parent.name}/{parts.name}"] = len(list(parts.glob("*.json")))
    return out


def degradations_present(spec: dict, out_dir: Path) -> list[str]:
    """Degradations with at least one result file, the spec's order first."""
    found = {p.stem.split("__", 1)[1] for p in out_dir.glob("*/*__*.json")}
    listed = [d for d in spec.get("degradations", []) if d in found]
    return listed + sorted(found - set(listed))


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build(spec: dict, out_dir: Path, repo: Path) -> str:
    detectors, benchmarks = spec["detectors"], spec["benchmarks"]
    expected = {b["key"]: b["n_generators"] for b in benchmarks}
    clean: dict[tuple[str, str], dict] = {}
    for d in detectors:
        for b in benchmarks:
            cell = read_cell(spec, Cell(d["key"], b["key"]), out_dir, repo)
            if cell:
                clean[d["key"], b["key"]] = cell
    mismatched = sorted(f"{b}/{d}" for (d, b), c in clean.items() if c["n_generators"] != expected[b])
    labels = {d["key"]: d["name"] + (f" {INTERNAL}" if d.get("internal_only") else "") for d in detectors}

    def shown(cell: dict | None, metric: str, key: tuple[str, str]) -> str:
        if cell is None:
            return "-"
        mark = (REUSED if cell["reused"] else "") + (MISMATCH if f"{key[1]}/{key[0]}" in mismatched else "")
        return fmt(cell["means"][metric]) + mark

    out = ["# Cross-evaluation matrix", "",
           "Every image detector on every benchmark, registry-default checkpoint, registry eval transform, full data, threshold 0.5; "
           "per-benchmark values are the unweighted mean over the benchmark's generators (`bench_eval` `means`). Generated by "
           "`benchmark/matrix_table.py` from the cell JSONs in `repro/matrix/<benchmark>/` and the `reuse` map of "
           "`repro/matrix/spec.json`.", "",
           f"* {REUSED} reused: an earlier `repro/bench_results` run of the same protocol, not re-run in the matrix.",
           f"* {INTERNAL} weights licence unclear or non-commercial: internal matrix only, keep out of anything public.",
           f"* {MISMATCH} the cell scored a different number of generators than the spec expects.",
           "* `-` not run (yet); see the failed and not-run sections.",
           "* `mean (n)` averages only the n benchmarks the detector has a cell for, so rows with different n are not comparable.",
           ""]

    # coverage
    rows = []
    for b in benchmarks:
        fresh = sum(1 for d in detectors if (c := clean.get((d["key"], b["key"]))) and not c["reused"])
        reused = sum(1 for d in detectors if (c := clean.get((d["key"], b["key"]))) and c["reused"])
        rows.append([b["title"], str(b["n_generators"]), str(fresh), str(reused), str(len(detectors) - fresh - reused)])
    total = [sum(int(r[i]) for r in rows) for i in (2, 3, 4)]
    rows.append(["**all**", "", f"**{total[0]}**", f"**{total[1]}**", f"**{total[2]}**"])
    out += ["## Coverage (clean cells)", "", md_table(["Benchmark", "Generators", "Run", "Reused", "Missing"], rows), ""]

    header = ["Detector"] + [b["key"] for b in benchmarks] + ["mean (n)"]
    for metric, title in METRICS:
        rows = []
        for d in detectors:
            cells = [clean.get((d["key"], b["key"])) for b in benchmarks]
            values = [c["means"][metric] for c in cells if c]
            avg = mean(values)
            rows.append([labels[d["key"]]] + [shown(c, metric, (d["key"], b["key"])) for c, b in zip(cells, benchmarks)]
                        + [f"{fmt(avg)} ({len(values)})" if values else "-"])
        out += [f"## {title}", "", md_table(header, rows), ""]

    # robustness
    degrades = degradations_present(spec, out_dir)
    if degrades:
        out += ["## Robustness (change versus the clean cell, percentage points)", "",
                "Each degradation is applied to every image (reals and fakes) before the detector's own transform "
                "(`bench_eval --degrade`). A change needs both the clean and the degraded cell; the number in brackets is how many "
                "benchmarks the change is averaged over.", ""]
        deg_cells = {(d["key"], b["key"], g): read_cell(spec, Cell(d["key"], b["key"], g), out_dir, repo)
                     for d in detectors for b in benchmarks for g in degrades}
        mismatched += sorted(Cell(d, b, g).name for (d, b, g), c in deg_cells.items() if c and c["n_generators"] != expected[b])

        def change(det: str, bench: str, g: str, metric: str) -> tuple[float, str] | None:
            """(degraded - clean, marks) when both cells exist; marks: dagger = the clean base is a reused cell, ! = the degraded cell
            scored an unexpected number of generators."""
            deg, base = deg_cells[det, bench, g], clean.get((det, bench))
            if not (deg and base):
                return None
            marks = (REUSED if base["reused"] else "") + (MISMATCH if deg["n_generators"] != expected[bench] else "")
            return deg["means"][metric] - base["means"][metric], marks

        for metric, title in (("mauroc", "mAUROC"), ("acc", "accuracy at 0.5")):
            rows = []
            for d in detectors:
                row = [labels[d["key"]]]
                present = False
                for g in degrades:
                    found = [c for b in benchmarks if (c := change(d["key"], b["key"], g, metric))]
                    present |= bool(found)
                    present_marks = {m for _, flags in found for m in flags}
                    marks = "".join(m for m in (REUSED, MISMATCH) if m in present_marks)
                    row.append(f"{fmt(mean([v for v, _ in found]), signed=True)}{marks} ({len(found)})" if found else "-")
                if present:
                    rows.append(row)
            if rows:
                out += [f"### Mean {title} change per detector", "",
                        f"{REUSED} some of the clean cells behind the mean are reused cells; {MISMATCH} some degraded cell scored an "
                        "unexpected number of generators.", "", md_table(["Detector"] + degrades, rows), ""]
        for g in degrades:
            rows = []
            for d in detectors:
                if not any(deg_cells[d["key"], b["key"], g] for b in benchmarks):
                    continue
                cells = []
                for b in benchmarks:
                    found = change(d["key"], b["key"], g, "mauroc")
                    deg = deg_cells[d["key"], b["key"], g]
                    cells.append(fmt(found[0], signed=True) + found[1] if found else (fmt(deg["means"]["mauroc"]) + "*" if deg else "-"))
                rows.append([labels[d["key"]]] + cells)
            if rows:
                out += [f"### {g}: mAUROC change per benchmark", "",
                        f"A value ending in `*` is the degraded mAUROC itself (the clean cell is missing); {REUSED} the clean cell is "
                        f"a reused one; {MISMATCH} the degraded cell scored an unexpected number of generators.", "",
                        md_table(["Detector"] + [b["key"] for b in benchmarks], rows), ""]

    failed = {name: rec for name, rec in failed_cells(out_dir).items() if still_failed(rec, out_dir)}
    if failed:
        out += ["## Failed units (latest error, retried by the next run)", "",
                md_table(["Unit", "Error"], [[f"`{n}`", rec["error"].replace("|", "/")[:200]] for n, rec in sorted(failed.items())]),
                ""]
    unmerged = unmerged_cells(out_dir)
    if unmerged:
        out += ["## Cells with parts but no merged result (still running, or run `benchmark/matrix.py --merge`)", "",
                md_table(["Cell", "Parts landed"], [[f"`{n}`", str(k)] for n, k in unmerged.items()]), ""]
    if mismatched:
        out += ["## Generator-count mismatches", "", ", ".join(f"`{m}`" for m in mismatched), ""]
    excluded = spec.get("excluded_detectors", {})
    if excluded:
        out += ["## Not run", "", md_table(["Detector", "Reason"], [[f"`{k}`", v] for k, v in excluded.items()]), ""]
    excluded = spec.get("excluded_benchmarks", {})
    if excluded:
        out += ["Benchmarks left out: " + "; ".join(f"`{k}` ({v})" for k, v in excluded.items()) + ".", ""]
    return "\n".join(out).rstrip("\n") + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", default=str(DEFAULT_OUT_ROOT / "spec.json"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT), help="folder with <benchmark>/<detector>[__<degrade>].json")
    ap.add_argument("--output", help="markdown file to write (default <out-dir>/MATRIX.md)")
    ap.add_argument("--repo-root", default=str(REPO), help="base of the spec's relative `reuse` paths")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    text = build(load_spec(Path(args.spec)), out_dir, Path(args.repo_root))
    output = Path(args.output) if args.output else out_dir / "MATRIX.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
