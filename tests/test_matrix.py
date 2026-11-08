"""Checks the cross-evaluation matrix driver (``benchmark/matrix.py``) and table (``benchmark/matrix_table.py``); CPU only, no
weights, no data: the cells run a stand-in for ``bench_eval.py``.

* the committed ``repro/matrix/spec.json``: its detectors are exactly the registry's image detectors minus the excluded ones,
  benchmark keys are unique, every ``reuse`` file exists and agrees with its detector and generator count,
* the cell list is deterministic, ordered benchmark-major, leaves out the reused clean cells (``--include-reused`` puts them
  back), and every ``--shard i/n`` split partitions it; bad shard / degrade / detector arguments are rejected,
* ``--dry-run`` prints the cells and commands and creates nothing,
* a run with a stand-in ``bench_eval``: the command carries the exact flags (``--out``, ``--results-log``, ``--tag``, ``--degrade``,
  ``--reals``), existing results are skipped (resume), a failing cell is logged to ``FAILED.jsonl`` with its error while the run goes
  on and is retried next time, no partial file is left, an unresolved data folder is a logged failure,
* many processes appending to one jsonl never interleave a line,
* the Synthbuster-style symlink layout is found by ``bench_eval.discover`` and pairs every generator with the shared reals,
* the table builds from fake cell JSONs: fresh beats reused, reused cells are marked, a degraded cell gives a robustness change,
  a failed cell is listed.
"""

from __future__ import annotations

import contextlib
import io
import json
import multiprocessing
import os
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

import matrix                                                         # noqa: E402
import matrix_table                                                   # noqa: E402
from matrix import Cell                                               # noqa: E402

FAKE_BENCH_EVAL = textwrap.dedent('''
    import json, os, sys
    a = sys.argv[1:]
    get = lambda k: a[a.index(k) + 1] if k in a else None
    with open(os.environ["FAKE_CALLS"], "a") as f:
        f.write(json.dumps(a) + "\\n")
    if get("--model") == os.environ.get("FAKE_FAIL_MODEL"):
        print("Traceback (most recent call last):\\nRuntimeError: boom")
        sys.exit(3)
    base = float(len(get("--model")) + len(get("--tag")))
    result = {"tag": get("--tag"), "model": get("--model"), "n_generators": int(os.environ.get("FAKE_NGEN", "2")),
              "means": {"acc": base, "map": base + 1, "mauroc": base + 2 - (5 if get("--degrade") else 0)},
              "degrade": get("--degrade")}
    open(get("--out"), "w").write(json.dumps(result))
    with open(get("--results-log"), "a") as f:
        f.write(json.dumps({"tag": get("--tag")}) + "\\n")
''')


def small_spec() -> dict:
    return {
        "version": 1, "degradations": ["jpeg75", "blur2"],
        "excluded_detectors": {"gone": "no weights"}, "excluded_benchmarks": {"nope": "not held"},
        "detectors": [{"key": "aa", "name": "Alpha"}, {"key": "bb", "name": "Beta", "internal_only": True},
                      {"key": "cc", "name": "Gamma"}],
        "benchmarks": [{"key": "plain", "title": "Plain", "profile": "pergen", "data": "Plain/test", "n_generators": 2},
                       {"key": "shared", "title": "Shared reals", "profile": "pergen", "data": "Fakes/test", "n_generators": 2,
                        "layout": "fakes_with_shared_reals", "reals": "Pool/0_real"}],
        "reuse": {"aa": {"plain": "old/aa_plain.json"}},
    }


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def check_committed_spec() -> None:
    from interface.backend.registry import MODELS
    spec = matrix.load_spec(matrix.DEFAULT_SPEC)
    image = {k for k, s in MODELS.items() if s.modality == "image"}
    keys = [d["key"] for d in spec["detectors"]]
    assert set(keys) == image - set(spec["excluded_detectors"]), (set(keys) ^ (image - set(spec["excluded_detectors"])))
    assert set(spec["excluded_detectors"]) == {"vib_net_official", "effort_chameleon"}
    assert len(spec["benchmarks"]) == 15 and len(keys) == 25
    for det, per_bench in spec["reuse"].items():
        for bench, rel in per_bench.items():
            data = json.loads((ROOT / rel).read_text())
            want = next(b["n_generators"] for b in spec["benchmarks"] if b["key"] == bench)
            assert data["model"] == det and data["n_generators"] == want, (det, bench, rel)
    for b in spec["benchmarks"]:
        assert b["profile"] in ("pergen", "forensynths", "ufd", "pool") and b["data"], b
    assert sum(len(v) for v in spec["reuse"].values()) == 143
    cells = matrix.build_cells(spec, keys, [b["key"] for b in spec["benchmarks"]], [None])
    assert len(cells) == 25 * 15 - 143, len(cells)


def check_cells_and_shards() -> None:
    spec = small_spec()
    cells = matrix.build_cells(spec, ["aa", "bb", "cc"], ["plain", "shared"], [None, "jpeg75"])
    assert cells == matrix.build_cells(spec, ["aa", "bb", "cc"], ["plain", "shared"], [None, "jpeg75"])    # deterministic
    names = [c.name for c in cells]
    assert names[:3] == ["plain/aa__jpeg75", "plain/bb", "plain/bb__jpeg75"], names      # aa/plain clean is reused: left out
    assert names[3:5] == ["plain/cc", "plain/cc__jpeg75"] and names[5].startswith("shared/aa"), names
    assert len(cells) == 12 - 1
    assert len(matrix.build_cells(spec, ["aa", "bb", "cc"], ["plain", "shared"], [None, "jpeg75"], include_reused=True)) == 12
    c = Cell("aa", "plain", "jpeg75")
    assert (c.tag, c.name, str(c.json_path(Path("/o")))) == ("mx_aa_plain_jpeg75", "plain/aa__jpeg75", "/o/plain/aa__jpeg75.json")
    assert Cell("aa", "plain").tag == "mx_aa_plain"
    # the user's order does not change the cell order
    assert matrix.split_list("cc,aa", ["aa", "bb", "cc"], "detector") == ["aa", "cc"]
    for n in (1, 2, 3, 5, 7, 40):
        parts = [matrix.shard_cells(cells, i, n) for i in range(n)]
        assert sorted(sum(parts, []), key=cells.index) == cells and sum(len(p) for p in parts) == len(cells), n
        assert matrix.shard_cells(cells, 0, n) == matrix.shard_cells(cells, 0, n)
    for bad in ("3/3", "1", "a/b", "-1/2", "0/0", "2/1"):
        try:
            matrix.parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert matrix.parse_shard("2/5") == (2, 5)
    assert matrix.parse_degrades("clean,jpeg75,jpeg75", spec) == [None, "jpeg75"]
    assert matrix.parse_degrades("all", spec) == [None, "jpeg75", "blur2"]
    assert matrix.parse_degrades("down0.5,social2", spec) == ["down0.5", "social2"]
    for bad in ("", ",", "jpeg", "rot90", "Social", "blur-1"):
        try:
            matrix.parse_degrades(bad, spec)
        except ValueError:
            continue
        raise AssertionError(bad)
    try:
        matrix.split_list("zz", ["aa"], "detector")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown detector accepted")
    assert matrix.parse_data_map("a=/x,b.reals=/y", None) == {"a": "/x", "b.reals": "/y"}


def write_spec(tmp: Path) -> Path:
    path = tmp / "spec.json"
    path.write_text(json.dumps(small_spec()))
    return path


def run_matrix(tmp: Path, extra: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = matrix.main(["--spec", str(tmp / "spec.json"), "--out-dir", str(tmp / "mx"), "--python", sys.executable,
                            "--bench-eval", str(tmp / "fake_bench_eval.py"), "--data-root", str(tmp / "data")] + extra)
    return code, out.getvalue()


def calls(tmp: Path) -> list[list[str]]:
    path = tmp / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def check_run_skip_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        write_spec(tmp)
        (tmp / "fake_bench_eval.py").write_text(FAKE_BENCH_EVAL)
        for rel in ("Plain/test", "Pool/0_real"):
            (tmp / "data" / rel).mkdir(parents=True)
        for g in ("g1", "g2"):
            (tmp / "data" / "Fakes/test" / g / "1_fake").mkdir(parents=True)
        os.environ["FAKE_CALLS"] = str(tmp / "calls.jsonl")
        os.environ.pop("FAKE_FAIL_MODEL", None)
        try:
            # dry run: lists, runs nothing, creates nothing
            code, text = run_matrix(tmp, ["--dry-run", "--degrade", "clean,jpeg75"])
            assert code == 0 and "plain/aa__jpeg75" in text and "plain/aa  " not in text and "--degrade jpeg75" in text, text
            assert "--reals" in text and "_layouts/shared" in text and not (tmp / "mx").exists() and not calls(tmp)
            code, text = run_matrix(tmp, ["--dry-run", "--include-reused", "--benchmarks", "plain", "--detectors", "aa"])
            assert "reused" in text and "todo" not in text, text
            # data root unknown -> UNRESOLVED in a dry run
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                matrix.main(["--spec", str(tmp / "spec.json"), "--dry-run", "--data-root", "", "--benchmarks", "plain"])
            assert "UNRESOLVED" in out.getvalue()

            # a real run, cc fails
            os.environ["FAKE_FAIL_MODEL"] = "cc"
            code, text = run_matrix(tmp, ["--benchmarks", "plain,shared", "--detectors", "aa,bb,cc", "--bs", "7"])
            assert code == 1, code
            mx = tmp / "mx"
            done = sorted(str(p.relative_to(mx)) for p in mx.glob("*/*.json"))
            assert done == ["plain/bb.json", "shared/aa.json", "shared/bb.json"], done          # aa/plain reused, cc failed
            assert not list(mx.glob("*/*.partial"))
            failed = [json.loads(line) for line in (mx / "FAILED.jsonl").read_text().splitlines()]
            assert sorted(f["cell"] for f in failed) == ["plain/cc", "shared/cc"], failed
            assert "boom" in failed[0]["error"] and failed[0]["rc"] == 3 and failed[0]["tail"] and failed[0]["cmd"]
            log = [json.loads(line) for line in (mx / "MATRIX_RESULTS.jsonl").read_text().splitlines()]
            assert sorted(entry["tag"] for entry in log) == ["mx_aa_shared", "mx_bb_plain", "mx_bb_shared"]
            seen = calls(tmp)
            assert len(seen) == 5, len(seen)                                                       # 3 ok + 2 failing
            first = next(c for c in seen if "mx_bb_plain" in c)
            assert first[first.index("--out") + 1].endswith("plain/bb.json.partial") and "--bs" in first and first[first.index("--bs") + 1] == "7"
            assert first[first.index("--results-log") + 1] == str(mx / "MATRIX_RESULTS.jsonl")
            assert first[first.index("--benchmark") + 1] == "pergen" and first[first.index("--data") + 1] == str(tmp / "data/Plain/test")
            assert "--degrade" not in first and "--reals" not in first
            shared = next(c for c in seen if "mx_aa_shared" in c)
            assert shared[shared.index("--data") + 1] == str(mx / "_layouts/shared")
            assert shared[shared.index("--reals") + 1] == str(tmp / "data/Pool/0_real")
            assert sorted(p.name for p in (mx / "_layouts/shared").iterdir()) == ["g1", "g2"]
            assert (mx / "_layouts/shared/g1/1_fake").is_symlink()
            data = json.loads((mx / "plain/bb.json").read_text())
            assert data["tag"] == "mx_bb_plain" and data["model"] == "bb"

            # resume: only the failed cells run again; with the failure gone they succeed
            os.environ.pop("FAKE_FAIL_MODEL")
            before = len(calls(tmp))
            code, text = run_matrix(tmp, ["--benchmarks", "plain,shared", "--detectors", "aa,bb,cc"])
            assert code == 0 and len(calls(tmp)) - before == 2 and text.count("--- skip") == 3, text
            assert (mx / "plain/cc.json").is_file() and (mx / "shared/cc.json").is_file()
            # shards of the degraded slice partition the work
            before = len(calls(tmp))
            for i in range(2):
                code, _ = run_matrix(tmp, ["--degrade", "jpeg75", "--shard", f"{i}/2"])
                assert code == 0
            ran = calls(tmp)[before:]
            assert len(ran) == 6 and len({c[c.index("--tag") + 1] for c in ran}) == 6, len(ran)    # 3 detectors x 2 benchmarks
            assert (mx / "plain/aa__jpeg75.json").is_file() and (mx / "shared/cc__jpeg75.json").is_file()
            assert all(c[c.index("--degrade") + 1] == "jpeg75" for c in ran)

            # unresolved data folder: logged, run continues, nothing launched
            n = len(calls(tmp))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = matrix.main(["--spec", str(tmp / "spec.json"), "--out-dir", str(tmp / "mx2"), "--benchmarks", "plain",
                                    "--detectors", "bb", "--data-root", "", "--python", sys.executable,
                                    "--bench-eval", str(tmp / "fake_bench_eval.py")])
            assert code == 1 and len(calls(tmp)) == n
            assert "no data folder" in json.loads((tmp / "mx2/FAILED.jsonl").read_text().splitlines()[0])["error"]
            # --data-map beats the data root
            os.environ["FAKE_NGEN"] = "2"
            alt = tmp / "alt"
            alt.mkdir()
            code, _ = run_matrix(tmp, ["--benchmarks", "plain", "--detectors", "aa", "--degrade", "blur2", "--data-map", f"plain={alt}"])
            last = calls(tmp)[-1]
            assert code == 0 and last[last.index("--data") + 1] == str(alt)
        finally:
            for k in ("FAKE_CALLS", "FAKE_FAIL_MODEL", "FAKE_NGEN"):
                os.environ.pop(k, None)


def _append_many(args: tuple[str, int]) -> None:
    path, worker = args
    for i in range(60):
        matrix.append_line(Path(path), {"worker": worker, "i": i, "pad": "x" * 300})


def check_concurrent_append() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "shared.jsonl"
        with multiprocessing.Pool(8) as pool:
            pool.map(_append_many, [(str(path), w) for w in range(8)])
        lines = path.read_text().splitlines()
        records = [json.loads(line) for line in lines]                       # a torn line would not parse
        assert len(records) == 8 * 60 and len({(r["worker"], r["i"]) for r in records}) == 8 * 60


def check_layout() -> None:
    import bench_eval
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        fakes, pool = tmp / "Synthbuster" / "test", tmp / "pool"
        pool.mkdir()
        for i in range(3):
            Image.new("RGB", (8, 8)).save(pool / f"real{i}.png")
        for g, n in (("dalle2", 2), ("firefly", 4)):
            (fakes / g / "1_fake").mkdir(parents=True)
            (fakes / g / "0_real").mkdir()                                    # the proxy reals the layout must not carry over
            Image.new("RGB", (8, 8)).save(fakes / g / "0_real" / "proxy.png")
            for i in range(n):
                Image.new("RGB", (8, 8)).save(fakes / g / "1_fake" / f"{i}.png")
        layout = tmp / "layout"
        matrix.ensure_layout(fakes, layout)
        matrix.ensure_layout(fakes, layout)                                   # idempotent
        found = bench_eval.discover("pergen", layout, pool, None)
        assert [(g, len(r), len(f)) for g, r, f in found] == [("dalle2", 3, 2), ("firefly", 3, 4)], found
        assert all(p.parent == pool for _, reals, _ in found for p in reals)


def check_table() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        spec = small_spec()
        out_dir = tmp / "mx"
        (out_dir / "plain").mkdir(parents=True)
        (out_dir / "shared").mkdir()
        (tmp / "old").mkdir()
        means = lambda a, b, c: {"means": {"acc": a, "map": b, "mauroc": c}, "n_generators": 2}      # noqa: E731
        (tmp / "old/aa_plain.json").write_text(json.dumps(means(70.0, 80.0, 90.0)))
        (out_dir / "plain/bb.json").write_text(json.dumps(means(60.0, 70.0, 80.0)))
        (out_dir / "plain/bb__jpeg75.json").write_text(json.dumps(means(50.0, 60.0, 65.0)))
        (out_dir / "plain/aa__jpeg75.json").write_text(json.dumps(means(55.0, 75.0, 85.0)))       # clean = the reused cell
        (out_dir / "shared/bb.json").write_text(json.dumps({"means": {"acc": 1, "map": 2, "mauroc": 3}, "n_generators": 9}))
        matrix.append_line(out_dir / "FAILED.jsonl", {"cell": "plain/cc", "detector": "cc", "benchmark": "plain", "degrade": None,
                                                      "error": "bench_eval exited 3: RuntimeError: boom | x"})
        matrix.append_line(out_dir / "FAILED.jsonl", {"cell": "plain/bb", "detector": "bb", "benchmark": "plain", "degrade": None,
                                                      "error": "stale: the cell has a result"})
        text = matrix_table.build(spec, out_dir, tmp)
        rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
                for line in text.split("## mAUROC (%)")[1].split("##")[0].splitlines() if line.startswith("| ") and "Detector" not in line}
        assert rows["Alpha"][:2] == ["90.0" + matrix_table.REUSED, "-"], rows["Alpha"]
        assert rows["Beta " + matrix_table.INTERNAL][0] == "80.0", rows                      # fresh result, no dagger
        assert rows["Beta " + matrix_table.INTERNAL][1] == "3.0" + matrix_table.MISMATCH, rows   # 9 generators, 2 expected
        assert rows["Gamma"] == ["-", "-", "-"] and rows["Alpha"][2] == "90.0 (1)"
        assert "-15.0 (1)" in text.split("### Mean mAUROC change per detector")[1], text       # bb: 65 - 80
        assert "-5.0 (1)" in text.split("### Mean mAUROC change per detector")[1]              # aa: 85 - 90 (reused clean)
        assert "plain/cc" in text.split("## Failed cells")[1] and "plain/bb`" not in text.split("## Failed cells")[1]
        assert "gone" in text.split("## Not run")[1] and "`shared/bb`" in text
        assert "| Plain | 2 | 1 | 1 | 1 |" in text and "| Shared reals | 2 | 1 | 0 | 2 |" in text      # coverage
        out = tmp / "MATRIX.md"
        spec_path = write_spec(tmp)
        assert matrix_table.main(["--spec", str(spec_path), "--out-dir", str(out_dir), "--output", str(out),
                                  "--repo-root", str(tmp)]) == 0 and out.read_text() == text
        assert text == matrix_table.build(spec, out_dir, tmp)                                  # deterministic


if __name__ == "__main__":
    check_committed_spec()
    check_cells_and_shards()
    check_run_skip_fail()
    check_concurrent_append()
    check_layout()
    check_table()
    print("matrix driver / table: all checks passed")
