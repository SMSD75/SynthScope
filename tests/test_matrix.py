"""Checks the cross-evaluation matrix driver (``benchmark/matrix.py``) and table (``benchmark/matrix_table.py``); CPU only, no
weights: the units run a stand-in for ``bench_eval.py`` over tiny fake benchmark folders.

* the committed ``repro/matrix/spec.json``: detectors == the registry's image detectors minus the excluded ones, native_res flags agree
  with the registry, the slow detectors / AIGIBench / Co-Spy-Bench are split, reuse files exist and agree; the three data maps,
* cells and units: deterministic, benchmark-major cells, reused cells left out, units largest cost first, chunking (time target, big
  generator alone, at most ``max_chunk_generators``, chunk names change with the listing), bad arguments rejected,
* merging: the merged cell of two ``--select`` runs of the real ``bench_eval.main`` (stub detector, 4 generators) has exactly the numbers
  of the whole run; parts that do not fit are rejected,
* running with the stand-in: exact flags, results and failure/results logs are per writer (no shared jsonl), chunks land under
  ``_parts`` and merge when the last one is there, resume skips, failures are logged (also a wrong generator count, which is not
  kept) and retried, an exception in one unit does not stop the others, the circuit breaker aborts; unresolved benchmarks are reported once and skipped (not failures),
  a lost claim does not overwrite a result, SIGTERM stops the child and releases the claim,
* claims: O_EXCL exclusivity, stale takeover, heartbeat; three real worker processes on one folder run every unit exactly once;
  static shards partition the units,
* the Synthbuster-style layout (hashed per data root) is found by ``bench_eval.discover``,
* the table: fresh beats reused, per-writer failure files (latest record wins), unmerged cells, robustness marks.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

import matrix                                                         # noqa: E402
import matrix_table                                                   # noqa: E402
from matrix import Cell                                               # noqa: E402

FAKE_BENCH_EVAL = textwrap.dedent('''
    import datetime, hashlib, json, os, sys, time
    from pathlib import Path
    a = sys.argv[1:]
    get = lambda k: a[a.index(k) + 1] if k in a else None
    with open(os.environ["FAKE_CALLS"], "a") as f:
        f.write(json.dumps(a) + "\\n")
    if os.environ.get("FAKE_PIDS"):
        with open(os.environ["FAKE_PIDS"], "a") as f:
            f.write(str(os.getpid()) + "\\n")
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
    model = get("--model")
    if model == os.environ.get("FAKE_FAIL_MODEL"):
        print("Traceback (most recent call last):\\nRuntimeError: boom")
        sys.exit(3)
    gens = get("--select").split(",") if get("--select") else sorted(p.name for p in Path(get("--data")).iterdir() if p.is_dir())
    if model == os.environ.get("FAKE_BAD_COUNT_MODEL"):
        gens = gens[:1]
    def metrics(g):
        h = int(hashlib.sha1((model + g + str(get("--degrade"))).encode()).hexdigest()[:6], 16)
        v = [round(40 + (h >> s) % 5000 / 100, 2) for s in (0, 3, 6)]
        return {"ap": v[0], "auroc": v[1], "real_acc": v[2], "fake_acc": v[0], "acc": round((v[2] + v[0]) / 2, 2)}
    per = {g: metrics(g) for g in gens}
    mean = lambda k: round(sum(m[k] for m in per.values()) / len(per), 2)
    result = {"tag": get("--tag"), "model": model, "ckpt": "registry-default", "model_config": None, "benchmark": get("--benchmark"),
              "data_root": get("--data"), "jpeg_fakes": None, "n_generators": len(per),
              "means": {"acc": mean("acc"), "map": mean("ap"), "mauroc": mean("auroc")}, "per_generator": per,
              "date": datetime.datetime.now().isoformat(timespec="seconds"), "synthscope_version": "fake"}
    if get("--degrade"):
        result["degrade"] = get("--degrade")
    open(get("--out"), "w").write(json.dumps(result))
    with open(get("--results-log"), "a") as f:
        f.write(json.dumps({"tag": get("--tag")}) + "\\n")
''')

# plain: 4 generators of 4 images; with bb at 1 image/s a generator costs 4 s, so 9 s chunks hold two of them
CHUNK_HOURS = "0.0025"


def small_spec() -> dict:
    return {
        "version": 1, "degradations": ["jpeg75", "blur2"], "chunk_hours": 2.0,
        "excluded_detectors": {"gone": "no weights"}, "excluded_benchmarks": {"nope": "not held"},
        "detectors": [{"key": "aa", "name": "Alpha", "imgs_per_s": 10}, {"key": "bb", "name": "Beta", "imgs_per_s": 1, "split": True,
                                                                         "internal_only": True},
                      {"key": "cc", "name": "Gamma", "imgs_per_s": 10}],
        "benchmarks": [{"key": "plain", "title": "Plain", "profile": "pergen", "data": "Plain/test", "n_generators": 4, "images": 16},
                       {"key": "shared", "title": "Shared reals", "profile": "pergen", "data": "Fakes/test", "n_generators": 2,
                        "images": 10, "layout": "fakes_with_shared_reals", "reals": "Pool/0_real"}],
        "reuse": {"aa": {"plain": "old/aa_plain.json"}},
    }


def make_data(data: Path) -> None:
    """Plain/test: g1..g4 with 2 reals + 2 fakes; Fakes/test: g1, g2 with 2 fakes (+ proxy reals); Pool/0_real: 3 shared reals."""
    for g in ("g1", "g2", "g3", "g4"):
        for kind in ("0_real", "1_fake"):
            (data / "Plain/test" / g / kind).mkdir(parents=True)
            for i in range(2):
                (data / "Plain/test" / g / kind / f"{i}.png").write_bytes(b"x")
    for g in ("g1", "g2"):
        for kind in ("0_real", "1_fake"):
            (data / "Fakes/test" / g / kind).mkdir(parents=True)
            for i in range(2):
                (data / "Fakes/test" / g / kind / f"{i}.png").write_bytes(b"x")
    (data / "Pool/0_real").mkdir(parents=True)
    for i in range(3):
        (data / "Pool/0_real" / f"{i}.png").write_bytes(b"x")


def fake_listing(profile, root, reals):
    return [("g1", 4), ("g2", 4), ("g3", 4), ("g4", 4)]


# ---- spec, cells, units -------------------------------------------------------------------------------------------------------

def check_committed_spec() -> None:
    from interface.backend.registry import MODELS
    spec = matrix.load_spec(matrix.DEFAULT_SPEC)
    image = {k for k, s in MODELS.items() if s.modality == "image"}
    keys = [d["key"] for d in spec["detectors"]]
    assert set(keys) == image - set(spec["excluded_detectors"]), (set(keys) ^ (image - set(spec["excluded_detectors"])))
    assert set(spec["excluded_detectors"]) == {"vib_net_official", "effort_chameleon"}
    assert len(spec["benchmarks"]) == 15 and len(keys) == 27   # 25 + probe, probe_resnet (merged 2026-11-09)
    assert {d["key"] for d in spec["detectors"] if d.get("native_res")} == {k for k in keys if MODELS[k].native_res}    # the registry's flag
    assert {d["key"] for d in spec["detectors"] if d.get("split")} == {"spai", "aide", "ltf", "omniaid", "omniaid_v1", "omniaid_genimage",
                                                                       "mirror", "bfree", "probe", "probe_resnet"}
    assert {b["key"] for b in spec["benchmarks"] if b.get("split")} == {"aigibench25", "cospybench"}
    assert all(d["imgs_per_s"] > 0 for d in spec["detectors"]) and spec["chunk_hours"] > 0
    for det, per_bench in spec["reuse"].items():
        for bench, rel in per_bench.items():
            data = json.loads((ROOT / rel).read_text())
            want = next(b["n_generators"] for b in spec["benchmarks"] if b["key"] == bench)
            assert data["model"] == det and data["n_generators"] == want, (det, bench, rel)
    for b in spec["benchmarks"]:
        assert b["profile"] in ("pergen", "forensynths", "ufd", "pool") and b["data"], b
    assert sum(len(v) for v in spec["reuse"].values()) == 149   # 143 + probe/probe_resnet x 3 wild/Synthbuster runs
    cells = matrix.build_cells(spec, keys, [b["key"] for b in spec["benchmarks"]], [None])
    assert len(cells) == 27 * 15 - 149, len(cells)
    # planning the whole matrix against empty benchmark folders with a fake listing
    listings = {"CoSpyBench": [(f"gen{i:03d}", 1000) for i in range(110)], "AIGIBench": [(f"s{i:02d}", 5000) for i in range(25)]}
    with tempfile.TemporaryDirectory() as tmp_:
        data = Path(tmp_) / "data"
        for b in spec["benchmarks"]:
            (data / b["data"]).mkdir(parents=True)
        (data / next(b for b in spec["benchmarks"] if b.get("reals"))["reals"]).mkdir(parents=True)
        units, chunked = matrix.plan_units(spec, cells, {}, str(data), Path(tmp_) / "lay", spec["chunk_hours"] * 3600,
                                           list_fn=lambda profile, root, reals: listings.get(root.parent.name, [("g", 100)]))
        assert len(units) >= len(cells) and not [u for u in units if u.error]
        costs = [u.cost for u in units]
        assert costs == sorted(costs, reverse=True)
        cospy = [u for u in units if u.cell.benchmark == "cospybench"]
        assert len(cospy) == 27 * 11 and max(len(u.generators) for u in cospy) == 10
        assert all(len(u.generators) <= 10 for u in cospy)
        spai = [u for u in units if u.cell == Cell("spai", "aigibench25")]
        clip = [u for u in units if u.cell == Cell("clip", "aigibench25")]
        assert len(spai) >= 5 and all(u.cost <= 2 * 3600 for u in spai) and sum(u.cost for u in spai) == 25 * 5000 / 3.3
        assert len(clip) == 1 and clip[0].chunk is None and clip[0].n_expected == 25      # 1.6 h estimated: not cut
        assert len({u.key for u in units}) == len(units)


def check_units_and_chunks() -> None:
    spec = small_spec()
    cells = matrix.build_cells(spec, ["aa", "bb", "cc"], ["plain", "shared"], [None, "jpeg75"])
    assert cells == matrix.build_cells(spec, ["aa", "bb", "cc"], ["plain", "shared"], [None, "jpeg75"])    # deterministic
    names = [c.name for c in cells]
    assert names[:3] == ["plain/aa__jpeg75", "plain/bb", "plain/bb__jpeg75"], names      # aa/plain clean is reused: left out
    assert len(cells) == 11 and len(matrix.build_cells(spec, ["aa", "bb", "cc"], ["plain", "shared"], [None, "jpeg75"], True)) == 12
    c = Cell("aa", "plain", "jpeg75")
    assert (c.tag, c.name, str(c.json_path(Path("/o")))) == ("mx_aa_plain_jpeg75", "plain/aa__jpeg75", "/o/plain/aa__jpeg75.json")
    assert str(c.parts_dir(Path("/o"))) == "/o/plain/_parts/aa__jpeg75"
    assert matrix.split_list("cc,aa", ["aa", "bb", "cc"], "detector") == ["aa", "cc"]
    assert matrix.parse_shard("2/5") == (2, 5)
    for bad in ("3/3", "1", "a/b", "-1/2", "0/0", "2/1"):
        try:
            matrix.parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert matrix.parse_degrades("clean,jpeg75,jpeg75", spec) == [None, "jpeg75"]
    assert matrix.parse_degrades("all", spec) == [None, "jpeg75", "blur2"]
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

    # chunking
    listing = [("a", 100), ("b", 100), ("c", 100), ("d", 1000), ("e", 100)]
    chunks = matrix.make_chunks(listing, 1.0, 250)
    assert [[g for g, _ in ch] for ch in chunks] == [["a", "b"], ["c"], ["d"], ["e"]], chunks      # 'd' alone, over the target
    assert [[g for g, _ in ch] for ch in matrix.make_chunks(listing, 1.0, 10000, 2)] == [["a", "b"], ["c", "d"], ["e"]]
    assert [[g for g, _ in ch] for ch in matrix.make_chunks(listing, 1.0, 10000)] == [["a", "b", "c", "d", "e"]]
    assert matrix.chunk_name(1, ["a", "b"]) == matrix.chunk_name(1, ["a", "b"]) != matrix.chunk_name(1, ["a", "c"])

    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        make_data(tmp / "data")
        units, chunked = matrix.plan_units(spec, cells, {}, str(tmp / "data"), tmp / "lay", 9.0, list_fn=fake_listing)
        again, _ = matrix.plan_units(spec, cells, {}, str(tmp / "data"), tmp / "lay", 9.0, list_fn=fake_listing)
        assert units == again                                                               # deterministic
        costs = [u.cost for u in units]
        assert costs == sorted(costs, reverse=True), costs                                  # largest first
        keys = [u.key for u in units]
        assert len(set(keys)) == len(keys)
        bb = [u for u in units if u.cell == Cell("bb", "plain")]
        assert [u.generators for u in bb] == [("g1", "g2"), ("g3", "g4")] and all(u.n_expected == 2 and u.cost == 8.0 for u in bb)
        assert bb[0].key.startswith("plain/bb#c00-") and chunked[Cell("bb", "plain")] == bb
        assert all(u.chunk is None and u.n_expected for u in units if u.cell.detector != "bb")    # only bb is split
        assert [u.n_expected for u in units if u.cell == Cell("cc", "plain")] == [4]
        # a changed chunk target gives other part names, so old parts are never mixed in
        other, _ = matrix.plan_units(spec, cells, {}, str(tmp / "data"), tmp / "lay", 5.0, list_fn=fake_listing)
        assert not {u.chunk for u in other if u.cell == Cell("bb", "plain")} & {u.chunk for u in bb}
        # the shards of the sorted unit list partition it
        for n in (1, 2, 3, 5, 40):
            parts = [matrix.shard_items(units, i, n) for i in range(n)]
            assert sorted(sum(parts, []), key=units.index) == units and sum(len(p) for p in parts) == len(units), n
        # no data root: error units, nothing raised, nothing listed
        units_err, _ = matrix.plan_units(spec, cells, {}, None, tmp / "lay", 9.0, list_fn=lambda *a: 1 / 0)
        assert len(units_err) == len(cells) and all("no data folder" in u.error for u in units_err)
        # listing errors become error units of the split cells only
        units_bad, _ = matrix.plan_units(spec, cells, {}, str(tmp / "data"), tmp / "lay", 9.0, list_fn=lambda *a: 1 / 0)
        assert {u.cell.detector for u in units_bad if u.error} == {"bb"}


def check_data_maps() -> None:
    spec = matrix.load_spec(matrix.DEFAULT_SPEC)
    benchmarks = {b["key"]: b for b in spec["benchmarks"]}
    for site, root in (("das6", "/local/ssalehid/cache/data0"), ("cn023", "/hddstore/ssalehi/SynthScope"),
                       ("cn029", "/ssdstore/ssalehi/SynthScope")):
        path = ROOT / "repro/matrix/datamaps" / f"{site}.json"
        raw = json.loads(path.read_text())
        assert "_comment" in raw and "_unverified" in raw, site
        mapping = matrix.parse_data_map(None, str(path))
        assert not any(k.startswith("_") for k in mapping)                                  # comments are dropped
        assert set(mapping) - {"synthbuster_coco.reals"} <= set(benchmarks), site
        for key, value in mapping.items():
            assert value.startswith(root), (site, key, value)
            if key in benchmarks:
                assert value.endswith(benchmarks[key]["data"]), (site, key, value)
        assert ("synthbuster_coco" in mapping) == ("synthbuster_coco.reals" in mapping)
    das = matrix.parse_data_map(None, str(ROOT / "repro/matrix/datamaps/das6.json"))
    assert das["ufd8"].startswith("/local/ssalehid/cache/data023/") and das["wildrf"].startswith("/local/ssalehid/cache/data029/")
    assert set(das) >= set(benchmarks) | {"synthbuster_coco.reals"}


# ---- merging ------------------------------------------------------------------------------------------------------------------

def check_merge_matches_bench_eval() -> None:
    """The real bench_eval.main (stub detector) over 4 generators in one run vs two --select runs merged: identical numbers."""
    import numpy as np
    import test_bench_eval_degrade as stub
    bench_eval = stub.bench_eval
    real = {name: getattr(bench_eval, name) for name in ("synthscope", "RESULTS_LOG")}
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        for g, seed in (("g1", 1), ("g2", 2), ("g3", 3), ("g4", 4)):
            for kind, off in (("0_real", 0), ("1_fake", 50)):
                (tmp / "bench" / g / kind).mkdir(parents=True)
                for i in range(5):
                    stub.synthetic((40 + i, 30), seed=seed * 7 + off + i).save(tmp / "bench" / g / kind / f"{i}.png")
        bench_eval.synthscope = types.SimpleNamespace(load=lambda model, device=None, **kw: stub.StubDetector(), __version__="stub")
        bench_eval.RESULTS_LOG = tmp / "log" / "MASTER.jsonl"

        def run(name: str, *extra: str) -> dict:
            with contextlib.redirect_stdout(io.StringIO()):
                bench_eval.main(["-m", "stub", "--benchmark", "pergen", "--data", str(tmp / "bench"), "--tag", name,
                                 "--out", str(tmp / f"{name}.json")] + list(extra))
            return json.loads((tmp / f"{name}.json").read_text())
        try:
            whole = run("whole")
            p1, p2 = run("p1", "--select", "g1,g2"), run("p2", "--select", "g3,g4")
            p3, p4 = run("p3", "--select", "g1"), run("p4", "--select", "g2,g3,g4")
            assert len(whole["per_generator"]) == 4 and len({m["auroc"] for m in whole["per_generator"].values()}) > 1
        finally:
            for name, value in real.items():
                setattr(bench_eval, name, value)
    for parts in ([p1, p2], [p3, p4], [p2, p1][::-1]):
        merged = matrix.merge_results(parts, ["g1", "g2", "g3", "g4"], "mx_x", ["a", "b"])
        assert merged["means"] == whole["means"], (merged["means"], whole["means"])
        assert merged["per_generator"] == whole["per_generator"] and list(merged["per_generator"]) == list(whole["per_generator"])
        assert merged["n_generators"] == 4 and merged["tag"] == "mx_x" and merged["merged_parts"] == ["a", "b"]
        for key in ("model", "ckpt", "model_config", "benchmark", "jpeg_fakes", "synthscope_version"):
            assert merged[key] == whole[key], key
        assert "degrade" not in merged and np.isclose(merged["means"]["mauroc"], whole["means"]["mauroc"])
    # parts that do not fit are rejected
    for parts, expected, why in (([p1], ["g1", "g2", "g3", "g4"], "missing"), ([p1, p2], ["g1", "g2", "g3"], "extra"),
                                 ([p1, p1], ["g1", "g2"], "duplicate"), ([p1, p2], ["g1", "g1", "g3", "g4"], "dup expected")):
        try:
            matrix.merge_results(parts, expected, "t", [])
        except ValueError:
            continue
        raise AssertionError(why)
    other = dict(p2, ckpt="elsewhere")
    try:
        matrix.merge_results([p1, other], ["g1", "g2", "g3", "g4"], "t", [])
    except ValueError as e:
        assert "ckpt" in str(e)
    else:
        raise AssertionError("parts with different checkpoints merged")
    assert matrix.merge_results([dict(p1, degrade="jpeg75"), dict(p2, degrade="jpeg75")], ["g1", "g2", "g3", "g4"], "t", [])["degrade"] == "jpeg75"


# ---- claims -------------------------------------------------------------------------------------------------------------------

def check_claims() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        claims = Path(tmp_) / "claims"
        first = matrix.try_claim(claims, "u1", 60)
        assert first and first.path.is_file() and first.owns() and not first.took_over
        assert matrix.try_claim(claims, "u1", 60) is None                                           # exclusive
        assert matrix.try_claim(claims, "u2", 60)                                                   # others are free
        old = time.time() - 120
        os.utime(first.path, (old, old))
        taken = matrix.try_claim(claims, "u1", 60)                                                  # stale: taken over
        assert taken and taken.path == first.path and taken.took_over and taken.token != first.token
        assert time.time() - taken.path.stat().st_mtime < 5
        assert taken.owns() and not first.owns()                                                    # ownership is by token
        assert matrix.try_claim(claims, "u1", 60) is None                                           # and live again
        assert not list(claims.glob("*.takeover"))
        # the former owner's heartbeat and exit leave the new owner's claim alone
        recent = time.time() - 10
        os.utime(taken.path, (recent, recent))
        with matrix.Heartbeat(first, 0.05):
            time.sleep(0.3)
            assert abs(taken.path.stat().st_mtime - recent) < 0.01, "a lost claim was touched"
        assert taken.path.is_file() and taken.owns(), "a lost claim was removed"
        # the owner's heartbeat keeps a long unit's claim fresh; the claim is removed on exit
        os.utime(taken.path, (old, old))
        with matrix.Heartbeat(taken, 0.05):
            time.sleep(0.4)
            assert time.time() - taken.path.stat().st_mtime < 5 and matrix.try_claim(claims, "u1", 60) is None
        assert not taken.path.exists()
        # a crashed takeover's old lock is cleared
        (claims / "u3").write_text("{}")
        os.utime(claims / "u3", (old - 100, old - 100))
        (claims / "u3.takeover").write_text("{}")
        os.utime(claims / "u3.takeover", (old - 100, old - 100))
        assert matrix.try_claim(claims, "u3", 60)
        # somebody else's live takeover lock is respected and never removed by us
        (claims / "u4").write_text("{}")
        os.utime(claims / "u4", (old, old))
        (claims / "u4.takeover").write_text(json.dumps({"token": "someone-else"}))
        assert matrix.try_claim(claims, "u4", 60) is None and (claims / "u4.takeover").is_file()
        # sweeping a dead worker's partial files: only the old ones of that result
        result = Path(tmp_) / "mx" / "b" / "aa.json"
        result.parent.mkdir(parents=True)
        for name, age in (("aa.json.partial.h.1", 5000), ("aa.json.partial.h.2", 10), ("bb.json.partial.h.1", 5000)):
            (result.parent / name).write_text("x")
            os.utime(result.parent / name, (time.time() - age, time.time() - age))
        assert matrix.sweep_partials(result, 1800) == 1
        assert sorted(p.name for p in result.parent.iterdir()) == ["aa.json.partial.h.2", "bb.json.partial.h.1"]


# ---- running ------------------------------------------------------------------------------------------------------------------

def run_matrix(tmp: Path, extra: list[str], expect_exit: int | None = None) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = matrix.main(["--spec", str(tmp / "spec.json"), "--out-root", str(tmp / "mx"), "--python", sys.executable,
                            "--bench-eval", str(tmp / "fake_bench_eval.py"), "--data-root", str(tmp / "data"),
                            "--chunk-hours", CHUNK_HOURS, "--heartbeat-seconds", "0.1"] + extra)
    if expect_exit is not None:
        assert code == expect_exit, (code, out.getvalue()[-3000:])
    return code, out.getvalue()


def calls(tmp: Path) -> list[list[str]]:
    path = tmp / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def tags(tmp: Path) -> list[str]:
    return [c[c.index("--tag") + 1] for c in calls(tmp)]


def setup(tmp: Path) -> None:
    (tmp / "spec.json").write_text(json.dumps(small_spec()))
    (tmp / "fake_bench_eval.py").write_text(FAKE_BENCH_EVAL)
    make_data(tmp / "data")
    os.environ["FAKE_CALLS"] = str(tmp / "calls.jsonl")


def failed_records(mx: Path) -> list[dict]:
    return [json.loads(line) for p in sorted(mx.glob("FAILED*.jsonl")) for line in p.read_text().splitlines()]


def check_run_chunks_and_logs() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        setup(tmp)
        for var in ("FAKE_FAIL_MODEL", "FAKE_BAD_COUNT_MODEL", "FAKE_SLEEP"):
            os.environ.pop(var, None)
        mx = tmp / "mx"
        try:
            # dry run: lists units with cost and commands; only the (tiny) layout symlinks are created
            code, text = run_matrix(tmp, ["--dry-run", "--degrade", "clean,jpeg75"], 0)
            assert "plain/bb#c00-" in text and "--select g1,g2" in text and "plain/aa__jpeg75" in text and "plain/aa  " not in text, text
            assert "--reals" in text and "_layouts/shared-" in text and "--degrade jpeg75" in text and "~  0.0 h" in text
            assert not calls(tmp) and sorted(p.name for p in mx.iterdir()) == ["_layouts"], list(mx.iterdir())

            # a run: bb is cut into chunks (merged afterwards), aa/cc are whole cells, cc fails
            os.environ["FAKE_FAIL_MODEL"] = "cc"
            code, text = run_matrix(tmp, ["--detectors", "aa,bb,cc"], 1)
            done = sorted(str(p.relative_to(mx)) for p in mx.glob("*/*.json"))
            assert done == ["plain/bb.json", "shared/aa.json", "shared/bb.json"], done          # aa/plain reused, cc failed
            assert len(list(mx.glob("plain/_parts/bb/*.json"))) == 2 and len(list(mx.glob("shared/_parts/bb/*.json"))) == 2
            assert not list(mx.rglob("*.partial*")) and not list((mx / "_claims").iterdir())     # nothing half-written, claims released
            merged = json.loads((mx / "plain/bb.json").read_text())
            assert merged["n_generators"] == 4 and merged["tag"] == "mx_bb_plain" and len(merged["merged_parts"]) == 2
            part = json.loads(next(mx.glob("plain/_parts/bb/c00-*.json")).read_text())
            assert part["tag"] == "mx_bb_plain__" + next(mx.glob("plain/_parts/bb/c00-*.json")).stem and set(part["per_generator"]) == {"g1", "g2"}
            whole = json.loads((mx / "shared/aa.json").read_text())
            assert whole["n_generators"] == 2 and whole["tag"] == "mx_aa_shared" and "merged_parts" not in whole
            # logs are per writer: no jsonl is shared between processes
            names = sorted(p.name for p in mx.glob("*.jsonl"))
            assert names == [f"FAILED.{matrix.WRITER}.jsonl", f"MATRIX_RESULTS.{matrix.WRITER}.jsonl"], names
            failed = failed_records(mx)
            assert sorted(f["cell"] for f in failed) == ["plain/cc", "shared/cc"] and failed[0]["rc"] == 3 and "boom" in failed[0]["error"]
            assert failed[0]["tail"] and failed[0]["cmd"] and failed[0]["writer"] == matrix.WRITER
            log = [json.loads(line)["tag"] for line in (mx / f"MATRIX_RESULTS.{matrix.WRITER}.jsonl").read_text().splitlines()]
            assert len(log) == 5 and "mx_aa_shared" in log and any(t.startswith("mx_bb_plain__c00-") for t in log), log
            first = next(c for c in calls(tmp) if c[c.index("--tag") + 1].startswith("mx_bb_plain__c00"))
            assert first[first.index("--select") + 1] == "g1,g2" and first[first.index("--bs") + 1] == "32"
            assert first[first.index("--out") + 1].endswith(".json.partial." + matrix.WRITER) and "_parts/bb/c00-" in first[first.index("--out") + 1]
            assert first[first.index("--results-log") + 1] == str(mx / f"MATRIX_RESULTS.{matrix.WRITER}.jsonl")
            shared = next(c for c in calls(tmp) if c[c.index("--tag") + 1] == "mx_aa_shared")
            assert "/_layouts/shared-" in shared[shared.index("--data") + 1] and "--select" not in shared
            assert shared[shared.index("--reals") + 1] == str(tmp / "data/Pool/0_real")
            n_before = len(calls(tmp))

            # resume: only the failed cc cells run again
            os.environ.pop("FAKE_FAIL_MODEL")
            code, text = run_matrix(tmp, ["--detectors", "aa,bb,cc"], 0)
            assert len(calls(tmp)) - n_before == 2, len(calls(tmp)) - n_before
            assert (mx / "plain/cc.json").is_file() and (mx / "shared/cc.json").is_file()
            n_before = len(calls(tmp))
            run_matrix(tmp, ["--detectors", "aa,bb,cc"], 0)
            assert len(calls(tmp)) == n_before                                                  # everything done: nothing runs

            # a degraded slice; a unit with a stale-by-nobody claim is skipped, and run once the claim is stale
            plain_aa = next(u for u in matrix.plan_units(small_spec(), [Cell("aa", "plain", "jpeg75")], {}, str(tmp / "data"),
                                                         tmp / "lay", 9.0)[0])
            claim = matrix.try_claim(mx / "_claims", plain_aa.claim_name, 1800)
            code, text = run_matrix(tmp, ["--detectors", "aa", "--benchmarks", "plain", "--degrade", "jpeg75"], 0)
            assert "claimed by another worker" in text and not (mx / "plain/aa__jpeg75.json").exists()
            old = time.time() - 4000
            os.utime(claim.path, (old, old))
            dead = mx / "plain/aa__jpeg75.json.partial.deadhost.7"          # left by the worker that died, plus a recent one of a live worker
            live = mx / "plain/aa__jpeg75.json.partial.livehost.8"
            for path, age in ((dead, 4000), (live, 5)):
                path.write_text("x")
                os.utime(path, (time.time() - age, time.time() - age))
            code, text = run_matrix(tmp, ["--detectors", "aa", "--benchmarks", "plain", "--degrade", "jpeg75"], 0)
            assert "took over the stale claim" in text and "removed 1 old partial" in text and not dead.exists() and live.exists()
            live.unlink()
            deg = json.loads((mx / "plain/aa__jpeg75.json").read_text())
            assert deg["degrade"] == "jpeg75" and deg["tag"] == "mx_aa_plain_jpeg75"
            last = calls(tmp)[-1]
            assert last[last.index("--degrade") + 1] == "jpeg75" and last[last.index("--tag") + 1] == "mx_aa_plain_jpeg75"
            # --data-map beats the data root; a map file's comment keys are ignored
            alt = tmp / "alt"
            (alt / "g9" / "0_real").mkdir(parents=True)
            (tmp / "map.json").write_text(json.dumps({"_comment": "x", "plain": str(alt)}))
            run_matrix(tmp, ["--detectors", "aa", "--benchmarks", "plain", "--degrade", "blur2", "--data-map-file", str(tmp / "map.json")], 1)
            last = calls(tmp)[-1]
            assert last[last.index("--data") + 1] == str(alt)                                  # (1 generator found, 4 expected: not kept)
            assert not (mx / "plain/aa__blur2.json").exists()
        finally:
            for var in ("FAKE_CALLS", "FAKE_FAIL_MODEL", "FAKE_BAD_COUNT_MODEL", "FAKE_SLEEP"):
                os.environ.pop(var, None)


def check_failures_and_breaker() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        setup(tmp)
        mx = tmp / "mx"
        try:
            # wrong generator count: the result is not kept
            os.environ["FAKE_BAD_COUNT_MODEL"] = "aa"
            run_matrix(tmp, ["--detectors", "aa", "--benchmarks", "plain", "--degrade", "clean,jpeg75"], 1)
            assert not list(mx.glob("plain/*.json")) and not list(mx.rglob("*.partial*"))
            records = failed_records(mx)
            assert len(records) == 1 and "scored 1 generators, expected 4" in records[0]["error"], records
            os.environ.pop("FAKE_BAD_COUNT_MODEL")

            # an exception inside one unit is logged and the next unit still runs
            real_run = matrix.run_command
            state = {"n": 0}

            def flaky(cmd, env):
                state["n"] += 1
                if state["n"] == 1:
                    raise OSError("disk went away")
                return real_run(cmd, env)
            matrix.run_command = flaky
            try:
                run_matrix(tmp, ["--detectors", "cc", "--benchmarks", "plain,shared"], 1)
            finally:
                matrix.run_command = real_run
            assert len(list(mx.glob("*/cc.json"))) == 1                                       # the other cell ran
            assert any("OSError: disk went away" in r["error"] for r in failed_records(mx))

            # the circuit breaker: the same failure twice aborts the worker before the rest runs
            os.environ["FAKE_FAIL_MODEL"] = "cc"
            n_before = len(calls(tmp))
            code, text = run_matrix(tmp, ["--detectors", "cc", "--benchmarks", "plain,shared", "--degrade", "jpeg75,blur2,social,social2",
                                          "--max-same-failures", "2", "--worker"], 2)
            assert "circuit breaker" in text and len(calls(tmp)) - n_before == 2, len(calls(tmp)) - n_before
            assert not list((mx / "_claims").iterdir())                                       # the claim of the failing unit is released

        finally:
            for var in ("FAKE_CALLS", "FAKE_FAIL_MODEL", "FAKE_BAD_COUNT_MODEL"):
                os.environ.pop(var, None)


def check_unresolved_benchmarks() -> None:
    """A worker that has only some of the benchmarks runs those: the rest is reported once per benchmark, is not a failure, and cannot trip
    the circuit breaker (it did: five 'no data folder' failures aborted the worker before its runnable units)."""
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        setup(tmp)
        try:
            (tmp / "map.json").write_text(json.dumps({"_comment": "only plain is here", "plain": str(tmp / "data/Plain/test")}))
            mx = tmp / "mx"
            code, text = run_matrix(tmp, ["--worker", "--degrade", "all", "--detectors", "aa,bb,cc", "--data-root", "",
                                          "--data-map-file", str(tmp / "map.json"), "--max-same-failures", "5"], 0)
            assert "ABORT" not in text and not list(mx.glob("FAILED*.jsonl")), text                      # nothing logged as failed
            lines = [line for line in text.splitlines() if "unresolved benchmark" in line]
            assert len(lines) == 1 and lines[0].startswith("[matrix] unresolved benchmark shared: 9 units skipped: no data folder"), lines
            assert not any("shared" in t for t in tags(tmp))                                             # none of them launched
            assert (mx / "plain/bb.json").is_file() and (mx / "plain/cc__blur2.json").is_file()          # the runnable ones all ran
            assert len(list(mx.glob("plain/*.json"))) == 8                                               # 3 detectors x 3 - the reused aa
            n_before = len(calls(tmp))
            # nothing runnable at all: exit 1 with a hint, still no failure records
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = matrix.main(["--spec", str(tmp / "spec.json"), "--out-root", str(tmp / "mx2"), "--benchmarks", "plain",
                                    "--detectors", "aa,bb", "--data-root", "", "--python", sys.executable,
                                    "--bench-eval", str(tmp / "fake_bench_eval.py")])
            assert code == 1 and len(calls(tmp)) == n_before and "nothing was runnable" in out.getvalue()
            assert not list((tmp / "mx2").glob("FAILED*.jsonl"))
            # the documented scenario with the real spec and a site map whose folders are not here
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = matrix.main(["--worker", "--data-map-file", str(ROOT / "repro/matrix/datamaps/cn029.json"), "--data-root", "",
                                    "--benchmarks", "wildrf,ditfake,cospybench", "--out-root", str(tmp / "mx3")])
            text = out.getvalue()
            assert code == 1 and "ABORT" not in text and not list((tmp / "mx3").glob("FAILED*.jsonl")), text
            assert [line.split(":")[0] for line in text.splitlines() if "unresolved benchmark" in line] == \
                ["[matrix] unresolved benchmark cospybench", "[matrix] unresolved benchmark ditfake", "[matrix] unresolved benchmark wildrf"]
        finally:
            os.environ.pop("FAKE_CALLS", None)


def check_claim_lost_and_signals() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        setup(tmp)
        mx = tmp / "mx"
        try:
            # a worker whose claim was taken over does not overwrite the complete result that is there
            unit = matrix.plan_units(small_spec(), [Cell("cc", "plain")], {}, str(tmp / "data"), tmp / "lay", 9.0)[0][0]
            final = unit.result_path(mx)
            final.parent.mkdir(parents=True)
            final.write_text('{"sentinel": 1}')
            claims = mx / "_claims"
            claims.mkdir()
            (claims / unit.claim_name).write_text(json.dumps({"token": "the-new-owner"}))
            lost = matrix.Claim(claims / unit.claim_name, "my-old-token")
            runner = matrix.Runner(mx, claims, mx / "r.jsonl", mx / "f.jsonl", sys.executable, tmp / "fake_bench_eval.py", 32, None,
                                   dict(os.environ), {})
            with contextlib.redirect_stdout(io.StringIO()) as out:
                assert runner.run_unit(unit, lost) is True
            assert json.loads(final.read_text()) == {"sentinel": 1} and not list(mx.rglob("*.partial*"))
            assert (claims / unit.claim_name).is_file() and len(calls(tmp)) == 1
            # with the claim still ours the result is renamed into place
            final.unlink()
            (claims / unit.claim_name).write_text(json.dumps({"token": "mine"}))
            with contextlib.redirect_stdout(io.StringIO()):
                assert runner.run_unit(unit, matrix.Claim(claims / unit.claim_name, "mine")) is True
            assert json.loads(final.read_text())["n_generators"] == 4

            # SIGTERM to the worker: the bench_eval child is terminated, the claim released, no partial file left
            os.environ["FAKE_SLEEP"] = "60"
            os.environ["FAKE_PIDS"] = str(tmp / "pids")
            mx2 = tmp / "mx2"
            cmd = [sys.executable, str(ROOT / "benchmark/matrix.py"), "--worker", "--spec", str(tmp / "spec.json"), "--out-root", str(mx2),
                   "--python", sys.executable, "--bench-eval", str(tmp / "fake_bench_eval.py"), "--data-root", str(tmp / "data"),
                   "--detectors", "aa", "--benchmarks", "plain", "--degrade", "jpeg75"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            deadline = time.time() + 60
            while not (tmp / "pids").exists() and time.time() < deadline:
                time.sleep(0.1)
            assert (tmp / "pids").exists(), "the stand-in never started"
            child = int((tmp / "pids").read_text().split()[0])
            assert list((mx2 / "_claims").iterdir())                                          # claimed while running
            proc.send_signal(15)
            proc.communicate(timeout=60)
            assert proc.returncode == 128 + 15, proc.returncode
            for _ in range(50):
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("the bench_eval child survived SIGTERM")
            assert not list((mx2 / "_claims").iterdir()) and not list(mx2.rglob("*.partial*")) and not list(mx2.glob("plain/*.json"))
        finally:
            for var in ("FAKE_CALLS", "FAKE_SLEEP", "FAKE_PIDS"):
                os.environ.pop(var, None)


def check_workers_and_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        setup(tmp)
        os.environ["FAKE_SLEEP"] = "0.25"
        try:
            # three worker processes on one folder: every unit exactly once, parts merged, per-writer logs
            cmd = [sys.executable, str(ROOT / "benchmark/matrix.py"), "--worker", "--spec", str(tmp / "spec.json"), "--out-root", str(tmp / "mx"),
                   "--python", sys.executable, "--bench-eval", str(tmp / "fake_bench_eval.py"), "--data-root", str(tmp / "data"),
                   "--chunk-hours", CHUNK_HOURS, "--heartbeat-seconds", "0.1", "--degrade", "clean,jpeg75"]
            procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(3)]
            outputs = [p.communicate(timeout=300)[0] for p in procs]
            assert all(p.returncode == 0 for p in procs), outputs
            seen = tags(tmp)
            units, chunked = matrix.plan_units(small_spec(), matrix.build_cells(small_spec(), ["aa", "bb", "cc"], ["plain", "shared"], [None, "jpeg75"]),
                                               {}, str(tmp / "data"), tmp / "mx" / "_layouts", float(CHUNK_HOURS) * 3600)
            assert sorted(seen) == sorted(u.tag for u in units), (len(seen), len(units))             # each unit ran exactly once
            assert len(set(seen)) == len(seen)
            assert sum("--- skip" in o and "claimed by another worker" in o for o in outputs) >= 1      # they really overlapped
            cells = sorted(str(p.relative_to(tmp / "mx")) for p in (tmp / "mx").glob("*/*.json"))
            assert len(cells) == 11 and "plain/bb__jpeg75.json" in cells and "shared/bb.json" in cells, cells
            assert len(list((tmp / "mx").glob("MATRIX_RESULTS.*.jsonl"))) >= 2                          # one log per worker process
            assert not (tmp / "mx" / "MATRIX_RESULTS.jsonl").exists() and not list((tmp / "mx").rglob("*.partial*"))
            # --merge on parts copied elsewhere merges them, and nothing is run
            mx2 = tmp / "mx2"
            for part in (tmp / "mx").glob("plain/_parts/bb/*.json"):
                (mx2 / "plain/_parts/bb").mkdir(parents=True, exist_ok=True)
                (mx2 / "plain/_parts/bb" / part.name).write_text(part.read_text())
            (mx2 / "plain/_parts/bb" / "c09-abcdef.json").write_text("{}")                      # a part the plan does not name
            n_before = len(calls(tmp))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert matrix.main(["--spec", str(tmp / "spec.json"), "--out-root", str(mx2), "--merge", "--benchmarks", "plain",
                                    "--detectors", "bb", "--data-root", str(tmp / "data"), "--chunk-hours", CHUNK_HOURS]) == 0
            assert len(calls(tmp)) == n_before and json.loads((mx2 / "plain/bb.json").read_text()) == \
                json.loads((tmp / "mx/plain/bb.json").read_text()), out.getvalue()
            assert "WARNING: 1 part files present but names do not match the current plan" in out.getvalue() and "c09-abcdef.json" in out.getvalue()

            # static shards of the sorted unit list: disjoint, complete
            os.environ.pop("FAKE_SLEEP")
            n_before = len(calls(tmp))
            ran = []
            for i in range(3):
                run_matrix(tmp, ["--shard", f"{i}/3", "--degrade", "blur2", "--out-root", str(tmp / "mxs")], 0)
                ran.append(set(tags(tmp)[n_before:]))
                n_before = len(calls(tmp))
            expected = {u.tag for u in matrix.plan_units(small_spec(), matrix.build_cells(small_spec(), ["aa", "bb", "cc"], ["plain", "shared"], ["blur2"]),
                                                         {}, str(tmp / "data"), tmp / "mxs" / "_layouts", float(CHUNK_HOURS) * 3600)[0]}
            assert not (ran[0] & ran[1] or ran[0] & ran[2] or ran[1] & ran[2]) and ran[0] | ran[1] | ran[2] == expected
        finally:
            for var in ("FAKE_CALLS", "FAKE_SLEEP"):
                os.environ.pop(var, None)


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
        # the listing the planner uses sees the same thing
        assert matrix.list_generators("pergen", layout, pool) == [("dalle2", 5), ("firefly", 7)]
        # nodes with other data roots get other layout folders
        sb = {"key": "synthbuster_coco", "data": "Synthbuster/test", "reals": "Pool", "layout": "fakes_with_shared_reals"}
        a = matrix.plan_benchmark(sb, {}, "/local/a", tmp / "lay")
        b = matrix.plan_benchmark(sb, {}, "/local/b", tmp / "lay")
        assert a.layout != b.layout and a.layout == matrix.plan_benchmark(sb, {}, "/local/a", tmp / "lay").layout
        assert a.layout.parent == tmp / "lay" and a.layout.name.startswith("synthbuster_coco-")


# ---- table --------------------------------------------------------------------------------------------------------------------

def check_table() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        spec = small_spec()
        out_dir = tmp / "mx"
        (out_dir / "plain").mkdir(parents=True)
        (out_dir / "shared").mkdir()
        (tmp / "old").mkdir()
        means = lambda a, b, c, n=4: {"means": {"acc": a, "map": b, "mauroc": c}, "n_generators": n}      # noqa: E731
        (tmp / "old/aa_plain.json").write_text(json.dumps(means(70.0, 80.0, 90.0)))
        (out_dir / "plain/bb.json").write_text(json.dumps(means(60.0, 70.0, 80.0)))
        (out_dir / "plain/bb__jpeg75.json").write_text(json.dumps(means(50.0, 60.0, 65.0)))
        (out_dir / "plain/aa__jpeg75.json").write_text(json.dumps(means(55.0, 75.0, 85.0, n=3)))    # clean = reused; 3 of 4 generators
        (out_dir / "shared/bb.json").write_text(json.dumps(means(1, 2, 3, n=9)))                     # 9 generators, 2 expected
        # failures from two writers: the latest record of a unit wins; fixed ones disappear; a chunk with its part present disappears
        matrix.append_line(out_dir / "FAILED.hostA.1.jsonl", {"time": "2026-11-09T10:00:00", "cell": "plain/cc", "detector": "cc",
                                                              "benchmark": "plain", "degrade": None, "chunk": None, "error": "old error"})
        matrix.append_line(out_dir / "FAILED.hostB.2.jsonl", {"time": "2026-11-09T11:00:00", "cell": "plain/cc", "detector": "cc",
                                                              "benchmark": "plain", "degrade": None, "chunk": None,
                                                              "error": "bench_eval exited 3: RuntimeError: boom | x"})
        matrix.append_line(out_dir / "FAILED.hostA.1.jsonl", {"time": "2026-11-09T10:00:00", "cell": "plain/bb", "detector": "bb",
                                                              "benchmark": "plain", "degrade": None, "chunk": None, "error": "stale"})
        matrix.append_line(out_dir / "FAILED.jsonl", {"time": "2026-11-08T10:00:00", "cell": "shared/cc#c01-abc", "detector": "cc",
                                                      "benchmark": "shared", "degrade": None, "chunk": "c01-abc", "error": "chunk gone"})
        (out_dir / "shared/_parts/cc").mkdir(parents=True)
        (out_dir / "shared/_parts/cc/c01-abc.json").write_text(json.dumps(means(1, 1, 1)))
        (out_dir / "shared/_parts/cc/c00-def.json").write_text(json.dumps(means(1, 1, 1)))
        text = matrix_table.build(spec, out_dir, tmp)
        rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
                for line in text.split("## mAUROC (%)")[1].split("##")[0].splitlines() if line.startswith("| ") and "Detector" not in line}
        assert rows["Alpha"][:2] == ["90.0" + matrix_table.REUSED, "-"], rows["Alpha"]
        assert rows["Beta " + matrix_table.INTERNAL][0] == "80.0", rows                      # fresh result, no dagger
        assert rows["Beta " + matrix_table.INTERNAL][1] == "3.0" + matrix_table.MISMATCH, rows   # 9 generators, 2 expected
        assert rows["Gamma"] == ["-", "-", "-"] and rows["Alpha"][2] == "90.0 (1)"
        summary = text.split("### Mean mAUROC change per detector")[1].split("###")[0]
        assert "-5.0" + matrix_table.REUSED + matrix_table.MISMATCH + " (1)" in summary, summary    # aa: reused base, 3 of 4 generators
        assert "-15.0 (1)" in summary                                                             # bb: fresh base, complete
        per_bench = text.split("### jpeg75: mAUROC change per benchmark")[1]
        assert "-5.0" + matrix_table.REUSED + matrix_table.MISMATCH in per_bench and "-15.0 |" in per_bench, per_bench
        assert "`plain/aa__jpeg75`" in text.split("## Generator-count mismatches")[1] and "`shared/bb`" in text
        failed = text.split("## Failed units")[1].split("##")[0]
        assert "boom" in failed and "old error" not in failed and "plain/bb`" not in failed and "chunk gone" not in failed, failed
        unmerged = text.split("## Cells with parts but no merged result")[1].split("##")[0]
        assert "`shared/cc` | 2 |" in unmerged
        assert "gone" in text.split("## Not run")[1]
        assert "| Plain | 4 | 1 | 1 | 1 |" in text and "| Shared reals | 2 | 1 | 0 | 2 |" in text      # coverage
        out = tmp / "MATRIX.md"
        spec_path = tmp / "spec.json"
        spec_path.write_text(json.dumps(spec))
        assert matrix_table.main(["--spec", str(spec_path), "--out-dir", str(out_dir), "--output", str(out),
                                  "--repo-root", str(tmp)]) == 0 and out.read_text() == text
        assert text == matrix_table.build(spec, out_dir, tmp)                                  # deterministic


if __name__ == "__main__":
    check_committed_spec()
    check_units_and_chunks()
    check_data_maps()
    check_merge_matches_bench_eval()
    check_claims()
    check_run_chunks_and_logs()
    check_failures_and_breaker()
    check_unresolved_benchmarks()
    check_claim_lost_and_signals()
    check_workers_and_shards()
    check_layout()
    check_table()
    print("matrix driver / table: all checks passed")
