"""CPU test of repro/video_agent_bench (make_subsets.py, run_units.py) on toy video trees (no model, no GPU).

Checks:
* limit mode (held benchmarks): the plan is bench_eval_video's ``--limit L`` selection (first L sorted reals and fakes
  per generator); a real pool hard-linked into every generator counts once in ``n_distinct`` (bench_eval_video's
  ``pool_key``); a fake-only benchmark with a separate real pool keys its reals ``@reals/<file>``;
* fraction mode (GenBuster): ceil(f * n) of every folder, the real pool once; refused when the generators do not share
  one real pool;
* ``stage`` + ``verify`` round trip: the staged copy mirrors the source paths, gives the same selection (fraction: all of
  it) with sha1-identical files; a changed byte or a missing file fails ``verify``;
* run_units: ``all`` expands to every benchmark's shards then its aggregation; the bench_eval_video arguments carry the
  staged data / reals paths, the limit (held only) and the per-benchmark tag and cache; an aggregation waits for its shards.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAB = ROOT / "repro" / "video_agent_bench"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, VAB / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ms = load("make_subsets")
ru = load("run_units")


def touch(d: Path, names) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(n.encode())
    return [d / n for n in names]


def pergen_tree(root: Path, n_real: int, fakes: dict) -> None:
    pool = touch(root / "_pool", [f"r{i:02d}.mp4" for i in range(n_real)])
    for g, n in fakes.items():
        (root / "test" / g / "0_real").mkdir(parents=True)
        for i, p in enumerate(pool):
            os.link(p, root / "test" / g / "0_real" / f"{i:05d}_{p.name}")
        touch(root / "test" / g / "1_fake", [f"{g}_{i:02d}.mp4" for i in range(n)])



def test_subsets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        pergen_tree(src / "Held", 10, {"gA": 8, "gB": 3})
        pergen_tree(src / "GB", 10, {"id_x": 10, "wild_y": 2})
        touch(src / "EC" / "1_fake" / "m1", [f"e{i}.mp4" for i in range(5)])
        touch(src / "EC" / "1_fake" / "m2", [f"f{i}.mp4" for i in range(5)])
        spec = {"benchmarks": [
            {"key": "held", "title": "h", "profile": "pergen", "data": "Held/test", "n_generators": 2, "limit": 4, "site": "029"},
            {"key": "ec", "title": "e", "profile": "fakeonly", "data": "EC/1_fake", "reals": "Held/test/gA/0_real",
             "n_generators": 2, "limit": 3, "site": "029"},
            {"key": "gb", "title": "g", "profile": "pergen", "data": "GB/test", "n_generators": 2, "fraction": 0.4,
             "site": "023"}]}
        saved = ms.SPEC, ms.SUBSETS
        ms.SPEC, ms.SUBSETS = tmp / "spec.json", tmp / "subsets"
        ms.SPEC.write_text(json.dumps(spec))
        try:
            assert ms.main(["plan", "--site", "029", "--src-root", str(src)]) == 0
            assert ms.main(["plan", "--site", "023", "--src-root", str(src)]) == 0
            held = json.loads((ms.SUBSETS / "held.json").read_text())
            assert [(g["name"], len(g["reals"]), len(g["fakes"])) for g in held["generators"]] == [("gA", 4, 4), ("gB", 4, 3)]
            assert held["generators"][0]["reals"][0] == "gA/0_real/00000_r00.mp4"
            assert held["n_distinct"] == 4 + 4 + 3 and held["n_distinct_real"] == 4 and held["limit"] == 4
            ec = json.loads((ms.SUBSETS / "ec.json").read_text())
            assert ec["generators"][0]["reals"] == ["@reals/00000_r00.mp4", "@reals/00001_r01.mp4", "@reals/00002_r02.mp4"]
            assert ec["n_distinct"] == 3 + 6
            gb = json.loads((ms.SUBSETS / "gb.json").read_text())
            assert [(g["name"], len(g["reals"]), len(g["fakes"])) for g in gb["generators"]] == [("id_x", 4, 4), ("wild_y", 4, 1)]
            assert gb["n_distinct"] == 4 + 5 and gb["fraction"] == 0.4
            # stage + verify
            dest = tmp / "dest"
            for key in ("held", "ec", "gb"):
                assert ms.main(["stage", key, "--src-root", str(src), "--dest", str(dest)]) == 0
                assert ms.main(["verify", key, str(dest / key)]) == 0, key
            assert (dest / "ec" / "Held/test/gA/0_real/00000_r00.mp4").is_file()
            assert sorted(p.name for p in (dest / "gb" / "GB/test/wild_y/1_fake").iterdir()) == ["wild_y_00.mp4"]
            (dest / "gb" / "GB/test/id_x/1_fake/id_x_00.mp4").write_bytes(b"changed")
            assert ms.main(["verify", "gb", str(dest / "gb")]) == 1
            (dest / "held" / "Held/test/gB/1_fake/gB_02.mp4").unlink()
            assert ms.main(["verify", "held", str(dest / "held")]) == 1
            # fraction mode needs one shared real pool
            touch(src / "GB" / "test" / "wild_y" / "0_real", ["extra.mp4"])
            try:
                ms.main(["plan", "--site", "023", "--src-root", str(src)])
                raise AssertionError("expected the shared-pool refusal")
            except SystemExit as e:
                assert "one real pool" in str(e)
        finally:
            ms.SPEC, ms.SUBSETS = saved
    print("make_subsets OK")


def test_run_units() -> None:
    spec = json.loads((VAB / "spec.json").read_text())
    units = ru.expand("all", spec, 4, set())
    keys = [b["key"] for b in spec["benchmarks"]]
    assert units[:5] == [f"wevis:{keys[0]}:{k}of4" for k in range(4)] + [f"wevis:{keys[0]}"]
    assert len(units) == 5 * len(keys)
    assert ru.expand("all", spec, 2, {"genbuster"}) == ["wevis:genbuster:0of2", "wevis:genbuster:1of2", "wevis:genbuster"]

    class A:
        subsets_root, cache_root, out_root, shards = "/S", "/C", "/O", 4
    ec = next(b for b in spec["benchmarks"] if b["key"] == "evalcrafter")
    argv = ru.bench_argv(ec, A)
    assert argv[argv.index("--data") + 1] == "/S/evalcrafter/EvalCrafter/test/1_fake"
    assert argv[argv.index("--reals") + 1] == "/S/evalcrafter/GenVideo/test/Crafter/0_real"
    assert argv[argv.index("--limit") + 1] == "18" and argv[argv.index("--tag") + 1] == "vab_evalcrafter_wevis"
    assert argv[argv.index("--cache-dir") + 1] == "/C/evalcrafter"
    gb = next(b for b in spec["benchmarks"] if b["key"] == "genbuster")
    assert "--limit" not in ru.bench_argv(gb, A) and "--reals" not in ru.bench_argv(gb, A)
    with tempfile.TemporaryDirectory() as tmp:
        A.out_root, A.lock_dir = tmp, tmp

        class NoBev:
            def main(self, argv):
                raise AssertionError("an aggregation must wait for its shards")
        assert ru.run_unit("wevis:genbuster", spec, A, NoBev()) == "waiting"
        calls = []

        class Bev:
            def main(self, argv):
                calls.append(argv)
                if "--out" in argv:
                    Path(argv[argv.index("--out") + 1]).write_text("{}")
                return 0
        for k in range(4):
            assert ru.run_unit(f"wevis:genbuster:{k}of4", spec, A, Bev()) == "done"
            assert calls[-1][calls[-1].index("--shard") + 1] == f"{k}/4"
        assert ru.run_unit("wevis:genbuster:0of4", spec, A, Bev()) == "exists"
        assert ru.run_unit("wevis:genbuster", spec, A, Bev()) == "done" and "--cache-only" in calls[-1]
        assert (Path(tmp) / "genbuster" / "wevis.json").is_file()
    print("run_units OK")


if __name__ == "__main__":
    test_subsets()
    test_run_units()
    print("test_video_agent_bench: all OK")
