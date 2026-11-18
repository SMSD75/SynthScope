"""Checks of the VLM benchmark campaign helpers (repro/vlm_bench; CPU only, no model weights).

* run_units ``jpegfakes<Q>``: the unit maps to ``bench_eval --jpeg-fakes Q`` (no ``--degrade``) and its own result file; and
  ``bench_eval.score_vlm`` with ``jpeg_q`` changes ONLY the fakes (indices >= ``jpeg_from``): every real reaches the scorer
  exactly as the model's loader returns it, every fake as the loader output after a JPEG q-Q round trip;
* compare.py with two VLM columns: best / best public / median over the comparators only, ``rank_*`` among the comparators plus
  the VLM itself, ``rank_all_*`` also counting the other VLM, and ``subset_means`` over the clean rows only;
* compare.py in-distribution flag: fakevlm's genimage_val100k row is marked ``in_distribution`` (another VLM's is not), and
  its subset mean is also given without that subset (``held_out``), while another VLM's mean has no such field;
* trace_check: a per-image log line belongs to the model of its unit's START line, so two VLMs that scored the same image of
  the same benchmark are checked against their own lines (and a wrong one is reported);
* compare.py ``n_unscored``: a parsed verdict without a verdict token, under the old label ("unparsed") or the new one;
* migrate_vlm_cache: a cache folder of a known older adb_vlm identity is carried into the current identity's folder without
  its parsed-but-unscored records, the result is what ``bench_eval.VLMCache`` reads there, a second run does nothing, an
  unknown identity is left alone, and a kept record that the verdict-offset hook would score differently stops it;
* migrate_vlm_cache, rule ``unchanged`` (main e5749ae -> detector/fakevlm): the e5749ae identities of adb_vlm and veritaspp
  hash to the folder names e5749ae wrote, differ from the current ones only in the code version, and the current code
  version is one the entries name (a later ``models/vlm`` edit fails here until it is checked and added); every record is
  carried with its line byte-equal (unscored ones too), bench_eval opens the new folder with the same records, an
  already-carried cf179eee30 source is skipped in favour of its e5749ae folder, and a target code version the entry does
  not name leaves the folder alone.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

import bench_eval                                                    # noqa: E402
from repro.vlm_bench import compare, migrate_vlm_cache, run_units, trace_check   # noqa: E402


def check_jpegfakes_unit() -> None:
    spec = run_units.load_spec()
    args = argparse.Namespace(vlm_model="adb_vlm", out_root="/o", subsets_root="/s", full_data="x", python="py", bs=32,
                              seed=0, cache_root="/c")
    cmd, out = run_units.command("vlm:aigcdetect17:jpegfakes96", spec, args)
    assert out == Path("/o/aigcdetect17/adb_vlm__jpegfakes96.json"), out
    assert cmd[cmd.index("--jpeg-fakes") + 1] == "96" and "--degrade" not in cmd, cmd
    assert cmd[cmd.index("--model") + 1] == "adb_vlm" and "--vlm-cache" in cmd
    cmd, out = run_units.command("vlm:ditfake:jpeg95", spec, args)
    assert cmd[cmd.index("--degrade") + 1] == "jpeg95" and "--jpeg-fakes" not in cmd and out.name == "adb_vlm__jpeg95.json"
    try:
        run_units.command("vlm:ditfake:jpegfakes", spec, args)
    except ValueError:
        pass
    else:
        raise AssertionError("a jpegfakes variant without a quality must be refused")


def check_jpegfakes_only_fakes(tmp: Path) -> None:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):                          # 0, 1 reals; 2, 3 fakes (noise PNGs: a JPEG round trip changes them)
        p = tmp / f"img{i}.png"
        Image.fromarray(rng.integers(0, 256, (48, 40, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    seen = []

    def vlm_score(spec, image, decoding="greedy", seed=None):
        seen.append(np.asarray(image.convert("RGB")).copy())
        return {"logit": 1.0, "verdict": "fake", "score_source": "stub", "new_tokens": 1, "finished": True,
                "verdict_token": "x", "input_size": image.size, "response": ""}

    det = types.SimpleNamespace(_engine=types.SimpleNamespace(vlm_score=vlm_score))
    spec = types.SimpleNamespace(vlm=types.SimpleNamespace(image_loader=lambda im: im.convert("RGB")))
    with contextlib.redirect_stdout(io.StringIO()):
        bench_eval.score_vlm(det, spec, paths, [p.name for p in paths], [0, 0, 1, 1], bench_eval.VLMCache(None, "t"),
                             decoding="greedy", seed=None, jpeg_q=96, jpeg_from=2)
    assert len(seen) == 4
    for i, p in enumerate(paths):
        rgb = Image.open(p).convert("RGB")
        if i < 2:
            assert np.array_equal(seen[i], np.asarray(rgb)), f"real {i} was changed"
        else:
            assert not np.array_equal(seen[i], np.asarray(rgb)), f"fake {i} was not JPEG-compressed"
            assert np.array_equal(seen[i], np.asarray(bench_eval._jpeg(rgb, 96))), f"fake {i} is not the q96 round trip"


def result(model: str, acc: float, mauroc: float, vlm: bool) -> dict:
    r = {"tag": f"t_{model}", "model": model, "means": {"acc": acc, "mauroc": mauroc, "map": mauroc},
         "per_generator": {"g": {"real_acc": 100.0, "fake_acc": 2 * acc - 100, "acc": acc}}, "limit": 5}
    if vlm:
        r["vlm"] = {"acc_pooled": acc, "acc_pooled_drop": acc, "n_images": 10, "n_unparsed": 1, "n_unfinished": 0,
                    "mean_tokens": 90.0, "mean_seconds_scored": 4.0}
        # image 0: a parsed verdict without a verdict token, under the label of the result JSONs written before "unscored"
        r["per_image"] = [{"key": f"g/{i}", "label": i % 2, "verdict": "fake" if i % 2 else "real", "logit": 1.0,
                           "score_source": "unparsed" if i == 0 else "verdict_token", "new_tokens": 90} for i in range(10)]
    return r


def check_compare_ranks(tmp: Path) -> None:
    res = tmp / "results"
    accs = {"b1": {"va": 80.0, "vb": 70.0, "c1": 90.0, "c2": 75.0, "c3": 60.0},
            "b2": {"va": 50.0, "vb": 95.0, "c1": 90.0, "c2": 75.0, "c3": 60.0}}
    for bench, row in accs.items():
        (res / bench).mkdir(parents=True)
        for model, acc in row.items():
            (res / bench / f"{model}.json").write_text(json.dumps(result(model, acc, acc + 1, model.startswith("v"))))
        (res / bench / "va__jpeg95.json").write_text(json.dumps(result("va", 10.0, 11.0, True)))
    rows = [compare.subset_row(b, "clean", ("va", "vb"), ["c1", "c2", "c3"], res, {"c1"}) for b in ("b1", "b2")]
    rows.append(compare.subset_row("b1", "jpeg95", ("va", "vb"), ["c1", "c2", "c3"], res, {"c1"}))
    r1, r2, deg = rows
    assert r1["best_acc"]["detector"] == "c1" and r1["best_public_acc"] == {"detector": "c2", "value": 75.0, "tied": ["c2"]}
    assert r1["median_acc"] == 75.0                          # over the comparators only, not the VLMs
    va, vb = r1["vlms"]["va"], r1["vlms"]["vb"]
    assert (va["rank_acc"], va["rank_all_acc"]) == (2, 2)    # c1 above; vb (70) below
    assert (vb["rank_acc"], vb["rank_all_acc"]) == (3, 4)    # c1, c2 above; va also above
    assert (r2["vlms"]["vb"]["rank_acc"], r2["vlms"]["va"]["rank_all_acc"]) == (1, 5)   # va: 3 comparators + vb above
    assert va["rank_mauroc"] == 2 and va["n_unparsed"] == 1 and va["n_unscored"] == 1 and va["limit"] == 5
    assert compare.n_unscored([{"verdict": "fake", "score_source": "unscored"}, {"verdict": None, "score_source": "unparsed"},
                               {"verdict": "real", "score_source": "verdict_word"}]) == 1
    assert set(deg["vlms"]) == {"va"} and deg["n_comparators"] == 0   # jpeg95 row: only va has a result
    means = compare.subset_means(rows, ("va", "vb", "vc"))
    assert means == {"va": {"n_subsets": 2, "acc": 65.0, "mauroc": 66.0, "n_images": 20, "n_unparsed": 2, "n_unscored": 2},
                     "vb": {"n_subsets": 2, "acc": 82.5, "mauroc": 83.5, "n_images": 20, "n_unparsed": 2,
                            "n_unscored": 2}}, means


def check_compare_in_distribution(tmp: Path) -> None:
    res = tmp / "results_ind"
    for bench, accs in {"genimage_val100k": {"fakevlm": 99.0, "va": 60.0, "c1": 70.0},
                        "wildrf": {"fakevlm": 50.0, "va": 80.0, "c1": 70.0}}.items():
        (res / bench).mkdir(parents=True)
        for model, acc in accs.items():
            (res / bench / f"{model}.json").write_text(json.dumps(result(model, acc, acc, model != "c1")))
    rows = [compare.subset_row(b, "clean", ("va", "fakevlm"), ["c1"], res) for b in ("genimage_val100k", "wildrf")]
    gi = rows[0]["vlms"]
    assert gi["fakevlm"]["in_distribution"] is True and "FakeClue" in gi["fakevlm"]["in_distribution_note"]
    assert "in_distribution" not in gi["va"] and "in_distribution" not in rows[1]["vlms"]["fakevlm"]
    means = compare.subset_means(rows, ("va", "fakevlm"))
    assert means["fakevlm"]["acc"] == 74.5 and means["fakevlm"]["in_distribution_subsets"] == ["genimage_val100k"]
    assert means["fakevlm"]["held_out"] == {"n_subsets": 1, "acc": 50.0, "mauroc": 50.0}, means
    assert "held_out" not in means["va"] and "in_distribution_subsets" not in means["va"]


def check_trace_model_keying(tmp: Path) -> None:
    res, logs = tmp / "tres", tmp / "tlogs"
    (res / "b").mkdir(parents=True)
    logs.mkdir()
    for model, verdict, logit in (("va", "fake", 2.5), ("vb", "real", -3.25)):
        r = {"tag": f"vb_b_{model}", "model": model, "means": {"map": 1.0, "mauroc": 2.0, "acc": 3.0},
             "per_image": [{"key": "g/1_fake/x.png", "verdict": verdict, "logit": logit, "new_tokens": 7}]}
        (res / "b" / f"{model}.json").write_text(json.dumps(r))
        (logs / f"adb_{model}.out").write_text(
            f"=== unit vlm:b START 2026-01-01 00:00:00 on h: py bench_eval.py --model {model} --benchmark pergen\n"
            f"[vlm:b]     vlm g/1_fake/x.png label=1 verdict={verdict} logit={logit:.4f} src=verdict_token tok=7 1.0s\n"
            f"[vlm:b] === RESULT vb_b_{model}: mAP=1.0 mAUROC=2.0 acc=3.0 over 1 generators -> x ===\n")
    run = ["--logs", str(logs), "--glob", "adb_*.out", "--results", str(res)]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert trace_check.main(run) == 0, out.getvalue()
    assert "2 result JSONs trace" in out.getvalue()
    with contextlib.redirect_stdout(io.StringIO()):
        assert trace_check.main(run + ["--models", "vb"]) == 0
    bad = json.loads((res / "b" / "vb.json").read_text())
    bad["per_image"][0]["verdict"] = "fake"                  # va's verdict: must not be matched against va's log line
    (res / "b" / "vb.json").write_text(json.dumps(bad))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert trace_check.main(run) == 1
    assert "MISMATCH b/vb.json" in out.getvalue() and "1 do not" in out.getvalue(), out.getvalue()


def check_migrate_cache(tmp: Path) -> None:
    old_id = migrate_vlm_cache.PREDECESSORS["adb_vlm"][0]["identity"]
    root = tmp / "cache"
    old = root / "b" / f"adb_vlm__greedy__jpegfakes96__{migrate_vlm_cache.identity_hash(old_id)}"
    old.mkdir(parents=True)
    (old / "identity.json").write_text(json.dumps(old_id))
    tagged = "<rationale>\nx\n</rationale>\n<answer>\nThis is an AI-generated image.\n</answer>"
    tagless = "The AI hint is absent.\n\nThis is an authentic image."
    recs = [{"key": "g/0_real/a.png", "sha1": "a", "verdict": "fake", "score_source": "verdict_token", "logit": 3.5, "response": tagged},
            {"key": "g/0_real/b.png", "sha1": "b", "verdict": "real", "score_source": "unparsed", "logit": 0.0, "response": tagless},
            {"key": "g/1_fake/c.png", "sha1": "c", "verdict": None, "score_source": "unparsed", "logit": 0.0, "response": "x"}]
    (old / "h.1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    unknown = dict(old_id, code_version="0000000000")                   # an identity the script does not know
    other = root / "b" / f"adb_vlm__greedy__{migrate_vlm_cache.identity_hash(unknown)}"
    other.mkdir()
    (other / "identity.json").write_text(json.dumps(unknown))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        migrate_vlm_cache.main(["--cache-root", str(root), "--model", "adb_vlm"])
    lines = {json.loads(x)["folder"]: json.loads(x) for x in out.getvalue().splitlines()}
    assert lines[str(other)]["status"] == "unknown identity, left alone"
    done = lines[str(old)]
    assert (done["status"], done["records"], done["kept"], done["dropped"]) == ("migrated", 3, 2, 1), done
    new_id = bench_eval.vlm_identity(bench_eval.get_spec("adb_vlm"), "greedy")
    variant = bench_eval.vlm_variant("adb_vlm", "greedy", None, None, 96, new_id)
    assert Path(done["to"]) == root / "b" / variant                       # the folder bench_eval opens for this unit
    with contextlib.redirect_stdout(io.StringIO()):
        cache = bench_eval.VLMCache(root / "b", variant, new_id)
    assert sorted(cache.records) == ["g/0_real/a.png", "g/1_fake/c.png"] and cache.get("g/0_real/b.png", "b") is None
    assert json.loads((root / "b" / variant / "MIGRATION.json").read_text())["dropped_unscored"] == ["g/0_real/b.png"]
    with contextlib.redirect_stdout(io.StringIO()) as out:                 # idempotent
        migrate_vlm_cache.main(["--cache-root", str(root), "--model", "adb_vlm"])
    assert {json.loads(x)["status"] for x in out.getvalue().splitlines()} == {"migrated earlier", "current",
                                                                              "unknown identity, left alone"}
    # a kept (scored) record whose verdict the parser reads from a tag-less line: its score would change -> refused
    bad = root / "c" / old.name
    bad.mkdir(parents=True)
    (bad / "identity.json").write_text(json.dumps(old_id))
    (bad / "h.1.jsonl").write_text(json.dumps(dict(recs[1], score_source="verdict_token")) + "\n")
    try:
        migrate_vlm_cache.migrate_folder(bad, "adb_vlm", dry_run=False)
    except SystemExit as e:
        assert "would be scored at the parser's verdict word" in str(e)
    else:
        raise AssertionError("a record the hook changes was carried over")


def check_migrate_unchanged(tmp: Path) -> None:
    mv = migrate_vlm_cache
    main_hashes = {("adb_vlm", "greedy"): "a8ce235283", ("veritaspp", "greedy"): "63de8d1257",
                   ("veritaspp", "official"): "c17b4d7728"}            # the folder hashes e5749ae's bench_eval wrote
    for (model, decoding), h in main_hashes.items():
        entry = next(e for e in mv.PREDECESSORS[model] if mv.identity_hash(e["identity"]) == h)
        cur = bench_eval.vlm_identity(bench_eval.get_spec(model), decoding)
        assert entry["rule"] == "unchanged" and entry["identity"]["code_version"] == "b490c64477"
        assert {k: v for k, v in cur.items() if k != "code_version"} == \
            {k: v for k, v in entry["identity"].items() if k != "code_version"}, (model, decoding, cur)
        assert cur["code_version"] in entry["to_code_versions"], (
            f"models/vlm/*.py changed (code version {cur['code_version']}): check that {model}'s scoring is unchanged and "
            "add the version to migrate_vlm_cache._FAKEVLM_BRANCH")
    old_id = mv._ADB_E5749AE
    root = tmp / "ucache"
    stem = "adb_vlm__greedy"
    cf = root / "b" / f"{stem}__{mv.identity_hash(mv.PREDECESSORS['adb_vlm'][0]['identity'])}"   # cf179eee30 source
    old = root / "b" / f"{stem}__{mv.identity_hash(old_id)}"
    for d, ident in ((cf, mv.PREDECESSORS["adb_vlm"][0]["identity"]), (old, old_id)):
        d.mkdir(parents=True)
        (d / "identity.json").write_text(json.dumps(ident))
    (cf / "h.1.jsonl").write_text('{"key": "g/x.png", "sha1": "x", "verdict": "real", "score_source": "verdict_token"}\n')
    (old / "MIGRATION.json").write_text(json.dumps({"from": str(cf)}))
    recs = [{"key": "g/0_real/a.png", "sha1": "a", "verdict": "fake", "score_source": "verdict_token", "logit": 0.1 + 0.2,
             "response": "<answer>\nThis is an AI-generated image.\n</answer> \u00e9"},
            {"key": "g/0_real/b.png", "sha1": "b", "verdict": "real", "score_source": "unscored", "logit": 0.0,
             "response": "This is an authentic image."},
            {"key": "g/1_fake/c.png", "sha1": "c", "verdict": None, "score_source": "unparsed", "logit": 0.0, "response": "x"}]
    stale = dict(recs[0], logit=-1.0)        # an earlier line of a.png (files read in sorted order): the later one wins
    (old / f"migrated.{mv.identity_hash(mv.PREDECESSORS['adb_vlm'][0]['identity'])}.jsonl").write_text(
        json.dumps(stale) + "\n" + json.dumps(recs[1]) + "\n")
    (old / "node9.2.jsonl").write_text(json.dumps(recs[0]) + "\n" + json.dumps(recs[2]) + "\n" + '{"key": "g/cut')
    with contextlib.redirect_stdout(io.StringIO()) as out:
        mv.main(["--cache-root", str(root), "--model", "adb_vlm"])
    lines = {json.loads(x)["folder"]: json.loads(x) for x in out.getvalue().splitlines()}
    assert lines[str(cf)]["status"].startswith("carried earlier into"), lines[str(cf)]
    done = lines[str(old)]
    assert (done["status"], done["records"], done["kept"], done["dropped"]) == ("migrated", 3, 3, 0), done
    new_id = bench_eval.vlm_identity(bench_eval.get_spec("adb_vlm"), "greedy")
    variant = bench_eval.vlm_variant("adb_vlm", "greedy", None, None, None, new_id)
    assert Path(done["to"]) == root / "b" / variant
    src = {json.loads(x)["key"]: x for f in sorted(old.glob("*.jsonl")) for x in f.read_text().splitlines()
           if not x.startswith('{"key": "g/cut')}
    got = [x for f in (root / "b" / variant).glob("*.jsonl") for x in f.read_text().splitlines()]
    assert sorted(got) == sorted(src.values()) and len(got) == 3            # byte-equal lines, the later a.png line
    with contextlib.redirect_stdout(io.StringIO()):
        cache = bench_eval.VLMCache(root / "b", variant, new_id)
    assert cache.records == {r["key"]: r for r in recs}
    assert json.loads((root / "b" / variant / "MIGRATION.json").read_text())["rule"] == "unchanged"
    # a target code version the entry does not name: left alone
    saved = mv._FAKEVLM_BRANCH[:]
    try:
        mv._FAKEVLM_BRANCH[:] = ["0000000000"]
        other = tmp / "ucache2" / "b" / old.name
        other.mkdir(parents=True)
        (other / "identity.json").write_text(json.dumps(old_id))
        res = mv.migrate_folder(other, "adb_vlm", dry_run=False)
        assert res["status"].startswith("no checked re-key"), res
    finally:
        mv._FAKEVLM_BRANCH[:] = saved


def main() -> None:
    check_jpegfakes_unit()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        check_jpegfakes_only_fakes(tmp)
        check_compare_ranks(tmp)
        check_compare_in_distribution(tmp)
        check_trace_model_keying(tmp)
        check_migrate_cache(tmp)
        check_migrate_unchanged(tmp)
    print("vlm_bench units: all checks passed")


def test_vlm_bench_units() -> None:          # pytest entry point; the file also runs as a script
    main()


if __name__ == "__main__":
    main()
