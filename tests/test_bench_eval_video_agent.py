"""CPU unit test of benchmark/bench_eval_video.py for an agentic video detector (``predict_kind="video_agent"``,
WeVisForensics), with a mock engine (no weights, no GPU, no video decoding).

Checks:
* cache metadata: the agent's checkpoint is a FOLDER, which is never hashed (``checkpoint_path`` is not even called):
  ``ckpt_sha256`` is the sha256 of the pinned release manifest (models/video/wevis/manifest.json) and ``agent`` carries the
  code / adapter / base revisions, the runtime pins, the audio flag and the verdict pairs; a cache written for another
  manifest is refused;
* accuracy is the agent's verdict: ``unparsed`` (no valid answer) is wrong for BOTH classes; AP / AUROC rank by the logit;
* tool_error, the official rule (``scripts/evaluate.py``): retried (once by default), then left out of the official
  accuracy's denominator with its count reported, and counted wrong in ``acc_strict``; a tool_error that succeeds on the
  retry is a normal row with ``retries.tool_error = 1``; the final record is cached, a ``--cache-only`` aggregation reads it
  as final and a scoring run tries it again (the official retry on every run);
* out of memory: retried (twice by default) and counted (``oom_retries``, ``n_retried_oom``); a video that still runs out
  of memory is a final "oom" row (excluded / strict wrong, like a tool_error, counted apart);
* a runner that died is retried; a video whose runner keeps dying is not cached, and its shard returns 2 (incomplete);
* the GPU of every row is recorded (per-video record, pooled ``gpus`` count, RESULT line);
* the pooled one-row-per-video block (official WeVisForensics accuracy, fake recall / F1) counts a real pool shared by
  several generators once;
* no training-free raw score is attempted for the agent (its runner client has a ``score`` method too);
* ``main`` end to end: two shards, then a ``--cache-only`` aggregation that scores nothing and writes the per-video
  records, the agent metadata and a RESULT line with acc_strict / tool_error / oom / unparsed / oom_retries / gpus; the
  per-video log lines carry verdict and logit; ``--per-video`` adds per-video records for an ordinary detector too.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

from models.video.wevis import wevis as wv                                       # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_eval_video", ROOT / "benchmark" / "bench_eval_video.py")
bev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bev)

GPU = "NVIDIA RTX A6000"


class AgentSpec:
    key, predict_kind, checkpoint, num_frames, video_sampler = "wevis", "video_agent", "wevisforensics_lora", 16, None
    transform, fps, frame_size, window_center, video_norm = None, None, None, False, None
    agent = wv.WEVIS


class PlainSpec:
    key, checkpoint, num_frames, video_sampler, transform = "mockdet", "mock.pth", 8, "mock_sampler", "T0"
    fps, frame_size, window_center, video_norm = None, 112, False, None


class Runner:
    """Stands in for WeVisRunner: it has ``info`` (the GPU) and a ``score`` method (which must NOT be taken for a
    training-free raw scorer)."""
    info = {"gpu": GPU}

    def score(self, *a, **k):
        raise AssertionError("bench_eval_video must go through engine.predict_video_clip")


class AgentEngine:
    """predict_video_clip reads the behaviour from the file name ``<id>__<verdict>__<logit>[__<fault>].mp4``:
    verdict real / fake / unparsed; fault ``toolerr`` (always), ``toolonce`` (first call), ``oomonce``, ``oomalways``,
    ``runneronce``, ``runneralways``."""
    device = "cpu"

    def __init__(self):
        self.calls, self.seen = [], {}

    def checkpoint_path(self, spec):
        raise AssertionError("the agent's checkpoint folder must not be resolved / hashed")

    def _get_model(self, spec):
        return Runner()

    def predict_video_clip(self, spec, path):
        name = Path(path).stem
        self.calls.append(name)
        n = self.seen[name] = self.seen.get(name, 0) + 1
        parts = name.split("__")
        verdict, logit = parts[1], float(parts[2])
        fault = parts[3] if len(parts) > 3 else ""
        if fault == "toolerr" or (fault == "toolonce" and n == 1):
            raise wv.WeVisToolError("ToolError: OCR worker failed")
        if fault == "oomalways" or (fault == "oomonce" and n == 1):
            raise wv.WeVisOOMError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if fault == "runneralways" or (fault == "runneronce" and n == 1):
            raise wv.WeVisRunnerError("the WeVisForensics runner exited (code -9)")
        p = 1.0 / (1.0 + np.exp(-logit)) if verdict != "unparsed" else 0.5
        return {"fake_probability": round(float(p), 6), "logit": logit if verdict != "unparsed" else 0.0,
                "verdict": verdict, "score_source": "verdict_token" if verdict != "unparsed" else "unparsed",
                "status": "answered" if verdict != "unparsed" else "invalid", "verdict_token": verdict,
                "trace": {"tool_calls": [["visual_appearance", [0.0, 2.0]]], "steps": [{"raw": "..."}]},
                "cost": {"seconds": 40.0, "model_calls": 5, "generated_tokens": 900, "peak_alloc_gib": 21.5}}


def touch(d: Path, names) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(n.encode())


def make_tree(root: Path) -> None:
    """Two generators sharing one real pool of 4 (one unparsed real, one real with a tool_error that never recovers, one
    with an OOM that recovers); genA fakes: right, unparsed, OOM-recovering, tool_error-once; genB: right, wrong, oom
    forever."""
    reals = ["r1__real__-15.0.mp4", "r2__unparsed__0.mp4", "r3__real__-14.0__toolerr.mp4", "r4__real__-16.0__oomonce.mp4"]
    for g in ("genA", "genB"):
        touch(root / g / "0_real", reals)
    touch(root / "genA" / "1_fake", ["a1__fake__15.0.mp4", "a2__unparsed__0.mp4", "a3__fake__14.0__oomonce.mp4",
                                     "a4__fake__13.0__toolonce.mp4"])
    touch(root / "genB" / "1_fake", ["b1__fake__16.0.mp4", "b2__real__-12.0.mp4", "b3__fake__12.0__oomalways.mp4"])


def test_cache_meta() -> None:
    eng = AgentEngine()
    meta = bev.cache_meta(eng, AgentSpec)                       # AgentEngine.checkpoint_path raises if it is called
    man = (ROOT / "models" / "video" / "wevis" / "manifest.json").read_bytes()
    assert meta["ckpt_sha256"] == hashlib.sha256(man).hexdigest() == meta["agent"]["manifest_sha256"]
    m = json.loads(man)
    assert meta["agent"]["code_commit"] == m["code"]["commit"] and meta["agent"]["base_revision"] == m["base"]["revision"]
    assert meta["agent"]["adapters_revision"] == m["adapters"]["revision"]
    assert meta["agent"]["has_audio"] is False and meta["agent"]["pins"]["transformers"] == "5.23.0"
    assert ["fake", "real"] in meta["agent"]["verdict_pairs"] and meta["model"] == "wevis"
    with tempfile.TemporaryDirectory() as tmp:
        bev.ScoreCache(tmp, "x", meta)
        other = json.loads(json.dumps(meta))
        other["ckpt_sha256"] = other["agent"]["manifest_sha256"] = "0" * 64
        try:
            bev.ScoreCache(tmp, "y", other)
            raise AssertionError("expected the cache_meta refusal")
        except SystemExit as e:
            assert "different setup" in str(e)
    print("cache_meta OK")


def test_score_agent_retries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        touch(tmp / "d" / "g" / "1_fake", ["x__fake__9.0__oomonce.mp4", "y__fake__9.0__toolerr.mp4",
                                          "z__real__-9.0__runneronce.mp4", "w__fake__9.0__runneralways.mp4",
                                          "v__fake__9.0__oomalways.mp4"])
        eng = AgentEngine()
        cache = bev.ScoreCache(tmp / "c", "t", bev.cache_meta(eng, AgentSpec))
        d = tmp / "d" / "g" / "1_fake"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rx = bev.score_one(eng, AgentSpec, d / "x__fake__9.0__oomonce.mp4", raw=False, cache=cache, run_tag="T")
            ry = bev.score_one(eng, AgentSpec, d / "y__fake__9.0__toolerr.mp4", cache=cache, run_tag="T")
            rz = bev.score_one(eng, AgentSpec, d / "z__real__-9.0__runneronce.mp4", cache=cache, run_tag="T")
            rw = bev.score_one(eng, AgentSpec, d / "w__fake__9.0__runneralways.mp4", cache=cache, run_tag="T")
            rv = bev.score_one(eng, AgentSpec, d / "v__fake__9.0__oomalways.mp4", cache=cache, run_tag="T")
        log = buf.getvalue()
        assert rx["verdict"] == "fake" and rx["retries"] == {"oom": 1, "tool_error": 0, "runner": 0} and rx["gpu"] == GPU
        assert eng.seen["x__fake__9.0__oomonce"] == 2
        assert ry["kind"] == "tool_error" and eng.seen["y__fake__9.0__toolerr"] == 2 and ry["retries"]["tool_error"] == 1
        assert rz["verdict"] == "real" and rz["retries"]["runner"] == 1
        assert rw["kind"] == "runner" and eng.seen["w__fake__9.0__runneralways"] == 2
        assert rv["kind"] == "oom" and eng.seen["v__fake__9.0__oomalways"] == 3 and rv["retries"]["oom"] == 2
        assert "agent[T] g/1_fake/x__fake__9.0__oomonce.mp4 RETRY kind=oom 1/2" in log
        assert "agent[T] g/1_fake/x__fake__9.0__oomonce.mp4 verdict=fake logit=9.0 " in log and f"gpu={GPU}" in log
        assert "FAILED kind=tool_error" in log and "FAILED kind=runner" in log
        # cached: success, final tool_error and oom; NOT the runner failure
        keys = {json.loads(l)["k"]: json.loads(l) for l in (tmp / "c" / "scores.t.jsonl").read_text().splitlines()}
        assert set(keys) == {"g/1_fake/x__fake__9.0__oomonce.mp4", "g/1_fake/y__fake__9.0__toolerr.mp4",
                             "g/1_fake/z__real__-9.0__runneronce.mp4", "g/1_fake/v__fake__9.0__oomalways.mp4"}
        traces = (tmp / "c" / "traces.t.jsonl").read_text().splitlines()
        assert len(traces) == 2 and json.loads(traces[0])["trace"]["tool_calls"] == [["visual_appearance", [0.0, 2.0]]]
        # a later scoring run retries the cached tool_error (official); an aggregation run reads it as final
        cache2 = bev.ScoreCache(tmp / "c", "t2", bev.cache_meta(eng, AgentSpec))
        with contextlib.redirect_stdout(io.StringIO()):
            bev.score_one(eng, AgentSpec, d / "y__fake__9.0__toolerr.mp4", cache=cache2)
        assert eng.seen["y__fake__9.0__toolerr"] == 4
        cache3 = bev.ScoreCache(tmp / "c", "t3", bev.cache_meta(eng, AgentSpec))
        cache3.only = True
        n = len(eng.calls)
        assert bev.score_one(eng, AgentSpec, d / "y__fake__9.0__toolerr.mp4", cache=cache3)["kind"] == "tool_error"
        assert bev.score_one(eng, AgentSpec, d / "x__fake__9.0__oomonce.mp4", cache=cache3)["verdict"] == "fake"
        assert len(eng.calls) == n
        try:
            bev.score_one(eng, AgentSpec, d / "w__fake__9.0__runneralways.mp4", cache=cache3)
            raise AssertionError("an uncached runner failure must stop a --cache-only aggregation")
        except SystemExit as e:
            assert "not in the score cache" in str(e)
        # retry counts from the command line
        eng2 = AgentEngine()
        with contextlib.redirect_stdout(io.StringIO()):
            r = bev.score_one(eng2, AgentSpec, d / "x__fake__9.0__oomonce.mp4", retries={"oom": 0, "tool_error": 0, "runner": 0})
        assert r["kind"] == "oom" and eng2.seen["x__fake__9.0__oomonce"] == 1
    print("score_agent retries OK")


def test_agent_metrics() -> None:
    real = lambda v, l: {"verdict": v, "logit": l, "p": float(1 / (1 + np.exp(-l))), "score_source": "verdict_token"}  # noqa: E731
    rrec = [real("real", -15.0), real("unparsed", 0.0), real("fake", 3.0)]
    frec = [real("fake", 15.0), real("unparsed", 0.0), real("real", -2.0), real("fake", 9.0)]
    rerr = [{"err": "x", "kind": "tool_error"}]
    ferr = [{"err": "y", "kind": "oom"}, {"err": "z", "kind": "tool_error"}]
    m = bev.agent_metrics(rrec, frec, rerr, ferr)
    assert m["real_acc"] == 33.33 and m["fake_acc"] == 50.0 and m["acc"] == 41.67      # unparsed wrong for both classes
    assert m["real_acc_strict"] == 25.0 and m["fake_acc_strict"] == 33.33 and m["acc_strict"] == 29.17
    assert m["unparsed_real"] == 1 and m["unparsed_fake"] == 1 and m["tool_error_real"] == 1 and m["tool_error_fake"] == 1
    assert m["oom_fake"] == 1 and m["oom_real"] == 0 and m["acc_by"] == "agent_verdict" and m["rank_by"] == "logit"
    from sklearn.metrics import roc_auc_score
    y = [0, 0, 0, 1, 1, 1, 1]
    assert m["auroc"] == round(roc_auc_score(y, [-15, 0, 3, 15, 0, -2, 9]) * 100, 2)
    pb = bev.pooled_block(AgentSpec, [dict(r, retries={"oom": 1}, gpu=GPU, seconds=40, model_calls=5) for r in rrec],
                          [dict(r, gpu=GPU, seconds=50, model_calls=7) for r in frec], [dict(e, gpu=GPU) for e in rerr],
                          [dict(e, gpu=GPU) for e in ferr])
    assert pb["n"] == 7 and pb["acc"] == round(100 * 3 / 7, 2) and pb["acc_strict"] == 30.0
    assert pb["fake_recall"] == 50.0 and pb["fake_f1"] == round(100 * 4 / (4 + 1 + 2), 2)
    assert pb["n_tool_error"] == 2 and pb["n_oom"] == 1 and pb["n_unparsed"] == 2 and pb["oom_retries"] == 3
    assert pb["gpus"] == {GPU: 10} and pb["mean_seconds"] == round((3 * 40 + 4 * 50) / 7, 2)
    print("agent metrics OK")


def test_main_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_tree(tmp / "data")
        eng = AgentEngine()
        saved = bev.get_engine, bev.get_spec, bev.REPRO, bev.RESULTS_LOG
        bev.get_engine, bev.get_spec = (lambda: eng), (lambda key: AgentSpec)
        bev.REPRO, bev.RESULTS_LOG = tmp, tmp / "M.jsonl"
        try:
            common = ["-m", "wevis", "--benchmark", "pergen", "--data", str(tmp / "data"), "--cache-dir", str(tmp / "cache")]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rcs = [bev.main(common + ["--tag", "ag", "--shard", f"{k}/2"]) for k in range(2)]
            assert rcs == [0, 0], rcs
            assert sorted(eng.seen) == sorted([f.stem for f in (tmp / "data").rglob("*.mp4") if "genB/0_real" not in str(f)])
            n_calls = len(eng.calls)
            out = tmp / "ag.json"
            with contextlib.redirect_stdout(buf):
                assert bev.main(common + ["--tag", "ag", "--cache-only", "--out", str(out)]) == 0
            assert len(eng.calls) == n_calls, "aggregation must not score"
            log = buf.getvalue()
            res = json.loads(out.read_text())
            ga, gb = res["per_generator"]["genA"], res["per_generator"]["genB"]
            # reals: r1 right, r2 unparsed (wrong), r3 tool_error (excluded / strict wrong), r4 right after an OOM retry
            assert ga["real_acc"] == round(200 / 3, 2) and ga["real_acc_strict"] == 50.0 and ga["tool_error_real"] == 1
            # genA fakes: a1 right, a2 unparsed, a3 right (OOM retried), a4 right (tool_error retried): 3/4
            assert ga["fake_acc"] == 75.0 and ga["fake_acc_strict"] == 75.0 and ga["unparsed_fake"] == 1
            # genB fakes: b1 right, b2 wrong, b3 OOM forever (excluded / strict wrong)
            assert gb["fake_acc"] == 50.0 and gb["fake_acc_strict"] == round(100 / 3, 2) and gb["oom_fake"] == 1
            assert res["means"]["acc"] == round((ga["acc"] + gb["acc"]) / 2, 2)
            assert res["means"]["acc_strict"] == round((ga["acc_strict"] + gb["acc_strict"]) / 2, 2)
            pb = res["means"]["pooled"]
            assert pb["n"] == 3 + 4 + 2 and pb["n_tool_error"] == 1 and pb["n_oom"] == 1   # the real pool counted once
            assert pb["acc"] == round(100 * (2 + 3 + 1) / 9, 2) and pb["acc_strict"] == round(100 * 6 / 11, 2)
            assert pb["oom_retries"] == 2 + 2 and pb["n_retried_oom"] == 3 and pb["tool_error_retries"] == 1 + 1
            assert pb["gpus"] == {GPU: 11}
            assert len(res["per_video"]) == 11 and all(r["gpu"] == GPU for r in res["per_video"])
            assert sum(r["label"] == 0 for r in res["per_video"]) == 4
            assert res["agent"]["manifest_sha256"] == res["ckpt_sha256"] and res["agent_retries"]["oom"] == 2
            assert "raw" not in json.dumps(res["per_video"])                          # no training-free raw score
            line = [l for l in log.splitlines() if l.startswith("=== RESULT ag:")][0]
            assert f"acc={res['means']['acc']} acc_strict={res['means']['acc_strict']} pooled_acc={pb['acc']}" in line
            assert "tool_error=1 oom=1 unparsed=2 oom_retries=4 gpus=NVIDIA_RTX_A6000 over 2 gens" in line
            assert "agent[ag] genA/1_fake/a1__fake__15.0.mp4 verdict=fake logit=15.0 src=verdict_token" in log
            # a runner that keeps dying: not cached, the shard reports itself incomplete
            touch(tmp / "data" / "genB" / "1_fake", ["b4__fake__11.0__runneralways.mp4"])
            with contextlib.redirect_stdout(io.StringIO()):
                rcs = [bev.main(common + ["--tag", "ag", "--shard", f"{k}/2"]) for k in range(2)]
            assert sorted(rcs) == [0, 2], rcs
            # an ordinary detector with --per-video: per-video records + pooled block, otherwise the usual JSON
            from test_bench_eval_video_cache import MockEngine, MockSpec, make_tree as plain_tree
            ckpt = tmp / "mock.pth"
            ckpt.write_bytes(b"w")
            plain_tree(tmp / "plain", MockSpec)
            peng = MockEngine(ckpt)
            bev.get_engine, bev.get_spec = (lambda: peng), (lambda key: MockSpec)
            with contextlib.redirect_stdout(io.StringIO()):
                assert bev.main(["-m", "waverep", "--benchmark", "pergen", "--data", str(tmp / "plain"), "--tag", "pl",
                                 "--out", str(tmp / "pl.json"), "--per-video"]) == 0
            pl = json.loads((tmp / "pl.json").read_text())
            assert "pooled" in pl["means"] and len(pl["per_video"]) == pl["means"]["pooled"]["n"] and "agent" not in pl
            assert all("k" in r and r["label"] in (0, 1) for r in pl["per_video"])
        finally:
            bev.get_engine, bev.get_spec, bev.REPRO, bev.RESULTS_LOG = saved
    print("main end to end OK")


if __name__ == "__main__":
    test_cache_meta()
    test_score_agent_retries()
    test_agent_metrics()
    test_main_end_to_end()
    print("test_bench_eval_video_agent: all OK")
