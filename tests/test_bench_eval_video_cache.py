"""CPU unit test of benchmark/bench_eval_video.py's score cache, sharding and metrics (mock engine, no weights, no video).

Checks:
* the shard partition: ``unique_videos`` lists each real pool once (the same pool hard-linked per generator) and every
  fake; the slices ``jobs[k::n]`` are disjoint and cover everything,
* cache resume: a second ``score_videos`` (also through a fresh ``ScoreCache`` that re-reads the files) calls the engine
  zero times; ``scores.<tag>.jsonl`` is only created on the first write,
* ``cache_meta.json``: written by the first use; a different model key, checkpoint hash, frame count or format is refused
  with a clear error; ``--cache-only`` refuses a cache without one,
* the raw logit ranks AP / AUROC only when every record has one (saturated, tied probabilities are separated by it), and a
  detector without ``logit`` gets exactly the old torchmetrics numbers,
* decode failures (``VideoDecodeError``) are counted and cached; any other exception (OOM, I/O) is counted separately,
  not cached and retried; a legacy error record (no ``kind``) is classified by its message; a failing raw score of a
  training-free detector drops the video from the raw list only,
* the real reader ``media.cv2_frames``: a missing / unopenable file is an OSError (not cached, retried), an empty file a
  decode failure (cached),
* ``main`` end to end with the mock engine: two shards, then a ``--cache-only`` aggregation with the same JSON means and
  no ``scores.agg*.jsonl``; the output JSON carries limit / reals / ckpt_sha256 / failed_to_decode / failed_other,
* a detector with its own decision rule (``own_verdict``, NSG-VD: fake iff the MMD statistic > 1): accuracy from the
  cached engine verdict (a statistic of exactly 1 is real although P(fake) = 0.5), the raw score is the logit (no second
  pass), records without a verdict are refused; every other detector keeps P(fake) >= 0.5 (the old code, inline), no
  verdict in its records and no ``acc_by`` in its JSON, also end to end,
* ``cache_meta`` hashes the pinned ffmpeg binary when its variable holds a bare command name (looked up on the PATH, not
  in the working directory).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

from backend import media                                                        # noqa: E402
from backend.media import VideoDecodeError                                       # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_eval_video", ROOT / "benchmark" / "bench_eval_video.py")
bev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bev)


class MockSpec:
    key, checkpoint, num_frames, video_sampler, transform = "mockdet", "mock.pth", 8, "mock_sampler", "T0"
    fps, frame_size, window_center, video_norm = None, 112, False, None


class MockEngine:
    """predict_video_clip reads the score from the file name: ``<name>__<logit>.mp4``; ``decode`` / ``oom`` / ``rawfail`` in
    the name make it fail the way the real engine would."""
    device = "cpu"

    def __init__(self, ckpt, with_logit=True, raw=False):
        self.ckpt, self.with_logit, self.raw, self.calls, self.oom_seen = ckpt, with_logit, raw, [], set()

    def checkpoint_path(self, spec):
        return self.ckpt

    def _get_model(self, spec):
        class M:
            pass
        m = M()
        if self.raw:
            m.score = lambda x: x
        return m

    def predict_video_clip(self, spec, path):
        self.calls.append(Path(path).name)
        name = Path(path).stem
        if "decode" in name:
            raise VideoDecodeError("Could not decode any frames from the video.")
        if "oom" in name:
            raise RuntimeError("CUDA out of memory")
        logit = float(name.split("__")[1])
        res = {"fake_probability": round(1 / (1 + math.exp(-logit)), 6)}
        if self.with_logit:
            res["logit"] = round(logit, 6)
        return res

    def video_raw_score(self, spec, path):
        if "rawfail" in Path(path).stem:
            raise RuntimeError("no raw score")
        return float(Path(path).stem.split("__")[1])


class MmdSpec(MockSpec):
    """An NSG-VD-like detector: the engine verdict is the official rule (statistic > 1)."""
    key, video_prob = "mockmmd", "mmd"


class MmdEngine(MockEngine):
    """``<name>__<statistic>.mp4``: P(fake) = sigmoid((s - 1) / 0.1), logit = s, verdict = fake iff s > 1 (NSG-VD)."""

    def predict_video_clip(self, spec, path):
        self.calls.append(Path(path).name)
        s = float(Path(path).stem.split("__")[1])
        return {"fake_probability": round(1 / (1 + math.exp(-(s - 1) / 0.1)), 6), "logit": round(s, 6),
                "verdict": "fake" if s > 1 else "real"}

    def video_raw_score(self, spec, path):
        raise AssertionError("an own_verdict detector's raw score is its logit: no second pass")


def old_acc(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    """The accuracy code of bench_eval_video.metrics before own_verdict (9be402e), inline."""
    preds = (probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"real_acc": round(real_acc, 2), "fake_acc": round(fake_acc, 2), "acc": round((real_acc + fake_acc) / 2, 2)}


def make_tree(root: Path, spec):
    """root/<gen>/{0_real,1_fake}/*.mp4 (empty files); reals shared by name between the generators (as hard links)."""
    names = {}
    for g in ("genA", "genB"):
        for cls, items in (("0_real", [f"r{i}__{-3 - i}" for i in range(6)]),
                           ("1_fake", [f"f{g}{i}__{2 + i}" for i in range(6)] + [f"{g}_decode__0"])):
            d = root / g / cls
            d.mkdir(parents=True, exist_ok=True)
            for n in items:
                (d / f"{n}.mp4").write_bytes(b"")
            names[(g, cls)] = sorted(d.glob("*.mp4"))
    return names


def test_shards_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"weights-v1")
        names = make_tree(tmp / "data", MockSpec)
        gens = bev.discover("pergen", tmp / "data", None, None)
        pool_key = lambda reals: (len(reals), reals[0].name, reals[-1].name)          # noqa: E731
        jobs = bev.unique_videos(gens, pool_key)
        # genA and genB have identical real file names (same pool key) -> counted once
        assert len(jobs) == 6 + 7 + 7, len(jobs)
        parts = [jobs[k::3] for k in range(3)]
        assert sorted(sum(parts, []), key=str) == sorted(jobs, key=str)
        assert sum(len(p) for p in parts) == len(set(sum(parts, [])))

        eng, meta = MockEngine(ckpt), None
        meta = bev.cache_meta(eng, MockSpec)
        cdir = tmp / "cache"
        c1 = bev.ScoreCache(cdir, "0of3", meta)
        assert not list(cdir.glob("scores.*.jsonl")), "scores file must only be created by the first write"
        recs, bad, other = bev.score_videos(eng, MockSpec, parts[0], False, c1)
        assert bad == sum("decode" in p.name for p in parts[0]) and other == 0
        n_calls = len(eng.calls)
        assert n_calls == len(parts[0]) and (cdir / "scores.0of3.jsonl").exists()
        # resume: through a fresh cache that re-reads the file, nothing is scored again (decode failures are cached too)
        eng2 = MockEngine(ckpt)
        c2 = bev.ScoreCache(cdir, "agg", meta)
        recs2, bad2, other2 = bev.score_videos(eng2, MockSpec, parts[0], False, c2)
        assert eng2.calls == [] and ([r['p'] for r in recs2], bad2, other2) == ([r['p'] for r in recs], bad, other)
        assert not (cdir / "scores.agg.jsonl").exists()
        # cache-only: refuses an uncached video, loudly
        c3 = bev.ScoreCache(cdir, "agg3", meta, write_meta=False)
        c3.only = True
        try:
            bev.score_videos(eng2, MockSpec, parts[1], False, c3)
            raise AssertionError("cache-only must refuse an uncached video")
        except SystemExit as e:
            assert "not in the score cache" in str(e)


def test_cache_meta_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"weights-v1")
        eng = MockEngine(ckpt)
        meta = bev.cache_meta(eng, MockSpec)
        assert meta["ckpt_sha256"] and meta["model"] == "mockdet" and meta["spec"]["num_frames"] == 8
        d = tmp / "c"
        bev.ScoreCache(d, "a", meta)                                    # first use writes cache_meta.json
        assert json.loads((d / "cache_meta.json").read_text())["ckpt_sha256"] == meta["ckpt_sha256"]
        bev.ScoreCache(d, "b", meta)                                    # same setup: fine

        def refused(m, what):
            try:
                bev.ScoreCache(d, "c", m)
            except SystemExit as e:
                assert "different setup" in str(e) and what in str(e), str(e)
                return
            raise AssertionError(f"a cache written for another {what} must be refused")

        refused(dict(meta, model="otherdet"), "model")
        ckpt.write_bytes(b"weights-v2-retrained")
        refused(bev.cache_meta(eng, MockSpec), "ckpt_sha256")
        refused(dict(meta, spec=dict(meta["spec"], num_frames=16)), "num_frames")
        refused(dict(meta, spec=dict(meta["spec"], transform="T1")), "transform")
        refused(dict(meta, cache_format=bev.CACHE_FORMAT + 1), "cache_format")
        try:                                                            # aggregation of a cache of unknown origin
            bev.ScoreCache(tmp / "nometa", "x", meta, write_meta=False)
            raise AssertionError("a cache without cache_meta.json must be refused when it may not write one")
        except SystemExit as e:
            assert "no cache_meta.json" in str(e)


def test_ranking_and_metrics():
    # probabilities saturate at 1.0 / 0.0 after the 6 dp rounding; the logits still rank
    logits = [-30.0, -25.0, -20.0, 20.0, 25.0, 30.0, -21.0, 22.0]
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0])                    # the last two are misranked on purpose
    recs = [{"p": round(1 / (1 + math.exp(-l)), 6), "logit": l} for l in logits]
    probs = torch.tensor([r["p"] for r in recs])
    assert len(set(probs.tolist())) == 2, "the rounded probabilities must tie in this case"
    rank = bev.rank_scores(recs)
    assert rank is not None and rank.dtype == np.float64
    by_prob, by_logit = bev.metrics(probs, labels), bev.metrics(probs, labels, rank)
    assert by_logit["auroc"] != by_prob["auroc"] and by_logit["acc"] == by_prob["acc"]   # accuracy always thresholds P(fake)
    # sklearn on the logit == the exact AUROC from a pairwise count
    pos, neg = rank[labels.numpy() == 1], rank[labels.numpy() == 0]
    exact = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]) * 100
    assert abs(by_logit["auroc"] - round(exact, 2)) < 0.006, (by_logit, exact)
    # ranking by logit only when EVERY record has one
    assert bev.rank_scores([{"p": 0.5}] + recs) is None and bev.rank_scores([]) is None
    # no logit: identical to the old torchmetrics path (the code of the previous bench_eval_video.metrics, inline)
    from torchmetrics.functional import auroc, average_precision
    rng = np.random.default_rng(0)
    y = torch.tensor((rng.random(300) < 0.4).astype(int))
    p = torch.round(torch.tensor(1 / (1 + np.exp(-(rng.normal(0, 3, 300) + 2 * (y.numpy() - 0.5)))), dtype=torch.float32) * 1e6) / 1e6
    new = bev.metrics(p, y)
    old = {"ap": round(float(average_precision(p, y, task="binary")) * 100, 2),
           "ap_real": round(float(average_precision(1 - p, 1 - y, task="binary")) * 100, 2),
           "auroc": round(float(auroc(p, y, task="binary")) * 100, 2)}
    assert all(new[k] == v for k, v in old.items()), (new, old)
    assert new == bev.metrics(p, y, None)


def test_failure_kinds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"w")
        eng = MockEngine(ckpt)
        meta = bev.cache_meta(eng, MockSpec)
        paths = [tmp / n for n in ("ok__1.0.mp4", "bad_decode__0.mp4", "bad_oom__0.mp4")]
        cache = bev.ScoreCache(tmp / "c", "t", meta)
        recs, bad, other = bev.score_videos(eng, MockSpec, paths, False, cache)
        assert (len(recs), bad, other) == (1, 1, 1)
        assert recs[0]["logit"] == 1.0 and "err" not in recs[0]
        lines = [json.loads(line) for line in (tmp / "c" / "scores.t.jsonl").read_text().splitlines()]
        kinds = {r["k"].split("/")[-1]: r.get("kind") for r in lines}
        assert kinds == {"ok__1.0.mp4": None, "bad_decode__0.mp4": "decode"}, kinds       # the OOM is not cached
        eng.calls.clear()
        bev.score_videos(eng, MockSpec, paths, False, cache)
        assert eng.calls == ["bad_oom__0.mp4"], eng.calls                                  # only the non-decode failure retried
        # a legacy error record (written before "kind" existed) is classified by the decoder's message
        assert bev.is_decode_error({"err": "ValueError: Could not decode any frames from the video."})
        assert not bev.is_decode_error({"err": "RuntimeError: CUDA out of memory"})
        assert bev.is_decode_error({"err": "x", "kind": "decode"}) and not bev.is_decode_error({"p": 0.5})
        # training-free detector: a failing raw score keeps the video (it only leaves the raw list)
        eng_raw = MockEngine(ckpt, with_logit=False, raw=True)
        cache2 = bev.ScoreCache(tmp / "c2", "t", bev.cache_meta(eng_raw, MockSpec))
        recs, bad, other = bev.score_videos(eng_raw, MockSpec, [tmp / "a__2.0.mp4", tmp / "rawfail__1.0.mp4"], True, cache2)
        assert (len(recs), bad, other) == (2, 0, 0)
        assert "raw" in recs[0] and "raw" not in recs[1] and "raw_err" in recs[1]
        assert bev.rank_scores(recs) is None                                                # no logit: P(fake) ranking


def test_missing_file_is_not_a_decode_failure():
    """The real reader (media.cv2_frames) separates I/O problems from decode failures: a missing or unopenable file is an
    OSError (not cached, retried), an empty file is a VideoDecodeError (cached)."""
    class ReaderEngine(MockEngine):
        def predict_video_clip(self, spec, path):               # the engine's check: no frame -> VideoDecodeError
            self.calls.append(Path(path).name)
            if not list(media.cv2_frames(path, limit=1)):
                raise VideoDecodeError("Could not decode any frames from the video.")
            return {"fake_probability": 0.5, "logit": 0.0}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"w")
        missing, empty, garbage = tmp / "missing.mp4", tmp / "empty.mp4", tmp / "garbage.mp4"
        empty.write_bytes(b"")
        garbage.write_bytes(b"this is not a video file")
        for p, exc in ((missing, FileNotFoundError), (garbage, OSError), (empty, VideoDecodeError)):
            try:
                list(media.cv2_frames(p))
                raise AssertionError(f"{p.name}: expected {exc.__name__}")
            except exc as e:
                assert exc is VideoDecodeError or not isinstance(e, VideoDecodeError), p.name
        assert issubclass(FileNotFoundError, OSError) and not issubclass(VideoDecodeError, OSError)
        eng = ReaderEngine(ckpt)
        cache = bev.ScoreCache(tmp / "c", "t", bev.cache_meta(eng, MockSpec))
        recs, bad, other = bev.score_videos(eng, MockSpec, [missing, garbage, empty], False, cache)
        assert (len(recs), bad, other) == (0, 1, 2), (bad, other)          # only the empty file is a decode failure
        lines = [json.loads(line) for line in (tmp / "c" / "scores.t.jsonl").read_text().splitlines()]
        assert {r["k"].split("/")[-1]: r["kind"] for r in lines} == {"empty.mp4": "decode"}   # missing / unopenable: not cached
        eng.calls.clear()
        bev.score_videos(eng, MockSpec, [missing, garbage, empty], False, cache)
        assert sorted(eng.calls) == ["garbage.mp4", "missing.mp4"], eng.calls            # retried; the decode failure is not


def test_list_out_needs_shard_and_base():
    for extra in ([], ["--shard", "0/2"], ["--base", "/x"]):
        try:
            bev.main(["-m", "waverep", "--benchmark", "pergen", "--data", "/nonexistent", "--list-out", "l.txt"] + extra)
            raise AssertionError(f"--list-out {extra}: expected an argparse error")
        except SystemExit as e:
            assert e.code == 2, (extra, e.code)                  # parser.error, not an AttributeError / TypeError


def test_main_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"weights")
        make_tree(tmp / "data", MockSpec)
        eng = MockEngine(ckpt)
        bev.get_engine, bev.get_spec = (lambda: eng), (lambda key: MockSpec)
        (tmp / "repro").mkdir()
        bev.REPRO, bev.RESULTS_LOG = tmp / "repro", tmp / "repro" / "MASTER_RESULTS.jsonl"
        common = ["-m", "waverep", "--benchmark", "pergen", "--data", str(tmp / "data"), "--cache-dir", str(tmp / "cache")]
        for k in range(3):
            assert bev.main(common + ["--tag", "t", "--shard", f"{k}/3"]) == 0
        n_scored = len(eng.calls)
        assert n_scored == 6 + 7 + 7, n_scored                                              # shared real pool scored once
        eng.calls.clear()
        out1 = tmp / "agg1.json"
        assert bev.main(common + ["--tag", "t_full", "--cache-only", "--out", str(out1)]) == 0
        assert eng.calls == [], "aggregation must not score"
        assert not list((tmp / "cache").glob("scores.agg*.jsonl")), "aggregation must not leave empty score files"
        res = json.loads(out1.read_text())
        assert res["failed_to_decode"] == {"fake": 2, "real_unique": 0} and res["failed_other"] == {"fake": 0, "real_unique": 0}
        assert res["limit"] is None and res["reals"] is None and res["ckpt_sha256"] == bev.cache_meta(eng, MockSpec)["ckpt_sha256"]
        for g in res["per_generator"].values():
            assert g["rank_by"] == "logit" and g["n_real"] == 6 and g["n_fake"] == 6 and g["failed_fake"] == 1
            assert "acc_by" not in g                     # a detector without its own decision rule: JSON unchanged
            assert g["auroc"] == 100.0, g                # every fake logit (2..7) outranks every real logit (-3..-8)
        out2 = tmp / "agg2.json"
        assert bev.main(common + ["--tag", "t_full2", "--cache-only", "--out", str(out2)]) == 0
        assert json.loads(out2.read_text())["means"] == res["means"]
        # a retrained checkpoint into the same cache dir is refused
        ckpt.write_bytes(b"retrained")
        try:
            bev.main(common + ["--tag", "t_full3", "--cache-only", "--out", str(tmp / "x.json")])
            raise AssertionError("expected the cache_meta refusal")
        except SystemExit as e:
            assert "different setup" in str(e)


def test_own_verdict():
    assert bev.own_verdict(MmdSpec) and not bev.own_verdict(MockSpec)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"w")
        # reals at 0.5 / 0.9 / exactly 1.0 (real by the official > 1), fakes just above 1, at 1.5 and 3
        paths = ([tmp / f"r{i}__{v}.mp4" for i, v in enumerate((0.5, 0.9, 1.0))]
                 + [tmp / f"f{i}__{v}.mp4" for i, v in enumerate((1.000001, 1.5, 3.0))])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        eng = MmdEngine(ckpt, raw=True)
        cache = bev.ScoreCache(tmp / "c", "t", bev.cache_meta(eng, MmdSpec))
        recs, bad, other = bev.score_videos(eng, MmdSpec, paths, True, cache)
        assert (len(recs), bad, other) == (6, 0, 0)
        assert [r["verdict"] for r in recs] == ["real"] * 3 + ["fake"] * 3
        assert all(r["raw"] == r["logit"] for r in recs)              # no video_raw_score call (it would raise)
        probs = torch.tensor([r["p"] for r in recs])
        assert float(probs[2]) == 0.5                                  # at the threshold P(fake) is 0.5 ...
        by_verdict = bev.metrics(probs, labels, bev.rank_scores(recs), bev.verdict_preds(recs))
        assert by_verdict["real_acc"] == 100.0 and by_verdict["acc"] == 100.0       # ... but the video is real
        assert bev.metrics(probs, labels, bev.rank_scores(recs))["real_acc"] == 66.67  # P >= 0.5 would call it fake
        # the verdicts are cached: a fresh cache reproduces them without scoring
        eng2 = MmdEngine(ckpt, raw=True)
        recs2, _, _ = bev.score_videos(eng2, MmdSpec, paths, True, bev.ScoreCache(tmp / "c", "t2", bev.cache_meta(eng2, MmdSpec)))
        assert eng2.calls == [] and [r["verdict"] for r in recs2] == [r["verdict"] for r in recs]
        try:                                                           # a record without a verdict is refused
            bev.verdict_preds(recs + [{"p": 0.5, "logit": 1.0}])
            raise AssertionError("records without a verdict must be refused")
        except SystemExit as e:
            assert "verdict" in str(e)
        # every other detector: no verdict in its records, accuracy exactly the old P(fake) >= 0.5 code
        eng3 = MockEngine(ckpt)
        recs3, _, _ = bev.score_videos(eng3, MockSpec, [tmp / "a__0.0.mp4", tmp / "b__-1.0.mp4", tmp / "c__2.0.mp4",
                                                        tmp / "d__0.0.mp4"], False, None)
        assert all("verdict" not in r for r in recs3)
        p3, y3 = torch.tensor([r["p"] for r in recs3]), torch.tensor([0, 0, 1, 1])
        assert float(p3[0]) == 0.5
        for rank in (None, bev.rank_scores(recs3)):
            m3 = bev.metrics(p3, y3, rank)
            assert {k: m3[k] for k in ("real_acc", "fake_acc", "acc")} == old_acc(p3, y3), (m3, old_acc(p3, y3))
        rng = np.random.default_rng(1)
        y = torch.tensor((rng.random(400) < 0.5).astype(int))
        p = torch.round(torch.tensor(1 / (1 + np.exp(-rng.normal(0, 2, 400))), dtype=torch.float32) * 1e3) / 1e3
        p[:20] = 0.5                                                    # ties at the threshold count fake, as before
        m = bev.metrics(p, y)
        assert {k: m[k] for k in ("real_acc", "fake_acc", "acc")} == old_acc(p, y)
    # end to end: an own_verdict detector's JSON says how accuracy was computed
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"w")
        for cls, vals in (("0_real", (0.5, 1.0)), ("1_fake", (1.2, 0.8))):
            d = tmp / "data" / "genA" / cls
            d.mkdir(parents=True)
            for i, v in enumerate(vals):
                (d / f"{cls}{i}__{v}.mp4").write_bytes(b"")
        eng = MmdEngine(ckpt, raw=True)
        saved = bev.get_engine, bev.get_spec, bev.REPRO, bev.RESULTS_LOG
        bev.get_engine, bev.get_spec = (lambda: eng), (lambda key: MmdSpec)
        bev.REPRO, bev.RESULTS_LOG = tmp, tmp / "MASTER_RESULTS.jsonl"
        try:
            out = tmp / "r.json"
            assert bev.main(["-m", "nsg_vd", "--benchmark", "pergen", "--data", str(tmp / "data"), "--tag", "v",
                             "--out", str(out)]) == 0
        finally:
            bev.get_engine, bev.get_spec, bev.REPRO, bev.RESULTS_LOG = saved
        g = json.loads(out.read_text())["per_generator"]["genA"]
        assert g["acc_by"] == "engine_verdict" and g["real_acc"] == 100.0 and g["fake_acc"] == 50.0, g
        assert g["fake_recall_at_fpr"] and g["rank_by"] == "logit"


def test_cache_meta_bare_ffmpeg_name():
    """$<ffmpeg_env> = a bare command name: cache_meta hashes the binary found on the PATH (not ./ffmpeg)."""
    class PinnedSpec(MockSpec):
        key, ffmpeg_version, ffmpeg_env = "mockpin", "4.2", "SYNTHSCOPE_FFMPEG_BENCH_TEST"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "mock.pth"
        ckpt.write_bytes(b"w")
        bindir, work = tmp / "bin", tmp / "work"
        bindir.mkdir()
        work.mkdir()
        exe = bindir / "ffmpeg-bench-test"
        exe.write_text("#!/bin/sh\necho 'ffmpeg version 4.2.2 Copyright (c) 2000-2019'\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        saved_env = {k: os.environ.get(k) for k in ("PATH", PinnedSpec.ffmpeg_env)}
        cwd = os.getcwd()
        os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
        os.environ[PinnedSpec.ffmpeg_env] = exe.name                  # bare name
        os.chdir(work)                                                # no such file in the working directory
        try:
            meta = bev.cache_meta(MockEngine(ckpt), PinnedSpec)
        finally:
            os.chdir(cwd)
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert meta["ffmpeg"] == {"version": "4.2.2", "sha256": bev._sha256(exe)}, meta["ffmpeg"]


if __name__ == "__main__":
    test_shards_and_resume()
    test_cache_meta_refusal()
    test_ranking_and_metrics()
    test_failure_kinds()
    test_missing_file_is_not_a_decode_failure()
    test_list_out_needs_shard_and_base()
    test_main_end_to_end()
    test_own_verdict()
    test_cache_meta_bare_ffmpeg_name()
    print("OK")
