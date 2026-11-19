"""Unified VIDEO benchmark eval for SynthScope video detectors (d3, split, restrav, stall, vina_video, waverep,
waverep_g4, nsg_vd, nsg_vd_pika, nsg_vd_unbalance).

Scores each video through the SAME verified synthscope engine path the web app uses
(engine.predict_video_clip -> native clip pipeline + per-model norm + fake-prob), computes
per-generator + mean acc@0.5 / AP / AUROC, and writes the SAME JSON schema as bench_eval.py
(repro/bench_results/<tag>.json + append repro/MASTER_RESULTS.jsonl) so the aggregator ingests it.

    python benchmark/bench_eval_video.py --model d3 --benchmark genvideo \
        --data /hddstore/ssalehi/SynthScope/d3_repro/genvideo_val/GenVideo-Val --limit 150 --tag d3_genvideo

    # one step: download the dataset with its loader if needed, then evaluate (D3's protocol: --limit 1000)
    python benchmark/bench_eval_video.py --model d3 --dataset videophy --root /data/VideoPhy --limit 1000

Each generator reports AP with fake as the positive class (``ap``) and with real as the positive class
(``ap_real``, what D3's official eval.py computes: ``average_precision_score(1 - y_true, std)``).

Resumable / parallel scoring (long benchmarks): ``--cache-dir DIR`` stores every scored video (P(fake), the raw logit
when the detector reports one, or the decode error) in DIR/scores.*.jsonl, keyed by the last three path components, and reuses it on the next run; ``--shard K/N`` scores only the K-th of N slices of the unique videos and stops
(run the N slices on N GPUs, then one final run without ``--shard`` reads every slice and writes the metrics).
A detector that reports a raw logit (WaveRep) is ranked by it (AP / AUROC), because the 6-dp rounded P(fake) ties
very confident videos at 0 / 1; accuracy still thresholds P(fake) at 0.5, except for a detector with its own decision
rule (``own_verdict``: NSG-VD, fake iff the MMD statistic > 1, so a statistic of exactly 1 is real although its P(fake)
is 0.5), whose accuracy uses the engine's verdict (cached per video); for NSG-VD the raw statistic is also the
training-free "raw" score, taken from the logit instead of a second diffusion pass. Videos that cannot be decoded
(backend.media.VideoDecodeError) are counted per generator (failed_real / failed_fake) and in the result JSON
(failed_to_decode: decode failures only, cached); any other error (OOM, I/O) is counted in failed_other, never cached
and retried; both kinds are left out of the metrics. The cache directory carries cache_meta.json (model key, sha256 of
the resolved checkpoint, frame count / sampler / transform, format version): a run whose setup differs is refused.
The result JSON also records limit, reals, ckpt_sha256 and cache_dir (and, for a detector with aux files or an ffmpeg
pin, the aux-file sha256s and the ffmpeg version + binary sha256).

Benchmark layouts:
  genvideo   <data>/Fake/<gen>/*.mp4 + <data>/Real/*.mp4 (shared real pool)   [GenVideo-Val]
  fakeonly   <data>/<gen>/*.mp4 (all fake) + --reals DIR of real videos        [EvalCrafter]
  pergen     <data>/<gen>/{0_real,1_fake}/*.mp4                                 [GenVidBench]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interface"))
from backend import config as backend_config   # noqa: E402
from backend.engine import get_engine, spec_ffmpeg   # noqa: E402
from backend.media import VideoDecodeError     # noqa: E402
from backend.registry import get_spec          # noqa: E402
from datasets_util import ensure_dataset        # noqa: E402
from video_layout import FAKE_DIRNAMES, REAL_DIRNAMES, VIDEO_EXTS, _find, _videos, discover   # noqa: E402,F401

REPRO = Path(__file__).resolve().parents[1] / "repro"
RESULTS_LOG = REPRO / "MASTER_RESULTS.jsonl"


def metrics(fake_probs: torch.Tensor, labels: torch.Tensor, rank: np.ndarray | None = None,
            preds: torch.Tensor | None = None):
    """acc@0.5 on ``fake_probs`` (or on ``preds``, 0/1 fake decisions, for a detector with its own decision rule); AP /
    AP_real / AUROC on ``fake_probs`` (torchmetrics, unchanged) or, when ``rank`` (a float64 raw score, higher = more
    likely fake, e.g. the logit) is given, on ``rank`` with scikit-learn, so that saturated probabilities do not tie."""
    from torchmetrics.functional import auroc, average_precision
    if labels.unique().numel() < 2:
        return None
    if rank is None:
        ap = float(average_precision(fake_probs, labels, task="binary")) * 100
        ap_real = float(average_precision(1 - fake_probs, 1 - labels, task="binary")) * 100
        au = float(auroc(fake_probs, labels, task="binary")) * 100
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y = labels.numpy()
        ap = float(average_precision_score(y, rank)) * 100
        ap_real = float(average_precision_score(1 - y, -rank)) * 100
        au = float(roc_auc_score(y, rank)) * 100
    preds = (fake_probs >= 0.5).int() if preds is None else preds.int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "ap_real": round(ap_real, 2), "auroc": round(au, 2), "real_acc": round(real_acc, 2),
            "fake_acc": round(fake_acc, 2), "acc": round((real_acc + fake_acc) / 2, 2)}


FPRS = (0.001, 0.01, 0.05)


CACHE_FORMAT = 2        # 2: error records carry "kind" ("decode" is cached, anything else is retried)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA256_MEMO: dict = {}


def _sha256_memo(path: Path) -> str:
    """``_sha256`` memoized by (resolved path, size, mtime), so a run hashes each big file (the NSG-VD diffusion model and
    reference set) at most once however often ``cache_meta`` is called; a file that changes is hashed again."""
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA256_MEMO:
        _SHA256_MEMO[key] = _sha256(path)
    return _SHA256_MEMO[key]


def is_agent(spec) -> bool:
    """An agentic video detector (``predict_kind="video_agent"``, WeVisForensics): the official agent in its own process,
    one investigation per video, with its own verdict and error kinds (see ``score_agent``)."""
    return getattr(spec, "predict_kind", "") == "video_agent"


def agent_meta(spec) -> dict:
    """What an agent's scores depend on, instead of a checkpoint hash: its checkpoint is a FOLDER (adapters on a separate
    base), so the cache is keyed by the pinned release manifest (``models/video/wevis/manifest.json``: the code commit and
    the sha256 of every code / adapter / base / OCR file, which the runner verifies before it serves) plus the run settings
    (audio flag, runtime pins, verdict-token pairs)."""
    from models.video.wevis import wevis as agent_mod
    man = agent_mod.load_manifest()
    cfg = spec.agent
    return {"manifest_sha256": _sha256_memo(agent_mod.MANIFEST), "code_commit": man["code"]["commit"],
            "adapters_revision": man["adapters"]["revision"], "base_revision": man["base"]["revision"],
            "has_audio": bool(cfg.has_audio), "pins": {k: v for k, v in cfg.pins},
            "verdict_pairs": [list(x) for x in cfg.verdict_pairs]}


def cache_meta(engine, spec) -> dict:
    """What a score cache is valid for: the detector key, the sha256 of the resolved checkpoint, and every spec field that
    changes a video's score (frame count, sampler, preprocessing); for a detector that declares them, the sha256 of its
    extra files (``aux_files``) and its pinned ffmpeg build (version + binary sha256). For an agent (``is_agent``) the
    checkpoint is a folder, which is never hashed: ``ckpt_sha256`` is the sha256 of its pinned release manifest and
    ``agent`` holds the pins (``agent_meta``)."""
    if is_agent(spec):
        am = agent_meta(spec)
        return {"model": spec.key, "ckpt_sha256": am["manifest_sha256"], "agent": am,
                "spec": {"num_frames": spec.num_frames, "video_sampler": spec.video_sampler},
                "cache_format": CACHE_FORMAT}
    ck = engine.checkpoint_path(spec)
    meta = {"model": spec.key,
            "ckpt_sha256": _sha256_memo(Path(ck)) if ck is not None else None,
            "spec": {"num_frames": spec.num_frames, "video_sampler": spec.video_sampler,
                     "transform": repr(spec.transform),
                     **{k: repr(getattr(spec, k, None)) for k in ("fps", "frame_size", "window_center", "video_norm")}},
            "cache_format": CACHE_FORMAT}
    # Only for detectors that declare them (NSG-VD), so the caches of the others keep their identity: the extra files the
    # model reads (diffusion model, reference set) and the pinned ffmpeg build its frames depend on.
    if getattr(spec, "aux_files", None):
        meta["aux_sha256"] = {name: _sha256_memo(backend_config.CHECKPOINTS_DIR / name)
                              for name in sorted(spec.aux_files.values())}
    if getattr(spec, "ffmpeg_version", ""):
        ff = spec_ffmpeg(spec)
        exe = shutil.which(ff["path"]) or ff["path"]     # a bare command name ("ffmpeg") is looked up on the PATH
        meta["ffmpeg"] = {"version": ff["version"], "sha256": _sha256_memo(Path(exe))}
    return meta


def own_verdict(spec) -> bool:
    """True for a detector whose decision is its engine ``verdict``, not P(fake) >= 0.5, and whose training-free raw score
    is its reported logit: NSG-VD (``video_prob == "mmd"``; official test_dMMD.py: fake iff the MMD statistic > 1, so a
    statistic of exactly 1 is real while P(fake) = sigmoid(0) = 0.5). False for every other detector (unchanged)."""
    return getattr(spec, "video_prob", "") == "mmd"


def verdict_preds(recs) -> torch.Tensor:
    """The cached engine verdicts as 0/1 fake decisions (``own_verdict`` detectors); refuses records without one."""
    if not all("verdict" in r for r in recs):
        raise SystemExit("score records without an engine verdict (cache written before verdicts were stored): rescore "
                         "into a new --cache-dir")
    return torch.tensor([int(r["verdict"] == "fake") for r in recs])


class ScoreCache:
    """Per-video score store shared by shards and re-runs: DIR/scores.<tag>.jsonl, one JSON per line, keyed by the last
    three components of the video path. All ``scores.*.jsonl`` in DIR are read; only ``scores.<tag>`` is written (created
    on the first write). DIR/cache_meta.json records what the scores are valid for (``cache_meta``): the first use writes
    it, a later use with a different model / checkpoint / spec / format is refused."""

    def __init__(self, cache_dir, tag: str, meta: dict | None = None, write_meta=True):
        self.rec, self.only, self.tag = {}, False, tag
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if meta is not None:
            self._check_meta(meta, write_meta)
        for f in sorted(self.dir.glob("scores.*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rec[r["k"]] = r
        self._out = self._traces = None

    def _check_meta(self, meta: dict, write: bool):
        mp = self.dir / "cache_meta.json"
        if not mp.exists() and write:
            tmp = mp.with_name(f".cache_meta.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, mp)                          # shards racing here all write the same content
        if not mp.exists():
            raise SystemExit(f"{self.dir} has no cache_meta.json: refusing to read scores of unknown origin")
        have = json.loads(mp.read_text())
        want = json.loads(json.dumps(meta))
        diff = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
        if diff:
            raise SystemExit(f"score cache {self.dir} was written for a different setup, refusing to reuse it "
                             f"(cache value, current value): {json.dumps(diff, sort_keys=True)}")

    def key(self, p: Path) -> str:
        return "/".join(p.parts[-3:])        # <split dir>/<generator or class>/<file>: the same on every node

    def get(self, p: Path):
        return self.rec.get(self.key(p))

    def put(self, p: Path, r: dict):
        r = {"k": self.key(p), **r}
        self.rec[r["k"]] = r
        if self._out is None:
            self._out = (self.dir / f"scores.{self.tag}.jsonl").open("a", buffering=1)
        self._out.write(json.dumps(r) + "\n")

    def put_trace(self, p: Path, trace) -> None:
        """An agent's full investigation (every turn's raw text, the tool observations) for ``p``: DIR/traces.<tag>.jsonl,
        kept apart from the scores so that reading the cache stays cheap."""
        if self._traces is None:
            self._traces = (self.dir / f"traces.{self.tag}.jsonl").open("a", buffering=1)
        self._traces.write(json.dumps({"k": self.key(p), "trace": trace}, default=str) + "\n")


def is_decode_error(r: dict) -> bool:
    """A genuine decode failure (cached). Records written before ``kind`` existed: the decoder's own message."""
    return "err" in r and (r.get("kind") == "decode" or ("kind" not in r and r["err"].startswith("ValueError: Could not decode")))


AGENT_RETRIES = {"oom": 2, "tool_error": 1, "runner": 1}
AGENT_FINAL = ("tool_error", "oom")       # cached; out of the official accuracy (with their count), wrong in the strict one


def is_agent_final_error(r: dict) -> bool:
    """An agent video that still failed after its retries with the official ``tool_error`` (or, after its OOM retries,
    out of memory): cached, left out of the official accuracy's denominator and counted wrong in the strict variant."""
    return "err" in r and r.get("kind") in AGENT_FINAL


def agent_gpu(engine, spec):
    """The GPU the agent's runner process runs on (its ``ready`` info), recorded on every row."""
    try:
        return (getattr(engine._get_model(spec), "info", None) or {}).get("gpu")
    except Exception:  # noqa: BLE001 — a runner that cannot start: no GPU name
        return None


def score_agent(engine, spec, p: Path, cache: ScoreCache | None = None, retries: dict | None = None,
                run_tag: str = "") -> dict:
    """One video through an agentic detector (``is_agent``). Success: {"p", "logit" (the verdict-token logit gap), "verdict"
    ("real" / "fake" / "unparsed": no valid answer, counted wrong for both classes), "score_source", "status",
    "verdict_token", "tool_calls", "seconds", "model_calls", "generated_tokens", "peak_alloc_gib", "gpu", "retries"} (the
    full trace goes to the cache's traces file). Errors are retried in place, per kind, up to ``retries`` (default
    ``AGENT_RETRIES``): "oom" (``WeVisOOMError``: the runner frees the GPU and the video is run again), "tool_error"
    (the official ``tool_error``: any other exception of the agent) and "runner" (the runner died or timed out: restarted on
    the next call). A video that still fails is returned as {"err", "kind", "retries", "gpu"}; "tool_error" / "oom"
    records are cached (final for a ``--cache-only`` aggregation; a scoring run retries them, as the official
    ``evaluate.py`` re-runs its tool_error rows), "runner" / "other" are not. Every attempt prints one log line."""
    retries = AGENT_RETRIES if retries is None else retries
    key = "/".join(p.parts[-3:])
    r = cache.get(p) if cache else None
    if r is not None:
        if "err" not in r or (cache.only and is_agent_final_error(r)):
            return r
        r = None                                         # a cached tool_error / oom: run again (official retry)
    if cache is not None and cache.only:
        raise SystemExit(f"--cache-only: {p} is not in the score cache")
    tries = {"oom": 0, "tool_error": 0, "runner": 0}
    while True:
        try:
            res = engine.predict_video_clip(spec, str(p))
            break
        except Exception as e:  # noqa: BLE001
            kind = getattr(e, "kind", None)
            kind = kind if kind in tries else "other"
            msg = f"{type(e).__name__}: {e}"[:300]
            if kind == "other" or tries[kind] >= retries.get(kind, 0):
                r = {"k": key, "err": msg, "kind": kind, "retries": dict(tries), "gpu": agent_gpu(engine, spec)}
                print(f"    agent[{run_tag}] {key} FAILED kind={kind} retries={_fmt_retries(tries)} gpu={r['gpu']}: {msg}",
                      flush=True)
                if cache is not None and kind in AGENT_FINAL:
                    cache.put(p, r)
                return r
            tries[kind] += 1
            print(f"    agent[{run_tag}] {key} RETRY kind={kind} {tries[kind]}/{retries[kind]}: {msg}", flush=True)
    cost, trace = res.get("cost") or {}, res.get("trace") or {}
    r = {"k": key, "p": res["fake_probability"], "logit": res["logit"], "verdict": res["verdict"],
         "score_source": res.get("score_source"), "status": res.get("status"), "verdict_token": res.get("verdict_token"),
         "tool_calls": trace.get("tool_calls"), "seconds": cost.get("seconds"), "model_calls": cost.get("model_calls"),
         "generated_tokens": cost.get("generated_tokens"), "peak_alloc_gib": cost.get("peak_alloc_gib"),
         "gpu": agent_gpu(engine, spec), "retries": dict(tries)}
    print(f"    agent[{run_tag}] {key} verdict={r['verdict']} logit={r['logit']!r} src={r['score_source']} "
          f"status={r['status']} calls={r['model_calls']} tok={r['generated_tokens']} s={r['seconds']} "
          f"peak={r['peak_alloc_gib']} gpu={r['gpu']} retries={_fmt_retries(tries)}", flush=True)
    if cache is not None:
        cache.put(p, r)
        cache.put_trace(p, trace)
    return r


def _fmt_retries(tries: dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in tries.items())


@torch.no_grad()
def score_one(engine, spec, p: Path, raw=False, cache: ScoreCache | None = None, retries: dict | None = None,
              run_tag: str = "") -> dict:
    """{"p": P(fake), "logit": raw logit (only if the detector reports one), "verdict": the engine's "fake" / "real" (only
    for an ``own_verdict`` detector), "raw": training-free score (if ``raw``; an ``own_verdict`` detector's logit, else
    engine.video_raw_score; if only that raises, "raw_err" and the video still counts)} or {"err", "kind"} when the video
    cannot be scored (kind "decode": cached; anything else, e.g. OOM / I/O: not cached, retried next time). An agentic
    detector goes through ``score_agent``."""
    if is_agent(spec):
        return score_agent(engine, spec, p, cache, retries, run_tag)
    r = cache.get(p) if cache else None
    if r is not None and ("err" in r or not raw or "raw" in r or "raw_err" in r):
        if "err" not in r or is_decode_error(r):
            return r
        r = None                                         # a legacy / non-decode error record: retry
    if r is None and cache is not None and cache.only:
        raise SystemExit(f"--cache-only: {p} is not in the score cache")
    try:
        if r is None:
            res = engine.predict_video_clip(spec, str(p))
            r = {"p": res["fake_probability"]}
            if "logit" in res:
                r["logit"] = res["logit"]
            if own_verdict(spec):
                r["verdict"] = res["verdict"]
        if raw:
            try:
                r["raw"] = r["logit"] if own_verdict(spec) else engine.video_raw_score(spec, str(p))
            except Exception as e:  # noqa: BLE001      # old behaviour: dropped from the raw scores only
                r["raw_err"] = f"{type(e).__name__}: {e}"[:300]
                print(f"    no raw score for {p.name}: {r['raw_err']}", flush=True)
    except Exception as e:  # noqa: BLE001
        r = {"err": f"{type(e).__name__}: {e}"[:300], "kind": "decode" if isinstance(e, VideoDecodeError) else "other"}
        print(f"    bad video {p.name} ({r['kind']}): {r['err']}", flush=True)
        if r["kind"] != "decode":
            return r
    if cache:
        cache.put(p, r)
    return r


def score_videos(engine, spec, paths, raw=False, cache: ScoreCache | None = None, label="", errs: list | None = None,
                 retries: dict | None = None, run_tag: str = ""):
    """Score ``paths`` -> (records of the videos that scored, decode failures, other failures). With ``errs`` (a list), an
    agent's final tool_error / oom records (``is_agent_final_error``) are appended to it instead of counted as other."""
    out, bad, other, t0 = [], 0, 0, time.time()
    every = 25 if is_agent(spec) else 250
    for i, p in enumerate(paths, 1):
        r = score_one(engine, spec, p, raw, cache, retries, run_tag)
        if "err" not in r:
            r.setdefault("k", "/".join(p.parts[-3:]))     # the cache key, also without a cache (per-video records)
            out.append(r)
        elif is_decode_error(r):
            bad += 1
        elif errs is not None and is_agent_final_error(r):
            errs.append(r)
        else:
            other += 1
        if i % every == 0:
            print(f"    {label} {i}/{len(paths)} ({(time.time() - t0) / i:.2f} s/video, {bad} decode + {other} other failures)",
                  flush=True)
    return out, bad, other


def rank_scores(recs) -> np.ndarray | None:
    """The float64 raw logits to rank by (AP / AUROC) when every record has one, else None (rank by the P(fake))."""
    if recs and all("logit" in r for r in recs):
        return np.array([r["logit"] for r in recs], dtype="float64")
    return None


def unique_videos(gens, pool_key) -> list[Path]:
    """The videos to score once: each distinct real pool (the same pool is hard-linked per generator) and every fake."""
    jobs, seen = [], set()
    for _, reals, fakes in gens:
        if pool_key(reals) not in seen:
            seen.add(pool_key(reals))
            jobs += reals
        jobs += fakes
    return jobs


def fake_recall_at_fpr(real_scores, fake_scores):
    """SPLIT's operating-point metric (official eval.py): the threshold is the (1 - FPR) quantile of the real scores
    (numpy method "higher"); fake recall = share of fakes scoring at or above it. In percent, per FPR."""
    real, fake = np.asarray(real_scores, dtype="float64"), np.asarray(fake_scores, dtype="float64")
    return {f"{a * 100:g}": round(float((fake >= np.quantile(real, 1 - a, method="higher")).mean()) * 100, 2) for a in FPRS}


def agent_metrics(rrec, frec, rerr, ferr):
    """Per-generator metrics of an agent (``is_agent``). AP / AP_real / AUROC rank the scored videos by the verdict-token
    logit gap (``metrics``); accuracy is the agent's own verdict: a video is right iff its verdict equals its label, so an
    unparsed answer is wrong for both classes. ``acc`` (official, ``scripts/evaluate.py``): final tool_error / oom videos
    (``rerr`` / ``ferr``) are left out of the denominator; ``acc_strict``: they count wrong."""
    labels = torch.tensor([0] * len(rrec) + [1] * len(frec))
    m = metrics(torch.tensor([r["p"] for r in rrec + frec]), labels,
                np.array([r["logit"] for r in rrec + frec], dtype="float64"))
    if m is None:
        return None
    rc, fc = sum(r["verdict"] == "real" for r in rrec), sum(r["verdict"] == "fake" for r in frec)
    ra, fa = 100.0 * rc / len(rrec), 100.0 * fc / len(frec)                 # both classes are present (metrics above)
    ras, fas = 100.0 * rc / (len(rrec) + len(rerr)), 100.0 * fc / (len(frec) + len(ferr))
    m.update(real_acc=round(ra, 2), fake_acc=round(fa, 2), acc=round((ra + fa) / 2, 2),
             real_acc_strict=round(ras, 2), fake_acc_strict=round(fas, 2), acc_strict=round((ras + fas) / 2, 2))
    m.update(acc_by="agent_verdict", rank_by="logit",
             unparsed_real=sum(r["verdict"] not in ("real", "fake") for r in rrec),
             unparsed_fake=sum(r["verdict"] not in ("real", "fake") for r in frec),
             unscored=sum(r.get("score_source") == "unscored" for r in rrec + frec),
             tool_error_real=sum(e["kind"] == "tool_error" for e in rerr),
             tool_error_fake=sum(e["kind"] == "tool_error" for e in ferr),
             oom_real=sum(e["kind"] == "oom" for e in rerr), oom_fake=sum(e["kind"] == "oom" for e in ferr))
    return m


def pooled_block(spec, reals, fakes, rerr=(), ferr=()) -> dict:
    """Accuracy over the DISTINCT videos, one row per video (a real pool shared by several generators counted once): the
    official WeVisForensics number (``scripts/evaluate.py`` / ``core/metrics.py``: fake is the positive class; recall and
    F1 of the fakes). The decision is the agent's / NSG-VD's verdict or P(fake) >= 0.5. ``acc`` leaves the final
    tool_error / oom videos out (official), ``acc_strict`` counts them wrong. For an agent also the error / retry counts,
    the GPU of every row and the mean cost."""
    def says_fake(r):
        return r["verdict"] == "fake" if (is_agent(spec) or own_verdict(spec)) else r["p"] >= 0.5

    def says_real(r):
        return r["verdict"] == "real" if (is_agent(spec) or own_verdict(spec)) else r["p"] < 0.5
    tp, fn = sum(says_fake(r) for r in fakes), sum(not says_fake(r) for r in fakes)
    tn, fp = sum(says_real(r) for r in reals), sum(says_fake(r) for r in reals)
    n, n_err = len(reals) + len(fakes), len(rerr) + len(ferr)
    pct = lambda a, b: round(100.0 * a / b, 2) if b else None             # noqa: E731
    out = {"n": n, "n_real": len(reals), "n_fake": len(fakes), "acc": pct(tp + tn, n), "acc_strict": pct(tp + tn, n + n_err),
           "real_acc": pct(tn, len(reals)), "fake_recall": pct(tp, len(fakes)),
           "fake_f1": pct(2 * tp, 2 * tp + fp + fn)}
    if is_agent(spec):
        rows = list(reals) + list(fakes)
        allr = rows + list(rerr) + list(ferr)
        gpus = {}
        for r in allr:
            gpus[str(r.get("gpu"))] = gpus.get(str(r.get("gpu")), 0) + 1
        out.update(n_tool_error=sum(e["kind"] == "tool_error" for e in list(rerr) + list(ferr)),
                   n_oom=sum(e["kind"] == "oom" for e in list(rerr) + list(ferr)),
                   n_unparsed=sum(r["verdict"] not in ("real", "fake") for r in rows),
                   n_unscored=sum(r.get("score_source") == "unscored" for r in rows),
                   oom_retries=sum((r.get("retries") or {}).get("oom", 0) for r in allr),
                   tool_error_retries=sum((r.get("retries") or {}).get("tool_error", 0) for r in allr),
                   runner_retries=sum((r.get("retries") or {}).get("runner", 0) for r in allr),
                   n_retried_oom=sum(bool((r.get("retries") or {}).get("oom")) for r in allr),
                   gpus=gpus,
                   mean_seconds=round(sum(r["seconds"] for r in rows) / len(rows), 2) if rows else None,
                   mean_model_calls=round(sum(r["model_calls"] for r in rows) / len(rows), 2) if rows else None,
                   max_peak_alloc_gib=max((r["peak_alloc_gib"] for r in rows if r.get("peak_alloc_gib") is not None),
                                          default=None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, choices=["d3", "split", "restrav", "stall", "vina_video", "waverep",
                                                                 "waverep_g4", "nsg_vd", "nsg_vd_pika", "nsg_vd_unbalance",
                                                                 "wevis"])
    ap.add_argument("--benchmark", choices=["genvideo", "fakeonly", "pergen"], help="layout (not needed with --dataset)")
    ap.add_argument("--data", help="benchmark root (not needed with --dataset)")
    ap.add_argument("--dataset", help="a supported video dataset (genvideo, videophy, vidprom, evalcrafter, ...): downloads "
                                      "it with its loader if missing, then evaluates its test split")
    ap.add_argument("--root", help="with --dataset: where the dataset lives / should be downloaded")
    ap.add_argument("--no-download", action="store_true", help="with --dataset: fail instead of downloading missing data")
    ap.add_argument("--reals", default=None)
    ap.add_argument("--select", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap videos per class per generator")
    ap.add_argument("--tag", help="run tag (output filename + log); default <model>_<dataset>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dir", default=None, help="store / reuse the per-video scores here (resumable, see above)")
    ap.add_argument("--cache-only", action="store_true", help="with --cache-dir: fail on a video that is not cached "
                                                              "instead of scoring it (the final aggregation run)")
    ap.add_argument("--list-out", default=None, help="with --shard: write the slice's video paths (relative to --base) to "
                                                     "this file and stop (no model needed); for staging only those files")
    ap.add_argument("--score-list", default=None, help="score exactly the videos listed in this file (paths relative to "
                                                       "--base) into --cache-dir and stop (a staged slice, no discovery)")
    ap.add_argument("--base", default=None, help="with --list-out / --score-list: the directory the listed paths are relative to")
    ap.add_argument("--shard", default=None, help="K/N: score only slice K of N of the unique videos into --cache-dir, "
                                                  "then stop (no metrics)")
    ap.add_argument("--per-video", action="store_true", help="also write every distinct video's record (key, label, "
                                                                 "score; for an agent its verdict, tool calls, cost, GPU) and the "
                                                                 "pooled one-row-per-video metrics (always on for an agent)")
    ap.add_argument("--results-log", default=str(RESULTS_LOG), help="append the summary line to this JSONL")
    ap.add_argument("--agent-oom-retries", type=int, default=AGENT_RETRIES["oom"],
                    help="agent: run a video again this often after an out-of-memory error")
    ap.add_argument("--agent-tool-retries", type=int, default=AGENT_RETRIES["tool_error"],
                    help="agent: run a video again this often after a tool_error (the official evaluate.py: once)")
    ap.add_argument("--agent-runner-retries", type=int, default=AGENT_RETRIES["runner"],
                    help="agent: run a video again this often after the runner died or timed out")
    args = ap.parse_args(argv)
    retries = {"oom": args.agent_oom_retries, "tool_error": args.agent_tool_retries, "runner": args.agent_runner_retries}
    if args.shard and not (args.cache_dir or args.list_out):
        raise SystemExit("--shard needs --cache-dir (or --list-out)")
    if args.list_out and not (args.shard and args.base):
        ap.error("--list-out needs --shard K/N and --base")

    if args.score_list:
        if not (args.cache_dir and args.base):
            raise SystemExit("--score-list needs --cache-dir and --base")
        engine, spec = get_engine(), get_spec(args.model)
        engine._get_model(spec)
        paths = [Path(args.base) / line.strip() for line in Path(args.score_list).read_text().splitlines() if line.strip()]
        cache = ScoreCache(args.cache_dir, Path(args.score_list).stem, cache_meta(engine, spec))
        print(f"score cache {args.cache_dir}: {len(cache.rec)} videos already scored; {len(paths)} listed | device={engine.device}",
              flush=True)
        raw = not is_agent(spec) and hasattr(engine._get_model(spec), "score")
        recs, bad, other = score_videos(engine, spec, paths, raw, cache, label="list", retries=retries,
                                        run_tag=Path(args.score_list).stem)
        print(f"=== LIST DONE {args.model} {Path(args.score_list).name}: scored {len(recs)} ok + {bad} decode failures + "
              f"{other} other failures of {len(paths)} ===", flush=True)
        return 0

    if args.dataset:
        droot = ensure_dataset(args.dataset, "video", args.root, download=not args.no_download)
        if args.dataset == "evalcrafter":                    # fake-only: test/1_fake/<generator>, reals via --reals
            args.data, args.benchmark = str(droot / "test" / "1_fake"), "fakeonly"
            if not args.reals:
                raise SystemExit("evalcrafter has no real videos: pass --reals DIR (D3 uses GenVideo-Val's MSR-VTT reals)")
        else:
            args.data, args.benchmark = str(droot / "test"), args.benchmark or "pergen"
        args.tag = args.tag or f"{args.model}_{args.dataset}"
    if not (args.data and args.benchmark and args.tag):
        raise SystemExit("pass --dataset, or --data, --benchmark and --tag")

    root = Path(args.data)
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root, Path(args.reals) if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake videos under {root} ({args.benchmark})")
        return 1
    if args.limit:
        gens = [(n, r[:args.limit], f[:args.limit]) for n, r, f in gens]
    pool_key = lambda reals: (len(reals), reals[0].name, reals[-1].name)    # the same pool hard-linked per generator
    if args.list_out:
        k, n = (int(x) for x in args.shard.split("/"))
        jobs = unique_videos(gens, pool_key)
        Path(args.list_out).write_text("".join(f"{p.relative_to(args.base)}\n" for p in jobs[k::n]))
        print(f"{args.list_out}: shard {k}/{n} = {len(jobs[k::n])} of {len(jobs)} unique videos")
        return 0
    engine = get_engine()
    spec = get_spec(args.model)
    engine._get_model(spec)
    print(f"=== bench_eval_video {args.tag} | {args.model} | {args.benchmark} | {len(gens)} gens "
          f"| device={engine.device} ===", flush=True)

    # training-free detectors: also fake recall at fixed FPR (not an agent, whose runner client also has a ``score``)
    raw = not is_agent(spec) and hasattr(engine._get_model(spec), "score")
    agent = is_agent(spec)
    if agent:
        print(f"agent {args.model}: GPU {agent_gpu(engine, spec)}, retries {retries}", flush=True)
    meta = cache_meta(engine, spec)
    cache = None
    if args.cache_dir:
        tag = f"{args.shard.replace('/', 'of')}" if args.shard else f"agg{os.getpid()}"
        cache = ScoreCache(args.cache_dir, tag, meta, write_meta=not args.cache_only)
        cache.only = args.cache_only
        print(f"score cache {args.cache_dir}: {len(cache.rec)} videos already scored", flush=True)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        jobs = unique_videos(gens, pool_key)
        mine = jobs[k::n]
        print(f"shard {k}/{n}: {len(mine)} of {len(jobs)} unique videos", flush=True)
        errs = []
        recs, bad, other = score_videos(engine, spec, mine, raw, cache, label=f"shard{k}", errs=errs, retries=retries,
                                        run_tag=args.tag)
        print(f"=== SHARD DONE {args.tag} {k}/{n}: scored {len(recs)} ok + {bad} decode failures + {other} other failures "
              f"of {len(mine)} ===" + (f" (agent: {len(errs)} final tool_error/oom)" if agent else ""), flush=True)
        if agent and other:
            return 2                                      # runner / other errors are not cached: the shard is incomplete
        return 0

    per_gen, agg = {}, {"ap": [], "ap_real": [], "auroc": [], "acc": []}
    pooled_fake_raw = []                                 # every generator's fakes, for the pooled ("Overall") recall
    real_cache = {}                                      # score the shared real pool once if it is identical across generators
    failed = {"fake": 0, "real_unique": 0}                 # decode failures (kind "decode") only
    failed_other = {"fake": 0, "real_unique": 0}           # any other error (OOM, I/O): dropped, never cached
    all_fakes, all_ferr = [], []                         # every generator's fakes (distinct), for the pooled block
    if agent:
        agg["acc_strict"] = []
    for name, reals, fakes in gens:
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        key = pool_key(reals)
        if key in real_cache:
            rrec, rbad, rother, rerr = real_cache[key]
        else:
            rerr = []
            rrec, rbad, rother = score_videos(engine, spec, reals, raw, cache, label=f"{name} real", errs=rerr,
                                              retries=retries, run_tag=args.tag)
            real_cache[key] = (rrec, rbad, rother, rerr)
            failed["real_unique"] += rbad
            failed_other["real_unique"] += rother
        ferr = []
        frec, fbad, fother = score_videos(engine, spec, fakes, raw, cache, label=f"{name} fake", errs=ferr,
                                          retries=retries, run_tag=args.tag)
        failed["fake"] += fbad
        failed_other["fake"] += fother
        all_fakes += [dict(r, gen=name) for r in frec]
        all_ferr += [dict(r, gen=name) for r in ferr]
        if agent:
            m = agent_metrics(rrec, frec, rerr, ferr)
            rank = "logit"
        else:
            probs = torch.tensor([r["p"] for r in rrec + frec])
            labels = torch.tensor([0] * len(rrec) + [1] * len(frec))
            rank = rank_scores(rrec + frec)                # the raw logit, not the rounded P(fake), when there is one
            preds = verdict_preds(rrec + frec) if own_verdict(spec) else None     # NSG-VD: the official MMD > 1
            m = metrics(probs, labels, rank, preds)
        if m is None:
            print(f"    skip {name}: one class"); continue
        m.update({"n_real": len(rrec), "n_fake": len(frec), "failed_real": rbad, "failed_fake": fbad,
                  "failed_other_real": rother, "failed_other_fake": fother,
                  "rank_by": "logit" if rank is not None else "fake_probability"})
        if not agent and preds is not None:                # only for own_verdict detectors (others: JSON unchanged)
            m["acc_by"] = "engine_verdict"
        if raw:
            rraw = [r["raw"] for r in rrec if "raw" in r]
            fraw = [r["raw"] for r in frec if "raw" in r]
            pooled_fake_raw += fraw
            m["fake_recall_at_fpr"] = fake_recall_at_fpr(rraw, fraw)
        per_gen[name] = m
        for k in agg:
            agg[k].append(m[k])
        print(f"    {name}: acc={m['acc']} AP={m['ap']} AP_real={m['ap_real']} AUROC={m['auroc']} "
              f"(n={len(rrec)}+{len(frec)}, failed {rbad}+{fbad} decode + {rother}+{fother} other, rank by {m['rank_by']})"
              + (f" acc_strict={m['acc_strict']} tool_error={m['tool_error_real']}+{m['tool_error_fake']} "
                 f"oom={m['oom_real']}+{m['oom_fake']} unparsed={m['unparsed_real']}+{m['unparsed_fake']}" if agent else ""),
              flush=True)

    if not per_gen:
        print("No generator produced metrics."); return 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2)
    means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "map_real": mean(agg["ap_real"]),
             "mauroc": mean(agg["auroc"])}
    if agent:
        means["acc_strict"] = mean(agg["acc_strict"])
    if raw:
        means["fake_recall_at_fpr"] = {k: mean([g["fake_recall_at_fpr"][k] for g in per_gen.values()])
                                       for k in next(iter(per_gen.values()))["fake_recall_at_fpr"]}
        # Overall: all fakes against the real pool(s) (each distinct pool once), as the SPLIT paper's "Overall" row
        pooled_real_raw = [r["raw"] for rrec, _, _, _ in real_cache.values() for r in rrec if "raw" in r]
        means["overall_fake_recall_at_fpr"] = fake_recall_at_fpr(pooled_real_raw, pooled_fake_raw)
    result = {"tag": args.tag, "model": args.model, "ckpt": "registry-default",
              "benchmark": f"video:{args.benchmark}", "data_root": str(root),
              "n_generators": len(per_gen), "means": means, "per_generator": per_gen, "failed_to_decode": failed,
              "failed_other": failed_other, "limit": args.limit, "reals": args.reals,
              "ckpt_sha256": meta["ckpt_sha256"], "cache_dir": args.cache_dir,
              "date": datetime.datetime.now().isoformat(timespec="seconds"),
              "synthscope_version": "video"}
    # Provenance of detectors that declare aux files / an ffmpeg pin (NSG-VD); absent for the others, so their JSON is unchanged.
    result.update({k: meta[k] for k in ("ffmpeg", "aux_sha256") if k in meta})
    if agent or args.per_video:                          # one row per distinct video (+ the pooled, official-style block)
        all_reals = [r for rrec, _, _, _ in real_cache.values() for r in rrec]
        all_rerr = [e for _, _, _, rerr in real_cache.values() for e in rerr]
        means["pooled"] = pooled_block(spec, all_reals, all_fakes, all_rerr, all_ferr)
        result["per_video"] = ([dict(r, label=0) for r in all_reals + all_rerr]
                               + [dict(r, label=1) for r in all_fakes + all_ferr])
    if agent:
        result["agent"] = meta["agent"]
        result["agent_retries"] = retries
    out = Path(args.out) if args.out else (REPRO / "bench_results" / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    with Path(args.results_log).open("a") as f:
        f.write(json.dumps({k: result[k] for k in
                ("date", "tag", "model", "ckpt", "benchmark", "data_root", "n_generators", "means")}) + "\n")
    extra = ""
    if "pooled" in means:
        pb = means["pooled"]
        extra = f" pooled_acc={pb['acc']} pooled_acc_strict={pb['acc_strict']} pooled_n={pb['n']}"
        if agent:
            extra = (f" acc_strict={means['acc_strict']}" + extra + f" tool_error={pb['n_tool_error']} oom={pb['n_oom']} "
                     f"unparsed={pb['n_unparsed']} oom_retries={pb['oom_retries']} "
                     f"gpus={'|'.join(sorted(g.replace(' ', '_') for g in pb['gpus']))}")
    print(f"\n=== RESULT {args.tag}: mAP={means['map']} mAP_real={means['map_real']} mAUROC={means['mauroc']} acc={means['acc']}"
          f"{extra} over {len(per_gen)} gens -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
