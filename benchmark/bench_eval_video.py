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
The result JSON also records limit, reals, ckpt_sha256 and cache_dir.

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

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".m4v"}
REPRO = Path(__file__).resolve().parents[1] / "repro"
RESULTS_LOG = REPRO / "MASTER_RESULTS.jsonl"
REAL_DIRNAMES = ("0_real", "real", "reals", "Real")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "Fake")


def _videos(d: Path) -> list[Path]:
    if not d or not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _find(parent: Path, names) -> Path | None:
    for n in names:
        if (parent / n).is_dir():
            return parent / n
    return None


def discover(benchmark: str, root: Path, shared_reals: Path | None, select):
    shared = _videos(shared_reals) if shared_reals else []
    out = []
    if benchmark == "genvideo":
        fake_root = _find(root, FAKE_DIRNAMES)
        real_root = _find(root, REAL_DIRNAMES)
        reals = _videos(real_root) or shared
        gens = sorted(p for p in fake_root.iterdir() if p.is_dir()) if fake_root else []
        for g in gens:
            if select and g.name not in select:
                continue
            fk = _videos(g)
            if fk and reals:
                out.append((g.name, reals, fk))
        return out
    if benchmark == "fakeonly":
        gens = sorted(p for p in root.iterdir() if p.is_dir())
        for g in gens:
            if select and g.name not in select:
                continue
            fk = _videos(g)
            if fk and shared:
                out.append((g.name, shared, fk))
        return out
    # pergen
    for g in sorted(p for p in root.iterdir() if p.is_dir()):
        if select and g.name not in select:
            continue
        reals = _videos(_find(g, REAL_DIRNAMES)) or shared
        fakes = _videos(_find(g, FAKE_DIRNAMES))
        if reals and fakes:
            out.append((g.name, reals, fakes))
    return out


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


def cache_meta(engine, spec) -> dict:
    """What a score cache is valid for: the detector key, the sha256 of the resolved checkpoint, and every spec field that
    changes a video's score (frame count, sampler, preprocessing); for a detector that declares them, the sha256 of its
    extra files (``aux_files``) and its pinned ffmpeg build (version + binary sha256)."""
    ck = engine.checkpoint_path(spec)
    meta = {"model": spec.key,
            "ckpt_sha256": _sha256(Path(ck).resolve()) if ck is not None else None,
            "spec": {"num_frames": spec.num_frames, "video_sampler": spec.video_sampler,
                     "transform": repr(spec.transform),
                     **{k: repr(getattr(spec, k, None)) for k in ("fps", "frame_size", "window_center", "video_norm")}},
            "cache_format": CACHE_FORMAT}
    # Only for detectors that declare them (NSG-VD), so the caches of the others keep their identity: the extra files the
    # model reads (diffusion model, reference set) and the pinned ffmpeg build its frames depend on.
    if getattr(spec, "aux_files", None):
        meta["aux_sha256"] = {name: _sha256((backend_config.CHECKPOINTS_DIR / name).resolve())
                              for name in sorted(spec.aux_files.values())}
    if getattr(spec, "ffmpeg_version", ""):
        ff = spec_ffmpeg(spec)
        exe = shutil.which(ff["path"]) or ff["path"]     # a bare command name ("ffmpeg") is looked up on the PATH
        meta["ffmpeg"] = {"version": ff["version"], "sha256": _sha256(Path(exe).resolve())}
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
        self._out = None

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


def is_decode_error(r: dict) -> bool:
    """A genuine decode failure (cached). Records written before ``kind`` existed: the decoder's own message."""
    return "err" in r and (r.get("kind") == "decode" or ("kind" not in r and r["err"].startswith("ValueError: Could not decode")))


@torch.no_grad()
def score_one(engine, spec, p: Path, raw=False, cache: ScoreCache | None = None) -> dict:
    """{"p": P(fake), "logit": raw logit (only if the detector reports one), "verdict": the engine's "fake" / "real" (only
    for an ``own_verdict`` detector), "raw": training-free score (if ``raw``; an ``own_verdict`` detector's logit, else
    engine.video_raw_score; if only that raises, "raw_err" and the video still counts)} or {"err", "kind"} when the video
    cannot be scored (kind "decode": cached; anything else, e.g. OOM / I/O: not cached, retried next time)."""
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


def score_videos(engine, spec, paths, raw=False, cache: ScoreCache | None = None, label=""):
    """Score ``paths`` -> (records of the videos that scored, decode failures, other failures)."""
    out, bad, other, t0 = [], 0, 0, time.time()
    for i, p in enumerate(paths, 1):
        r = score_one(engine, spec, p, raw, cache)
        if "err" not in r:
            out.append(r)
        elif is_decode_error(r):
            bad += 1
        else:
            other += 1
        if i % 250 == 0:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, choices=["d3", "split", "restrav", "stall", "vina_video", "waverep",
                                                                 "waverep_g4", "nsg_vd", "nsg_vd_pika", "nsg_vd_unbalance"])
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
    args = ap.parse_args(argv)
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
        recs, bad, other = score_videos(engine, spec, paths, hasattr(engine._get_model(spec), "score"), cache, label="list")
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

    raw = hasattr(engine._get_model(spec), "score")     # training-free detectors: also fake recall at fixed FPR
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
        recs, bad, other = score_videos(engine, spec, mine, raw, cache, label=f"shard{k}")
        print(f"=== SHARD DONE {args.tag} {k}/{n}: scored {len(recs)} ok + {bad} decode failures + {other} other failures "
              f"of {len(mine)} ===", flush=True)
        return 0

    per_gen, agg = {}, {"ap": [], "ap_real": [], "auroc": [], "acc": []}
    pooled_fake_raw = []                                 # every generator's fakes, for the pooled ("Overall") recall
    real_cache = {}                                      # score the shared real pool once if it is identical across generators
    failed = {"fake": 0, "real_unique": 0}                 # decode failures (kind "decode") only
    failed_other = {"fake": 0, "real_unique": 0}           # any other error (OOM, I/O): dropped, never cached
    for name, reals, fakes in gens:
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        key = pool_key(reals)
        if key in real_cache:
            rrec, rbad, rother = real_cache[key]
        else:
            rrec, rbad, rother = score_videos(engine, spec, reals, raw, cache, label=f"{name} real")
            real_cache[key] = (rrec, rbad, rother)
            failed["real_unique"] += rbad
            failed_other["real_unique"] += rother
        frec, fbad, fother = score_videos(engine, spec, fakes, raw, cache, label=f"{name} fake")
        failed["fake"] += fbad
        failed_other["fake"] += fother
        probs = torch.tensor([r["p"] for r in rrec + frec])
        labels = torch.tensor([0] * len(rrec) + [1] * len(frec))
        rank = rank_scores(rrec + frec)                    # the raw logit, not the rounded P(fake), when there is one
        preds = verdict_preds(rrec + frec) if own_verdict(spec) else None     # NSG-VD: the official MMD > 1
        m = metrics(probs, labels, rank, preds)
        if m is None:
            print(f"    skip {name}: one class"); continue
        m.update({"n_real": len(rrec), "n_fake": len(frec), "failed_real": rbad, "failed_fake": fbad,
                  "failed_other_real": rother, "failed_other_fake": fother,
                  "rank_by": "logit" if rank is not None else "fake_probability"})
        if preds is not None:                              # only for own_verdict detectors (others: JSON unchanged)
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
              f"(n={len(rrec)}+{len(frec)}, failed {rbad}+{fbad} decode + {rother}+{fother} other, rank by {m['rank_by']})", flush=True)

    if not per_gen:
        print("No generator produced metrics."); return 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2)
    means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "map_real": mean(agg["ap_real"]),
             "mauroc": mean(agg["auroc"])}
    if raw:
        means["fake_recall_at_fpr"] = {k: mean([g["fake_recall_at_fpr"][k] for g in per_gen.values()])
                                       for k in next(iter(per_gen.values()))["fake_recall_at_fpr"]}
        # Overall: all fakes against the real pool(s) (each distinct pool once), as the SPLIT paper's "Overall" row
        pooled_real_raw = [r["raw"] for rrec, _, _ in real_cache.values() for r in rrec if "raw" in r]
        means["overall_fake_recall_at_fpr"] = fake_recall_at_fpr(pooled_real_raw, pooled_fake_raw)
    result = {"tag": args.tag, "model": args.model, "ckpt": "registry-default",
              "benchmark": f"video:{args.benchmark}", "data_root": str(root),
              "n_generators": len(per_gen), "means": means, "per_generator": per_gen, "failed_to_decode": failed,
              "failed_other": failed_other, "limit": args.limit, "reals": args.reals,
              "ckpt_sha256": meta["ckpt_sha256"], "cache_dir": args.cache_dir,
              "date": datetime.datetime.now().isoformat(timespec="seconds"),
              "synthscope_version": "video"}
    out = Path(args.out) if args.out else (REPRO / "bench_results" / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    with RESULTS_LOG.open("a") as f:
        f.write(json.dumps({k: result[k] for k in
                ("date", "tag", "model", "ckpt", "benchmark", "data_root", "n_generators", "means")}) + "\n")
    print(f"\n=== RESULT {args.tag}: mAP={means['map']} mAP_real={means['map_real']} mAUROC={means['mauroc']} acc={means['acc']} "
          f"over {len(per_gen)} gens -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
