"""Unified VIDEO benchmark eval for SynthScope video detectors (d3, split, restrav, stall, vina_video, waverep,
waverep_g4).

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
very confident videos at 0 / 1; accuracy still thresholds P(fake) at 0.5. Videos that fail to decode are counted
per generator and in the result JSON (they are left out of the metrics).

Benchmark layouts:
  genvideo   <data>/Fake/<gen>/*.mp4 + <data>/Real/*.mp4 (shared real pool)   [GenVideo-Val]
  fakeonly   <data>/<gen>/*.mp4 (all fake) + --reals DIR of real videos        [EvalCrafter]
  pergen     <data>/<gen>/{0_real,1_fake}/*.mp4                                 [GenVidBench]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interface"))
from backend.engine import get_engine          # noqa: E402
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


def metrics(fake_probs: torch.Tensor, labels: torch.Tensor, rank: np.ndarray | None = None):
    """acc@0.5 on ``fake_probs``; AP / AP_real / AUROC on ``fake_probs`` (torchmetrics, unchanged) or, when ``rank`` (a
    float64 raw score, higher = more likely fake, e.g. the logit) is given, on ``rank`` with scikit-learn, so that
    saturated probabilities do not tie."""
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
    preds = (fake_probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "ap_real": round(ap_real, 2), "auroc": round(au, 2), "real_acc": round(real_acc, 2),
            "fake_acc": round(fake_acc, 2), "acc": round((real_acc + fake_acc) / 2, 2)}


FPRS = (0.001, 0.01, 0.05)


class ScoreCache:
    """Per-video score store shared by shards and re-runs: DIR/scores.<tag>.jsonl, one JSON per line, keyed by the
    last three components of the video path. All ``scores.*.jsonl`` in DIR are read; only ``scores.<tag>`` is written."""

    def __init__(self, cache_dir, tag: str):
        self.rec, self.only = {}, False
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("scores.*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rec[r["k"]] = r
        self.out = (d / f"scores.{tag}.jsonl").open("a", buffering=1)

    def key(self, p: Path) -> str:
        return "/".join(p.parts[-3:])        # <split dir>/<generator or class>/<file>: the same on every node

    def get(self, p: Path):
        return self.rec.get(self.key(p))

    def put(self, p: Path, r: dict):
        r = {"k": self.key(p), **r}
        self.rec[r["k"]] = r
        self.out.write(json.dumps(r) + "\n")


@torch.no_grad()
def score_one(engine, spec, p: Path, raw=False, cache: ScoreCache | None = None, tag="") -> dict:
    """{"p": P(fake), "logit": raw logit (only if the detector reports one), "raw": training-free score (if ``raw``)}
    or {"err": message} when the video cannot be decoded / scored."""
    r = cache.get(p) if cache else None
    if r is not None and ("err" in r or not raw or "raw" in r):
        return r
    if r is None and cache is not None and cache.only:
        raise SystemExit(f"--cache-only: {p} is not in the score cache")
    try:
        res = engine.predict_video_clip(spec, str(p)) if r is None else {"fake_probability": r["p"], **({"logit": r["logit"]} if "logit" in r else {})}
        r = {"p": res["fake_probability"]}
        if "logit" in res:
            r["logit"] = res["logit"]
        if raw:
            r["raw"] = engine.video_raw_score(spec, str(p))
    except Exception as e:  # noqa: BLE001
        r = {"err": f"{type(e).__name__}: {e}"[:300]}
        print(f"    bad video {p.name}: {r['err']}", flush=True)
    if cache:
        cache.put(p, r)
    return r


def score_videos(engine, spec, paths, raw=False, cache: ScoreCache | None = None, label=""):
    """Score ``paths`` -> (records of the videos that scored, number that failed to decode / score)."""
    out, bad, t0 = [], 0, time.time()
    for i, p in enumerate(paths, 1):
        r = score_one(engine, spec, p, raw, cache)
        if "err" in r:
            bad += 1
        else:
            out.append(r)
        if i % 250 == 0:
            print(f"    {label} {i}/{len(paths)} ({(time.time() - t0) / i:.2f} s/video, {bad} failed)", flush=True)
    return out, bad


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
                                                                 "waverep_g4"])
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

    if args.score_list:
        if not (args.cache_dir and args.base):
            raise SystemExit("--score-list needs --cache-dir and --base")
        engine, spec = get_engine(), get_spec(args.model)
        engine._get_model(spec)
        paths = [Path(args.base) / line.strip() for line in Path(args.score_list).read_text().splitlines() if line.strip()]
        cache = ScoreCache(args.cache_dir, Path(args.score_list).stem)
        print(f"score cache {args.cache_dir}: {len(cache.rec)} videos already scored; {len(paths)} listed | device={engine.device}",
              flush=True)
        recs, bad = score_videos(engine, spec, paths, hasattr(engine._get_model(spec), "score"), cache, label="list")
        print(f"=== LIST DONE {args.model} {Path(args.score_list).name}: scored {len(recs)} ok + {bad} failed of {len(paths)} ===",
              flush=True)
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
    cache = None
    if args.cache_dir:
        tag = f"{args.shard.replace('/', 'of')}" if args.shard else f"agg{os.getpid()}"
        cache = ScoreCache(args.cache_dir, tag)
        cache.only = args.cache_only
        print(f"score cache {args.cache_dir}: {len(cache.rec)} videos already scored", flush=True)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        jobs = unique_videos(gens, pool_key)
        mine = jobs[k::n]
        print(f"shard {k}/{n}: {len(mine)} of {len(jobs)} unique videos", flush=True)
        recs, bad = score_videos(engine, spec, mine, raw, cache, label=f"shard{k}")
        print(f"=== SHARD DONE {args.tag} {k}/{n}: scored {len(recs)} ok + {bad} failed of {len(mine)} ===", flush=True)
        return 0

    per_gen, agg = {}, {"ap": [], "ap_real": [], "auroc": [], "acc": []}
    pooled_fake_raw = []                                 # every generator's fakes, for the pooled ("Overall") recall
    real_cache = {}                                      # score the shared real pool once if it is identical across generators
    failed = {"fake": 0, "real_unique": 0}
    for name, reals, fakes in gens:
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        key = pool_key(reals)
        if key in real_cache:
            rrec, rbad = real_cache[key]
        else:
            rrec, rbad = score_videos(engine, spec, reals, raw, cache, label=f"{name} real")
            real_cache[key] = (rrec, rbad)
            failed["real_unique"] += rbad
        frec, fbad = score_videos(engine, spec, fakes, raw, cache, label=f"{name} fake")
        failed["fake"] += fbad
        probs = torch.tensor([r["p"] for r in rrec + frec])
        labels = torch.tensor([0] * len(rrec) + [1] * len(frec))
        rank = None
        if all("logit" in r for r in rrec + frec) and rrec + frec:    # rank by the raw logit, not the rounded P(fake)
            rank = np.array([r["logit"] for r in rrec + frec], dtype="float64")
        m = metrics(probs, labels, rank)
        if m is None:
            print(f"    skip {name}: one class"); continue
        m.update({"n_real": len(rrec), "n_fake": len(frec), "failed_real": rbad, "failed_fake": fbad,
                  "rank_by": "logit" if rank is not None else "fake_probability"})
        if raw:
            rraw = [r["raw"] for r in rrec]
            fraw = [r["raw"] for r in frec]
            pooled_fake_raw += fraw
            m["fake_recall_at_fpr"] = fake_recall_at_fpr(rraw, fraw)
        per_gen[name] = m
        for k in agg:
            agg[k].append(m[k])
        print(f"    {name}: acc={m['acc']} AP={m['ap']} AP_real={m['ap_real']} AUROC={m['auroc']} "
              f"(n={len(rrec)}+{len(frec)}, failed {rbad}+{fbad}, rank by {m['rank_by']})", flush=True)

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
        pooled_real_raw = [r["raw"] for rrec, _ in real_cache.values() for r in rrec]
        means["overall_fake_recall_at_fpr"] = fake_recall_at_fpr(pooled_real_raw, pooled_fake_raw)
    result = {"tag": args.tag, "model": args.model, "ckpt": "registry-default",
              "benchmark": f"video:{args.benchmark}", "data_root": str(root),
              "n_generators": len(per_gen), "means": means, "per_generator": per_gen, "failed_to_decode": failed,
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
