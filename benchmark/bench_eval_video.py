"""Unified VIDEO benchmark eval for SynthScope video detectors (d3, restrav, stall).

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

Benchmark layouts:
  genvideo   <data>/Fake/<gen>/*.mp4 + <data>/Real/*.mp4 (shared real pool)   [GenVideo-Val]
  fakeonly   <data>/<gen>/*.mp4 (all fake) + --reals DIR of real videos        [EvalCrafter]
  pergen     <data>/<gen>/{0_real,1_fake}/*.mp4                                 [GenVidBench]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
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


def metrics(fake_probs: torch.Tensor, labels: torch.Tensor):
    from torchmetrics.functional import auroc, average_precision
    if labels.unique().numel() < 2:
        return None
    ap = float(average_precision(fake_probs, labels, task="binary")) * 100
    ap_real = float(average_precision(1 - fake_probs, 1 - labels, task="binary")) * 100
    au = float(auroc(fake_probs, labels, task="binary")) * 100
    preds = (fake_probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "ap_real": round(ap_real, 2), "auroc": round(au, 2), "real_acc": round(real_acc, 2),
            "fake_acc": round(fake_acc, 2), "acc": round((real_acc + fake_acc) / 2, 2)}


FPRS = (0.001, 0.01, 0.05)


@torch.no_grad()
def score_videos(engine, spec, paths, raw=False):
    """P(fake) per video; with ``raw`` (training-free detectors with a ``score``), the raw score oriented so that higher
    means generated instead, which keeps the order at the extremes where probabilities saturate."""
    probs = []
    for p in paths:
        try:
            probs.append(engine.video_raw_score(spec, str(p)) if raw else engine.predict_video_clip(spec, str(p))["fake_probability"])
        except Exception as e:  # noqa: BLE001
            print(f"    bad video {p.name}: {e}")
    return probs


def fake_recall_at_fpr(real_scores, fake_scores):
    """SPLIT's operating-point metric (official eval.py): the threshold is the (1 - FPR) quantile of the real scores
    (numpy method "higher"); fake recall = share of fakes scoring at or above it. In percent, per FPR."""
    real, fake = np.asarray(real_scores, dtype="float64"), np.asarray(fake_scores, dtype="float64")
    return {f"{a * 100:g}": round(float((fake >= np.quantile(real, 1 - a, method="higher")).mean()) * 100, 2) for a in FPRS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, choices=["d3", "split", "restrav", "stall"])
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
    args = ap.parse_args(argv)

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

    engine = get_engine()
    spec = get_spec(args.model)
    engine._get_model(spec)
    root = Path(args.data)
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root, Path(args.reals) if args.reals else None, select)
    if not gens:
        print(f"No generators with real+fake videos under {root} ({args.benchmark})")
        return 1
    print(f"=== bench_eval_video {args.tag} | {args.model} | {args.benchmark} | {len(gens)} gens "
          f"| device={engine.device} ===", flush=True)

    per_gen, agg = {}, {"ap": [], "ap_real": [], "auroc": [], "acc": []}
    pooled_fake_raw = []                                 # every generator's fakes, for the pooled ("Overall") recall
    raw = hasattr(engine._get_model(spec), "score")     # training-free detectors: also fake recall at fixed FPR
    # score the shared real pool once if it is identical across generators
    real_cache = {}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        rk = (id(reals) if args.limit is None else None)
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        key = (len(reals), reals[0].name, reals[-1].name)    # the same pool hard-linked per generator is scored once
        if key in real_cache:
            rp = real_cache[key]
        else:
            rp = score_videos(engine, spec, reals)
            real_cache[key] = rp
        fp = score_videos(engine, spec, fakes)
        if raw:
            rraw = real_cache.setdefault(("raw",) + key, None) or score_videos(engine, spec, reals, raw=True)
            real_cache[("raw",) + key] = rraw
            fraw = score_videos(engine, spec, fakes, raw=True)
            pooled_fake_raw += fraw
        probs = torch.tensor(rp + fp)
        labels = torch.tensor([0] * len(rp) + [1] * len(fp))
        m = metrics(probs, labels)
        if m is None:
            print(f"    skip {name}: one class"); continue
        if raw:
            m["fake_recall_at_fpr"] = fake_recall_at_fpr(rraw, fraw)
        per_gen[name] = m
        for k in agg:
            agg[k].append(m[k])
        print(f"    {name}: acc={m['acc']} AP={m['ap']} AP_real={m['ap_real']} AUROC={m['auroc']}", flush=True)

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
        pooled_real_raw = [s for k, v in real_cache.items() if k[0] == "raw" for s in v]
        means["overall_fake_recall_at_fpr"] = fake_recall_at_fpr(pooled_real_raw, pooled_fake_raw)
    result = {"tag": args.tag, "model": args.model, "ckpt": "registry-default",
              "benchmark": f"video:{args.benchmark}", "data_root": str(root),
              "n_generators": len(per_gen), "means": means, "per_generator": per_gen,
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
