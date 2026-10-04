"""Unified VIDEO benchmark eval for SynthScope video detectors (d3, restrav, stall).

Scores each video through the SAME verified synthscope engine path the web app uses
(engine.predict_video_clip -> native clip pipeline + per-model norm + fake-prob), computes
per-generator + mean acc@0.5 / AP / AUROC, and writes the SAME JSON schema as bench_eval.py
(repro/bench_results/<tag>.json + append repro/MASTER_RESULTS.jsonl) so the aggregator ingests it.

    python benchmark/bench_eval_video.py --model d3 --benchmark genvideo \
        --data /hddstore/ssalehi/AGID/d3_repro/genvideo_val/GenVideo-Val --limit 150 --tag d3_genvideo

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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interface"))
from backend.engine import get_engine          # noqa: E402
from backend.registry import get_spec          # noqa: E402

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
    au = float(auroc(fake_probs, labels, task="binary")) * 100
    preds = (fake_probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "auroc": round(au, 2), "real_acc": round(real_acc, 2),
            "fake_acc": round(fake_acc, 2), "acc": round((real_acc + fake_acc) / 2, 2)}


@torch.no_grad()
def score_videos(engine, spec, paths):
    probs = []
    for p in paths:
        try:
            probs.append(engine.predict_video_clip(spec, str(p))["fake_probability"])
        except Exception as e:  # noqa: BLE001
            print(f"    bad video {p.name}: {e}")
    return probs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, choices=["d3", "restrav", "stall"])
    ap.add_argument("--benchmark", required=True, choices=["genvideo", "fakeonly", "pergen"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--reals", default=None)
    ap.add_argument("--select", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap videos per class per generator")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

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

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    # score the shared real pool once if it is identical across generators
    real_cache = {}
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        rk = (id(reals) if args.limit is None else None)
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        key = tuple(reals[:1]) + (len(reals),)
        if key in real_cache:
            rp = real_cache[key]
        else:
            rp = score_videos(engine, spec, reals)
            real_cache[key] = rp
        fp = score_videos(engine, spec, fakes)
        probs = torch.tensor(rp + fp)
        labels = torch.tensor([0] * len(rp) + [1] * len(fp))
        m = metrics(probs, labels)
        if m is None:
            print(f"    skip {name}: one class"); continue
        per_gen[name] = m
        for k in agg:
            agg[k].append(m[k])
        print(f"    {name}: acc={m['acc']} AP={m['ap']} AUROC={m['auroc']}", flush=True)

    if not per_gen:
        print("No generator produced metrics."); return 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2)
    means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "mauroc": mean(agg["auroc"])}
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
    print(f"\n=== RESULT {args.tag}: mAP={means['map']} mAUROC={means['mauroc']} acc={means['acc']} "
          f"over {len(per_gen)} gens -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
