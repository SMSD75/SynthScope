"""NSG-VD on GenVideo-Val: paper vs released CSVs vs the official code on our data vs SynthScope (Task 2 of the
integration). Writes repro/nsg_vd_benchmark_genvideo_val.json and prints the per-generator tables.

Inputs
  --official DIR   JSONs of SynthScope_shared/nsgvd_official_bench.py (the OFFICIAL test_dMMD.main, unmodified, seed 42,
                   shuffled 24-video batches; per-video statistic, id and batch recorded): <label>_<ckpt>_L<len>.json
  --ss-cache DIR   bench_eval_video score caches, one sub-folder per key: cache_<key>_<ffmpeg>/scores.*.jsonl
  --csv DIR        the released results (NSG-VD repo assets/nsg-vd-test_results/*.csv)

Protocols
  L500  test.yaml as released: first 500 test ids per generator, MSR-VTT test reals from index 200 (500), Sora: 51 test
        fakes (the whole released Sora test split) + 56 reals. = the released CSVs' protocol.
  L120  first 120 of every list (Sora 51 + 56): the per-generator count of the paper's NSG-VD rows (Tables 1-3: e.g.
        68.33 = 82/120), an APPROXIMATION - the paper does not say how its 120 were chosen, and its Sora row has 56 fakes
        (94.64 = 53/56) while the released Sora test split has 51.

Metrics as test_dMMD.py: verdict fake iff statistic > 1; recall / precision / F1 on the fake class; accuracy pooled over
real + fake; AUROC of the raw statistic; the "Avg." column is the mean over the 10 generators. SynthScope's metrics are
recomputed here in the same definitions from its per-video statistic (bench_eval_video's own JSON reports balanced
accuracy and AP, which differ for Sora's 56 + 51).
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

GENS = ["ModelScope", "MorphStudio", "MoonValley", "HotShot", "Show_1", "Gen2", "Crafter", "Lavie", "Sora", "WildScrape"]
METRICS = ["recall", "accuracy", "f1", "auroc", "precision"]
KEYS = {"nsg_vd": "standard-SEINE-mp", "nsg_vd_pika": "standard-Pika-mp", "nsg_vd_unbalance": "unbalance-SEINE-mp"}
CKPTS = ["standard-SEINE-mp", "standard-Pika-mp", "unbalance-SEINE-mp", "standard-SEINE-d", "standard-Pika-d",
         "unbalance-SEINE-d"]
# arXiv 2510.08073 v2, NSG-VD (Ours) rows, generators in GENS order + Avg. Table 1 = Kinetics-400 + Pika, Table 2 =
# Kinetics-400 + SEINE, Table 3 = 10,000 Kinetics-400 + 1,000 SEINE (MMD-MP; the paper reports no MMD-D model).
PAPER = {
    "standard-Pika-mp": {"table": "Table 1",
        "recall":   [68.33, 98.33, 100.00, 92.50, 87.50, 80.00, 98.33, 94.17, 78.57, 82.50, 88.02],
        "accuracy": [81.67, 98.33, 96.67, 91.67, 90.83, 88.33, 95.83, 94.17, 88.39, 88.75, 91.46],
        "f1":       [78.85, 98.33, 96.77, 91.74, 90.52, 87.27, 95.93, 94.17, 87.13, 88.00, 90.87],
        "auroc":    [92.26, 98.66, 98.15, 94.45, 96.38, 94.83, 98.16, 97.41, 96.40, 94.73, 96.14]},
    "standard-SEINE-mp": {"table": "Table 2",
        "recall":   [91.67, 100.00, 100.00, 100.00, 100.00, 98.33, 100.00, 97.50, 94.64, 89.17, 97.13],
        "accuracy": [82.50, 88.33, 89.58, 84.58, 86.25, 87.08, 86.67, 87.92, 89.29, 78.33, 86.05],
        "f1":       [83.97, 89.55, 90.57, 86.64, 87.91, 88.39, 88.24, 88.97, 89.83, 80.45, 87.45],
        "auroc":    [90.67, 97.62, 98.38, 95.88, 96.69, 97.87, 97.64, 95.09, 96.14, 88.65, 95.46]},
    "unbalance-SEINE-mp": {"table": "Table 3",
        "recall":   [85.83, 99.17, 100.00, 99.17, 97.50, 95.83, 99.17, 91.67, 82.14, 81.67, 93.21],
        "accuracy": [84.58, 92.50, 93.75, 89.58, 89.58, 90.83, 92.50, 90.00, 86.61, 81.67, 89.16],
        "f1":       [84.77, 92.97, 94.12, 90.49, 90.35, 91.27, 92.97, 90.16, 85.98, 81.67, 89.48],
        "auroc":    [90.76, 98.18, 98.18, 95.03, 95.48, 96.97, 96.53, 95.11, 95.73, 87.13, 94.91]},
}
CSV_NAME = {"standard-SEINE-mp": "standard-SEINE-mmd-mp", "standard-Pika-mp": "standard-Pika-mmd-mp",
            "unbalance-SEINE-mp": "unbalance-SEINE-mmd-mp", "standard-SEINE-d": "standard-SEINE-mmd-d",
            "standard-Pika-d": "standard-Pika-mmd-d", "unbalance-SEINE-d": "unbalance-SEINE-mmd-d"}


def official_metrics(real: np.ndarray, fake: np.ndarray) -> dict:
    """test_dMMD.test_dMMD's metrics (x100) from per-video statistics."""
    s = np.concatenate([real, fake])
    y = np.concatenate([np.zeros(len(real), int), np.ones(len(fake), int)])
    p = (s > 1).astype(int)
    return {"recall": recall_score(y, p) * 100, "accuracy": accuracy_score(y, p) * 100, "f1": f1_score(y, p) * 100,
            "auroc": roc_auc_score(y, s) * 100, "precision": precision_score(y, p) * 100,
            "n_real": len(real), "n_fake": len(fake)}


def with_avg(per_gen: dict) -> dict:
    out = {g: {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()} for g, m in per_gen.items()}
    if all(g in per_gen for g in GENS):
        out["Avg."] = {k: round(float(np.mean([per_gen[g][k] for g in GENS])), 2) for k in METRICS}
    return out


def read_csv(path: Path) -> dict:
    rows = list(csv.reader(path.open()))
    names = rows[0][1:]
    out = {n: {} for n in names}
    for row in rows[1:]:
        for n, v in zip(names, row[1:]):
            out[n][row[0].lower()] = float(v)
    return out


def read_official(path: Path) -> dict:
    """{generator: {"real": {id: s}, "fake": {id: s}, "metrics_logged": as test_dMMD returned them (x100),
    "<cls>_n_dataset": dataset sizes}}."""
    d = json.loads(path.read_text())
    out = {}
    for g in d["generators"]:
        rec = {"metrics_logged": {k: v * 100 for k, v in g["official_metrics"].items()}}
        for cls in ("real", "fake"):
            rec[cls] = dict(zip(g[cls]["ids"], g[cls]["scores"]))
            rec[cls + "_n_dataset"] = g[cls]["n_dataset"]
        out[g["generator"]] = rec
    return out


def read_ss(cache_dir: Path) -> dict:
    """{generator: {"real": {id: (logit, verdict)}, "fake": {...}}} from bench_eval_video score caches (keys
    <Gen>/<0_real|1_fake>/<pos>_<id>.mp4); the shared real pool was scored under one generator's folder: reals are
    looked up by id."""
    by_id = {"real": {}, "fake": {}}
    errors = []
    for f in sorted(cache_dir.glob("scores.*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            gen, cls, name = r["k"].split("/")
            vid = name.split("_", 1)[1].rsplit(".", 1)[0]
            label = "real" if cls == "0_real" else "fake"
            if "err" in r:
                errors.append(r)
                continue
            by_id[label][(gen if label == "fake" else "") + "/" + vid] = (r["logit"], r["verdict"])
    return by_id, errors


def metrics_of(run_gen: dict) -> dict:
    return official_metrics(np.array(list(run_gen["real"].values())), np.array(list(run_gen["fake"].values())))


def per_video_diff(a: dict, b: dict) -> dict:
    """Per-video |statistic difference| between two official runs (same generator keys)."""
    d = [abs(a[g][c][v] - b[g][c][v]) for g in a if g in b for c in ("real", "fake") for v in a[g][c] if v in b[g][c]]
    flips = sum((a[g][c][v] > 1) != (b[g][c][v] > 1) for g in a if g in b for c in ("real", "fake") for v in a[g][c]
                if v in b[g][c])
    return {"n": len(d), "max_abs_diff": float(max(d)), "mean_abs_diff": float(np.mean(d)), "verdict_flips": int(flips)}


def official_seed_spread(runs: dict) -> dict:
    """The official test_dMMD.main with other seeds (= other shuffles, so other 24-video batches): metrics per run, the
    per-generator max - min across runs, per-video differences vs the seed-42 run; ``rerun42`` = seed 42 again."""
    base = runs["seed42 (warm)"]
    out = {"note": "official test_dMMD.main, same cached NSG features; the seed only changes the DataLoader shuffles "
                   "(batch composition of the net / MMD_batch2 forward)",
           "metrics": {name: with_avg({g: metrics_of(r[g]) for g in GENS if g in r}) for name, r in runs.items()},
           "per_video_vs_seed42": {name: per_video_diff(base, r) for name, r in runs.items() if r is not base},
           "logged_vs_recomputed_max_abs_diff": {name: max(abs(r[g]["metrics_logged"][m] - metrics_of(r[g])[m])
                                                           for g in r for m in METRICS) for name, r in runs.items()}}
    names = [n for n in runs if not n.startswith("rerun")]
    out["spread_max_minus_min"] = {g: {m: round(max(out["metrics"][n][g][m] for n in names)
                                                 - min(out["metrics"][n][g][m] for n in names), 2) for m in METRICS}
                                   for g in GENS + ["Avg."]}
    return out


def zero_shot(cache_root: Path, split_zip: str | None) -> dict:
    """nsg_vd on other benchmarks (bench_eval_video --dataset <d> --limit 1000; EvalCrafter with GenVideo-Val's MSR-VTT
    reals), in test_dMMD's definitions. The real pools are the first 1,000 MSR-VTT files by name; some of them are in
    the 100-video reference set (scored against a set that holds themselves), so each row is also given without them."""
    ref = set()
    if split_zip:
        with zipfile.ZipFile(split_zip) as z:
            val = [x.strip() for x in z.read("split/real/MSR-VTT/val_ids.txt").decode().splitlines() if x.strip()]
        ref = set([v.split(".")[0].lower() for v in val if v.endswith(".mp4")][:100])   # the shipped reference set
    out = {"note": "ZERO-SHOT reference rows (not a paper protocol): nsg_vd (standard-SEINE-mp), the shipped reference "
                   "set (100 MSR-VTT val videos) for every dataset, ffmpeg 4.2.2, --limit 1000 per class per generator; "
                   "metrics as test_dMMD (fake iff MMD > 1, pooled accuracy)"}
    for ds in ("vidprom", "videophy", "evalcrafter"):
        cdir = cache_root / f"cache_nsg_vd_{ds}"
        if not cdir.is_dir():
            continue
        reals, fakes, failed = {}, {}, []
        for f in sorted(cdir.glob("scores.*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                a, b, name = r["k"].split("/")        # <gen>/<0_real|1_fake>/<file>, or 1_fake/<gen>/<file> (fake-only)
                if "err" in r:
                    failed.append(r["k"])
                    continue
                if b in ("0_real", "real", "reals", "Real"):
                    reals[name.split("_", 1)[1].rsplit(".", 1)[0].lower()] = r["logit"]
                else:
                    fakes.setdefault(b if a in ("1_fake", "fake", "fakes", "Fake") else a, []).append(r["logit"])
        overlap = sorted(set(reals) & ref)
        rows = {}
        for g, fk in sorted(fakes.items()):
            rr = np.array(list(reals.values()))
            rows[g] = {"all_reals": {k: round(v, 2) if isinstance(v, float) else v
                                     for k, v in official_metrics(rr, np.array(fk)).items()}}
            if overlap:
                rr2 = np.array([v for k, v in reals.items() if k not in ref])
                rows[g]["without_reference_reals"] = {k: round(v, 2) if isinstance(v, float) else v
                                                      for k, v in official_metrics(rr2, np.array(fk)).items()}
        avg = {variant: {m: round(float(np.mean([rows[g][variant][m] for g in rows])), 2) for m in METRICS}
               for variant in ("all_reals", "without_reference_reals") if all(variant in rows[g] for g in rows)}
        out[ds] = {"n_reals": len(reals), "reals_in_reference_set": len(overlap), "decode_failures": failed,
                   "per_generator": rows, "Avg.": avg}
    return out if len(out) > 1 else {}


def ss_scores(by_id, gen, real_ids, fake_ids):
    real = [by_id["real"]["/" + v] for v in real_ids if "/" + v in by_id["real"]]
    fake = [by_id["fake"][gen + "/" + v] for v in fake_ids if gen + "/" + v in by_id["fake"]]
    return real, fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--ss-cache", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lists", required=True, help="nsgvd_bench_farm.py's lists.json (official id order per generator)")
    ap.add_argument("--ffmpeg", default="ffmpeg-linux64-v4.2.2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-zip", default=None, help="the released split.zip: with it, the zero-shot rows are also given "
                                                       "without the reals that are in the 100-video reference set")
    ap.add_argument("--per-video-out", default=None, help="also write the per-video statistics (L500, seed 42) aligned "
                                                          "to the official id lists")
    args = ap.parse_args()
    off_dir, lists = Path(args.official), json.loads(Path(args.lists).read_text())
    res = {"protocols": {"L500": "test.yaml: 500 per generator (Sora 51 + 56)",
                         "L120": "first 120 of every list (Sora 51 + 56): paper per-generator count, approximation"},
           "metric_definitions": "test_dMMD.py: fake iff statistic > 1; recall/precision/F1 on fakes; pooled accuracy; "
                                 "AUROC of the statistic; Avg. = mean of the 10 generators",
           "checkpoints": {}}
    for ck in CKPTS:
        entry = {"released_csv": read_csv(Path(args.csv) / f"{CSV_NAME[ck]}.csv")}     # incl. its own "Avg."
        if ck in PAPER:
            entry["paper"] = {"table": PAPER[ck]["table"],
                              **{g: {m: PAPER[ck][m][i] for m in ("recall", "accuracy", "f1", "auroc")}
                                 for i, g in enumerate(GENS + ["Avg."])}}
        key = next((k for k, v in KEYS.items() if v == ck), None)
        by_id = None
        if key:
            cdir = Path(args.ss_cache) / f"cache_{key}_{args.ffmpeg}"
            if cdir.is_dir():
                by_id, errors = read_ss(cdir)
                entry["synthscope_key"] = key
                entry["synthscope_failed"] = [e["k"] + ": " + e["err"] for e in errors]
        for proto, n in (("L500", 500), ("L120", 120)):
            p = off_dir / f"warm_{ck}_L{n}.json"
            off = read_official(p) if p.exists() else None
            prot = {}
            if off:
                raw = {g: metrics_of(off[g]) for g in GENS if g in off}
                prot["official_on_our_data"] = with_avg(raw)
                prot["official_logged_vs_recomputed_max_abs_diff"] = max(
                    abs(off[g]["metrics_logged"][m] - raw[g][m]) for g in off for m in METRICS)
                prot["official_counts"] = {g: [off[g]["real_n_dataset"], off[g]["fake_n_dataset"]] for g in off}
            if by_id is not None:
                ss, deltas = {}, {}
                for g in GENS:
                    real_ids, fake_ids = lists[g]["real"][:n], lists[g]["fake"][:n]
                    real, fake = ss_scores(by_id, g, real_ids, fake_ids)
                    if not real or not fake:
                        continue
                    m = official_metrics(np.array([r[0] for r in real]), np.array([f[0] for f in fake]))
                    vr = np.array([int(r[1] == "fake") for r in real] + [int(f[1] == "fake") for f in fake])
                    lr = np.array([int(r[0] > 1) for r in real] + [int(f[0] > 1) for f in fake])
                    m["verdict_vs_rounded_logit_disagreements"] = int((vr != lr).sum())
                    ss[g] = m
                    if off and g in off:                        # per video: official (shuffled batches) vs SynthScope
                        pairs = [(off[g][c][v], by_id[c][("" if c == "real" else g) + "/" + v][0])
                                 for c, ids in (("real", real_ids), ("fake", fake_ids)) for v in ids
                                 if v in off[g][c] and (("" if c == "real" else g) + "/" + v) in by_id[c]]
                        a = np.array(pairs)
                        deltas[g] = {"n": len(a), "max_abs_diff": float(np.abs(a[:, 0] - a[:, 1]).max()),
                                     "mean_abs_diff": float(np.abs(a[:, 0] - a[:, 1]).mean()),
                                     "verdict_agree": int(((a[:, 0] > 1) == (a[:, 1] > 1)).sum())}
                prot["synthscope"] = with_avg(ss)
                if deltas:
                    prot["per_video_official_vs_synthscope"] = deltas
                    prot["metric_delta_synthscope_minus_official"] = {
                        g: {m: round(prot["synthscope"][g][m] - prot["official_on_our_data"][g][m], 2) for m in METRICS}
                        for g in GENS + ["Avg."] if g in prot["synthscope"] and g in prot["official_on_our_data"]}
            if off:                                     # the official code against itself: other seeds / a rerun
                runs = {"seed42 (warm)": off}
                for f in sorted(off_dir.glob(f"seed*_{ck}_L{n}.json")) + sorted(off_dir.glob(f"rerun42_{ck}_L{n}.json")):
                    runs[f.name.split("_")[0]] = read_official(f)
                if len(runs) > 1:
                    prot["official_seed_runs"] = official_seed_spread(runs)
            if prot:
                entry[proto] = prot
        res["checkpoints"][ck] = entry
        if ck in KEYS.values():                     # the official code, L500 vs L120 batches, same cached features
            a, b = off_dir / f"warm_{ck}_L500.json", off_dir / f"warm_{ck}_L120.json"
            if a.exists() and b.exists():
                entry["official_L500_vs_L120_same_videos"] = per_video_diff(read_official(b), read_official(a))
    # ffmpeg 9.0.2 instead of the pinned 4.2.2 (SynthScope path, nsg_vd, the generators that were scored with it)
    alt = Path(args.ss_cache) / "cache_nsg_vd_ffmpeg-linux-x86_64-v9.0.2"
    base = Path(args.ss_cache) / f"cache_nsg_vd_{args.ffmpeg}"
    if alt.is_dir() and base.is_dir():
        a_ids, _ = read_ss(alt)
        b_ids, _ = read_ss(base)
        gens = sorted({k.split("/")[0] for k in a_ids["fake"]})
        sens = {"note": "SynthScope path, nsg_vd, ffmpeg 9.0.2 (imageio-ffmpeg 0.8.1) vs the pinned 4.2.2: frames are "
                        "re-encoded JPEGs of the same frame indices (repro/nsg_vd_ffmpeg_builds_genvideo75.json)",
                "generators": gens}
        for n in (500, 120):
            per = {}
            for g in gens:
                real_ids, fake_ids = lists[g]["real"][:n], lists[g]["fake"][:n]
                ra, fa = ss_scores(a_ids, g, real_ids, fake_ids)
                rb, fb = ss_scores(b_ids, g, real_ids, fake_ids)
                ma = official_metrics(np.array([r[0] for r in ra]), np.array([f[0] for f in fa]))
                mb = official_metrics(np.array([r[0] for r in rb]), np.array([f[0] for f in fb]))
                sa = np.array([r[0] for r in ra] + [f[0] for f in fa])
                sb = np.array([r[0] for r in rb] + [f[0] for f in fb])
                per[g] = {"ffmpeg_9.0.2": {k: round(v, 2) if isinstance(v, float) else v for k, v in ma.items()},
                          "ffmpeg_4.2.2": {k: round(v, 2) if isinstance(v, float) else v for k, v in mb.items()},
                          "delta_9.0.2_minus_4.2.2": {m: round(ma[m] - mb[m], 2) for m in METRICS},
                          "per_video": {"n": len(sa), "max_abs_diff": float(np.abs(sa - sb).max()),
                                        "mean_abs_diff": float(np.abs(sa - sb).mean()),
                                        "verdict_flips": int(((sa > 1) != (sb > 1)).sum())}}
            sens[f"L{n}"] = per
        res["ffmpeg_9.0.2_sensitivity"] = sens
    zs = zero_shot(Path(args.ss_cache), args.split_zip)
    if zs:
        res["zero_shot"] = zs
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    if args.per_video_out:
        pv = {"note": "per-video MMD statistic (fake iff > 1), L500 protocol, ids in the official list order "
                      "(lists[gen][cls]); official = test_dMMD.main seed 42 (each generator's reals are scored in that "
                      "generator's own shuffled batches); synthscope = bench_eval_video served path (one score per video)",
              "lists": {g: {c: lists[g][c][:56 if g == "Sora" else 500] for c in ("real", "fake")} for g in GENS},
              "official": {}, "synthscope": {}}
        for ck in CKPTS:
            f = off_dir / f"warm_{ck}_L500.json"
            if f.exists():
                off = read_official(f)
                pv["official"][ck] = {g: {c: [round(off[g][c][v], 6) if v in off[g][c] else None
                                              for v in pv["lists"][g][c]] for c in ("real", "fake")} for g in GENS}
        for key in KEYS:
            cdir = Path(args.ss_cache) / f"cache_{key}_{args.ffmpeg}"
            if cdir.is_dir():
                by_id, _ = read_ss(cdir)
                pv["synthscope"][key] = {g: {c: [by_id[c].get(("" if c == "real" else g) + "/" + v, (None,))[0]
                                                 for v in pv["lists"][g][c]] for c in ("real", "fake")} for g in GENS}
        Path(args.per_video_out).write_text(json.dumps(pv, separators=(",", ":")) + "\n")
    for ck, e in res["checkpoints"].items():
        print(f"\n## {ck}" + (f" ({e.get('synthscope_key')})" if e.get("synthscope_key") else ""))
        for m in ("recall", "accuracy", "f1", "auroc"):
            rows = [("released CSV", e["released_csv"])]
            if "L500" in e:
                rows += [(f"official L500", e["L500"].get("official_on_our_data", {})),
                         (f"SynthScope L500", e["L500"].get("synthscope", {}))]
            if "paper" in e:
                rows.append((f"paper {e['paper']['table']}", e["paper"]))
            if "L120" in e:
                rows += [(f"official L120", e["L120"].get("official_on_our_data", {})),
                         (f"SynthScope L120", e["L120"].get("synthscope", {}))]
            print(f"| {m} | " + " | ".join(g[:6] for g in GENS + ["Avg."]) + " |")
            for name, t in rows:
                if t:
                    print(f"| {name} | " + " | ".join(f"{t[g][m]:.2f}" if g in t and m in t[g] else "-" for g in GENS + ["Avg."]) + " |")


if __name__ == "__main__":
    main()
