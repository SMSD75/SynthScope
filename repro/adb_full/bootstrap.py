"""Bootstrap 95% confidence intervals for fused ``adb_full`` rows (``repro/adb_full/results/<row>.json``): the full method,
the VLM alone and DDA alone, and their paired differences, on the same images.

Resampling: stratified by (generator, class), with replacement within each stratum, so every replicate keeps the row's
images per generator and class. The same resampled images serve all three detectors (paired). Per replicate: balanced
accuracy per generator (fused: the official decision, an unparsed answer wrong; VLM: the parsed verdict; DDA: P >= 0.5),
AUROC per generator (fused score / VLM logit / DDA P, rank-based with ties as half), then their means over generators,
as the headline numbers are computed. 2,000 replicates, seed 0; CI = the 2.5 and 97.5 percentiles.

    python repro/adb_full/bootstrap.py genimage_10k_jpegfakes96 genimage_subset_jpegfakes96

``--pair BEFORE AFTER`` (the same images scored twice, e.g. the f4b4794 and the cf168db VLM request): one resampling of the
shared images drives both rows, so the CI of AFTER - BEFORE is paired; for the full method, the VLM alone and DDA alone
(DDA is the same records in both rows, so its difference is 0). Output ``bootstrap_<AFTER>__vs__<BEFORE>.json``.

    python repro/adb_full/bootstrap.py --pair genimage_10k_jpegfakes96 genimage_10k_jpegfakes96_cf168db
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
B = 2000
SEED = 0
DETECTORS = ("adb_full", "vlm", "dda")
PAIRS = (("adb_full", "dda"), ("adb_full", "vlm"), ("vlm", "dda"))


def auroc(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUROC per replicate row: pos [B, n1], neg [B, n0] scores (Mann-Whitney, ties count half)."""
    n1, n0 = pos.shape[1], neg.shape[1]
    ranks = np.apply_along_axis(rankdata, 1, np.concatenate([pos, neg], axis=1))
    return (ranks[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def gen_arrays(rows: list[dict]) -> dict:
    out = {}
    for label in (0, 1):
        rs = [r for r in rows if r["label"] == label]
        want = "fake" if label else "real"
        out[label] = {
            "correct": {"adb_full": np.array([r["verdict"] == want for r in rs]),
                        "vlm": np.array([r["vlm_verdict"] == want for r in rs]),
                        "dda": np.array([(r["dda_prob"] >= 0.5) == bool(label) for r in rs])},
            "score": {"adb_full": np.array([r["score"] for r in rs]), "vlm": np.array([r["vlm_logit"] for r in rs]),
                      "dda": np.array([r["dda_prob"] for r in rs])}}
    return out


def ci(x: np.ndarray) -> list[float]:
    return [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]


def run(row: str) -> dict:
    res = json.loads((HERE / "results" / f"{row}.json").read_text())
    gens = sorted({r["generator"] for r in res["per_image"]})
    rng = np.random.default_rng(SEED)
    acc = {d: np.zeros((B, len(gens))) for d in DETECTORS}
    auc = {d: np.zeros((B, len(gens))) for d in DETECTORS}
    point_acc = {d: [] for d in DETECTORS}
    point_auc = {d: [] for d in DETECTORS}
    for gi, g in enumerate(gens):
        a = gen_arrays([r for r in res["per_image"] if r["generator"] == g])
        idx = {lab: rng.integers(0, len(a[lab]["correct"]["vlm"]), (B, len(a[lab]["correct"]["vlm"]))) for lab in (0, 1)}
        for d in DETECTORS:
            real, fake = a[0]["correct"][d], a[1]["correct"][d]
            acc[d][:, gi] = 50 * (real[idx[0]].mean(axis=1) + fake[idx[1]].mean(axis=1))
            point_acc[d].append(50 * (real.mean() + fake.mean()))
            s0, s1 = a[0]["score"][d], a[1]["score"][d]
            auc[d][:, gi] = 100 * auroc(s1[idx[1]], s0[idx[0]])
            point_auc[d].append(float(100 * auroc(s1[None, :], s0[None, :])[0]))
    out = {"row": row, "n_images": len(res["per_image"]), "generators": gens, "replicates": B, "seed": SEED,
           "method": "stratified by (generator, class), paired across detectors; means over generators", "detectors": {},
           "paired_differences": {}, "per_generator": {}}
    for d in DETECTORS:
        m_acc, m_auc = acc[d].mean(axis=1), auc[d].mean(axis=1)
        out["detectors"][d] = {"acc": round(float(np.mean(point_acc[d])), 2), "acc_ci95": ci(m_acc),
                               "mauroc": round(float(np.mean(point_auc[d])), 2), "mauroc_ci95": ci(m_auc)}
    for x, y in PAIRS:
        da = acc[x].mean(axis=1) - acc[y].mean(axis=1)
        du = auc[x].mean(axis=1) - auc[y].mean(axis=1)
        out["paired_differences"][f"{x}-{y}"] = {
            "acc": round(float(np.mean(point_acc[x]) - np.mean(point_acc[y])), 2), "acc_ci95": ci(da),
            "mauroc": round(float(np.mean(point_auc[x]) - np.mean(point_auc[y])), 2), "mauroc_ci95": ci(du)}
    for gi, g in enumerate(gens):
        out["per_generator"][g] = {d: {"acc": round(float(point_acc[d][gi]), 2), "acc_ci95": ci(acc[d][:, gi])}
                                   for d in DETECTORS}
    return out


def run_pair(before: str, after: str) -> dict:
    rows = {}
    for name in (before, after):
        res = json.loads((HERE / "results" / f"{name}.json").read_text())
        rows[name] = {(r["generator"], r["key"]): r for r in res["per_image"]}
    if rows[before].keys() != rows[after].keys():
        raise SystemExit(f"{before} and {after} differ in images ({len(rows[before].keys() ^ rows[after].keys())} not shared)")
    for k, r in rows[before].items():
        if (r["label"], r["dda_prob"]) != (rows[after][k]["label"], rows[after][k]["dda_prob"]):
            raise SystemExit(f"{k}: label or DDA record differs between {before} and {after}")
    gens = sorted({g for g, _ in rows[before]})
    rng = np.random.default_rng(SEED)
    acc = {(n, d): np.zeros((B, len(gens))) for n in (before, after) for d in DETECTORS}
    auc = {(n, d): np.zeros((B, len(gens))) for n in (before, after) for d in DETECTORS}
    point_acc = {(n, d): [] for n in (before, after) for d in DETECTORS}
    point_auc = {(n, d): [] for n in (before, after) for d in DETECTORS}
    changed = {"vlm_verdict": 0, "verdict": 0}
    for gi, g in enumerate(gens):
        keys = sorted(k for k in rows[before] if k[0] == g)
        arrays = {n: gen_arrays([rows[n][k] for k in keys]) for n in (before, after)}
        size = {lab: len(arrays[before][lab]["correct"]["vlm"]) for lab in (0, 1)}
        idx = {lab: rng.integers(0, size[lab], (B, size[lab])) for lab in (0, 1)}
        for n in (before, after):
            a = arrays[n]
            for d in DETECTORS:
                real, fake = a[0]["correct"][d], a[1]["correct"][d]
                acc[n, d][:, gi] = 50 * (real[idx[0]].mean(axis=1) + fake[idx[1]].mean(axis=1))
                point_acc[n, d].append(50 * (real.mean() + fake.mean()))
                s0, s1 = a[0]["score"][d], a[1]["score"][d]
                auc[n, d][:, gi] = 100 * auroc(s1[idx[1]], s0[idx[0]])
                point_auc[n, d].append(float(100 * auroc(s1[None, :], s0[None, :])[0]))
        for k in keys:
            for f in changed:
                changed[f] += rows[before][k][f] != rows[after][k][f]
    out = {"before": before, "after": after, "n_images": len(rows[before]), "generators": gens, "replicates": B,
           "seed": SEED, "method": "paired: one stratified (generator, class) resampling of the shared images for both "
                                   "rows; means over generators; difference = after - before",
           "changed_decisions": changed, "rows": {}, "after_minus_before": {}}
    for n in (before, after):
        out["rows"][n] = {d: {"acc": round(float(np.mean(point_acc[n, d])), 2), "acc_ci95": ci(acc[n, d].mean(axis=1)),
                              "mauroc": round(float(np.mean(point_auc[n, d])), 2), "mauroc_ci95": ci(auc[n, d].mean(axis=1))}
                          for d in DETECTORS}
    for d in DETECTORS:
        da = acc[after, d].mean(axis=1) - acc[before, d].mean(axis=1)
        du = auc[after, d].mean(axis=1) - auc[before, d].mean(axis=1)
        out["after_minus_before"][d] = {
            "acc": round(float(np.mean(point_acc[after, d]) - np.mean(point_acc[before, d])), 2), "acc_ci95": ci(da),
            "mauroc": round(float(np.mean(point_auc[after, d]) - np.mean(point_auc[before, d])), 2), "mauroc_ci95": ci(du)}
    out["per_generator"] = {g: {d: round(float(point_acc[after, d][gi] - point_acc[before, d][gi]), 2) for d in DETECTORS}
                            for gi, g in enumerate(gens)}
    return out


def main(argv: list[str]) -> int:
    if argv[:1] == ["--pair"]:
        if len(argv) != 3:
            raise SystemExit("usage: bootstrap.py --pair BEFORE AFTER")
        out = run_pair(argv[1], argv[2])
        path = HERE / f"bootstrap_{argv[2]}__vs__{argv[1]}.json"
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps({"pair": [argv[1], argv[2]], **out["after_minus_before"], "changed": out["changed_decisions"]}),
              flush=True)
        return 0
    for row in argv or ["genimage_10k_jpegfakes96"]:
        out = run(row)
        path = HERE / f"bootstrap_{row}.json"
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps({"row": row, **out["detectors"], "paired": out["paired_differences"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
