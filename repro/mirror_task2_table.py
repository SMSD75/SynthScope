"""MIRROR Task 2: paper vs official-code-on-our-data vs SynthScope, built from repro/bench_results/mirror_*.json and the
official inference.py CSVs in repro/bench_results/mirror_official_code/<jobid>/. Prints markdown. Every measured number
comes from those files; the paper/README numbers are the constants below (official README "13 standard + in-the-wild
benchmarks" table, DINOv3-H+ column = the released checkpoint; paper Tables 1/2/5/7/8/9 for the DINOv3-L context).

    python repro/mirror_task2_table.py            # summary + aggregates + per-subset tables
"""
import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BR = REPO / "repro" / "bench_results"
OFFICIAL_DIR = BR / "mirror_official_code"

# official README, DINOv3-H+ column (the released checkpoint-h-cur.pth)
PAPER_H = {"AIGCDetect": 97.3, "GenImage": 99.8, "UFD": 92.4, "Synthbuster": 99.2, "EvalGEN": 99.8, "DRCT-2M": 93.0,
           "Chameleon": 94.6, "SynthWildX": 95.1, "WildRF": 97.8, "AIGIBench": 94.9, "CO-SPY": 97.4, "RRDataset": 88.3,
           "BFree-Online": 97.6}
# README DINOv3-L column (= paper Tables 1/2 DINOv3 rows)
PAPER_L_MEAN = {"AIGCDetect": 91.7, "GenImage": 94.2, "UFD": 88.2, "Synthbuster": 98.1, "EvalGEN": 99.0,
                "DRCT-2M": 93.0,
                "Chameleon": 90.7, "SynthWildX": 93.1, "WildRF": 96.7, "AIGIBench": 90.5, "CO-SPY": 91.3,
                "RRDataset": 78.9, "BFree-Online": 83.0}
# paper per-subset tables (DINOv3-L only): Table 5 (AIGCDetect), 7 (UFD), 8 (Synthbuster), 9 (EvalGEN), 2 (in the wild)
PAPER_L = {
    "AIGCDetect": {"ADM": 75.6, "DALLE2": 88.9, "Glide": 93.1, "Midjourney": 92.5, "VQDM": 95.9, "biggan": 97.2,
                   "cyclegan": 73.3, "gaugan": 95.8, "progan": 96.8, "sd_xl": 99.8, "stable_diffusion_v_1_4": 99.9,
                   "stable_diffusion_v_1_5": 99.8, "stargan": 87.6, "stylegan": 91.9, "stylegan2": 91.1,
                   "whichfaceisreal": 78.9, "wukong": 99.8},
    "UFD": {"seeingdark": 63.3, "san": 80.6, "crn": 76.4, "imle": 76.9, "guided": 79.8, "ldm_200": 98.9,
            "ldm_200_cfg": 98.8, "ldm_100": 99.0, "glide_100_27": 95.1, "glide_50_27": 94.7, "glide_100_10": 95.6,
            "dalle": 98.6},
    "Synthbuster": {"dalle2": 98.0, "dalle3": 99.9, "firefly": 91.6, "glide": 96.5, "midjourney-v5": 98.3,
                    "stable-diffusion-1-3": 99.8, "stable-diffusion-1-4": 100.0, "stable-diffusion-2": 99.2,
                    "stable-diffusion-xl": 99.5},
    "EvalGEN": {"Flux": 99.9, "GoT": 99.8, "Infinity": 99.5, "NOVA": 99.9, "OmniGen": 95.7},
    "SynthWildX": {"dalle3": 95.9, "firefly": 88.4, "midjourney_v5": 94.9},
    "WildRF": {"facebook": 97.1, "reddit": 96.6, "twitter": 96.4},
    "AIGIBench": {"SocialRF": 87.6, "CommunityAI": 93.4},
}
# UniversalFakeDetect subset lists: paper Table 7 (12), official inference.py DATASET_CONFIGS 'UnivFD' (13) and
# 'ojha' (12)
UFD_LISTS = {
    "paper Table 7 (12)": ["seeingdark", "san", "crn", "imle", "guided", "ldm_200", "ldm_200_cfg", "ldm_100",
                           "glide_100_27", "glide_50_27", "glide_100_10", "dalle"],
    "official `UnivFD` (13)": ["deepfake", "seeingdark", "san", "ldm_200_cfg", "ldm_200", "ldm_100", "imle", "guided",
                               "glide_50_27", "glide_100_27", "glide_100_10", "dalle", "crn"],
    "official `ojha` (12)": ["deepfake", "san", "crn", "imle", "guided", "ldm_100", "ldm_200", "ldm_200_cfg",
                             "glide_50_27", "glide_100_10", "glide_100_27", "dalle"],
}
# images per class per RRDataset scenario (bench_eval log mirrorbench_279554.out): pooled B.Acc as the official code
# scores a benchmark without a DATASET_CONFIGS entry (one folder, every image)
RR_N = {"original": (8500, 8500), "redigital": (8499, 8500), "transfer": (8500, 8500)}
MISSING = {"DRCT-2M": "not in SynthScope", "Chameleon": "by request only, not held",
           "CO-SPY": "in-the-wild part not held (SynthScope's cospy loader is the synthetic Co-Spy-Bench)"}


def load(tag):
    p = BR / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def mean(xs):
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)


def official_csvs():
    """{benchmark: {row: {metric: %}}} from every inference.py CSV under OFFICIAL_DIR (values there are fractions).
    Row 'MEAN' is the official mean over subsets; a pooled benchmark has the single row 'ALL'."""
    out = {}
    for f in sorted(glob.glob(str(OFFICIAL_DIR / "*" / "*.csv"))):
        bench = Path(f).name.rsplit("_", 2)[0]
        with open(f) as fh:
            rows = {r["Dataset"]: {k: float(v) * 100 for k, v in r.items() if k != "Dataset"}
                    for r in csv.DictReader(fh)}
        out.setdefault(bench, {}).update(rows)
    return out


OFF = official_csvs()


def off_bacc(bench, row):
    return OFF.get(bench, {}).get(row, {}).get("Bal_Acc")


def acc(tag, sub=None):
    d = load(tag)
    if d is None:
        return None
    return d["means"]["acc"] if sub is None else d["per_generator"].get(sub, {}).get("acc")


def ufd_means():
    a, b = load("mirror_ufd_diffusion8"), load("mirror_ufd_cnndet5")
    if not (a and b):
        return {}
    pg = {**a["per_generator"], **b["per_generator"]}
    return {name: mean([pg[g]["acc"] for g in subs]) for name, subs in UFD_LISTS.items()}


def synthbuster_extrule():
    sb, sbx = load("mirror_synthbuster_coco"), load("mirror_synthbuster_coco_firefly_extrule")
    if not (sb and sbx):
        return None
    pg = dict(sb["per_generator"])
    pg["firefly"] = sbx["per_generator"]["firefly"]
    return mean([m["acc"] for m in pg.values()])


def synthbuster_official_firefly_mean():
    """Synthbuster mean with firefly from the official code (the other 8 subsets: SynthScope, which does not differ from
    the official rule for them: no extension/format mismatch)."""
    sb, o = load("mirror_synthbuster_coco"), off_bacc("Synthbuster", "firefly")
    if not sb or o is None:
        return None
    return mean([o if g == "firefly" else m["acc"] for g, m in sb["per_generator"].items()])


def rr_pooled():
    rr = load("mirror_rrdataset")
    if not rr:
        return None, None, None
    pg = rr["per_generator"]
    nr, nf = sum(n[0] for n in RR_N.values()), sum(n[1] for n in RR_N.values())
    ra = sum(pg[s]["real_acc"] * RR_N[s][0] for s in RR_N) / nr
    fa = sum(pg[s]["fake_acc"] * RR_N[s][1] for s in RR_N) / nf
    return (ra + fa) / 2, ra, fa


def summary():
    ufd = ufd_means()
    rrp = rr_pooled()[0]
    sbx = synthbuster_extrule()
    sbo = synthbuster_official_firefly_mean()
    rows = [  # benchmark, official code on our data, SynthScope (served), SynthScope value used in the aggregates, note
        ("AIGCDetect", None, acc("mirror_aigcdetect"), acc("mirror_aigcdetect"), "17 subsets, full data"),
        ("GenImage", None, acc("mirror_genimage_val100k"), acc("mirror_genimage_val100k"),
         f"full val (100k); test release `mirror_genimage_full` {fmt(acc('mirror_genimage_full'))}"),
        ("UFD", None, ufd.get("official `UnivFD` (13)"), ufd.get("official `UnivFD` (13)"),
         "official `UnivFD` list (13); " + "; ".join(f"{k} {fmt(v)}" for k, v in ufd.items() if "13" not in k)),
        ("Synthbuster", sbo, sbx, sbx,
         f"vs 5000 MSCOCO reals; headline = the official extension-based JPEG-q96 rule (firefly is 1000 JPEGs named "
         f".png); the served pipeline keys on the decoded format: {fmt(acc('mirror_synthbuster_coco'))}; "
         f"official column = official firefly + our 8 others"),
        ("EvalGEN", None, acc("mirror_evalgen"), acc("mirror_evalgen"), "5 subsets vs MSCOCO reals"),
        ("DRCT-2M", None, None, None, "**missing**: " + MISSING["DRCT-2M"]),
        ("Chameleon", None, None, None, "**missing**: " + MISSING["Chameleon"]),
        ("SynthWildX", off_bacc("Synthwildx", "MEAN"), acc("mirror_synthwildx"), acc("mirror_synthwildx"),
         "1713 of 2000 images still online"),
        ("WildRF", off_bacc("WildRF", "MEAN"), acc("mirror_wildrf_nojfif"), acc("mirror_wildrf_nojfif"),
         f"official file set (no .jfif); all files {fmt(acc('mirror_wildrf'))}"),
        ("AIGIBench", None, acc("mirror_aigibench"), acc("mirror_aigibench"), "SocialRF + CommunityAI"),
        ("CO-SPY", None, None, None, "**missing**: " + MISSING["CO-SPY"]),
        ("RRDataset", None, rrp, rrp, f"pooled folder B.Acc (official scoring); mean of 3 scenarios "
                                      f"{fmt(acc('mirror_rrdataset'))}"),
        ("BFree-Online", off_bacc("B-Free", "ALL"), acc("mirror_bfree_online"), acc("mirror_bfree_online"),
         "807 of ~1400 images still online"),
    ]
    print("### Summary (B.Acc %, threshold 0.5)\n")
    print("| benchmark | paper DINOv3-L | paper DINOv3-H+ (released ckpt) | official code on our data | SynthScope | "
          "SynthScope − H+ | note |")
    print("|---|---:|---:|---:|---:|---:|---|")
    used = {}
    for b, o, s, u, note in rows:
        used[b] = u
        d = None if s is None else s - PAPER_H[b]
        print(f"| {b} | {PAPER_L_MEAN[b]:.1f} | {PAPER_H[b]:.1f} | {fmt(o)} | {fmt(s)} | "
              f"{'—' if d is None else f'{d:+.2f}'} | {note} |")
    print("| (extra) RealChain | — | — | — | " f"{fmt(acc('mirror_realchain'))} | — | not in the paper; 7 subsets |")
    return used


def aggregates(used):
    print("\n### Aggregates\n")
    std = ["AIGCDetect", "GenImage", "UFD", "Synthbuster", "EvalGEN"]
    print(f"* Six-standard-benchmark average: paper H+ {mean([PAPER_H[k] for k in std + ['DRCT-2M']]):.2f} "
          f"(derived from the README; L: Table 1 94.0). We hold 5 of 6 (no DRCT-2M): SynthScope "
          f"{fmt(mean([used[k] for k in std]))} vs paper H+ on the same 5 {mean([PAPER_H[k] for k in std]):.2f}.")
    cols = {}
    for s in ("dalle3", "firefly", "midjourney_v5"):
        cols[f"SynthWildX/{s}"] = acc("mirror_synthwildx", s)
    for s in ("facebook", "reddit", "twitter"):
        cols[f"WildRF/{s}"] = acc("mirror_wildrf_nojfif", s)
    for s in ("SocialRF", "CommunityAI"):
        cols[f"AIGIBench/{s}"] = acc("mirror_aigibench", s)
    cols["RRDataset (pooled)"] = used["RRDataset"]
    cols["BFree-Online"] = used["BFree-Online"]
    w = {"SynthWildX": 3, "WildRF": 3, "AIGIBench": 2, "RRDataset": 1, "BFree-Online": 1}
    paper10 = sum(PAPER_H[k] * n for k, n in w.items()) / 10
    w16 = dict(w, Chameleon=1, **{"CO-SPY": 5})
    paper16 = sum(PAPER_H[k] * n for k, n in w16.items()) / 16
    print(f"* In-the-wild 16-column score: paper H+ {paper16:.2f} (derived from the README per-benchmark means "
          f"weighted by their column counts; L: Table 2 91.2). We hold 10 of 16 columns (no Chameleon, no CO-SPY "
          f"in-the-wild x5): "
          f"SynthScope {fmt(mean(list(cols.values())))} vs paper H+ on the same 10 columns {paper10:.2f}.")
    print("  * columns: " + ", ".join(f"{k} {fmt(v)}" for k, v in cols.items()))
    rrp, ra, fa = rr_pooled()
    if rrp is not None:
        print(f"* RRDataset pooled: real acc {ra:.2f}, fake acc {fa:.2f} -> {rrp:.2f}; per scenario: " +
              ", ".join(f"{g} {m['acc']:.2f}" for g, m in load("mirror_rrdataset")["per_generator"].items()))


def per_subset(bench, tag, official_name=None, row_map=None):
    d = load(tag)
    if d is None:
        print(f"\n(no {tag}.json)\n")
        return
    pg = d["per_generator"]
    off = OFF.get(official_name, {}) if official_name else {}
    row_map = row_map or {}
    variant = d.get("variant")
    print(f"\n#### {bench} (`{tag}`{', variant ' + json.dumps(variant) if variant else ''})\n")
    print("| subset | paper L | official code B.Acc | SynthScope B.Acc | real acc | fake acc | AP | AUROC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for g, m in pg.items():
        o = off.get(row_map.get(g, g), {}).get("Bal_Acc")
        print(f"| {g} | {fmt(PAPER_L.get(bench, {}).get(g), 1)} | {fmt(o)} | {m['acc']:.2f} | {m['real_acc']:.2f} | "
              f"{m['fake_acc']:.2f} | {m['ap']:.2f} | {m['auroc']:.2f} |")
    if len(pg) > 1:
        o = off.get("MEAN", {}).get("Bal_Acc")
        print(f"| **mean** | {fmt(PAPER_L_MEAN.get(bench), 1)} | {fmt(o)} | **{d['means']['acc']:.2f}** | | | "
              f"{d['means']['map']:.2f} | {d['means']['mauroc']:.2f} |")


def main():
    used = summary()
    aggregates(used)
    print("\n### Per subset")
    for bench, tag, oname, rmap in [
            ("AIGCDetect", "mirror_aigcdetect", None, None), ("GenImage", "mirror_genimage_val100k", None, None),
            ("GenImage", "mirror_genimage_full", None, None), ("UFD", "mirror_ufd_diffusion8", None, None),
            ("UFD", "mirror_ufd_cnndet5", None, None), ("Synthbuster", "mirror_synthbuster_coco", None, None),
            ("Synthbuster", "mirror_synthbuster_coco_firefly_extrule", "Synthbuster", None),
            ("EvalGEN", "mirror_evalgen", None, None), ("SynthWildX", "mirror_synthwildx", "Synthwildx", None),
            ("WildRF", "mirror_wildrf", None, None), ("WildRF", "mirror_wildrf_nojfif", "WildRF", None),
            ("WildRF", "mirror_wildrf_amp", None, None), ("AIGIBench", "mirror_aigibench", None, None),
            ("RRDataset", "mirror_rrdataset", None, None),
            ("RRDataset", "mirror_rrdataset_original_extrule", None, None),
            ("BFree-Online", "mirror_bfree_online", "B-Free", {"viral": "ALL"}),
            ("RealChain", "mirror_realchain", None, None)]:
        per_subset(bench, tag, oname, rmap)


if __name__ == "__main__":
    main()
