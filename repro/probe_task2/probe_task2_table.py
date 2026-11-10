"""PROBE Task 2: paper (Tables 1-4) vs official-code-on-our-data vs SynthScope bench_eval, from the saved result JSONs.
Prints markdown tables and writes <out>.json. Paper numbers are PROBE-DINOv2 / PROBE-ResNet50 bAcc from arXiv 2605.24906.
usage: python probe_task2_table.py WT_DIR SCORES_DIR OFFICIAL_DIR OUT_JSON"""
import glob
import json
import os
import sys

import numpy as np

WT, SCORES, OFF, OUT = sys.argv[1:5]
BR = WT + "/repro/bench_results"

# paper bAcc per subset (Tables 2, 3, 4), in the order of the paper's columns; subset names map to our folder names
PAPER = {
    "genimage_val100k": {"order": ["midjourney", "sdv1.4", "sdv1.5", "adm", "glide", "wukong", "vqdm", "biggan"], "avg_table1": {"probe": 96.8, "probe_resnet": 78.4},
        "probe": [98.8, 99.4, 99.4, 91.0, 92.1, 99.5, 98.9, 95.0], "probe_resnet": [90.7, 99.4, 99.2, 60.0, 63.0, 99.4, 66.9, 49.4]},
    "synthbuster_coco": {"order": ["dalle2", "dalle3", "firefly", "glide", "midjourney-v5", "stable-diffusion-1-3", "stable-diffusion-1-4", "stable-diffusion-2", "stable-diffusion-xl"],
        "avg_table1": {"probe": 97.5, "probe_resnet": 79.1},
        "probe": [94.6, 99.4, 99.5, 86.7, 99.5, 99.5, 99.5, 99.5, 99.5], "probe_resnet": [48.3, 90.2, 46.9, 57.4, 93.2, 94.9, 94.9, 93.1, 93.0]},
    "synthwildx": {"order": ["dalle3", "firefly", "midjourney_v5"], "avg_table1": {"probe": 96.4, "probe_resnet": 75.6},
        "probe": [96.5, 96.4, 96.5], "probe_resnet": [90.3, 56.7, 79.7]},
    "wildrf": {"order": ["facebook", "reddit", "twitter"], "avg_table1": {"probe": 96.9, "probe_resnet": 81.3},
        "probe": [98.4, 93.4, 99.0], "probe_resnet": [87.8, 72.9, 83.3]},
    "aigibench": {"order": ["CommunityAI", "SocialRF"], "avg_table1": {"probe": 89.9, "probe_resnet": 70.4},
        "probe": [89.8, 89.9], "probe_resnet": [71.0, 69.9]},
}
MISSING = {"chameleon": {"probe": 86.6, "probe_resnet": 74.0}, "aigi_quality_paradox": {"probe": 92.9, "probe_resnet": 87.7}}
out = {"paper_missing": MISSING, "benchmarks": {}}
for bench, p in PAPER.items():
    out["benchmarks"][bench] = {}
    for model in ("probe", "probe_resnet"):
        f = f"{BR}/{model}_{bench}.json"
        if not os.path.exists(f):
            print(f"| {bench} | {model} | NOT RUN |")
            continue
        d = json.load(open(f))
        per = d["per_generator"]
        rows = []
        for name, pv in zip(p["order"], p[model]):
            m = per.get(name)
            rows.append({"subset": name, "paper": pv, "synthscope": m["acc"] if m else None,
                         "real_acc": m["real_acc"] if m else None, "fake_acc": m["fake_acc"] if m else None, "ap": m["ap"] if m else None})
        out["benchmarks"][bench][model] = {"rows": rows, "paper_avg_table1": p["avg_table1"][model], "paper_mean_of_listed": round(float(np.mean(p[model])), 2),
                                           "synthscope_mean": d["means"]["acc"], "synthscope_map": d["means"]["map"], "tag": d["tag"], "data_root": d["data_root"],
                                           "extra_subsets_in_run": sorted(set(per) - set(p["order"]))}
json.dump(out, open(OUT, "w"), indent=1)
for bench, bm in out["benchmarks"].items():
    for model, r in bm.items():
        print(f"\n### {bench} / {model}  (tag {r['tag']})  paper Table1 avg {r['paper_avg_table1']} | SynthScope mean {r['synthscope_mean']}")
        print("| subset | paper | SynthScope | real | fake |")
        print("|---|---|---|---|---|")
        for x in r["rows"]:
            print(f"| {x['subset']} | {x['paper']} | {x['synthscope']} | {x['real_acc']} | {x['fake_acc']} |")
