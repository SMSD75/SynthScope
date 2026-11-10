"""Official evaluate_{dino,resnet}.py vs SynthScope bench_eval on the same files (WildRF, SynthWildX): per subset bAcc of both,
file-set differences, per-file P(fake) differences. usage: python probe_task2_official_cmp.py SCORES_DIR OFFICIAL_DIR JOBID OUT_JSON BENCH_RESULTS_DIR"""
import json
import sys

import numpy as np

SC, OF, JOB, OUT, BR = sys.argv[1:6]   # BR = <worktree>/repro/bench_results
res = {}
for model, v in (("probe", "dino"), ("probe_resnet", "resnet")):
    for bench in ("wildrf", "synthwildx"):
        sc = json.load(open(f"{SC}/{model}_{bench}.json"))
        of = json.load(open(f"{OF}/probe_official_{v}_{bench}_{JOB}.json"))
        ours = {g["name"]: dict(zip(g["paths"], zip(g["probs"], g["labels"]))) for g in sc["gens"]}
        rows = {}
        for name, o in of["subsets"].items():
            s = ours[name]
            op = dict(zip(o["paths"], zip(o["probs"], o["labels"])))
            common = sorted(set(s) & set(op))
            a = np.array([s[p][0] for p in common]); b = np.array([op[p][0] for p in common]); y = np.array([s[p][1] for p in common])
            def bacc(p, yy):
                return ((p[yy == 0] < 0.5).mean() + (p[yy == 1] >= 0.5).mean()) * 50
            sy = np.array([x[1] for x in s.values()]); sp = np.array([x[0] for x in s.values()])
            rows[name] = {"n_synthscope": len(s), "n_official": len(op), "n_common": len(common),
                          "only_synthscope": sorted({p.rsplit('.', 1)[-1].lower() for p in set(s) - set(op)}), "n_only_synthscope": len(set(s) - set(op)),
                          "n_only_official": len(set(op) - set(s)),
                          "synthscope_bacc_all": round(float(bacc(sp, sy)), 2), "official_bacc_all": round(o["full"]["bacc"], 2),
                          "synthscope_bacc_common": round(float(bacc(a, y)), 2), "official_bacc_common": round(float(bacc(b, y)), 2),
                          "official_overall_acc_balanced_subsample": round(o["official"]["acc"], 2),
                          "official_subsample_n": [o["official"]["n_real"], o["official"]["n_fake"]],
                          "official_ap": round(o["official"]["ap"], 2),
                          "max_abs_dP": float(np.abs(a - b).max()), "mean_abs_dP": float(np.abs(a - b).mean()),
                          "flips_at_0.5": int(((a >= 0.5) != (b >= 0.5)).sum())}
        res[f"{model}_{bench}"] = {"rows": rows, "official_mean_bacc_all": round(float(np.mean([r["official_bacc_all"] for r in rows.values()])), 2),
                                   "official_mean_overall_acc": round(float(np.mean([r["official_overall_acc_balanced_subsample"] for r in rows.values()])), 2),
                                   "synthscope_mean": json.load(open(f"{BR}/{model}_{bench}.json"))["means"]["acc"]}
json.dump(res, open(OUT, "w"), indent=1)
for k, r in res.items():
    print(f"\n## {k}: SynthScope mean {r['synthscope_mean']} | official bAcc(all files) {r['official_mean_bacc_all']} | official Overall (seed-42 balanced) {r['official_mean_overall_acc']}")
    for n, x in r["rows"].items():
        print(f"  {n}: SS {x['synthscope_bacc_all']} (n={x['n_synthscope']}) | OFF bAcc {x['official_bacc_all']} (n={x['n_official']}) | OFF Overall {x['official_overall_acc_balanced_subsample']} "
              f"subsample {x['official_subsample_n']} | common n={x['n_common']}: SS {x['synthscope_bacc_common']} OFF {x['official_bacc_common']} "
              f"maxdP {x['max_abs_dP']:.2e} flips {x['flips_at_0.5']} | only-SS {x['n_only_synthscope']} {x['only_synthscope']}")
