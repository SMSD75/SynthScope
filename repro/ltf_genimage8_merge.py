"""Merge the per-subset-group LTF GenImage runs into the two result files (SynthScope bench_eval + official evaluate.py).

SynthScope parts: repro/bench_results/ltf_part_*.json (benchmark/bench_eval.py, one job per subset group).
Official parts:   <official_out>/full_<subset>/summary.json (ltf_ref/evaluate.py, fusion branch, threshold 0.5).

    python repro/ltf_genimage8_merge.py --official-out /path/to/ltf_off_out --suffix genimage8 \
        --jobs-sd 277009,277011,... --jobs-off 277010,...
"""
import argparse
import datetime
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "bench_results"
MASTER = ROOT / "MASTER_RESULTS.jsonl"
SUBSETS = ("midjourney", "sdv1.4", "sdv1.5", "adm", "glide", "wukong", "vqdm", "biggan")


def mean(values):
    return round(sum(values) / len(values), 2)


def means_of(per_gen):
    return {"acc": mean([m["acc"] for m in per_gen.values()]), "map": mean([m["ap"] for m in per_gen.values()]),
            "mauroc": mean([m["auroc"] for m in per_gen.values()])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-out", required=True)
    ap.add_argument("--parts-glob", default=str(RESULTS / "ltf_part_*.json"))
    ap.add_argument("--off-prefix", default="full")
    ap.add_argument("--suffix", default="genimage8")
    ap.add_argument("--data-root", default="/ssdstore/ssalehi/SynthScope/GenImage_full/test")
    ap.add_argument("--jobs-sd", default="")
    ap.add_argument("--jobs-off", default="")
    args = ap.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")

    sd = {}
    for part in sorted(glob.glob(args.parts_glob)):
        sd.update(json.loads(Path(part).read_text())["per_generator"])
    off, counts = {}, {}
    for name in SUBSETS:
        path = Path(args.official_out) / f"{args.off_prefix}_{name}" / "summary.json"
        if not path.is_file():
            continue
        (subset, branches), = json.loads(path.read_text())["subsets"].items()
        fusion = branches["fusion"]
        counts[name] = (fusion["real_images"], fusion["fake_images"])
        off[name] = {"ap": round(fusion["ap"] * 100, 2), "auroc": round(fusion["auc"] * 100, 2),
                     "real_acc": round((1 - fusion["real_fpr"]) * 100, 2), "fake_acc": round(fusion["fake_recall"] * 100, 2),
                     "acc": round(fusion["accuracy"] * 100, 2)}
    outputs = {
        "synthscope": (sd, "registry-default (ltf.pth = official weights/epoch_20.pth)", "benchmark/bench_eval.py --model ltf",
                       args.jobs_sd),
        "official": (off, "official ltf_ref/evaluate.py (weights/epoch_20.pth, fusion branch)", "ltf_ref/evaluate.py",
                     args.jobs_off),
    }
    for kind, (per_gen, ckpt, how, jobs) in outputs.items():
        if set(per_gen) != set(SUBSETS):
            print(f"{kind}: missing {sorted(set(SUBSETS) - set(per_gen))}; not written")
            continue
        per_gen = {k: per_gen[k] for k in sorted(per_gen)}
        result = {"tag": f"ltf_{kind}_{args.suffix}", "model": "ltf", "ckpt": ckpt, "benchmark": "pergen",
                  "data_root": args.data_root, "jpeg_fakes": None, "n_generators": len(per_gen),
                  "means": means_of(per_gen), "per_generator": per_gen, "run_with": how, "slurm_jobs": jobs,
                  "date": now}
        if kind == "official":
            result["images_per_subset_real_fake"] = counts
        out = RESULTS / f"{result['tag']}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        with MASTER.open("a") as f:
            f.write(json.dumps({k: result[k] for k in ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                                                        "n_generators", "means")}) + "\n")
        print(kind, result["means"], "->", out)


if __name__ == "__main__":
    main()
