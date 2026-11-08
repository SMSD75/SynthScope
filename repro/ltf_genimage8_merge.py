"""Merge the per-subset-group LTF GenImage runs into the two result files (SynthScope bench_eval + official evaluate.py).

SynthScope parts: one bench_eval.py JSON per job, kept outside the repo in SynthScope_shared (``--parts-glob``, required):
                  ltf_parts/ltf_part_*.json for GenImage_full/test (33k), ltf_fullparts/ltf_fullpart_*.json for the full val.
Official parts:   <official_out>/full_<subset>/summary.json (ltf_ref/evaluate.py, fusion branch, threshold 0.5); the
                  official run exists for the 33k set only, so the full-val merge omits --official-out.

    python repro/ltf_genimage8_merge.py --official-out SynthScope_shared/ltf_off_out --suffix genimage8 \
        --parts-glob 'SynthScope_shared/ltf_parts/ltf_part_*.json' --jobs-sd 277009,277011,... --jobs-off 277010,...

Full val (100,000 images; SynthScope only, so no --official-out):

    python repro/ltf_genimage8_merge.py --suffix genimage_val100k \
        --parts-glob 'SynthScope_shared/ltf_fullparts/ltf_fullpart_*.json' \
        --data-root /ssdstore/ssalehi/SynthScope/GenImage_val_full/test --jobs-sd 277066,277011,277068,...,277073 \
        --note '...'

MASTER_RESULTS.jsonl gets one line per tag; a tag that is already there is skipped (the result JSON is still rewritten),
with a warning if its means differ from the existing line.
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


def master_means():
    """tag -> means of the lines already in MASTER_RESULTS.jsonl."""
    if not MASTER.is_file():
        return {}
    lines = (json.loads(line) for line in MASTER.read_text().splitlines() if line.strip())
    return {line["tag"]: line.get("means") for line in lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-out", default=None, help="folder of the official full_<subset>/summary.json (omit: no official file)")
    ap.add_argument("--parts-glob", required=True, help="glob of the per-job SynthScope bench_eval JSONs")
    ap.add_argument("--off-prefix", default="full")
    ap.add_argument("--suffix", default="genimage8")
    ap.add_argument("--data-root", default="/ssdstore/ssalehi/SynthScope/GenImage_full/test")
    ap.add_argument("--jobs-sd", default="")
    ap.add_argument("--jobs-off", default="")
    ap.add_argument("--note", default="", help="free text stored in the SynthScope result JSON")
    args = ap.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")

    sd = {}
    for part in sorted(glob.glob(args.parts_glob)):
        sd.update(json.loads(Path(part).read_text())["per_generator"])
    off, counts = {}, {}
    for name in SUBSETS:
        path = Path(args.official_out or "") / f"{args.off_prefix}_{name}" / "summary.json"
        if not args.official_out or not path.is_file():
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
        if kind == "official" and not args.official_out:
            continue                                       # omitted on purpose (no official run for this set)
        if set(per_gen) != set(SUBSETS):
            print(f"{kind}: missing {sorted(set(SUBSETS) - set(per_gen))}; not written")
            continue
        per_gen = {k: per_gen[k] for k in sorted(per_gen)}
        result = {"tag": f"ltf_{kind}_{args.suffix}", "model": "ltf", "ckpt": ckpt, "benchmark": "pergen",
                  "data_root": args.data_root, "jpeg_fakes": None, "n_generators": len(per_gen),
                  "means": means_of(per_gen), "per_generator": per_gen, "run_with": how, "slurm_jobs": jobs,
                  "date": now}
        if args.note and kind == "synthscope":
            result["note"] = args.note
        if kind == "official":
            result["images_per_subset_real_fake"] = counts
        out = RESULTS / f"{result['tag']}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        existing = master_means()
        if result["tag"] in existing:
            print(f"{result['tag']} is already in {MASTER.name}: not appended")
            if existing[result["tag"]] != result["means"]:
                print(f"WARNING: its means {existing[result['tag']]} differ from the new {result['means']}; "
                      f"{MASTER.name} was not updated")
        else:
            with MASTER.open("a") as f:
                f.write(json.dumps({k: result[k] for k in ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                                                            "n_generators", "means")}) + "\n")
        print(kind, result["means"], "->", out)


if __name__ == "__main__":
    main()
