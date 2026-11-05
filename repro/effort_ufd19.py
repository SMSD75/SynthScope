"""Combine the two bench_eval runs of one Effort checkpoint (11 ForenSynths GAN sets + 8 UFD diffusion sets) into the paper's
19-subset UniversalFakeDetect table (Effort paper Tables 2/3: mAP and mean accuracy at threshold 0.5, mean over the 19 subsets).

    python repro/effort_ufd19.py effort_ufd_step18000 [effort_ufd_step21000 ...]

Reads repro/bench_results/<tag>_gan11.json and <tag>_ufd8.json, writes repro/bench_results/<tag>_ufd19.json and appends a line
to repro/MASTER_RESULTS.jsonl. Nothing is computed beyond the means of the per-subset numbers those runs wrote.
"""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "bench_results"
# The official subset order (UniversalFakeDetect_Benchmark/dataset_paths.py): key in the paper -> folder name in the data.
SUBSETS = {"progan": "progan", "cyclegan": "cyclegan", "biggan": "biggan", "stylegan": "stylegan", "gaugan": "gaugan",
           "stargan": "stargan", "deepfake": "deepfake", "sitd": "seeingdark", "san": "san", "crn": "crn", "imle": "imle",
           "guided": "guided", "ldm_200": "ldm_200", "ldm_200_cfg": "ldm_200_cfg", "ldm_100": "ldm_100",
           "glide_100_27": "glide_100_27", "glide_50_27": "glide_50_27", "glide_100_10": "glide_100_10", "dalle": "dalle"}


def combine(tag: str) -> dict:
    gan, dif = (json.loads((RESULTS / f"{tag}_{part}.json").read_text()) for part in ("gan11", "ufd8"))
    pool = {**gan["per_generator"], **dif["per_generator"]}
    per = {key: pool[folder] for key, folder in SUBSETS.items()}

    def mean(field: str) -> float:
        return round(sum(v[field] for v in per.values()) / len(per), 2)

    return {"tag": f"{tag}_ufd19", "model": gan["model"], "ckpt": gan["ckpt"], "model_config": gan.get("model_config"),
            "benchmark": "ufd19", "data_root": f"{gan['data_root']} + {dif['data_root']}", "n_generators": len(per),
            "means": {"acc": mean("acc"), "map": mean("ap"), "mauroc": mean("auroc")}, "per_generator": per,
            "sources": [f"{tag}_gan11", f"{tag}_ufd8"], "date": datetime.datetime.now().isoformat(timespec="seconds")}


def main(tags: list[str]) -> None:
    for tag in tags:
        result = combine(tag)
        (RESULTS / f"{result['tag']}.json").write_text(json.dumps(result, indent=2) + "\n")
        with (ROOT / "MASTER_RESULTS.jsonl").open("a") as f:
            keys = ("date", "tag", "model", "ckpt", "model_config", "benchmark", "data_root", "n_generators", "means")
            f.write(json.dumps({k: result[k] for k in keys}) + "\n")
        print(result["tag"], result["means"])


if __name__ == "__main__":
    main(sys.argv[1:])
