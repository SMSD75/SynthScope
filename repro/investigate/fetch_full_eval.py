"""Re-fetch eval benchmarks at FULL size (no per-class cap) for the data-fidelity audit.

The eval-only loaders in dataset.py hardcode a per-class cap (GenImage 1500, others 1000) and
take the FIRST cap images alphabetically — i.e. a biased fraction of the paper's full test set.
This re-downloads the source archive and re-restructures with cap=None into a *_full sibling dir
so we can re-evaluate on the complete set without disturbing the existing capped dirs.

Usage: python fetch_full_eval.py <which>   where which in {genimage, aigcdetect, commgen15}
"""
import sys, shutil
from pathlib import Path

sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/SynthScope")
from dataset import _extract_archives_into, _restructure_image_pergen  # noqa: E402


def fetch_genimage(base="/hddstore/ssalehi/SynthScope/GenImage_full"):
    from huggingface_hub import hf_hub_download
    base = Path(base)
    dl = base / "_dl_full"; dl.mkdir(parents=True, exist_ok=True)
    print("downloading genimage_test.zip (25GB)...", flush=True)
    z = hf_hub_download(repo_id="jzousz/GenImage", filename="genimage_test.zip",
                        repo_type="dataset", local_dir=str(dl))
    print("extracting...", flush=True)
    staging = _extract_archives_into(Path(z).parent, base / "_staging_full")
    inner = next((p for p in staging.rglob("test") if p.is_dir()), staging)
    dest = base / "test_full"
    print(f"restructuring (cap=None) -> {dest}", flush=True)
    _restructure_image_pergen(inner, dest, cap=None, rename=lambda n: n.replace("_imagenet", ""))
    shutil.rmtree(base / "_staging_full", ignore_errors=True)
    print("done. per-gen counts:", flush=True)
    for g in sorted(dest.iterdir()):
        if g.is_dir():
            r = len(list((g / "0_real").glob("*"))) if (g / "0_real").is_dir() else 0
            f = len(list((g / "1_fake").glob("*"))) if (g / "1_fake").is_dir() else 0
            print(f"  {g.name}: real={r} fake={f}", flush=True)


def fetch_aigcdetect(base="/hddstore/ssalehi/SynthScope/AIGCDetectBenchmark"):
    base = Path(base)
    dl = base / "_dl_full"; dl.mkdir(parents=True, exist_ok=True)
    url = ("https://modelscope.cn/api/v1/datasets/aemilia/AIGCDetectionBenchmark/"
           "repo?Revision=master&FilePath=AIGCDetectionBenchMark%2Ftest_set.zip")
    import subprocess
    zp = dl / "test_set.zip"
    if not zp.exists():
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(zp), url], check=True)
    staging = _extract_archives_into(dl, base / "_staging_full")
    dest = base / "test_full"
    _restructure_image_pergen(staging, dest, cap=None)
    shutil.rmtree(base / "_staging_full", ignore_errors=True)
    for g in sorted(dest.iterdir()):
        if g.is_dir():
            r = len(list((g / "0_real").glob("*"))) if (g / "0_real").is_dir() else 0
            f = len(list((g / "1_fake").glob("*"))) if (g / "1_fake").is_dir() else 0
            print(f"  {g.name}: real={r} fake={f}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "genimage"
    {"genimage": fetch_genimage, "aigcdetect": fetch_aigcdetect}[which]()
