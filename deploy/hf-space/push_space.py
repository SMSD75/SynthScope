"""Assemble the Hugging Face Space from this repository and (only with --confirm) upload it.

    python deploy/hf-space/push_space.py --space SMSD75/SynthScope              # dry run: list what would be uploaded
    python deploy/hf-space/push_space.py --space SMSD75/SynthScope --confirm    # create/update the Space

The Space gets the tracked repository files (minus job logs), with this folder's Dockerfile at the root and SPACE_README.md as
its README.md (the Space card). Needs a Hugging Face token with write access (HF_TOKEN or `huggingface-cli login`).
After the first upload, set SYNTHSCOPE_HUB_REPO (and HF_TOKEN as a secret, if the weights repo is private) in the Space settings,
and pick the hardware (e.g. "CPU upgrade").
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXCLUDE = ("repro/logs/",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="owner/name of the Space")
    ap.add_argument("--private", action="store_true", help="create the Space as private (it can be made public later)")
    ap.add_argument("--hardware", default="cpu-upgrade",
                    help="Space hardware when creating it (Docker Spaces on free 'cpu-basic' need a PRO subscription)")
    ap.add_argument("--sleep", type=int, default=3600, help="seconds without visitors before the Space sleeps (paid hardware)")
    ap.add_argument("--confirm", action="store_true", help="actually upload (default: dry run)")
    a = ap.parse_args()

    files = [f for f in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
             if f and not f.startswith(EXCLUDE) and (ROOT / f).is_file()]
    size = sum((ROOT / f).stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e6:.1f} MB from the repository, plus Dockerfile and the Space card")
    if not a.confirm:
        print("dry run: nothing uploaded (add --confirm to upload)")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True,
                    space_hardware=a.hardware, space_sleep_time=a.sleep)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in files:
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, tmp / f)
        shutil.copy2(HERE / "Dockerfile", tmp / "Dockerfile")
        shutil.copy2(HERE / "SPACE_README.md", tmp / "README.md")
        api.upload_folder(folder_path=str(tmp), repo_id=a.space, repo_type="space",
                          commit_message="Update SynthScope Space from the GitHub repository")
    print(f"uploaded: https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
