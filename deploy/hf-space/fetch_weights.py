"""Download the detector weights the public Space may serve, from a Hugging Face model repo.

Configure the Space with two settings (Space -> Settings -> Variables and secrets):
  SYNTHSCOPE_HUB_REPO   model repo holding <key>.pth files (it can be private)       e.g. SMSD75/synthscope-checkpoints
  HF_TOKEN              secret; a read token for that repo (only needed if the repo is private)
SYNTHSCOPE_SPACE_DETECTORS (comma list) overrides which detectors are fetched. Missing files are skipped, so a detector
without weights simply shows as unavailable on the page. Training-free detectors (D3) need no file.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Detectors whose upstream licenses allow a free public demo. NPR's repository has no license; its first author gave
# written permission (2026-10-11) to use NPR.pth in this free, non-commercial demo with credit (see LICENSES.md).
# B-Free's GRIP license allows informational and nonprofit use, which this free demo is.
DEFAULT = ["clip", "cnn", "npr", "safe", "aide", "co_spy", "fat_former", "pgc", "cure", "dda", "d3qe", "omniaid", "bfree",
           "vib_net", "dinov2", "adapted_dino", "restrav"]


def main() -> None:
    repo = os.environ.get("SYNTHSCOPE_HUB_REPO")
    if not repo:
        print("[weights] SYNTHSCOPE_HUB_REPO is not set: starting without detector weights", flush=True)
        return
    keys = [k for k in os.environ.get("SYNTHSCOPE_SPACE_DETECTORS", ",".join(DEFAULT)).split(",") if k]
    from synthscope import hub
    token = os.environ.get("HF_TOKEN") or None
    ckpt_dir = Path(os.environ.get("SYNTHSCOPE_CKPT_DIR", "~/ckpts")).expanduser()
    # D3 is training-free (no .pth) but turns its raw score into a probability with a small calibration file
    if not (ckpt_dir / "d3_calibration.json").exists():
        hub._maybe_download_aux("d3", repo, token)
    for key in keys:
        if (ckpt_dir / f"{key}.pth").exists():
            continue
        try:
            hub.download(key, repo_id=repo, token=token)
            print(f"[weights] {key}: ok", flush=True)
        except Exception as e:                      # keep starting: the page marks the detector as unavailable
            print(f"[weights] {key}: skipped ({type(e).__name__})", flush=True)


if __name__ == "__main__":
    main()
