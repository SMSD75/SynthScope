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
# B-Free's GRIP license allows informational and nonprofit use, which this free demo is; Effort and STALL are CC BY-NC
# 4.0 (STALL's DINOv3 backbone is under the DINOv3 License).
DEFAULT = ["clip", "cnn", "npr", "safe", "aide", "co_spy", "fat_former", "pgc", "cure", "dda", "d3qe", "omniaid", "bfree", "effort",
           "vib_net", "dinov2", "adapted_dino", "restrav", "stall"]
DINOV3_FILE = "dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


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
    # STALL is training-free but runs on the gated DINOv3 ViT-L/16 backbone, kept in the weights repo with its license
    # (DINOv3 License: use and redistribution allowed with the license and attribution); DINO_V3_WEIGHTS points at it.
    if "stall" in keys and os.environ.get("DINO_V3_WEIGHTS") and not Path(os.environ["DINO_V3_WEIGHTS"]).exists():
        from huggingface_hub import hf_hub_download
        try:
            for name in (DINOV3_FILE, "dinov3/LICENSE.md"):
                hf_hub_download(repo_id=repo, filename=name, local_dir=str(ckpt_dir), token=token)
            print("[weights] stall: DINOv3 backbone ok", flush=True)
        except Exception as e:
            print(f"[weights] stall: skipped ({type(e).__name__}: {str(e)[:400]})", flush=True)
    keys = [k for k in keys if k != "stall"]       # no <key>.pth of its own
    for key in keys:
        if (ckpt_dir / f"{key}.pth").exists():
            continue
        try:
            hub.download(key, repo_id=repo, token=token)
            print(f"[weights] {key}: ok", flush=True)
        except Exception as e:                      # keep starting: the page marks the detector as unavailable
            print(f"[weights] {key}: skipped ({type(e).__name__}: {str(e)[:400]})", flush=True)


if __name__ == "__main__":
    main()
