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
# 4.0 (STALL's and the DINOv3 probe's backbone is under the DINOv3 License); SPAI is Apache-2.0 but its MFM backbone is
# under the S-Lab License (non-commercial use, which this free demo is).
DEFAULT = ["clip", "cnn", "npr", "safe", "aide", "co_spy", "fat_former", "pgc", "cure", "dda", "d3qe", "omniaid", "omniaid_v1",
           "omniaid_genimage", "bfree", "effort", "effort_chameleon", "moejepa", "vib_net", "dinov2", "dinov3", "adapted_dino", "spai",
           "restrav", "stall"]
DINOV3_FILE = "dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# Backbones that are not a detector's own <key>.pth, fetched with their licenses when one of those detectors is served.
BACKBONES = {"stall": [DINOV3_FILE, "dinov3/LICENSE.md"], "dinov3": [DINOV3_FILE, "dinov3/LICENSE.md"],
             "spai": ["mfm/mfm_pretrain_vit_base.pth", "mfm/LICENSE_S-Lab.txt", "licenses/SPAI_LICENSE.txt"]}


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
    # Gated / separately licensed backbones, kept in the weights repo with their licenses: DINOv3 ViT-L/16 for STALL and
    # the DINOv3 probe (DINO_V3_WEIGHTS / DINO_V3_WEIGHTS_DIR point at it), MFM ViT-B/16 for SPAI (SYNTHSCOPE_MFM_CKPT).
    from huggingface_hub import hf_hub_download
    for name in sorted({f for k in keys for f in BACKBONES.get(k, [])}):
        if (ckpt_dir / name).exists():
            continue
        try:
            hf_hub_download(repo_id=repo, filename=name, local_dir=str(ckpt_dir), token=token)
            print(f"[weights] backbone {name}: ok", flush=True)
        except Exception as e:
            print(f"[weights] backbone {name}: skipped ({type(e).__name__}: {str(e)[:400]})", flush=True)
    keys = [k for k in keys if k != "stall"]       # STALL is training-free: no <key>.pth of its own
    for key in keys:
        if (ckpt_dir / hub.repo_filename(key)).exists():
            continue
        try:
            hub.download(key, repo_id=repo, token=token)
            print(f"[weights] {key}: ok", flush=True)
        except Exception as e:                      # keep starting: the page marks the detector as unavailable
            print(f"[weights] {key}: skipped ({type(e).__name__}: {str(e)[:400]})", flush=True)


if __name__ == "__main__":
    main()
