#!/bin/bash
# Space entry point: fetch the approved detector weights, then serve the page on the port Spaces expect (7860).
set -e
python deploy/hf-space/fetch_weights.py
# Openly licensed datasets for the "Browse datasets" tab, downloaded in the background (the tab appears once one is ready).
# SYNTHSCOPE_SPACE_DATASETS="" turns this off.
export SYNTHSCOPE_DATA_DIR=${SYNTHSCOPE_DATA_DIR:-$HOME/data}
(
  for spec in ${SYNTHSCOPE_SPACE_DATASETS-image:commgen15:CommGen15 video:evalcrafter:EvalCrafter}; do
    IFS=: read -r modality name folder <<< "$spec"
    python benchmark/download_dataset.py "$name" --modality "$modality" --root "$SYNTHSCOPE_DATA_DIR/$folder" \
      > "$HOME/download_$name.log" 2>&1 && echo "[data] $name ready" || echo "[data] $name failed (see $HOME/download_$name.log)"
  done
) &
exec python -m uvicorn interface.backend.app:app --host 0.0.0.0 --port 7860
