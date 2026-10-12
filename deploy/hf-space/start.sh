#!/bin/bash
# Space entry point: fetch the approved detector weights, then serve the page on the port Spaces expect (7860).
set -e
python deploy/hf-space/fetch_weights.py
# Data for the "Browse datasets" tab. By default a small demo pack (a few files per generator from several openly licensed
# benchmarks, ~a few hundred MB) is fetched from the Hugging Face dataset repo SYNTHSCOPE_DEMO_DATA_REPO. Setting
# SYNTHSCOPE_SPACE_DATASETS (e.g. "image:commgen15:CommGen15") instead downloads those full datasets with their loaders.
# Both run in the background; the tab appears once a dataset is ready.
export SYNTHSCOPE_DATA_DIR=${SYNTHSCOPE_DATA_DIR:-$HOME/data}
(
  if [ -n "${SYNTHSCOPE_SPACE_DATASETS:-}" ]; then
    for spec in $SYNTHSCOPE_SPACE_DATASETS; do
      IFS=: read -r modality name folder <<< "$spec"
      python benchmark/download_dataset.py "$name" --modality "$modality" --root "$SYNTHSCOPE_DATA_DIR/$folder" \
        > "$HOME/download_$name.log" 2>&1 && echo "[data] $name ready" || echo "[data] $name failed (see $HOME/download_$name.log)"
    done
  elif [ -n "${SYNTHSCOPE_DEMO_DATA_REPO-SMSD75/synthscope-demo-data}" ]; then
    python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ.get('SYNTHSCOPE_DEMO_DATA_REPO', 'SMSD75/synthscope-demo-data'), repo_type='dataset', local_dir=os.environ['SYNTHSCOPE_DATA_DIR'], token=os.environ.get('HF_TOKEN'))" \
      > "$HOME/download_demo_data.log" 2>&1 && echo "[data] demo pack ready" || echo "[data] demo pack failed (see $HOME/download_demo_data.log)"
  fi
) &
exec python -m uvicorn interface.backend.app:app --host 0.0.0.0 --port 7860
