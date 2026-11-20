#!/bin/bash
#SBATCH --job-name=wevis_gbdl
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/wevis_gbdl_%j.out
# Download GenBuster-Bench and GenBuster-Bench++ with their SynthScope loaders (dataset.py GenBusterBenchDataset /
# GenBusterBenchPPDataset: pinned HF revisions, anonymous) to ivi-cn023's node-local /hddstore, then list the layout.
#   sbatch repro/video_agent_bench/download_genbuster.sbatch          (from the repo root)
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
PY=/home/ssalehi/miniconda3/envs/SynthScope/bin/python
WT=${SYNTHSCOPE_REPO:-$SLURM_SUBMIT_DIR}
S=/hddstore/ssalehi/SynthScope
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 | cut -c1-80) dirty=$(git -C $WT status --short --untracked-files=no | wc -l)"
cd $WT
export PYTHONPATH=$WT:$WT/interface:$WT/benchmark PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
for pair in "genbuster GenBuster-Bench" "genbuster_pp GenBuster-Bench-plusplus"; do
  set -- $pair
  echo "=== $1 -> $S/$2 $(date +%T)"
  $PY -W ignore -c "from datasets_util import ensure_dataset; print(ensure_dataset('$1', 'video', '$S/$2'))"
  cat $S/$2/SOURCE.json
  for g in $S/$2/test/*/; do echo "  $(basename $g): $(ls $g/0_real | wc -l) real, $(ls $g/1_fake | wc -l) fake"; done
  du -sh $S/$2
done
echo "done $(date)"
