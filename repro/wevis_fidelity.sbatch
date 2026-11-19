#!/bin/bash
#SBATCH --job-name=wevisfid
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/wevis_fidelity_%j.out
# WeVisForensics fidelity (repro/wevis_fidelity.py) on one cees A5000, from the shared staged copies of the Task 0 gate
# (SynthScope_shared/wevis_gate/stage: base + adapters + PP-OCRv5 models, sha256-verified; the clips):
#   sbatch --nodelist=ivi-cn027 repro/wevis_fidelity.sbatch          (submit from the worktree root)
# 1. the official backend (demo/wvf_local.py, unmodified) on 18 clips, under the isolated dependencies;
# 2. SynthScope's engine (runner process, sha256 + strict adapters, verdict logit during generation + a direct call);
# 3. the comparison -> repro/wevis_fidelity.json in the worktree.
# The worktree's committed HEAD is copied first (git archive) and run from the copy.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
G=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared
ST=$G/wevis_gate/stage
DEPS=$G/vlm_pydeps
REF=$G/vlm_ref/WeVisForensics
PY=/home/ssalehi/miniconda3/envs/SynthScope/bin/python
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/wevis_fidelity.py" ] || { echo "$WT is not the SynthScope repo root (submit from it, or set SYNTHSCOPE_REPO)"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
[ "$(git -C $REF rev-parse --short=7 HEAD)" = 43ae1c0 ] || { echo "official clone is not at 43ae1c0"; exit 5; }
J=$G/wevis_gate/fid/$SLURM_JOB_ID
SHA=$(git -C $WT rev-parse HEAD)
W=$J/code
mkdir -p $W $J/tmp $J/cache && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
free -g | head -2

# shared by both runs (the runner inherits them): offline, the staged files, the same OCR thread count and caches
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME=$J/cache/hf XDG_CACHE_HOME=$J/cache TMPDIR=$J/tmp TRITON_CACHE_DIR=$ST/_cache/triton
export PADDLE_PDX_CACHE_HOME=$ST/ckpts/_cache/paddlex PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True WVF_OCR_THREADS=6
cd $W

echo "=== staged files: sha256 $(date +%T) ==="
for d in qwen3.5-4b wevisforensics_lora; do (cd $ST/ckpts/$d && sha256sum -c --quiet SHA256SUMS) && echo "$d sha256 OK"; done
$PY repro/wevis_fidelity.py list

echo "=== 1. official backend (demo/wvf_local.py, unmodified), has_audio=False $(date +%T) ==="
WVF_BASE=$ST/ckpts/qwen3.5-4b WVF_ADAPTERS=$ST/ckpts/wevisforensics_lora WVF_WORK_DIR=$J/ref_work WVF_OCR_DEVICE=cpu \
  PYTHONPATH=$DEPS/wevis:$DEPS/wevis_ocr $PY repro/wevis_fidelity.py reference --out $J/reference.json

echo "=== 2. SynthScope engine (runner process) $(date +%T) ==="
SYNTHSCOPE_CKPT_DIR=$ST/ckpts SYNTHSCOPE_CACHE_DIR=$J/cache SYNTHSCOPE_WEVIS_PYTHONPATH=$DEPS/wevis:$DEPS/wevis_ocr \
  SYNTHSCOPE_WEVIS_CODE=$REF SYNTHSCOPE_WEVIS_PADDLEX=$ST/ckpts/_cache/paddlex PYTHONPATH=$W \
  $PY repro/wevis_fidelity.py ours --out $J/ours.json

echo "=== 3. comparison $(date +%T) ==="
$PY repro/wevis_fidelity.py compare --reference $J/reference.json --ours $J/ours.json \
  --log "repro/logs/wevis_fidelity_${SLURM_JOB_ID}.out" --out $WT/repro/wevis_fidelity.json
echo "done $(date)"
cd / && rm -rf $W $J/tmp && echo "removed the code copy (kept $J: reference.json, ours.json, frame caches)"
