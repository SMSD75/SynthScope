#!/bin/bash
#SBATCH --job-name=wevis_bench
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/wevis_%x_%j.out
# WeVisForensics (wevis) on the video_agent_bench subsets: one worker of repro/video_agent_bench/run_units.py on one RTX
# A6000 of ivi-cn023 (every wevis row of the benchmark runs on this GPU type). Several workers share the shards through
# flock locks in $R/locks; the agent's runner stays loaded from one unit to the next.
#   sbatch --job-name=wevis_w0 repro/video_agent_bench/wevis_worker.sbatch          (from the worktree root; the committed
#   HEAD is copied first and run from the copy)
#   MODE=pilot LIST=<file of paths relative to $R/subsets> sbatch ... : score only those videos (--score-list) into
#   $R/cache_pilot and stop (cost estimate).
#   CMP=1: also run repro/video_agent_bench/comparators.sh in the background on the same GPU (the cheap detectors).
# Layout on ivi-cn023:/hddstore/ssalehi/SynthScope/video_agent_bench ($R): subsets/ (staged, make_subsets.py verify),
# ckpts/ (base + adapters + PP-OCRv5, sha256-checked; the runner verifies every pinned file again), cache/wevis/<bench>,
# out/<bench>/, locks/, code/<job>, work/<job> (frame caches).
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
G=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared
R=/hddstore/ssalehi/SynthScope/video_agent_bench
DEPS=$G/vlm_pydeps
REF=$G/vlm_ref/WeVisForensics
PY=/home/ssalehi/miniconda3/envs/SynthScope/bin/python
WT=${SYNTHSCOPE_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$WT/repro/video_agent_bench/run_units.py" ] || { echo "$WT is not the SynthScope worktree root"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
[ "$(git -C $REF rev-parse --short=7 HEAD)" = 43ae1c0 ] || { echo "official clone is not at 43ae1c0"; exit 5; }
SHA=$(git -C $WT rev-parse HEAD)
W=$R/code/$SLURM_JOB_ID
mkdir -p $W $R/work/$SLURM_JOB_ID/tmp $R/locks $R/out && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
free -g | head -2

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export HF_HOME=$R/work/$SLURM_JOB_ID/hf XDG_CACHE_HOME=$R/work/$SLURM_JOB_ID TMPDIR=$R/work/$SLURM_JOB_ID/tmp
export TRITON_CACHE_DIR=$G/wevis_gate/stage/_cache/triton WVF_OCR_THREADS=6
export SYNTHSCOPE_CKPT_DIR=$R/ckpts SYNTHSCOPE_CACHE_DIR=$R/work/$SLURM_JOB_ID \
  SYNTHSCOPE_WEVIS_PYTHONPATH=$DEPS/wevis:$DEPS/wevis_ocr SYNTHSCOPE_WEVIS_CODE=$REF \
  SYNTHSCOPE_WEVIS_PADDLEX=$R/ckpts/_cache/paddlex PYTHONPATH=$W
cd $W

if [ "${CMP:-0}" = 1 ]; then
  echo "=== comparators in the background $(date +%T)"
  W=$W R=$R bash $W/repro/video_agent_bench/comparators.sh > >(sed -u 's/^/[cmp] /') 2>&1 &
  CMP_PID=$!
fi

if [ "${MODE:-units}" = pilot ]; then
  echo "=== pilot: ${LIST:?} $(date +%T)"
  $PY -W ignore benchmark/bench_eval_video.py -m wevis --score-list $LIST --base $R/subsets --cache-dir $R/cache_pilot
else
  $PY -W ignore repro/video_agent_bench/run_units.py --units ${UNITS:-all} --benchmarks "${BENCHES:-}" --shards ${SHARDS:-4} \
    --subsets-root $R/subsets --cache-root $R/cache/wevis --out-root $R/out --lock-dir $R/locks || echo "run_units rc=$?"
fi
if [ -n "${CMP_PID:-}" ]; then echo "=== waiting for the comparators $(date +%T)"; wait $CMP_PID || echo "comparators rc=$?"; fi
echo "done $(date)"
cd / && rm -rf $W $R/work/$SLURM_JOB_ID
