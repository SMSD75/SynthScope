#!/bin/bash
# Re-eval D3 with the faithful protocol: X-CLIP-B/16 encoder + 10% edge-crop (registry now sets both).
# GenVideo-Val + EvalCrafter, on cn023 where the video data is local.
#SBATCH --job-name=d3_xclip --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=03:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/d3_xclip_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export HF_HOME=/home/ssalehi/.cache/huggingface PYTHONUNBUFFERED=1   # X-CLIP-B/16 cached here
H=/hddstore/ssalehi/AGID
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
run(){ local tag="$1" bench="$2" data="$3"; shift 3
  [ -e "$data" ] || { echo "[skip] $tag no data $data"; return; }
  echo "===== $tag ($bench) ====="
  local full=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/${tag}_${SLURM_JOB_ID}.full.log
  $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval_video.py --model d3 \
    --benchmark "$bench" --data "$data" --tag "$tag" "$@" > "$full" 2>&1
  local rc=$?
  grep -aE "RESULT|: acc=|=== bench_eval" "$full" | tail -18
  [ $rc -ne 0 ] && { echo "[FAIL rc=$rc]"; tail -25 "$full"; }; }
run d3_xclip_genvideo   genvideo "$H/d3_repro/genvideo_val/GenVideo-Val"
run d3_xclip_evalcrafter fakeonly "$H/EvalCrafter/test/1_fake"
echo "=== D3 XCLIP EVAL COMPLETE $(date) ==="
