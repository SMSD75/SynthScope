#!/bin/bash
#SBATCH --job-name=pgc_evfix --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/pgc_evfix_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
CONDA=/home/ssalehi/miniconda3/bin/conda
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/.cache_shared/hf PYTHONUNBUFFERED=1
H=/hddstore/ssalehi/SynthScope
echo "NODE=$(hostname) $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
run(){ local tag="$1"; shift; local full=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/${tag}_${SLURM_JOB_ID}.full.log
  echo "===== $tag ====="
  $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model pgc \
    --benchmark forensynths --data "$H/CNNDetection/test" --bs 32 --device cuda --tag "$tag" "$@" > "$full" 2>&1
  grep -aE "RESULT|: acc=|=== bench_eval" "$full" | tail -16; }
# served official weights (registry default) — the one that matters
run pgc_evalfix_official
# step900 reproduction ckpt
run pgc_evalfix_step900 --ckpt /ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/pgc_400step/step900_acc66_ap94.pth
echo "=== PGC EVAL-FIX FORENSYNTHS DONE $(date) ==="
