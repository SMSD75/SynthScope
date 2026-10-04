#!/bin/bash
#SBATCH --job-name=safe_reev
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/safe_reeval_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
CONDA=/home/ssalehi/miniconda3/bin/conda
CK=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/safe_full/model_18.pth
DATA=""
for d in /ssdstore /nvmestore /scratch; do [ -d "$d/ssalehi/SynthScope/CNNDetection/test" ] && DATA="$d/ssalehi/SynthScope/CNNDetection/test" && break; done
[ -n "$DATA" ] || { echo "NO ForenSynths test"; exit 3; }
echo "[reeval] SynthScope-trained SAFE model_18, crop-only transform (current registry), data=$DATA"
$CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model safe --ckpt "$CK" \
  --benchmark forensynths --data "$DATA" --bs 64 --device cuda --tag safe_synthscopetrained_cropfix 2>&1 | tail -25
echo "=== SAFE RE-EVAL (crop) COMPLETE $(date) ==="
