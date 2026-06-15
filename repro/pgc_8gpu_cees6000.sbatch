#!/bin/bash
#SBATCH --job-name=pgc_sdv14_8
#SBATCH --partition=cees6000
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --nodelist=ivi-cn031
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_sdv14_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
echo "NODE=$(hostname)  GPUS=$CUDA_VISIBLE_DEVICES  $(date)"

$CONDA run --no-capture-output -n AGID python -c \
  "import torch; assert torch.cuda.is_available(); print('CUDA OK,', torch.cuda.device_count(), 'GPUs')" || { echo "CUDA FAIL"; exit 1; }

# --- data: REUSE pre-staged GenImage if present (cn030/31 /nvmestore), else ssh-stream from cn023 ---
DATA=""
for d in /nvmestore /ssdstore /scratch /local; do
  [ -d "$d/ssalehi/AGID/GenImage/train" ] && DATA="$d/ssalehi/AGID/GenImage" && break
done
if [ -n "$DATA" ]; then
  echo "[data] REUSING pre-staged $DATA ($(du -sh $DATA 2>/dev/null | cut -f1))"
else
  SCR=""; for d in /nvmestore /ssdstore /scratch /local /tmp; do [ -d "$d" ] && [ -w "$d" ] && SCR="$d/ssalehi/pgc_sdv14_${SLURM_JOB_ID}" && break; done
  mkdir -p "$SCR"; echo "[data] ssh-streaming 88G GenImage train from cn023 -> $SCR"
  T0=$(date +%s)
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn023 \
    'tar -C /hddstore/ssalehi/AGID/GenImage -cf - train' | tar -C "$SCR" -xf - \
    || { echo "STREAM FAILED"; exit 1; }
  DATA="$SCR"; echo "[data] staged in $((($(date +%s)-T0)/60)) min"
fi

export PGC_DATA_ROOT="$DATA/train"
export PGC_N=400000 PGC_EPOCHS=${PGC_EPOCHS:-100} PGC_BS=16
export PGC_OUT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_sdv14_8gpu/model.pth
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf
export PYTHONUNBUFFERED=1
NG=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "[train] DDP on $NG GPUs (GenImage SD-v1.4 regime)"
$CONDA run --no-capture-output -n AGID accelerate launch \
  --multi_gpu --num_processes $NG --mixed_precision bf16 --main_process_port 29572 \
  repro/train_pgc_ddp.py
echo "[train] done $(date)"
echo "=== SDV14 8GPU PIPELINE COMPLETE ==="
