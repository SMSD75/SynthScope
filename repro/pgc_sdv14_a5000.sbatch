#!/bin/bash
#SBATCH --job-name=pgc_sdv14_a5
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:8
#SBATCH --exclude=ivi-cn020,ivi-cn029
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_sdv14_a5_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
HS=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/sdv14_handshake
mkdir -p "$HS"
echo "NODE=$(hostname)  GPUS=$CUDA_VISIBLE_DEVICES  $(date)"

$CONDA run --no-capture-output -n SynthScope python -c \
  "import torch; assert torch.cuda.is_available(); print('CUDA OK,', torch.cuda.device_count(), 'GPUs')" || { echo "CUDA FAIL"; exit 1; }

# --- data: pre-staged? else handshake: announce node, wait for cn023 push ---
DATA=""
for d in /ssdstore /nvmestore /scratch /local; do
  [ -d "$d/ssalehi/AGID/GenImage/train/real" ] && DATA="$d/ssalehi/AGID/GenImage" && break
done
if [ -z "$DATA" ]; then
  SCRBASE=""; for d in /ssdstore /nvmestore /scratch /local; do [ -d "$d" ] && [ -w "$d" ] && SCRBASE="$d" && break; done
  DEST="$SCRBASE/ssalehi/AGID/GenImage"; mkdir -p "$DEST"
  hostname > "$HS/NODE"; rm -f "$HS/DATA_READY"
  echo "[data] requested push to $(hostname):$DEST — waiting for cn023"
  until [ -f "$HS/DATA_READY" ]; do sleep 30; done
  DATA="$DEST"
fi
echo "[data] using $DATA ($(du -sh $DATA 2>/dev/null | cut -f1))"

export PGC_DATA_ROOT="$DATA/train"
export PGC_N=400000 PGC_EPOCHS=100 PGC_BS=16
export PGC_OUT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_sdv14_8gpu/model.pth
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf
export PYTHONUNBUFFERED=1
NG=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "[train] DDP on $NG GPUs (GenImage SD-v1.4, exact recipe)"
$CONDA run --no-capture-output -n SynthScope accelerate launch \
  --multi_gpu --num_processes $NG --mixed_precision bf16 --main_process_port 29573 \
  repro/train_pgc_ddp.py
echo "=== SDV14-A5000 PIPELINE COMPLETE ==="
