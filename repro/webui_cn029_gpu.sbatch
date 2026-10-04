#!/bin/bash
# GPU web UI on a FREE cees node (cn029). cn023 holds all detector checkpoints on node-local
# /hddstore but its GPUs are booked 4+ days, so we PULL the served checkpoints from cn023 into
# cn029 /ssdstore (allowed: we hold the CPU-UI job 188131 on cn023, so cn029->cn023 ssh passes
# pam_slurm_adopt), then serve on the cn029 GPU. ~17G one-time pull, excludes the big _train/.
#SBATCH --job-name=agid_webui_gpu --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=1-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/webui_gpu_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
SRC=ivi-cn023
CKPT=/ssdstore/ssalehi/AGID/checkpoints
DINO=/ssdstore/ssalehi/AGID/dinov3_checkpoints
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
mkdir -p "$CKPT" "$DINO"
echo "=== pull served checkpoints from $SRC (exclude _train) ==="
rsync -a --exclude='*/' "$SRC:/hddstore/ssalehi/AGID/checkpoints/" "$CKPT/" || { echo "rsync ckpts FAILED"; exit 1; }
rsync -a "$SRC:/hddstore/ssalehi/AGID/dinov3_checkpoints/" "$DINO/" || { echo "rsync dinov3 FAILED"; exit 1; }
echo "staged: $(ls $CKPT/*.pth 2>/dev/null | wc -l) ckpts, dinov3=$(ls $DINO/*.pth 2>/dev/null | wc -l)"

export AGID_CKPT_DIR="$CKPT"
export AGID_MFM_CKPT=/ssdstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth
export DINO_V3_WEIGHTS="$DINO/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf
export AGID_PORT=${AGID_PORT:-8021} AGID_HOST=0.0.0.0

# CUDA precheck (a5000 on cees is reliable, but verify)
ok=0
for i in 1 2 3; do
  $CONDA run --no-capture-output -n AGID python -c "import torch; assert torch.cuda.is_available(); (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item(); print('CUDA OK', torch.cuda.get_device_name(0))" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = 1 ] || { echo "CUDA NEVER CAME UP"; exit 1; }

echo "[webui] serving on http://$(hostname):$AGID_PORT  (GPU, CKPT_DIR=$AGID_CKPT_DIR)"
echo "[webui] tunnel:  ssh -N -L ${AGID_PORT}:$(hostname):${AGID_PORT} ssalehi@ivi-h1.science.uva.nl"
exec bash interface/run_backend.sh
