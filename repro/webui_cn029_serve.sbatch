#!/bin/bash
# Serve-only GPU web UI on cn029 (checkpoints + datasets ALREADY staged on /ssdstore by the
# deref job). No rsync here -- re-pulling with -a would clobber the materialized files with
# dangling symlinks. Same node + port (8021) so the existing ssh tunnel keeps working.
#SBATCH --job-name=agid_webui_gpu --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=1-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/webui_serve_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export AGID_CKPT_DIR=/ssdstore/ssalehi/AGID/checkpoints
export AGID_MFM_CKPT=/ssdstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth
export DINO_V3_WEIGHTS=/ssdstore/ssalehi/AGID/dinov3_checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf
export AGID_PORT=${AGID_PORT:-8021} AGID_HOST=0.0.0.0
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES PORT=$AGID_PORT $(date)"
echo "ckpts: $(ls $AGID_CKPT_DIR/*.pth 2>/dev/null | wc -l) | dataset roots checked under /hddstore->/ssdstore mirror"
ok=0
for i in 1 2 3; do
  $CONDA run --no-capture-output -n AGID python -c "import torch; assert torch.cuda.is_available(); (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item(); print('CUDA OK', torch.cuda.get_device_name(0))" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = 1 ] || { echo "CUDA NEVER CAME UP"; exit 1; }
echo "[webui] tunnel:  ssh -N -L ${AGID_PORT}:$(hostname):${AGID_PORT} ssalehi@ivi-h1.science.uva.nl"
exec bash interface/run_backend.sh
