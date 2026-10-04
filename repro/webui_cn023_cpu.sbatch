#!/bin/bash
# Serve the AGID web UI (FastAPI/uvicorn) on cn023 CPU-only. cn023's GPUs are fully allocated,
# but it is the only node with all detector checkpoints (node-local /hddstore). Single-image /
# clip inference on CPU is a few seconds — fine for the interactive demo, and frees no GPU from
# other users (more courteous than the 2-GPU allowance).
#SBATCH --job-name=agid_webui_cpu
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/webui_cpu_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export AGID_CKPT_DIR=/hddstore/ssalehi/AGID/checkpoints
export AGID_MFM_CKPT=/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf
export AGID_PORT=${AGID_PORT:-8021} AGID_HOST=0.0.0.0
export CUDA_VISIBLE_DEVICES=""          # force CPU; config.device() auto-detects -> cpu
echo "NODE=$(hostname) (CPU-only) PORT=$AGID_PORT $(date)"
$CONDA run --no-capture-output -n AGID python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[webui] serving on http://$(hostname):$AGID_PORT  (CKPT_DIR=$AGID_CKPT_DIR, device=cpu)"
echo "[webui] tunnel:  ssh -N -L ${AGID_PORT}:$(hostname):${AGID_PORT} ssalehi@ivi-h1.science.uva.nl   then open http://localhost:${AGID_PORT}"
exec bash interface/run_backend.sh
