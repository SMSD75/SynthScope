#!/bin/bash
# Hold a cn023 GPU + serve the SynthScope web UI (FastAPI/uvicorn) on it. Reads checkpoints from the
# node-local /hddstore dir (where the official PGC/SAFE get deployed), so the UI serves the
# official-weights PGC/SAFE once job 185922's swap completes.
#SBATCH --job-name=synthscope_webui
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/webui_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
CONDA=/home/ssalehi/miniconda3/bin/conda
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/SynthScope/checkpoints
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/.cache_shared/hf
export SYNTHSCOPE_PORT=${SYNTHSCOPE_PORT:-8021} SYNTHSCOPE_HOST=0.0.0.0
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES PORT=$SYNTHSCOPE_PORT $(date)"

# CUDA precheck (cn023 flaky init)
ok=0
for i in 1 2 3 4 5; do
  $CONDA run --no-capture-output -n SynthScope python -c \
    "import torch; assert torch.cuda.is_available(); (torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum().item(); print('CUDA OK')" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = 1 ] || { echo "CUDA NEVER CAME UP"; exit 1; }

echo "[webui] serving on http://$(hostname):$SYNTHSCOPE_PORT  (CKPT_DIR=$SYNTHSCOPE_CKPT_DIR)"
echo "[webui] access from your machine:  ssh -N -L 8021:$(hostname):8021 <you>@<login-host>   then open http://localhost:8021"
exec bash interface/run_backend.sh
