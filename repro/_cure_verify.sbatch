#!/bin/bash
# CuRe integration: smoke test + exact-equivalence verification vs the OFFICIAL model (any free cees node).
#SBATCH --job-name=cure_verify --partition=cees --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=6 --mem=24G --time=00:40:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/cure_verify_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
REF=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/cure_ref
SCR=/ssdstore; [ -w $SCR ] || SCR=/tmp; mkdir -p $SCR/$USER/cure_hf && export HF_HOME=$SCR/$USER/cure_hf PYTHONUNBUFFERED=1   # node-local: keep 2.7GB off the /ivi quota
export PYTHONPATH=$REF/albu:$REF/CuRe
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
$CONDA run --no-capture-output -n AGID python -c "import torch;assert torch.cuda.is_available();print('CUDA OK',torch.cuda.get_device_name(0))" || exit 1
echo "################ smoke test ################"
$CONDA run --no-capture-output -n AGID python tests/test_cure_smoke.py 2>&1 | grep -avE "Warning|warn|nvml|raw_cnt"
echo "################ equivalence vs official ################"
$CONDA run --no-capture-output -n AGID python repro/cure_verify_load.py \
  --adapter $REF/CuRe/weights/cure_adapter.pt --images $REF/equiv_imgs --limit 60 --device cuda 2>&1 | grep -avE "Warning|warn|nvml|raw_cnt|Fetching|it/s\]"
echo "=== CURE VERIFY DONE $(date) ==="
