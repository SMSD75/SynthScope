#!/bin/bash
#SBATCH --job-name=spai_interp --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:0 --cpus-per-task=4 --mem=24G --time=00:20:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/spai_interp_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export AGID_CKPT_DIR=/hddstore/ssalehi/AGID/checkpoints          # spai.pth (official-remapped) local on cn023
export AGID_MFM_CKPT=/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf PYTHONUNBUFFERED=1
echo "NODE=$(hostname) $(date)  spai.pth -> $(readlink -f $AGID_CKPT_DIR/spai.pth 2>/dev/null)"
$CONDA run --no-capture-output -n AGID python repro/_interp_spai_test.py 2>&1 | grep -avE "warn|Warning"
echo "=== SPAI INTERP TEST DONE $(date) ==="
