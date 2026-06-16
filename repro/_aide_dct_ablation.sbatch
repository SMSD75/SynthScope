#!/bin/bash
#SBATCH --job-name=aide_ablat
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/aide_ablat_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch PYTHONUNBUFFERED=1
H=/hddstore/ssalehi/AGID
GCK=$H/repro/official_ckpts/aide/AIDE_checkpoints/GenImage_train.pth
[ -f "$GCK" ] || GCK=$H/repro/official_ckpts/aide/AIDE_checkpoints/sd14_train.pth
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES ckpt=$GCK $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n AGID python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
echo "##### GenImage-8, official GenImage weights — NATIVE-res DCT (official preprocessing) #####"
$CONDA run --no-capture-output -n AGID python repro/_aide_dct_ablation.py --data "$H/GenImage_full/test" --ckpt "$GCK" --benchmark pergen --limit 200 --mode native
echo "##### GenImage-8, official GenImage weights — RESIZE-256 before DCT (mimics AGID class) #####"
$CONDA run --no-capture-output -n AGID python repro/_aide_dct_ablation.py --data "$H/GenImage_full/test" --ckpt "$GCK" --benchmark pergen --limit 200 --mode resize256
echo "=== AIDE DCT ABLATION COMPLETE $(date) ==="
