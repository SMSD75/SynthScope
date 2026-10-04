#!/bin/bash
# Reproduce OFFICIAL SAFE results on ForenSynths via the faithful vendored eval (fc1 head + crop-256,
# no-resize, no-norm), to confirm the official checkpoint is fine and the all-fake was our load path.
#SBATCH --job-name=safe_repro
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/safe_repro_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
CKPT=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/official_ckpts_shared/safe_checkpoint-best.pth
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
# locate ForenSynths test data (node-local on cn029)
DATA=""
for d in /ssdstore /nvmestore /scratch /local; do
  [ -d "$d/ssalehi/AGID/CNNDetection/test" ] && DATA="$d/ssalehi/AGID/CNNDetection/test" && break
done
[ -n "$DATA" ] || { echo "NO ForenSynths test on this node (checked /ssdstore etc.) -- redirect to cn023"; exit 3; }
echo "[data] $DATA ($(ls $DATA | wc -l) generators)"
# CUDA precheck
ok=0; for i in 1 2 3; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 4; done
[ "$ok" = 1 ] || { echo "CUDA FAIL"; exit 1; }
echo "===== OFFICIAL SAFE on ForenSynths (faithful vendored path; expect ~98 mAP, NOT all-fake) ====="
$CONDA run --no-capture-output -n SynthScope python repro/official_evals/safe_official_eval.py \
  --benchmark forensynths --data "$DATA" --bs 32 --device cuda \
  --ckpt "$CKPT" --tag safe_official_repro 2>&1 | tail -30
echo "=== SAFE REPRO COMPLETE $(date) ==="
