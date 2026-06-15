#!/bin/bash
# Early-signal eval for the const-LR PGC run (185056): evaluates each milestone snapshot on the THREE
# hard generators (deepfake/san/seeingdark) as it lands, ~9h before the in-job full ForenSynths eval.
# Uses the shared-FS hard-3 test set (no data staging) + 1 GPU on the 'all' partition under the
# linuxusers account (separate GPU cap -> does NOT touch the cees 8-GPU cap held by 185056).
#SBATCH --job-name=pgc_cl_eval
#SBATCH --partition=all
#SBATCH --account=linuxusers
#SBATCH --gres=gpu:1
#SBATCH --exclude=ivi-cn003,ivi-cn021
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=40:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_constlr_hard3_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
TEST=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval/test   # deepfake/san/seeingdark (shared FS)
CKDIR=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_8gpu_constlr
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf
export PYTHONUNBUFFERED=1
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
echo "hard-3 test gens: $(ls $TEST 2>/dev/null | tr '\n' ' ')"

ok=0
for i in 1 2 3 4 5; do
  $CONDA run --no-capture-output -n AGID python -c \
    "import torch; assert torch.cuda.is_available(); (torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum().item(); print('CUDA OK')" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = 1 ] || { echo "CUDA NEVER CAME UP"; exit 1; }

for ep in 10 25 50 75 100; do
  ck=$CKDIR/model_ep${ep}.pth
  echo "[wait] ep$ep -> $ck  $(date)"
  until [ -f "$ck" ]; do sleep 120; done
  sleep 25   # let the snapshot finish flushing to shared FS
  echo "[eval] ep$ep hard-3  $(date)"
  $CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py \
    --model pgc --benchmark forensynths --data "$TEST" --bs 32 \
    --ckpt "$ck" --tag pgc_constlr_ep${ep}_hard3 2>&1 | tail -25
done
echo "=== CONST-LR HARD-3 TRAJECTORY DONE $(date) ==="
