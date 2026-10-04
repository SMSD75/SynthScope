#!/bin/bash
# PGC ProGAN reproduction — OFFICIAL-MATCHED near-constant LR (PGC_TMAX_MULT=4 -> cosine traverses
# only 1/4, LR stays ~5e-5, exactly as the released train.py does). Everything else identical to the
# annealing run (same 720k ProGAN data on cn029:/ssdstore, eff batch 128, MLP-LoRA, smoothing 0.1,
# seed 42). Tests whether full LR annealing was erasing the hard-generator transfer.
#SBATCH --job-name=pgc_constlr
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/pgc_constlr_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
CONDA=/home/ssalehi/miniconda3/bin/conda
echo "NODE=$(hostname)  GPUS=$CUDA_VISIBLE_DEVICES  $(date)"

$CONDA run --no-capture-output -n SynthScope python -c \
  "import torch; assert torch.cuda.is_available(); print('CUDA OK,', torch.cuda.device_count(), 'GPUs')" || { echo "CUDA FAIL"; exit 1; }

# data: MUST reuse pre-staged node-local copy on cn029 (cn023 is down -> no ssh-stream fallback)
DATA=""
for d in /ssdstore /nvmestore /scratch /local; do
  [ -d "$d/ssalehi/SynthScope/CNNDetection/train" ] && DATA="$d/ssalehi/SynthScope/CNNDetection" && break
done
[ -n "$DATA" ] || { echo "[data] no pre-staged CNNDetection on this node -- ABORT"; exit 1; }
echo "[data] REUSING $DATA ($(du -sh $DATA/train 2>/dev/null | cut -f1) train)"

OUTDIR=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/pgc_8gpu_constlr
mkdir -p "$OUTDIR"
export PGC_DATA_ROOT="$DATA/train"
export PGC_N=400000 PGC_EPOCHS=100 PGC_BS=16
export PGC_TMAX_MULT=4          # replicate official near-constant ~5e-5 LR
export PGC_MILESTONES=10,25,50,75,100
export PGC_OUT="$OUTDIR/model.pth"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/.cache_shared/hf
export PYTHONUNBUFFERED=1
NG=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "[train] DDP on $NG GPUs (ProGAN, const-LR / official-matched schedule)"
$CONDA run --no-capture-output -n SynthScope accelerate launch \
  --multi_gpu --num_processes $NG --mixed_precision bf16 --main_process_port 29572 \
  repro/train_pgc_ddp.py
echo "[train] done $(date)"

# in-job ForenSynths eval of every milestone snapshot (robust; no detached relay)
if [ -d "$DATA/test" ]; then
  for ck in model_ep10 model_ep25 model_ep50 model_ep75 model_ep100; do
    [ -f "$OUTDIR/$ck.pth" ] || { echo "[eval] missing $ck.pth, skip"; continue; }
    echo "[eval] === $ck on ForenSynths ==="
    $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
      --model pgc --benchmark forensynths --data "$DATA/test" --bs 32 \
      --ckpt "$OUTDIR/$ck.pth" --tag "pgc_constlr_${ck#model_}"
  done
else
  echo "[eval] no $DATA/test -- skipping in-job eval"
fi
echo "=== PGC CONST-LR PIPELINE COMPLETE $(date) ==="
