#!/bin/bash
# Full-data PGC eval with CUDA pre-check (cn023 flakily fails CUDA init -> silent CPU fallback).
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
CKPT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_full/model.pth
ok=0
for i in 1 2 3 4 5; do
  if CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n SynthScope python -c \
    "import torch; assert torch.cuda.is_available(), 'no cuda'; x=torch.randn(4,4,device='cuda'); (x@x).sum().item(); print('CUDA OK')"; then
    ok=1; break
  fi
  echo "CUDA init failed (attempt $i)"; sleep 5
done
[ "$ok" = "1" ] || { echo "CUDA NEVER CAME UP — aborting eval"; exit 1; }
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
  --ckpt "$CKPT" --tag pgc_full_progan_forensynths
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --model pgc --benchmark ufd --data /hddstore/ssalehi/AGID/universialfakedetection/test \
  --ckpt "$CKPT" --tag pgc_full_progan_ufd
echo "FULL EVAL COMPLETE"
