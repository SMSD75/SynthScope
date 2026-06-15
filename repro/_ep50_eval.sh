#!/bin/bash
# ep50 snapshot eval — sleeps until the snapshot exists (~09:00) and GPU4 is free, then runs.
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
until [ -f checkpoints_retrain/pgc_8gpu/model_ep50.pth ]; do sleep 300; done
while ps -eo cmd | grep "benchmark/bench_eval.py" | grep -qv grep; do sleep 60; done
ok=0
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n AGID python -c \
    "import torch; assert torch.cuda.is_available(); (torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum().item(); print('CUDA OK')" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = "1" ] || { echo "CUDA NEVER CAME UP"; exit 1; }
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
  --ckpt /ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_8gpu/model_ep50.pth \
  --tag pgc_8gpu_ep50
echo "=== EP50 EVAL COMPLETE ==="
