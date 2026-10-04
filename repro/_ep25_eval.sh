#!/bin/bash
# ep25 snapshot eval — waits for ep10 eval to finish (strict GPU4 sequencing), then runs.
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
CONDA=/home/ssalehi/miniconda3/bin/conda
until grep -qa "EP10 EVAL COMPLETE\|CUDA NEVER" repro/logs/ep10_eval.log 2>/dev/null; do sleep 120; done
while ps -eo cmd | grep "benchmark/bench_eval.py" | grep -qv grep; do sleep 60; done
ok=0
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n SynthScope python -c \
    "import torch; assert torch.cuda.is_available(); (torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum().item(); print('CUDA OK')" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = "1" ] || { echo "CUDA NEVER CAME UP"; exit 1; }
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/SynthScope/CNNDetection/test \
  --ckpt /ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/pgc_8gpu/model_ep25.pth \
  --tag pgc_8gpu_ep25
echo "=== EP25 EVAL COMPLETE ==="
