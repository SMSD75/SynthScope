#!/bin/bash
# ep100 (final) eval — sleeps until the 100-epoch run finishes ([done] in its log), then evals the
# final checkpoint on ForenSynths. The formal endpoint of the PGC exact-recipe reproduction.
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
until grep -qa "\[done\]" repro/logs/pgc_8gpu_184771.out 2>/dev/null; do sleep 600; done
while ps -eo cmd | grep "benchmark/bench_eval.py" | grep -qv grep; do sleep 60; done
ok=0
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n SynthScope python -c \
    "import torch; assert torch.cuda.is_available(); (torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum().item(); print('CUDA OK')" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = "1" ] || { echo "CUDA NEVER CAME UP"; exit 1; }
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
  --ckpt /ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_8gpu/model.pth \
  --tag pgc_8gpu_ep100
echo "=== EP100 EVAL COMPLETE ==="
