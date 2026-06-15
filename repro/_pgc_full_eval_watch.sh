#!/bin/bash
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
LOG=repro/logs/pgc_full_progan.log
CKPT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_full/model.pth
CONDA=/home/ssalehi/miniconda3/bin/conda
# wait for full training to finish (GPU 4 frees), then eval on GPU 4
until grep -qa "\[done\] saved" "$LOG" 2>/dev/null; do sleep 120; done
echo "=== full-data training done; ForenSynths AUROC eval ==="
CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
  --ckpt "$CKPT" --tag pgc_full_progan_forensynths 2>&1 | tail -2
CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py \
  --model pgc --benchmark ufd --data /hddstore/ssalehi/AGID/universialfakedetection/test \
  --ckpt "$CKPT" --tag pgc_full_progan_ufd 2>&1 | tail -2
echo "=== FULL EVAL COMPLETE ==="
