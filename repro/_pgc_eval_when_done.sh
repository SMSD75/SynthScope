#!/bin/bash
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
LOG=repro/logs/pgc_fast_progan.log
CKPT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_fast/model.pth
until grep -qE "\[done\] saved|Traceback|CUDA out of memory" "$LOG" 2>/dev/null; do sleep 60; done
if ! grep -q "\[done\] saved" "$LOG"; then echo "TRAINING DID NOT FINISH CLEANLY"; tail -5 "$LOG"; exit 0; fi
echo "=== training done; running AUROC eval on new ProGAN checkpoint ==="
CONDA=/home/ssalehi/miniconda3/bin/conda
CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
  --ckpt "$CKPT" --tag pgc_fast_progan_forensynths 2>&1 | tail -3
CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py \
  --model pgc --benchmark ufd --data /hddstore/ssalehi/AGID/universialfakedetection/test \
  --ckpt "$CKPT" --tag pgc_fast_progan_ufd 2>&1 | tail -3
echo "=== EVAL COMPLETE ==="
