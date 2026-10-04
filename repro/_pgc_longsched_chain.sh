#!/bin/bash
# After the full-data run + its eval complete, continue training from the final full-data ckpt
# for 30 more epochs (long-schedule test) on the 24k subsample, then eval ForenSynths AUROC.
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
until grep -qa "FULL EVAL COMPLETE" repro/logs/pgc_full_eval.log 2>/dev/null; do sleep 180; done
echo "=== full-data eval done; starting long-schedule continuation (30 epochs, 24k) on GPU 4 ==="
PGC_REGIME=progan PGC_N=12000 PGC_EPOCHS=30 PGC_BS=128 \
  PGC_INIT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_full/model.pth \
  PGC_OUT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_long/model.pth \
  CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n SynthScope python repro/train_pgc_fast.py 2>&1 | grep -aE "\[init\]|\[data\]|\[epoch|\[done\]"
echo "=== continuation done; ForenSynths AUROC eval ==="
CUDA_VISIBLE_DEVICES=4 $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
  --ckpt /ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_long/model.pth \
  --tag pgc_longsched_forensynths 2>&1 | tail -2
echo "=== LONG-SCHEDULE CHAIN COMPLETE ==="
