#!/bin/bash
# Long-schedule continuation: 30 more epochs (24k subsample) from the 4-epoch full-data checkpoint,
# then ForenSynths AUROC eval. Strictly sequential on cn023 GPU 4 (48GiB cgroup: never stack jobs).
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
until [ -s repro/bench_results/pgc_full_progan_ufd.json ]; do sleep 120; done
while ps -eo cmd | grep "benchmark/bench_eval.py" | grep -qv grep; do sleep 60; done
echo "=== starting long-schedule continuation (30 epochs, 24k) on GPU 4 ==="
PGC_REGIME=progan PGC_N=12000 PGC_EPOCHS=30 PGC_BS=128 \
  PGC_INIT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_full/model.pth \
  PGC_OUT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_long/model.pth \
  CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n SynthScope python repro/train_pgc_fast.py
echo "=== continuation finished; ForenSynths AUROC eval ==="
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --model pgc --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test \
  --ckpt /ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_long/model.pth \
  --tag pgc_longsched_forensynths
echo "=== LONG-SCHEDULE CHAIN COMPLETE ==="
