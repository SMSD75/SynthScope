#!/bin/bash
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/AGID
export AGID_CKPT_DIR=/hddstore/ssalehi/AGID/checkpoints HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4
echo "### NPR_table1 ForenSynths $(date)"
conda run --no-capture-output -n AGID python benchmark/bench_eval.py --model npr --ckpt /hddstore/ssalehi/AGID/repro/ckpts/NPR_table1.pth --benchmark forensynths --data /hddstore/ssalehi/AGID/CNNDetection/test --bs 16 --tag npr_table1_forensynths 2>&1 | grep -E "RESULT|: acc=" | tail -15
echo "### SAFE DiTFake $(date)"
conda run --no-capture-output -n AGID python benchmark/bench_eval.py --model safe --benchmark pergen --data /hddstore/ssalehi/AGID/DiTFake/test --bs 16 --limit 2000 --tag safe_ditfake 2>&1 | grep -E "RESULT|: acc=" | tail -6
echo "### SEQ DONE $(date)"
