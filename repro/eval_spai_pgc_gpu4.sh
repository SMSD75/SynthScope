#!/bin/bash
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/SynthScope/checkpoints HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch SYNTHSCOPE_MFM_CKPT=/hddstore/ssalehi/SynthScope/mfm/mfm_pretrain_vit_base.pth CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4
G=/hddstore/ssalehi/SynthScope/GenImage_full/test_full
echo "### SPAI-v2 GenImage-8 $(date)"; conda run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model spai --ckpt "checkpoints_retrain/spai/model_1.pth" --benchmark pergen --data $G --limit 1500 --tag spai_v2_g8 2>&1 | grep -E "RESULT" | tail -1
echo "### PGC-long GenImage-8 $(date)"; conda run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model pgc --ckpt "" --benchmark pergen --data $G --limit 1500 --tag pgc_long_g8 2>&1 | grep -E "RESULT" | tail -1
echo "### SPAI/PGC EVALS DONE $(date)"
