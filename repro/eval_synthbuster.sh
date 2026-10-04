#!/bin/bash
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/SynthScope/checkpoints HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch SYNTHSCOPE_MFM_CKPT=/hddstore/ssalehi/SynthScope/mfm/mfm_pretrain_vit_base.pth CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4
D=/hddstore/ssalehi/SynthScope/Synthbuster/test
for M in spai co_spy safe; do
  echo "### $M synthbuster"; conda run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model $M --benchmark pergen --data $D --bs 16 --limit 1000 --tag ${M}_synthbuster 2>&1 | grep -E "RESULT" | tail -1
done
echo "### SYNTHBUSTER EVALS DONE"
