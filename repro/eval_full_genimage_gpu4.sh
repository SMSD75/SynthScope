#!/bin/bash
# Re-eval the headline GenImage-8 detectors on the FULL val set (6000/class, vs the old 1500/class cap)
# to satisfy the full-data requirement and check whether the cap biased any number.
# cn023 GPU 4 (legal). Sequential. Tags *_fullgenimage8.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/SynthScope/checkpoints
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch
export SYNTHSCOPE_MFM_CKPT=/hddstore/ssalehi/SynthScope/mfm/mfm_pretrain_vit_base.pth
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4
DATA=/hddstore/ssalehi/SynthScope/GenImage_full/test_full
RUN(){ echo "### $1 $(date +%H:%M:%S)"; conda run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --benchmark pergen --data "$DATA" "${@:2}" 2>&1 | grep -vE "libpng|Warning|warn" | tail -14; }

RUN safe          --model safe                                                   --tag safe_fullgenimage8
RUN pgc           --model pgc                                                    --tag pgc_fullgenimage8
RUN vib_net       --model vib_net                                                --tag vib_fullgenimage8
RUN co_spy        --model co_spy                                                 --tag cospy_fullgenimage8
RUN aide_genimage --model aide --ckpt checkpoints_retrain/aide_genimage/model_4.pth --tag aide_fullgenimage8
RUN spai          --model spai                                                   --tag spai_fullgenimage8
echo "=== FULL GENIMAGE EVAL BATCH DONE $(date) ==="
