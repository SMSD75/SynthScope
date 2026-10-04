#!/bin/bash
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/SynthScope/checkpoints HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4
CK=/hddstore/ssalehi/SynthScope/repro/ckpts/NPR_table1.pth
RUN(){ echo "### $1"; conda run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model npr --ckpt $CK "${@:2}" 2>&1 | grep -E "RESULT" | tail -1; }
RUN npr_t1_freqnet --benchmark forensynths --data /hddstore/ssalehi/SynthScope/freqnet --native-res --bs 1 --tag npr_t1_freqnet
RUN npr_t1_diffusionforensics --benchmark pergen --data /hddstore/ssalehi/SynthScope/DiffusionForensics/test --bs 16 --tag npr_t1_diffusionforensics
RUN npr_t1_ufd --benchmark ufd --data /hddstore/ssalehi/SynthScope/universialfakedetection --bs 16 --tag npr_t1_ufd
RUN npr_t1_diffusion1kstep --benchmark pergen --data /hddstore/ssalehi/SynthScope/Diffusion1kStep/test --bs 16 --tag npr_t1_diffusion1kstep
echo "### NPR VERIFY DONE"
