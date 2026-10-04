#!/bin/bash
# CuRe on GenImage-8 (its own SD-v1.4 training distribution) with the copy staged on cn029.
#SBATCH --job-name=cure_gi8 --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=8 --mem=24G --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/cure_gi8_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
CONDA=/home/ssalehi/miniconda3/bin/conda
REF=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/cure_ref
S=/ssdstore/ssalehi/SynthScope
mkdir -p $S/cure_ckpt $S/.cache_hf_cure
export HF_HOME=$S/.cache_hf_cure PYTHONUNBUFFERED=1 SYNTHSCOPE_CKPT_DIR=$S/cure_ckpt
cp $REF/CuRe/weights/cure_adapter.pt $SYNTHSCOPE_CKPT_DIR/cure.pth      # OFFICIAL adapter, unchanged
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
tag=cure_repro_genimage8; full=repro/logs/${tag}_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model cure --benchmark pergen \
  --data $S/GenImage_full/test --bs 16 --device cuda --limit 1500 --tag $tag > $full 2>&1
grep -aE "RESULT|: acc=|=== bench_eval" $full | tail -20
echo "=== CURE GENIMAGE8 DONE $(date) ==="
