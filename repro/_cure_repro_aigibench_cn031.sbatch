#!/bin/bash
# CuRe on AIGIBench (paper: 92.8 balanced acc) with the full extracted copy on cn031 /nvmestore.
# --limit 1500 = first 1500 images per class per generator (25 generators, ~4500/class available).
#SBATCH --job-name=cure_aigi --partition=cees6000 --account=ceesusers --nodelist=ivi-cn031
#SBATCH --gres=gpu:rtx_6000:1 --cpus-per-task=8 --mem=24G --time=03:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/cure_aigi_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
REF=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/cure_ref
S=/nvmestore/ssalehi/AGID
mkdir -p $S/cure_ckpt $S/.cache_hf_cure
export HF_HOME=$S/.cache_hf_cure PYTHONUNBUFFERED=1 SYNTHSCOPE_CKPT_DIR=$S/cure_ckpt
cp $REF/CuRe/weights/cure_adapter.pt $SYNTHSCOPE_CKPT_DIR/cure.pth      # OFFICIAL adapter, unchanged
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
tag=cure_repro_aigibench; full=repro/logs/${tag}_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model cure --benchmark pergen \
  --data $S/AIGIBench/test --bs 16 --device cuda --limit 1500 --tag $tag > $full 2>&1
grep -aE "RESULT|: acc=|=== bench_eval" $full | tail -40
echo "=== CURE AIGIBENCH DONE $(date) ==="
