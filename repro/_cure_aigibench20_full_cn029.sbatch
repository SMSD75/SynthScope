#!/bin/bash
# CuRe on the paper's 20 AIGIBench full-image-synthesis subsets (arXiv 2609.14316 Table 2), FULL test data
# (no --limit). Official adapter through the SynthScope port; runs from the trending/cure worktree.
#SBATCH --job-name=cure_aigi20 --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=8 --mem=24G --time=05:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/cure_aigi20_%j.out
set -u
W=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/cure_wt
cd $W
CONDA=/home/ssalehi/miniconda3/bin/conda
S=/ssdstore/ssalehi/SynthScope
export HF_HOME=$S/.cache_hf_cure PYTHONUNBUFFERED=1 SYNTHSCOPE_CKPT_DIR=$S/cure_ckpt PYTHONPATH=$W
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) | code=$(git log --oneline -1)"
SEL=ProGAN,R3GAN,StyleGAN3,StyleGAN-XL,StyleSwim,WFIR,DALLE-3,FLUX1-dev,GLIDE,Imagen3,Midjourney,SD3,SDXL,BLIP,Infinite_ID,InstantID,IP_Adapter,PhotoMaker,CommunityAI,SocialRF
tag=cure_repro_aigibench20_full; full=repro/logs/${tag}_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model cure --benchmark pergen \
  --data $S/AIGIBench/test --select $SEL --bs 16 --device cuda --tag $tag > $full 2>&1
grep -aE "RESULT|: acc=|=== bench_eval|skip " $full | tail -30
echo "=== CURE AIGIBENCH20 FULL DONE $(date) ==="
