#!/bin/bash
# Complete AIGIBench on cn029 (its copy had only 13/25 generator zips, nothing extracted): download the 12
# missing test zips from the public HF dataset, extract all 25, run CuRe (official adapter) with --limit 1500.
#SBATCH --job-name=cure_aigi29 --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=8 --mem=24G --time=06:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/cure_aigi29_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
REF=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/cure_ref
S=/ssdstore/ssalehi/AGID
A=$S/AIGIBench; R=$A/test
mkdir -p $S/cure_ckpt $S/.cache_hf_cure
export HF_HOME=$S/.cache_hf_cure PYTHONUNBUFFERED=1 AGID_CKPT_DIR=$S/cure_ckpt
cp $REF/CuRe/weights/cure_adapter.pt $AGID_CKPT_DIR/cure.pth      # OFFICIAL adapter, unchanged
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"; df -h /ssdstore | tail -1
echo "=== 1/3 download missing test zips (resumable) ==="
$CONDA run --no-capture-output -n AGID python - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download
R=Path("$R")
missing=["Midjourney","PhotoMaker","ProGAN","R3GAN","SD3","SDXL","SimSwap","SocialRF","StyleGAN-XL","StyleGAN3","StyleSwim","WFIR"]
todo=[f"test/{g}.zip" for g in missing if not (R/f"{g}.zip").exists()]
print("downloading", len(todo), "zips", flush=True)
if todo:
    snapshot_download("HorizonTEL/AIGIBench", repo_type="dataset", allow_patterns=todo, local_dir="$A", max_workers=4)
print("download done", flush=True)
PY
echo "=== 2/3 extract all zips (parallel x6) $(date) ==="
cd $R
ls *.zip | xargs -P 6 -I{} sh -c 'unzip -q -n "{}" && echo "extracted {}" || echo "UNZIP FAILED {}"'
n=$(ls -d */ | wc -l); echo "generator dirs: $n"
echo "=== 3/3 evaluate $(date) ==="
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
tag=cure_repro_aigibench; full=repro/logs/${tag}_cn029_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py --model cure --benchmark pergen \
  --data $R --bs 16 --device cuda --limit 1500 --tag $tag > $full 2>&1
grep -aE "RESULT|: acc=|=== bench_eval|skip " $full | tail -45
echo "=== CURE AIGIBENCH CN029 DONE $(date) ==="
