#!/bin/bash
#SBATCH --job-name=pgc_off_eval
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --exclude=ivi-cn001,ivi-cn005,ivi-cn006,ivi-cn012,ivi-cn019
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_official_eval_%j.out
# Control experiment: OFFICIAL PGC code + OFFICIAL ProGAN checkpoint on OUR copies of the three
# verdict generators (deepfake/san/seeingdark). Expected if data+eval are equivalent: ~96.5/99.1/87.5
# AUROC — matching what the same weights produce through the SynthScope harness.
set -u
S=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval
CONDA=/home/ssalehi/miniconda3/bin/conda
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
until [ -f $S/READY ]; do sleep 20; done
$CONDA run --no-capture-output -n SynthScope python -c \
  "import torch; assert torch.cuda.is_available(); print('CUDA OK')" || { echo "CUDA FAIL"; exit 1; }
cd $S/repo
export PYTHONUNBUFFERED=1
$CONDA run --no-capture-output -n SynthScope python test.py \
  --name test_hard3_official \
  --checkpoint $S/PGC_train_progan_ckpt.pth \
  --test_root $S/test \
  --dino_variant dinov2-large \
  --dino_pretrained_root $S/pretrained_dino \
  --lora_rank 8 --lora_alpha 1.0 \
  --devices 0 --batch_size 32
echo "=== OFFICIAL EVAL COMPLETE ==="
