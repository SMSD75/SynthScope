#!/bin/bash
#SBATCH --job-name=aide_faith --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=1-00:00:00 --cpus-per-task=24 --mem=120G --gres=gpu:4
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/retrain_aide_faithful_%j.log
# AIDE retrain with the FIXED faithful preprocessing (raw [0,1] 256 in; model applies CLIP norm to the
# semantic ConvNeXt-XXL branch and ImageNet norm to the SRM/HPF patch branch, DCT in pixel space) — the
# divergence that capped adm/vqdm at chance. 4-GPU DeepSpeed ZeRO-2 on GenImage SD-v1.4 (staged on cn029).
# After: eval on cn023 GPUs 4&7 (aide bench_eval / aide_official_eval) vs official 97.12 — target adm 58->~94.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
DATA=/ssdstore/ssalehi/SynthScope; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/aide_faithful; mkdir -p "$SAVE"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface   # shared so convnext (4.8G) caches once
export TORCH_HOME=$DATA/.cache/torch TRITON_CACHE_DIR=$DATA/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
[ -d "$DATA/GenImage/train/real" ] || { echo "GenImage not staged on $(hostname)"; exit 1; }
echo "node=$(hostname) data=$DATA/GenImage real=$(ls $DATA/GenImage/train/real|wc -l) $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml \
  --num_processes 4 --main_process_port 29670 \
  main.py trainer@TrainerConfig=image/aide model@ModelConfig=image/aide dataset@DatasetConfig=image/genimage \
  DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_256,gaussian_blur,jpeg_compression,random_crop_256,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_256,to_dtype]" \
  DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 DatasetConfig.num_workers=12 \
  ++TrainerConfig.epochs=5 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=aide_faithful \
  2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== AIDE FAITHFUL RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
