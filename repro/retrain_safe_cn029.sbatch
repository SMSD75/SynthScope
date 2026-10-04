#!/bin/bash
#SBATCH --job-name=safe_rt --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=7-00:00:00 --cpus-per-task=14 --mem=32G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/retrain_safe_%j.log
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
DATA=/ssdstore/ssalehi/SynthScope; SC=/ssdstore/ssalehi/.cache; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/safe; mkdir -p "$SAVE" "$SC"/{huggingface,torch,triton}
export HF_HOME=$SC/huggingface TORCH_HOME=$SC/torch TRITON_CACHE_DIR=$SC/triton OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES data=$DATA/CNNDetection $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29611 \
  main.py trainer@TrainerConfig=image/safe ModelConfig.model_type=safe ModelConfig.freeze_backbone=false \
  dataset@DatasetConfig=image/cnndetection DatasetConfig.root_dir=$DATA/CNNDetection \
  DatasetConfig.train_transform_list="[random_crop_256_pad,random_horizontal_flip,random_rotation_180,color_jitter_safe,to_dtype,random_mask_safe]" \
  DatasetConfig.eval_transform_list="[center_crop_256,to_dtype]" \
  DatasetConfig.train_batch_size=64 DatasetConfig.val_batch_size=128 DatasetConfig.test_batch_size=128 DatasetConfig.num_workers=8 \
  ++TrainerConfig.epochs=12 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=safe_retrain 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== SAFE RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
