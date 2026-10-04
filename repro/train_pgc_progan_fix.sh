#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=4
export HF_HOME=/hddstore/ssalehi/SynthScope/.cache/huggingface
export TORCH_HOME=/hddstore/ssalehi/SynthScope/.cache/torch
export OMP_NUM_THREADS=4
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch \
  --config_file accelerate_config.yaml --num_processes 1 main.py \
  trainer@TrainerConfig=image/pgc \
  model@ModelConfig=image/pgc \
  dataset@DatasetConfig=image/cnndetection \
  TrainerConfig.epochs=3 \
  TrainerConfig.save=true \
  ModelConfig.backbone_name=facebook/dinov2-large \
  DatasetConfig.train_batch_size=128 \
  DatasetConfig.val_batch_size=256 \
  DatasetConfig.test_batch_size=256 \
  DatasetConfig.train_transform_list=[pad_random_crop_224,random_horizontal_flip,color_jitter_pgc,to_dtype,normalize_dinov2] \
  DatasetConfig.eval_transform_list=[center_crop_224,to_dtype,normalize_dinov2] \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] \
  logger.name=pgc_progan_fix logger.mode=offline
