#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

# accelerate launch --config_file accelerate_config.yaml --num_processes 4 main.py
# accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py

accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    ModelConfig.bottleneck_dim=64 \
    ModelConfig.model_type=dinov3 \
    ModelConfig.use_adapter=false \
    DatasetConfig.train_transform_list=[resize_short_256,gaussian_blur,jpeg_compression,random_crop_256,random_horizontal_flip,to_dtype,normalize_dinov3] \
    DatasetConfig.eval_transform_list=[resize_short_256,center_crop_256,to_dtype,normalize_dinov3] \
    ModelConfig.backbone_name=dinov3_vitl16