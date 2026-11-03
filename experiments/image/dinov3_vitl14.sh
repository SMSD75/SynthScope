#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

# accelerate launch --config_file accelerate_config.yaml --num_processes 4 main.py
# accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py

accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    model@ModelConfig=image/dinov3 \
    ModelConfig.bottleneck_dim=64 \
    ModelConfig.use_adapter=false \
    ModelConfig.backbone_name=dinov3_vitl16