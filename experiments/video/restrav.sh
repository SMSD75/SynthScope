#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# ReStraV trains a small MLP on DINOv2 ViT-S/14 trajectory-geometry features (distance +
# curvature). The DINOv2 backbone is frozen; only the MLP trains. Frames are decoded to
# 224x224 in [0,1] (24 frames over 2s) and normalized by the dataset transform pipeline
# (normalize_dinov2 = ImageNet stats, what DINOv2 expects). The video benchmark must be
# present under DatasetConfig.root_dir; this is a template (data download is out of scope).
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 1 main.py \
    trainer@TrainerConfig=video/restrav \
    model@ModelConfig=video/restrav \
    dataset@DatasetConfig=video/vidprom \
    DatasetConfig.train_batch_size=8 \
    DatasetConfig.val_batch_size=8 \
    DatasetConfig.test_batch_size=8 \
    DatasetConfig.num_frames=24 \
    DatasetConfig.train_transform_list=[normalize_dinov2] \
    DatasetConfig.eval_transform_list=[normalize_dinov2] \
    metrics=[real_accuracy,fake_accuracy,AP,AUC] \
    logger.name=restrav \
    logger.mode=online \
    logger.tags=[AGID,ReStraV,vidprom]
