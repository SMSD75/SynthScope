#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# ReStraV trains a small MLP on DINOv2 ViT-S/14 trajectory-geometry features (distance +
# curvature). The DINOv2 backbone is frozen; only the MLP trains. Frames are decoded to
# 224x224 over a CENTER 2 s window (24 frames, BICUBIC) and fed to DINOv2 as RAW [0,1] with
# *no* ImageNet normalization -- this matches the official ReStraV repo verbatim
# (dinov2_features.preprocess = frames/255 + BICUBIC resize, no Normalize) and the deployed
# serving spec (registry video_norm = identity). Do NOT add normalize_dinov2 here: the
# detector is served raw [0,1], so normalizing only at train time creates a train/serve
# mismatch that collapses cross-generator transfer. Official optimizer is plain Adam (no
# weight decay), 10 epochs -- matched below. The video benchmark must be present under
# DatasetConfig.root_dir; this is a template (data download is out of scope).
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 main.py \
    trainer@TrainerConfig=video/restrav \
    model@ModelConfig=video/restrav \
    dataset@DatasetConfig=video/vidprom \
    TrainerConfig.epochs=10 \
    TrainerConfig.weight_decay=0.0 \
    DatasetConfig.train_batch_size=8 \
    DatasetConfig.val_batch_size=8 \
    DatasetConfig.test_batch_size=8 \
    DatasetConfig.num_frames=24 \
    DatasetConfig.train_transform_list=[] \
    DatasetConfig.eval_transform_list=[] \
    metrics=[real_accuracy,fake_accuracy,AP,AUC] \
    logger.name=restrav \
    logger.mode=online \
    logger.tags=[SynthScope,ReStraV,vidprom]
