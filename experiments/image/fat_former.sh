#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

# Official FatFormer (Michel-liu/FatFormer, utils/dataset.py) normalizes with **ImageNet**
# mean/std -- NOT CLIP stats -- despite the CLIP ViT-L/14 backbone. `normalize_dinov2` is the
# ImageNet Normalize in this repo, so we use it here to match the official preprocessing (and the
# deployed official-weights eval, which reproduces 98.39 acc / 99.70 mAP with ImageNet norm).
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=image/fat_former \
    model@ModelConfig=image/fat_former \
    dataset@DatasetConfig=image/aigibench \
    DatasetConfig.train_batch_size=32 \
    DatasetConfig.val_batch_size=32 \
    DatasetConfig.test_batch_size=32 \
    DatasetConfig.train_transform_list=[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype,normalize_dinov2] \
    DatasetConfig.eval_transform_list=[resize_short_256,center_crop_224,to_dtype,normalize_dinov2] \
    logger.name=fat_former \
    logger.mode=online \
    logger.tags=[SynthScope,FatFormer,aigibench]
