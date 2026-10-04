#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

# VIB-Net uses a frozen CLIP ViT-L/14 backbone (CLIP normalization, 224 crops).
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=image/vib_net \
    model@ModelConfig=image/vib_net \
    dataset@DatasetConfig=image/genimage \
    DatasetConfig.train_batch_size=128 \
    DatasetConfig.val_batch_size=512 \
    DatasetConfig.test_batch_size=512 \
    DatasetConfig.train_transform_list=[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip] \
    DatasetConfig.eval_transform_list=[resize_short_256,center_crop_224,to_dtype,normalize_clip] \
    logger.name=vib_net \
    logger.mode=online \
    logger.tags=[SynthScope,VIBNet,genimage]
