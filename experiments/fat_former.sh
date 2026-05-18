#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=fat_former \
    model@ModelConfig=fat_former \
    dataset@DatasetConfig=aigibench \
    DatasetConfig.train_batch_size=32 \
    DatasetConfig.val_batch_size=32 \
    DatasetConfig.test_batch_size=32 \
    DatasetConfig.train_transform_list=[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip] \
    DatasetConfig.eval_transform_list=[resize_short_256,center_crop_224,to_dtype,normalize_clip] \
    logger.name=fat_former \
    logger.mode=online \
    logger.tags=[AGID,FatFormer,aigibench]
