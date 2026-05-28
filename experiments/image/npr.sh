#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

# NPR trains a small ResNet from scratch on neighboring-pixel residuals; feed [0,1] images (to_dtype only).
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=image/npr \
    model@ModelConfig=image/npr \
    dataset@DatasetConfig=image/genimage \
    DatasetConfig.train_batch_size=128 \
    DatasetConfig.val_batch_size=512 \
    DatasetConfig.test_batch_size=512 \
    DatasetConfig.train_transform_list=[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype] \
    DatasetConfig.eval_transform_list=[resize_short_256,center_crop_224,to_dtype] \
    logger.name=npr \
    logger.mode=online \
    logger.tags=[AGID,NPR,genimage]
