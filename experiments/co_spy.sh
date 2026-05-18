#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=co_spy \
    model@ModelConfig=co_spy \
    DatasetConfig.train_batch_size=16 \
    DatasetConfig.val_batch_size=16 \
    DatasetConfig.test_batch_size=16 \
    DatasetConfig.train_transform_list=[resize_short_384,gaussian_blur,jpeg_compression,random_crop_384,random_horizontal_flip,to_dtype] \
    DatasetConfig.eval_transform_list=[resize_short_384,center_crop_384,to_dtype] \
    logger.name=co_spy \
    logger.mode=online \
    logger.tags=[AGID,CoSpy,genimage]
