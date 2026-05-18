#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=safe \
    dataset@DatasetConfig=aigibench \
    ModelConfig.model_type=safe \
    ModelConfig.freeze_backbone=false \
    DatasetConfig.train_batch_size=64 \
    DatasetConfig.val_batch_size=64 \
    DatasetConfig.test_batch_size=64 \
    DatasetConfig.train_transform_list=[random_crop_256_pad,random_horizontal_flip,random_rotation_180,color_jitter_safe,to_dtype,random_mask_safe] \
    DatasetConfig.eval_transform_list=[center_crop_256,to_dtype] \
    TrainerConfig.epochs=100 \
    TrainerConfig.lr=0.0005 \
    TrainerConfig.weight_decay=0.05 \
    TrainerConfig.label_smoothing=0.1 \
    TrainerConfig.warmup.mode=on \
    TrainerConfig.warmup.warmup_ratio=0.2 \
    TrainerConfig.warmup.lr_scheduler=cosine \
    logger.name=safe \
    logger.mode=online \
    logger.tags=[AGID,SAFE,aigibench]
