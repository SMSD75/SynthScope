#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=image/spai \
    model@ModelConfig=image/spai \
    DatasetConfig.train_batch_size=32 \
    DatasetConfig.val_batch_size=32 \
    DatasetConfig.test_batch_size=32 \
    logger.name=spai \
    logger.mode=online \
    logger.tags=[SynthScope,SPAI,genimage]
