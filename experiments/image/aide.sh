#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

# Five native-resolution views per image (aide.yaml lists): batch 8 and 5 workers bound memory, as in the run behind
# the README's SynthScope_train checkpoint (repro/retrain_aide_native_v2.sbatch).
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=image/aide \
    model@ModelConfig=image/aide \
    dataset@DatasetConfig=image/genimage \
    DatasetConfig.train_batch_size=8 \
    DatasetConfig.num_workers=5 \
    DatasetConfig.val_batch_size=16 \
    DatasetConfig.test_batch_size=16 \
    logger.name=aide \
    logger.mode=online \
    logger.tags=[SynthScope,AIDE,genimage]
