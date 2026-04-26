#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2,3,5
export OMP_NUM_THREADS=1

accelerate launch --config_file accelerate_config.yaml --num_processes 4 main.py
# Just for testing
accelerate launch --config_file accelerate_config.yaml --num_processes 1 main.py
accelerate launch --config_file accelerate_config.yaml --num_processes 4 main.py