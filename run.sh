#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=2,3,4,5
export OMP_NUM_THREADS=4

accelerate launch --config_file accelerate_config.yaml --num_processes 4 main.py