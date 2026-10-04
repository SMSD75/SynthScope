#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# STALL (Ben Hayun et al., CVPR'26) is training-free: a frozen DINOv3 ViT-L/16 extracts
# per-frame features, which are scored against a VATEX real-video calibration via two whitened
# isotropic-Gaussian likelihood branches (spatial + temporal). Frames are decoded square at 224
# in [0,1] and normalized with ImageNet stats -- normalize_dinov2 here, because the official
# DINOv3 transform uses ImageNet mean/std (NOT normalize_dinov3). AP/AUROC are the meaningful
# metrics. The video benchmark must be present under DatasetConfig.root_dir.
#
# DINOv3 ViT-L/16 backbone is GATED: point these at your local clone + weights.
export DINO_V3_REPO_DIR=${DINO_V3_REPO_DIR:-/home/ssalehi/.cache/torch/hub/facebookresearch_dinov3}
export DINO_V3_WEIGHTS=${DINO_V3_WEIGHTS:-/hddstore/ssalehi/SynthScope/dinov3_checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 main.py \
    trainer@TrainerConfig=video/stall \
    model@ModelConfig=video/stall \
    dataset@DatasetConfig=video/genvideo \
    ModelConfig.backbone_name=dinov3_vitl16 \
    DatasetConfig.test_batch_size=8 \
    DatasetConfig.num_frames=16 \
    DatasetConfig.frame_size=224 \
    DatasetConfig.train_transform_list=[normalize_dinov2] \
    DatasetConfig.eval_transform_list=[normalize_dinov2] \
    metrics=[real_accuracy,fake_accuracy,AP,AUC] \
    logger.name=stall \
    logger.mode=online \
    logger.tags=[SynthScope,STALL,genvideo]
