#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# D3 is training-free: a frozen CLIP ViT-B/16 extracts per-frame features, and the per-video
# second-order std becomes the detection score. Frames are decoded to 224x224 in [0,1] and
# normalized by the dataset transform pipeline (normalize_clip here, matched to the CLIP
# backbone -- switch to normalize_dinov2 if you set a DINOv2/ResNet backbone). AP/AUROC are the
# meaningful metrics. The video benchmark must be present under DatasetConfig.root_dir;
# this is a template (data download is out of scope).
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 main.py \
    trainer@TrainerConfig=video/d3 \
    model@ModelConfig=video/d3 \
    dataset@DatasetConfig=video/genvideo \
    ModelConfig.backbone_name=clip \
    ModelConfig.first_order=l2 \
    DatasetConfig.test_batch_size=8 \
    DatasetConfig.num_frames=16 \
    DatasetConfig.fps=8 \
    DatasetConfig.train_transform_list=[normalize_clip] \
    DatasetConfig.eval_transform_list=[normalize_clip] \
    metrics=[real_accuracy,fake_accuracy,AP,AUC] \
    logger.name=d3 \
    logger.mode=online \
    logger.tags=[SynthScope,D3,genvideo]
