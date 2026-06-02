#!/bin/bash
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1

# PGC (Zhou et al., 2026): LoRA-tuned DINOv2-Large RGB stream + YCbCr-quantization-residual CNN
# stream, fused by peak-guided calibration. The residual is computed INSIDE the model, so the
# dataset feeds a normal ImageNet-normalized image (normalize_dinov2 = what DINOv2 expects). The
# residual is resolution-sensitive, so we use CROP-only transforms (no resize) to preserve the
# per-pixel quantization pattern. Trained here on ProGAN (CNNDetection); swap the dataset for
# GenImage to match the paper's SDv1.4 setting.
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=image/pgc \
    model@ModelConfig=image/pgc \
    dataset@DatasetConfig=image/cnndetection \
    ModelConfig.backbone_name=facebook/dinov2-large \
    DatasetConfig.train_batch_size=32 \
    DatasetConfig.val_batch_size=64 \
    DatasetConfig.test_batch_size=64 \
    DatasetConfig.train_transform_list=[random_crop_224,random_horizontal_flip,to_dtype,normalize_dinov2] \
    DatasetConfig.eval_transform_list=[center_crop_224,to_dtype,normalize_dinov2] \
    metrics=[real_accuracy,fake_accuracy,AP,AUC] \
    logger.name=pgc \
    logger.mode=online \
    logger.tags=[AGID,PGC,cnndetection]
