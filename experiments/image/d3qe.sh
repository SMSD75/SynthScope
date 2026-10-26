#!/bin/bash
# D3QE (ICCV 2025): the official training recipe, reproduced in SynthScope.
#   data      ARForensics train (100k LlamaGen vs 100k ImageNet) and val (10k + 10k); downloaded on first use
#   model     frozen LlamaGen VQ-16 + frozen CLIP ViT-L/14; only the D3AT transformer and the heads are trained
#   train     short side 256 -> CNNDetection blur/JPEG (p 0.1 each) -> RandomCrop 256 -> flip -> [0, 1], no normalization
#   val       short side 256 -> the same blur/JPEG at fixed settings (sigma 1.5, PIL q65) -> CenterCrop 256
#   optimizer AdamW lr 8e-5, weight decay 0.01, batch 32 on ONE GPU (BatchNorm in the heads), 10 epochs,
#             best-validation-accuracy checkpoint, learning rate /10 after 5 epochs without improvement
# Evaluate the result with:  python benchmark/bench_eval.py --model d3qe --ckpt <save_path>/model_best.pth --dataset arforensics
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/ARForensics}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/d3qe}

# Plain single-process run (no `accelerate launch`): the repo's accelerate_config.yaml uses DeepSpeed with gradient
# clipping 1.0, which the official recipe does not have.
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python main.py \
    trainer@TrainerConfig=image/d3qe \
    model@ModelConfig=image/d3qe \
    dataset@DatasetConfig=image/arforensics \
    DatasetConfig.root_dir=$DATA \
    DatasetConfig.train_batch_size=32 \
    DatasetConfig.val_batch_size=64 \
    "+DatasetConfig.train_transform_list=[resize_short_256,blur_jpeg_cnndet,random_crop_256,random_horizontal_flip,to_dtype]" \
    "+DatasetConfig.val_transform_list=[resize_short_256,blur_jpeg_cnndet_val,center_crop_256,to_dtype]" \
    "DatasetConfig.eval_transform_list=[resize_short_256,center_crop_256,to_dtype]" \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=d3qe \
    logger.mode=offline \
    "logger.tags=[SynthScope,D3QE,arforensics]"
