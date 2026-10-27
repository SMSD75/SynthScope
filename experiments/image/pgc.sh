#!/bin/bash
# PGC (Zhou et al., 2026): the official ProGAN training recipe, reproduced in SynthScope.
#   data      ForenSynths / CNNDetection ProGAN train, all 20 classes (720k images = the UniversalFakeDetect train set)
#   train     flip -> ColorJitter(0.15, 0.15, 0.15, 0.05) p0.2 -> pad + RandomCrop 224 (no resize: the YCbCr residual
#             is resolution-sensitive) -> ImageNet normalization; the residual is computed inside the model
#   optimizer AdamW lr 5e-5 (constant over this budget), weight decay 0.05, label smoothing 0.1, batch 32 x 4
#             accumulation on ONE GPU, gradient clipping 1.0, seed 42
#   budget    2000 micro-batches with a checkpoint every 100 (model_step<N>.pth): the released checkpoint is ~400 steps
#             in and transfer to unseen generators erodes within the first epoch, so the early curve is evaluated
# Evaluate:  python benchmark/bench_eval.py --model pgc --ckpt <save_path>/model_step<N>.pth --benchmark forensynths --data <CNNDetection/test>
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/CNNDetection}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/pgc}

# Plain single-process run (no `accelerate launch`): gradient accumulation and clipping are done by the trainer's
# step budget exactly as in the official code.
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python main.py \
    trainer@TrainerConfig=image/pgc \
    model@ModelConfig=image/pgc \
    dataset@DatasetConfig=image/cnndetection \
    DatasetConfig.root_dir=$DATA \
    DatasetConfig.train_batch_size=32 \
    DatasetConfig.val_batch_size=64 \
    "DatasetConfig.train_transform_list=[random_horizontal_flip,color_jitter_pgc,pad_random_crop_224,to_dtype,normalize_dinov2]" \
    "DatasetConfig.eval_transform_list=[center_crop_224,to_dtype,normalize_dinov2]" \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=pgc \
    logger.mode=offline \
    "logger.tags=[SynthScope,PGC,progan20]"
