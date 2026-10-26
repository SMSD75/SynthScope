#!/bin/bash
# NPR (CVPR 2024): the official training recipe, reproduced in SynthScope.
#   data      ForenSynths / CNNDetection ProGAN train + val, 4-class setting (car, cat, chair, horse)
#   train     Resize((256, 256)) -> RandomCrop 224 -> flip -> ImageNet normalization (no blur/JPEG: the official
#             data_augment call is commented out); NPR's neighboring-pixel residual is computed inside the model
#   val       Resize((256, 256)) -> CenterCrop 224 -> ImageNet normalization
#   optimizer Adam lr 2e-4 (AdamW, weight decay 0), batch 32 on ONE GPU, 50 epochs, lr x0.9 every 10 epochs
#   checkpoint model_last.pth (the official code releases the last epoch); model_best.pth is kept for reference
# Evaluate with:  python benchmark/bench_eval.py --model npr --ckpt <save_path>/model_last.pth --benchmark forensynths --data <CNNDetection/test>
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/CNNDetection}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/npr}

# Plain single-process run (no `accelerate launch`): the repo's accelerate_config.yaml uses DeepSpeed with gradient
# clipping 1.0, which the official recipe does not have.
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python main.py \
    trainer@TrainerConfig=image/npr \
    model@ModelConfig=image/npr \
    dataset@DatasetConfig=image/cnndetection \
    DatasetConfig.root_dir=$DATA \
    "+DatasetConfig.train_classes=[car,cat,chair,horse]" \
    DatasetConfig.train_batch_size=32 \
    DatasetConfig.val_batch_size=64 \
    "DatasetConfig.train_transform_list=[resize_256,random_crop_224,random_horizontal_flip,to_dtype,normalize_dinov2]" \
    "DatasetConfig.eval_transform_list=[resize_256,center_crop_224,to_dtype,normalize_dinov2]" \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=npr \
    logger.mode=offline \
    "logger.tags=[SynthScope,NPR,progan4]"
