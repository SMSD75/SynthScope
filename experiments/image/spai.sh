#!/bin/bash
# SPAI (Karageorgiou et al., CVPR 2025): the official training recipe (configs/spai.yaml + the README's train command),
# reproduced in SynthScope.
#   data      GRIP's latent-diffusion training set (180k LDM fakes) + 90k COCO train2017 + 90k LSUN (CNNDetection ProGAN-train
#             reals) as listed by the set's real_coco.txt / real_lsun.txt; validation = its valid split (20k + 20k),
#             augmented once offline (SPAITrainDataset.augment_val, built on first use with DatasetConfig.prefetch=true)
#   train     4 independent draws of the official albumentations transform (RandomResizedCrop 224 scale 0.2-1, flips,
#             rotation +-90 p0.5, Gaussian blur p0.5, Gaussian noise p0.5, JPEG 50-100 p0.5) side by side: the Spectral
#             Context Attention sees them as 4 patches; the MFM ViT-B/16 backbone is frozen (drop path 0.1 in training)
#   optimizer AdamW 5e-4, weight decay 0.05 (none on biases / 1-D), batch 72, 35 epochs, 5-epoch warmup from 2.5e-7,
#             cosine to 2.5e-7 per step, fp16 mixed precision, seed 0; the checkpoint with the lowest validation loss
#             (model_best.pth) is the one the official README evaluates
#   memory    batch 72 needs a 48 GB GPU; GRAD_ACCUM=2 runs two micro-batches of 36 on a 24 GB GPU with the same schedule
# Evaluate:  python benchmark/bench_eval.py --model spai --ckpt <save_path>/model_best.pth --benchmark synthbuster --data <...>
# Requires:  pip install "albumentations==1.4.14" "albucore==0.0.16" (the versions the official code pins)
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1
export ACCELERATE_MIXED_PRECISION=fp16      # the official run uses APEX AMP O2 (fp16)
DATA=${DATA:-/nvmestore/ssalehi/SynthScope}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/spai}
GRAD_ACCUM=${GRAD_ACCUM:-1}

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python main.py \
    trainer@TrainerConfig=image/spai \
    model@ModelConfig=image/spai \
    dataset@DatasetConfig=image/spai_train \
    DatasetConfig.root_dir=$DATA/spai_train \
    DatasetConfig.coco_dir=$DATA/COCO/train2017 \
    DatasetConfig.cnndetection_dir=$DATA/CNNDetection \
    DatasetConfig.train_batch_size=$((72 / GRAD_ACCUM)) \
    TrainerConfig.timm_cosine.grad_accum_steps=$GRAD_ACCUM \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=spai \
    logger.mode=offline \
    "logger.tags=[SynthScope,SPAI,ldm]"
