#!/bin/bash
# DDA (NeurIPS 2025): the official training recipe, reproduced in SynthScope.
#   data      DDA-Training-Set: 118,288 MSCOCO train2017 reals + their SD-2.0 VAE reconstructions (dataset `dda_train`);
#             each pair: matched JPEG of the fake (p 0.5), pixel/frequency blending (p 0.2 each), down/up-resized copies,
#             and ONE shared random augmentation for its six images (official ComposedTransforms), crop 336, CLIP norm
#   model     DINOv2 ViT-L/14 + LoRA rank 8 / alpha 1 on attn.qkv/proj and mlp.fc1/fc2 + linear head (from torch.hub)
#   loss      0.5 x BCE + 0.5 x contrastive (pos margin 0, neg margin 1) on the CLS features
#   optimizer AdamW 1e-4, wd 0, 2 pairs x 4 accumulation on ONE GPU, cosine T_max 1000 optimizer steps, 1 epoch
# Evaluate:  python benchmark/bench_eval.py --model dda --ckpt <save_path>/model_final.pth --dataset wildrf
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/DDATrain}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/dda}

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python main.py \
    trainer@TrainerConfig=image/dda \
    model@ModelConfig=image/dda \
    dataset@DatasetConfig=image/dda_train \
    DatasetConfig.root_dir=$DATA \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=dda \
    logger.mode=offline \
    "logger.tags=[SynthScope,DDA,dda_train]"
