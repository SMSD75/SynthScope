#!/bin/bash
# AIDE (ICLR 2025): the official training recipe (shilinyan99/AIDE main_finetune.py, data/datasets.py + the paper's
# GenImage setting), reproduced in SynthScope.
#   data      GenImage SD v1.4 train (162k ImageNet reals + 162k SD v1.4 fakes); validation = the SD v1.4 val split
#   model     frozen OpenCLIP ConvNeXt-XXL (laion2b soup) + two ImageNet ResNet-50 on the SRM-filtered DCT patches + MLP head
#   train     at native resolution: Gaussian blur (3x3, sigma 0.1-3, p 0.1), JPEG (QF 30-100, p 0.1), DCT selection of
#             4 patches, every view resized to 256 (no flip, no crop). The one deviation: the long side is capped at
#             1024 before the DCT (dataloader-worker memory). The validation images get the same perturbations, as in
#             the official run (it validates through the training dataset class).
#   optimizer AdamW 5e-4 at effective batch 256, weight decay 0, 5 epochs, per-iteration cosine to 1e-6, no warmup,
#             label smoothing 0.1, fp32, no gradient clipping, no EMA
#   batch     the official 8 GPUs x 32 becomes 4 GPUs x 8 x grad_accum 8 = 256 (a 24 GB GPU fits a micro-batch of 8); the
#             BatchNorm statistics are synchronized over the 4 GPUs (32 images, as on each official GPU)
#   checkpoint model_best.pth = the epoch with the highest validation accuracy (the official max_accuracy); every epoch
#             is kept as model_<epoch>.pth
# Plain DDP (accelerate --multi_gpu): the repo's accelerate_config.yaml uses DeepSpeed with gradient clipping 1.0, which the
# official recipe does not have. Resume: rerun with the same SAVE (the state is saved after every epoch).
# Evaluate:  python benchmark/bench_eval.py --model aide --ckpt <save_path>/model_best.pth --benchmark genimage
# Requires:  pip install kornia==0.7.2   (the official perturbations)
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/GenImage}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/aide}
NUM_GPUS=${NUM_GPUS:-4}
MICRO=${MICRO:-8}                                   # micro-batch per GPU
if [ $((256 % (NUM_GPUS * MICRO))) -ne 0 ]; then
    echo "NUM_GPUS x MICRO = $((NUM_GPUS * MICRO)) must divide the effective batch 256" >&2; exit 1
fi
ACCUM=$((256 / (NUM_GPUS * MICRO)))                 # effective batch 256
WORKERS=${WORKERS:-5}
# wandb runs offline by default (batch nodes have no login, like the other recipe scripts); LOGGER_MODE=online to sync.

/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch --multi_gpu \
    --num_processes $NUM_GPUS --mixed_precision no main.py \
    trainer@TrainerConfig=image/aide \
    model@ModelConfig=image/aide \
    dataset@DatasetConfig=image/genimage \
    DatasetConfig.root_dir=$DATA \
    DatasetConfig.train_batch_size=$MICRO \
    DatasetConfig.num_workers=$WORKERS \
    DatasetConfig.val_batch_size=16 \
    DatasetConfig.test_batch_size=16 \
    TrainerConfig.timm_cosine.grad_accum_steps=$ACCUM \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=aide \
    logger.mode=${LOGGER_MODE:-offline} \
    "logger.tags=[SynthScope,AIDE,genimage]" \
    "$@"
