#!/bin/bash
# FatFormer (CVPR 2024): the training recipe of the paper (Sec. 4.1; the authors released no training code), in SynthScope.
#   data      ForenSynths / CNNDetection ProGAN train + val, the released 4-class setting (car, cat, chair, horse; 144k train images)
#   model     CLIP ViT-L/14 (both towers frozen) + forgery-aware adapters at blocks 7/15/23 (num_vit_adapter 3), language-guided
#             alignment (8 context tokens + patch-based enhancer) and text-guided interactor: only those are trained
#   train     Resize((256, 256)) -> RandomCrop 224 -> flip -> ImageNet normalization ("no other augmentations": no blur/JPEG)
#   val       Resize((256, 256)) -> CenterCrop 224 -> ImageNet normalization
#   loss      cross-entropy on the summed logits (Eq. 10), fp32
#   optimizer Adam (AdamW with weight decay 0), lr 4e-4, total batch 256, 25 epochs, lr x0.9 every 10 epochs, no clipping
#   batch     256 = NGPU x MICRO x ACCUM; the defaults 4 x 32 x 2 fit 24 GB GPUs (ACCUM is derived from NGPU and MICRO)
#   checkpoint model_last.pth (the last epoch, taken as the released model); model_best.pth (best validation accuracy) for reference
# Evaluate with:  python benchmark/bench_eval.py --model fat_former --ckpt <save_path>/model_last.pth --benchmark forensynths --data <CNNDetection/test>
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/CNNDetection}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/fat_former}
NGPU=${NGPU:-4}
MICRO=${MICRO:-32}
TOTAL_BATCH=256
if (( TOTAL_BATCH % (NGPU * MICRO) != 0 )); then echo "NGPU x MICRO must divide $TOTAL_BATCH" >&2; exit 1; fi
ACCUM=$((TOTAL_BATCH / (NGPU * MICRO)))

# Plain DDP (accelerate_config_ddp.yaml), not the repo's DeepSpeed accelerate_config.yaml: that one clips gradients at 1.0,
# which the paper's recipe does not mention.
if (( NGPU > 1 )); then
    LAUNCH="accelerate launch --config_file accelerate_config_ddp.yaml --num_processes $NGPU"
else
    LAUNCH="python"
fi
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope $LAUNCH main.py \
    trainer@TrainerConfig=image/fat_former \
    model@ModelConfig=image/fat_former \
    dataset@DatasetConfig=image/cnndetection \
    DatasetConfig.root_dir=$DATA \
    "+DatasetConfig.train_classes=[car,cat,chair,horse]" \
    DatasetConfig.train_batch_size=$MICRO \
    DatasetConfig.val_batch_size=64 \
    TrainerConfig.lr_step.grad_accum_steps=$ACCUM \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=fat_former \
    logger.mode=offline \
    "logger.tags=[SynthScope,FatFormer,progan4]" \
    "$@"
