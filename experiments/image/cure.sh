#!/bin/bash
# CuRe (arXiv 2609.14316): the paper's three-stage training recipe (Sec. 3, 4.1-4.2, Appendix A); the official release
# (github.com/manic-cui/CuRe) has inference code only, so this follows the paper. See models/image/cure/README.md.
#   data      GenImage SD v1.4 train: 162,000 ImageNet reals + 162,000 SD v1.4 images (PNG -> JPEG q96 to match the reals)
#   model     Perception Encoder Core L/14 @336 (facebook/PE-Core-L14-336, frozen) + LoRA on Q and V of all 24 layers
#             (rank 64, alpha 128, dropout 0); feature = mean of the 576 patch tokens (1024-d)
#   stage 1   ratio regression: x_a = (1 - a) x_r + a x_g, a ~ U(0, 1), independent real / generated endpoints; linear
#             ratio head + sigmoid, BCE with soft target a; AdamW lr 1e-5, weight decay 0.1 LoRA / 0.01 head, 5 epochs,
#             per-GPU batch 64, 2 warmup epochs then cosine to 1e-6 (main.py, conf/trainer/image/cure.yaml)
#   stage 2   frozen encoder; 20,000 endpoint pairs per trajectory type RF / RR / FF on {0, .2, .4, .6, .8, 1}; slopes,
#             trace-normalised C_F, C_M, C_S, generalised eigenproblem -> W (1024 x 128), mu (train_stages.py stage2)
#   stage 3   Linear(128, 2) on W^T (z - mu), dropout 0.05, CE, AdamW lr 2e-4, weight decay 0.1, batch 256, 5 epochs
#             (train_stages.py stage3) -> $SAVE/stage23/cure_trained.pth, the released adapter's 100 keys
#   batch     per-GPU 64 = MICRO x ACCUM (MICRO 64 + GRAD_CKPT=true on a 48 GB GPU; e.g. MICRO 16 x ACCUM 4 on 24 GB);
#             the paper does not state its GPU count (global batch = NGPU x 64)
# Plain DDP with fp16 autocast + loss scaling, no gradient clipping (not the repo's DeepSpeed accelerate_config.yaml, which
# clips at 1.0). Stage 1 resumes after its last finished epoch when rerun with the same SAVE.
# Evaluate:  python benchmark/bench_eval.py --model cure --ckpt $SAVE/stage23/cure_trained.pth --dataset wildrf
set -eo pipefail
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/GenImage}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/cure}
NGPU=${NGPU:-1}
MICRO=${MICRO:-64}
GRAD_CKPT=${GRAD_CKPT:-true}
WORKERS=${WORKERS:-8}
STAGES=${STAGES:-"1 2 3"}
PY=${PY:-/home/ssalehi/miniconda3/envs/SynthScope/bin/python}
if (( 64 % MICRO != 0 )); then echo "MICRO must divide the per-GPU batch 64" >&2; exit 1; fi
ACCUM=$((64 / MICRO))
MULTI=""; (( NGPU > 1 )) && MULTI="--multi_gpu"

if [[ " $STAGES " == *" 1 "* ]]; then
    "$(dirname $PY)/accelerate" launch $MULTI --num_processes $NGPU --mixed_precision fp16 main.py \
        trainer@TrainerConfig=image/cure \
        model@ModelConfig=image/cure \
        dataset@DatasetConfig=image/cure_train \
        ModelConfig.training_stage=ratio \
        ModelConfig.grad_checkpointing=$GRAD_CKPT \
        DatasetConfig.root_dir=$DATA \
        DatasetConfig.train_batch_size=$MICRO \
        DatasetConfig.num_workers=$WORKERS \
        TrainerConfig.timm_cosine.grad_accum_steps=$ACCUM \
        TrainerConfig.save_path=$SAVE/stage1 \
        "metrics=[]" \
        logger.name=cure \
        logger.mode=${LOGGER_MODE:-offline} \
        "logger.tags=[SynthScope,CuRe,genimage_sd14]" \
        "$@"
fi
if [[ " $STAGES " == *" 2 "* ]]; then
    $PY -m models.image.cure.train_stages stage2 --lora $SAVE/stage1/model_last.pth --data $DATA --out $SAVE/stage23 \
        --workers $WORKERS ${STAGE2_ARGS:-}
fi
if [[ " $STAGES " == *" 3 "* ]]; then
    $PY -m models.image.cure.train_stages stage3 --lora $SAVE/stage1/model_last.pth --data $DATA --out $SAVE/stage23 \
        --workers $WORKERS ${STAGE3_ARGS:-}
fi
