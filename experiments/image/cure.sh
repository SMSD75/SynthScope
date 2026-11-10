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
# clips at 1.0); the accelerate launch is pinned by an explicit
# minimal config file + flags (ACC_ARGS), independent of the host's default accelerate config.
# Stage 1 resumes after its last finished epoch when rerun with the same SAVE.
# Stage-1 epoch: every epoch is saved (stage1/model_<k>.pth). The last epoch over-adapts to SD v1.4; the SynthScope-trained
# checkpoints that partially reproduce the paper use the epoch picked by the pre-registered, non-benchmark rule of
# repro/cure_stage1/epoch_select.py (epoch 0 in both runs), set with LORA=$SAVE/stage1/model_<k>.pth for stages 2-3. Optional
# DDA-style training augmentation (an assumption; the paper states none): append DatasetConfig.train_augment=dda to stage 1.
# See models/image/cure/README.md "Trained checkpoint".
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
LORA=${LORA:-}                  # stage-1 checkpoint for stages 2-3 (default: $SAVE/stage1/model_last.pth)
PY=${PY:-/home/ssalehi/miniconda3/envs/SynthScope/bin/python}
if (( 64 % MICRO != 0 )); then echo "MICRO must divide the per-GPU batch 64" >&2; exit 1; fi
ACCUM=$((64 / MICRO))
# STEP_MULT (default 1): accumulate STEP_MULT x the per-GPU 64 per optimizer step, i.e. global batch NGPU x 64 x STEP_MULT.
# Only for the stage-1 investigation's R2 run (global 512 = 2 GPUs x 64 x 4, the paper's per-GPU 64 on an assumed 8 GPUs);
# the recipe default is 1 (global NGPU x 64).
STEP_MULT=${STEP_MULT:-1}
ACCUM=$((ACCUM * STEP_MULT))
MULTI=""; (( NGPU > 1 )) && MULTI="--multi_gpu"
# Every accelerate setting is explicit (a minimal config file AND flags), so a default accelerate config on the host
# (DeepSpeed, dynamo, other precision) cannot change the run: single machine, no dynamo, fp16 autocast + loss scaling
# (CuRe.features always autocasts to fp16).
ACC_CFG=$(mktemp --suffix=.yaml)
trap 'rm -f "$ACC_CFG"' EXIT
DIST=MULTI_GPU; (( NGPU > 1 )) || DIST="'NO'"
cat > "$ACC_CFG" <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: $DIST
downcast_bf16: 'no'
dynamo_config: {dynamo_backend: 'NO'}
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: $NGPU
rdzv_backend: static
same_network: true
use_cpu: false
YAML
ACC_ARGS="--config_file $ACC_CFG --num_machines 1 --dynamo_backend no --mixed_precision fp16 --num_processes $NGPU $MULTI"

if [[ " $STAGES " == *" 1 "* ]]; then
    "$(dirname $PY)/accelerate" launch $ACC_ARGS main.py \
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
    $PY -m models.image.cure.train_stages stage2 --lora ${LORA:-$SAVE/stage1/model_last.pth} --data $DATA --out $SAVE/stage23 \
        --workers $WORKERS ${STAGE2_ARGS:-}
fi
if [[ " $STAGES " == *" 3 "* ]]; then
    $PY -m models.image.cure.train_stages stage3 --lora ${LORA:-$SAVE/stage1/model_last.pth} --data $DATA --out $SAVE/stage23 \
        --workers $WORKERS ${STAGE3_ARGS:-}
fi
