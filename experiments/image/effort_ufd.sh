#!/bin/bash
# Effort (Yan et al., ICML 2025): the official UniversalFakeDetect_Benchmark training recipe (ProGAN -> UFD tables 2/3),
# reproduced in SynthScope. Source: YZY-stack/Effort-AIGI-Detection, UniversalFakeDetect_Benchmark/{script/train_svd.sh,
# train.py, models/trainer.py, data/datasets.py}.
#   data      ForenSynths / CNNDetection ProGAN train, all 20 classes (720k images = the UniversalFakeDetect train set)
#   train     RandomCrop 224 (from the 256x256 images) -> flip -> CLIP normalization (--data_aug is off: no blur/JPEG)
#   model     CLIP ViT-L/14, every self-attention projection split by SVD (rank 1023 frozen, rank-1 residual trained)
#             + the classifier head (N(0, 0.02) init); nothing else trains
#   loss      the official 1-logit BCE, exactly: row 0 of the 2-logit head is pinned at zero (head_single_logit), so the
#             cross-entropy on the logits [0, z] equals BCEWithLogits(z); see the loss / head row of the table in
#             models/image/effort/README.md (Training)
#   optimizer AdamW lr 2e-4 constant, weight decay 0, betas (0.9, 0.999), batch 48 (no accumulation) on ONE GPU, no clipping, seed 0, fp32
#   budget    24,000 iterations of the batch of 48 (measured to fit a 24 GB card: peak 19.8 GiB, smoke job 275708),
#             a checkpoint every 3,000 iterations (model_step<N>.pth); the official test uses iteration 18000
# Evaluate:  python benchmark/bench_eval.py --model effort --ckpt <save_path>/model_step18000.pth ...
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/CNNDetection}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/effort_ufd}

# Plain single-process run (no `accelerate launch`): the repo's accelerate_config.yaml uses DeepSpeed with gradient
# clipping 1.0, which the official recipe does not have.
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python main.py \
    trainer@TrainerConfig=image/effort_ufd \
    model@ModelConfig=image/effort_ufd \
    dataset@DatasetConfig=image/cnndetection \
    DatasetConfig.root_dir=$DATA \
    DatasetConfig.train_batch_size=48 \
    DatasetConfig.val_batch_size=64 \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=effort_ufd \
    logger.mode=offline \
    "logger.tags=[SynthScope,Effort,progan20]" \
    "$@"      # extra Hydra overrides, e.g. DatasetConfig.prefetch=false
