#!/bin/bash
# Train one AGID detector with checkpoint-saving enabled, into the central checkpoint dir's
# staging area. Recipes mirror experiments/image/*.sh; epochs are reduced for turnaround and
# overridable. After training, run promote_checkpoint.py to publish <key>.pth.
#
#   bash interface/scripts/train_model.sh <model_key> <gpus> <port> [epochs]
#   e.g. bash interface/scripts/train_model.sh clip 0,1 29510 2
#
# Notes:
#  * save_policy is forced to "min" (lowest val loss) because the trainer initialises
#    best_val_loss=+inf, so the repo default save_policy=max never writes a checkpoint.
#  * Runs in the AGID conda env; logs to $CKPT_DIR/_train/<key>/train.log.
set -uo pipefail

MODEL="${1:?usage: train_model.sh <model_key> <gpus> <port> [epochs]}"
GPUS="${2:?need GPU list, e.g. 0,1}"
PORT="${3:?need master port, e.g. 29510}"
EPOCHS_OVERRIDE="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
CONDA="${CONDA_EXE:-/home/ssalehi/miniconda3/bin/conda}"
CKPT_DIR="${AGID_CKPT_DIR:-/hddstore/ssalehi/AGID/checkpoints}"
CACHE="${AGID_CACHE_DIR:-/hddstore/ssalehi/AGID/.cache}"
SAVE_DIR="$CKPT_DIR/_train/$MODEL"
mkdir -p "$SAVE_DIR"

export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export TRITON_CACHE_DIR="$CACHE/triton"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DINO_V3_REPO_DIR="${DINO_V3_REPO_DIR:-/home/ssalehi/.cache/torch/hub/facebookresearch_dinov3}"
export DINO_V3_WEIGHTS_DIR="${DINO_V3_WEIGHTS_DIR:-/hddstore/ssalehi/AGID/dinov3_checkpoints}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")

# Common transform lists.
T_CLIP_TRAIN="[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]"
T_CLIP_EVAL="[resize_short_256,center_crop_224,to_dtype,normalize_clip]"
T_IMNET_TRAIN="[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype,normalize_dinov2]"
T_IMNET_EVAL="[resize_short_256,center_crop_224,to_dtype,normalize_dinov2]"
T_RAW_TRAIN="[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype]"
T_RAW_EVAL="[resize_short_256,center_crop_224,to_dtype]"

OV=()  # Hydra overrides
case "$MODEL" in
  clip)
    OV=(ModelConfig.model_type=clip ModelConfig.backbone_name=CLIP:ViT-L/14
        ModelConfig.use_adapter=false ModelConfig.bottleneck_dim=64 ModelConfig.freeze_backbone=true
        dataset@DatasetConfig=image/cnndetection
        DatasetConfig.train_transform_list=$T_CLIP_TRAIN DatasetConfig.eval_transform_list=$T_CLIP_EVAL)
    EPOCHS=${EPOCHS_OVERRIDE:-2} ;;
  dinov2)
    OV=(ModelConfig.model_type=dinov2 ModelConfig.backbone_name=dinov2_vitl14
        ModelConfig.use_adapter=false ModelConfig.bottleneck_dim=64 ModelConfig.freeze_backbone=true
        dataset@DatasetConfig=image/cnndetection
        DatasetConfig.train_transform_list=$T_IMNET_TRAIN DatasetConfig.eval_transform_list=$T_IMNET_EVAL)
    EPOCHS=${EPOCHS_OVERRIDE:-2} ;;
  dinov3)
    # Frozen DINOv3 ViT-L/16 probe (weights at $DINO_V3_WEIGHTS_DIR). DINOv3 normalization + 256 crops.
    OV=(ModelConfig.model_type=dinov3 ModelConfig.backbone_name=dinov3_vitl16
        ModelConfig.use_adapter=false ModelConfig.bottleneck_dim=64 ModelConfig.freeze_backbone=true
        dataset@DatasetConfig=image/cnndetection
        DatasetConfig.train_transform_list="[resize_short_256,gaussian_blur,jpeg_compression,random_crop_256,random_horizontal_flip,to_dtype,normalize_dinov3]"
        DatasetConfig.eval_transform_list="[resize_short_256,center_crop_256,to_dtype,normalize_dinov3]")
    EPOCHS=${EPOCHS_OVERRIDE:-2} ;;
  pgc)
    # LoRA-DINOv2-Large + residual stream. Train on the ~120K ProGAN subset for turnaround;
    # crop-only (no resize) preserves the quantization residual; normalize_dinov2 = ImageNet.
    OV=(trainer@TrainerConfig=image/pgc model@ModelConfig=image/pgc
        dataset@DatasetConfig=image/cospy_subset
        DatasetConfig.train_batch_size=32 DatasetConfig.val_batch_size=64 DatasetConfig.test_batch_size=64
        DatasetConfig.train_transform_list="[random_crop_224,random_horizontal_flip,to_dtype,normalize_dinov2]"
        DatasetConfig.eval_transform_list="[center_crop_224,to_dtype,normalize_dinov2]")
    EPOCHS=${EPOCHS_OVERRIDE:-2} ;;
  adapted_dino)
    # Adapters at every block retain backbone activations for backprop -> small batch.
    OV=(ModelConfig.model_type=adapted_dino ModelConfig.backbone_name=dinov2_vitl14
        ModelConfig.use_adapter=true ModelConfig.bottleneck_dim=64 ModelConfig.freeze_backbone=true
        dataset@DatasetConfig=image/cnndetection
        DatasetConfig.train_batch_size=64 DatasetConfig.val_batch_size=256 DatasetConfig.test_batch_size=256
        DatasetConfig.train_transform_list=$T_IMNET_TRAIN DatasetConfig.eval_transform_list=$T_IMNET_EVAL)
    EPOCHS=${EPOCHS_OVERRIDE:-2} ;;
  cnn)
    # Full ResNet-50 fine-tune (backbone trains) -> moderate batch.
    OV=(ModelConfig.model_type=cnn ModelConfig.freeze_backbone=false
        dataset@DatasetConfig=image/cnndetection
        DatasetConfig.train_batch_size=128 DatasetConfig.val_batch_size=512 DatasetConfig.test_batch_size=512
        DatasetConfig.train_transform_list=$T_IMNET_TRAIN DatasetConfig.eval_transform_list=$T_IMNET_EVAL)
    EPOCHS=${EPOCHS_OVERRIDE:-3} ;;
  vib_net)
    OV=(trainer@TrainerConfig=image/vib_net model@ModelConfig=image/vib_net
        dataset@DatasetConfig=image/genimage
        DatasetConfig.train_transform_list=$T_CLIP_TRAIN DatasetConfig.eval_transform_list=$T_CLIP_EVAL)
    EPOCHS=${EPOCHS_OVERRIDE:-3} ;;
  spai)
    OV=(trainer@TrainerConfig=image/spai model@ModelConfig=image/spai
        dataset@DatasetConfig=image/spai_ldm
        DatasetConfig.train_transform_list=$T_RAW_TRAIN DatasetConfig.eval_transform_list=$T_RAW_EVAL)
    EPOCHS=${EPOCHS_OVERRIDE:-5} ;;
  co_spy)
    OV=(trainer@TrainerConfig=image/co_spy model@ModelConfig=image/co_spy
        dataset@DatasetConfig=image/cospy_subset
        DatasetConfig.train_transform_list="[resize_short_384,gaussian_blur,jpeg_compression,random_crop_384,random_horizontal_flip,to_dtype]"
        DatasetConfig.eval_transform_list="[resize_short_384,center_crop_384,to_dtype]")
    EPOCHS=${EPOCHS_OVERRIDE:-3} ;;
  safe)
    # AIGIBench (the paper recipe) is not on this node; train SAFE from scratch on ProGAN instead.
    OV=(trainer@TrainerConfig=image/safe ModelConfig.model_type=safe ModelConfig.freeze_backbone=false
        dataset@DatasetConfig=image/cnndetection
        DatasetConfig.train_transform_list="[random_crop_256_pad,random_horizontal_flip,random_rotation_180,color_jitter_safe,to_dtype,random_mask_safe]"
        DatasetConfig.eval_transform_list="[center_crop_256,to_dtype]")
    EPOCHS=${EPOCHS_OVERRIDE:-10} ;;
  restrav)
    OV=(trainer@TrainerConfig=video/restrav model@ModelConfig=video/restrav
        dataset@DatasetConfig=video/genvideo DatasetConfig.root_dir=/hddstore/ssalehi/AGID/restrav_data
        DatasetConfig.num_frames=24 DatasetConfig.num_workers=8
        DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=16 DatasetConfig.test_batch_size=16
        ++TrainerConfig.test_interval=5
        DatasetConfig.train_transform_list="[normalize_dinov2]" DatasetConfig.eval_transform_list="[normalize_dinov2]")
    EPOCHS=${EPOCHS_OVERRIDE:-10} ;;
  *) echo "unknown model: $MODEL"; exit 2 ;;
esac

LOG="$SAVE_DIR/train.log"
echo "=== train $MODEL on GPUs=$GPUS ($NPROC proc) port=$PORT epochs=$EPOCHS ===" | tee "$LOG"
echo "save_path=$SAVE_DIR" | tee -a "$LOG"

# CUDA init can flake on this node's old kernel; retry the pre-check before launching.
for try in 1 2 3 4 5; do
  if "$CONDA" run -n AGID python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1,device='cuda')" 2>/dev/null; then
    break
  fi
  echo "[precheck] CUDA not ready (try $try), retrying..." | tee -a "$LOG"; sleep 5
done

"$CONDA" run --no-capture-output -n AGID accelerate launch \
  --config_file accelerate_config.yaml --num_processes "$NPROC" --main_process_port "$PORT" \
  main.py "${OV[@]}" \
  ++TrainerConfig.epochs="$EPOCHS" \
  ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 \
  ++TrainerConfig.save_path="$SAVE_DIR" \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] \
  logger.mode=offline logger.name="train_$MODEL" \
  2>&1 | grep --line-buffered -v -E "libpng (error|warning)" | tee -a "$LOG"

echo "=== done $MODEL (exit ${PIPESTATUS[0]}) ===" | tee -a "$LOG"
