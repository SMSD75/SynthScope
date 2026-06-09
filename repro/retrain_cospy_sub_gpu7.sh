#!/bin/bash
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/AGID
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch TRITON_CACHE_DIR=/hddstore/ssalehi/.triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7
SAVE=/hddstore/ssalehi/AGID/checkpoints/_train/cospy_genimage_sub; mkdir -p "$SAVE"
echo "=== cospy SUBSET retrain start $(date) ==="
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29655 \
  main.py trainer@TrainerConfig=image/co_spy model@ModelConfig=image/co_spy dataset@DatasetConfig=image/genimage \
  DatasetConfig.root_dir=/hddstore/ssalehi/AGID/GenImage_sub \
  DatasetConfig.train_transform_list="[resize_short_384,gaussian_blur,jpeg_compression,random_crop_384,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_384,center_crop_384,to_dtype]" \
  DatasetConfig.train_batch_size=24 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 DatasetConfig.num_workers=8 DatasetConfig.prefetch=false \
  ++TrainerConfig.epochs=5 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path="$SAVE" \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=cospy_sub 2>&1 | grep --line-buffered -vE "libpng|Testing:|Training:|Validation:"
echo "=== COSPY SUBSET DONE (exit ${PIPESTATUS[0]}) $(date) ==="
