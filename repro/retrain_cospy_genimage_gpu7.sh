#!/bin/bash
# Co-Spy gap-closing retrain on GenImage SD-v1.4 (the correct domain for the sd-v1_4 Co-Spy variant:
# SigLIP-384 + SD-VAE-residual). The prior retrain used cospy_subset = ProGAN (wrong domain for the
# SD-VAE artifact branch) -> 72.72 ceiling. Arch audit confirmed the SynthScope Co-Spy class MATCHES the
# official sd-v1_4 detector, so the gap is training data. GenImage SDv1.4 = 162k/162k on disk.
# cn023 GPU 7 (legal), single process, raw [0,1] 384 input (model normalizes internally).
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch
export TRITON_CACHE_DIR=/hddstore/ssalehi/.triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7
SAVE=/hddstore/ssalehi/SynthScope/checkpoints/_train/cospy_genimage; mkdir -p "$SAVE"
echo "=== cospy genimage retrain start $(date) node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES ==="
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope accelerate launch \
  --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29655 \
  main.py trainer@TrainerConfig=image/co_spy model@ModelConfig=image/co_spy dataset@DatasetConfig=image/genimage \
  DatasetConfig.root_dir=/hddstore/ssalehi/SynthScope/GenImage \
  DatasetConfig.train_transform_list="[resize_short_384,gaussian_blur,jpeg_compression,random_crop_384,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_384,center_crop_384,to_dtype]" \
  DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 \
  DatasetConfig.num_workers=8 DatasetConfig.prefetch=false \
  ++TrainerConfig.epochs=3 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min \
  ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path="$SAVE" \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=cospy_genimage \
  2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== COSPY GENIMAGE RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
