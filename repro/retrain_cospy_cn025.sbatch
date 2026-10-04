#!/bin/bash
#SBATCH --job-name=cospy_rt --partition=cees --account=ceesusers --nodelist=ivi-cn025
#SBATCH --time=2-00:00:00 --cpus-per-task=10 --mem=40G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/retrain_cospy_%j.log
# Co-Spy gap-closing retrain on the cospy_subset (faithful Co-Spy class). Waits for the VIB job to stage
# data onto cn025. Conservative batch for the 384px SigLIP+VAE forward. Saves to /ivi shared.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
SCR=$(for d in /nvmestore /ssdstore /hddstore /sddstore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done)
DATA=$SCR/ssalehi/SynthScope; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/co_spy; mkdir -p "$SAVE"
export HF_HOME=$SCR/ssalehi/.cache/huggingface TORCH_HOME=$SCR/ssalehi/.cache/torch TRITON_CACHE_DIR=$SCR/ssalehi/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
for i in $(seq 1 240); do [ -f /ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/DATA_READY_cn025 ] && [ -d "$DATA/spai_train/cospy_subset/train" ] && break; echo "[wait] data not ready ($i)"; sleep 30; done
echo "node=$(hostname) scratch=$SCR data=$DATA/spai_train/cospy_subset $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29621 \
  main.py trainer@TrainerConfig=image/co_spy model@ModelConfig=image/co_spy dataset@DatasetConfig=image/cospy_subset DatasetConfig.root_dir=$DATA/spai_train/cospy_subset \
  DatasetConfig.train_transform_list="[resize_short_384,gaussian_blur,jpeg_compression,random_crop_384,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_384,center_crop_384,to_dtype]" \
  DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 DatasetConfig.num_workers=10 \
  ++TrainerConfig.epochs=8 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=cospy_retrain 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== COSPY RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
