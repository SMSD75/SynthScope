#!/bin/bash
#SBATCH --job-name=aideP_rt --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=1-12:00:00 --cpus-per-task=24 --mem=60G --gres=gpu:3
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/retrain_aideP_%j.log
# AIDE ProGAN gap-closer, 3-GPU DeepSpeed. Replaces the slow single-GPU cn023 run (was ep0 after 5h).
# CNNDetection train is on cn029 /ssdstore. epochs=3 (enough to close the ForenSynths gap by the deadline).
# Saves to checkpoints_retrain/aide_progan.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
DATA=/ssdstore/ssalehi/SynthScope; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/aide_progan; mkdir -p "$SAVE"
export HF_HOME=$DATA/.cache/huggingface TORCH_HOME=$DATA/.cache/torch TRITON_CACHE_DIR=$DATA/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
[ -d "$DATA/CNNDetection/train" ] || { echo "!!! no CNNDetection/train on $(hostname)"; exit 1; }
echo "node=$(hostname) gpus=3 data=$DATA/CNNDetection $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 3 --main_process_port 29654 \
  main.py trainer@TrainerConfig=image/aide model@ModelConfig=image/aide dataset@DatasetConfig=image/cnndetection DatasetConfig.root_dir=$DATA/CNNDetection \
  DatasetConfig.train_transform_list="[resize_short_256,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_224,to_dtype,normalize_clip]" \
  DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 DatasetConfig.num_workers=6 \
  ++TrainerConfig.epochs=3 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=aide_progan_mgpu 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== AIDE-PROGAN 3GPU DONE (exit ${PIPESTATUS[0]}) $(date) ==="
