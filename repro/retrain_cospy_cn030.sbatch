#!/bin/bash
#SBATCH --job-name=cospy_rt --partition=cees6000 --account=ceesusers --constraint=6000 --nodelist=ivi-cn030
#SBATCH --time=4-00:00:00 --cpus-per-task=14 --mem=32G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_cospy_%j.log
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
SCR=$(for d in /hddstore /ssdstore /sddstore; do [ -d "$d/ssalehi" ] && echo "$d" && break; done); DATA=$SCR/ssalehi/AGID; SC=$SCR/ssalehi/.cache
CONDA=/home/ssalehi/miniconda3/bin/conda; SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/co_spy; mkdir -p "$SAVE" "$SC"/{huggingface,torch,triton}
export HF_HOME=$SC/huggingface TORCH_HOME=$SC/torch TRITON_CACHE_DIR=$SC/triton OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) scratch=$SCR data=$DATA/spai_train/cospy_subset $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29621 \
  main.py trainer@TrainerConfig=image/co_spy model@ModelConfig=image/co_spy dataset@DatasetConfig=image/cospy_subset DatasetConfig.root_dir=$DATA/spai_train/cospy_subset \
  DatasetConfig.train_transform_list="[resize_short_384,gaussian_blur,jpeg_compression,random_crop_384,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_384,center_crop_384,to_dtype]" \
  DatasetConfig.train_batch_size=32 DatasetConfig.val_batch_size=64 DatasetConfig.test_batch_size=64 DatasetConfig.num_workers=8 \
  ++TrainerConfig.epochs=8 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=cospy_retrain 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== COSPY RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
