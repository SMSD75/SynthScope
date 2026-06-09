#!/bin/bash
#SBATCH --job-name=aide_rt --partition=cees6000 --account=ceesusers --constraint=6000 --nodelist=ivi-cn031
#SBATCH --time=4-00:00:00 --cpus-per-task=14 --mem=40G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_aide_%j.log
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
SCR=$(for d in /hddstore /ssdstore /sddstore; do [ -d "$d/ssalehi" ] && echo "$d" && break; done); DATA=$SCR/ssalehi/AGID; SC=$SCR/ssalehi/.cache
CONDA=/home/ssalehi/miniconda3/bin/conda; SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/aide; mkdir -p "$SAVE" "$SC"/{huggingface,torch,triton}
export HF_HOME=$SC/huggingface TORCH_HOME=$SC/torch TRITON_CACHE_DIR=$SC/triton OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) scratch=$SCR data=$DATA/GenImage $(date)"
$CONDA run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29631 \
  main.py trainer@TrainerConfig=image/aide model@ModelConfig=image/aide dataset@DatasetConfig=image/genimage DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_256,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_224,to_dtype,normalize_clip]" \
  DatasetConfig.train_batch_size=32 DatasetConfig.val_batch_size=64 DatasetConfig.test_batch_size=64 DatasetConfig.num_workers=8 \
  ++TrainerConfig.epochs=5 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=aide_retrain 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== AIDE RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
