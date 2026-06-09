#!/bin/bash
#SBATCH --job-name=pgc_long --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=1-00:00:00 --cpus-per-task=10 --mem=40G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_pgc_long_%j.log
# PGC longer-train on GenImage SD-v1.4 (staged on cn029 /ssdstore). The promoted pgc.pth was only 1 epoch
# (97.62; OOD gap on adm/vqdm). Paper uses ~100 epochs; train 25 here, keep best-AP. dinov2-large backbone.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
DATA=/ssdstore/ssalehi/AGID; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_genimage_long; mkdir -p "$SAVE"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface
export TORCH_HOME=$DATA/.cache/torch TRITON_CACHE_DIR=$DATA/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
[ -d "$DATA/GenImage/train/real" ] || { echo "GenImage not staged on $(hostname)!"; exit 1; }
echo "node=$(hostname) data=$DATA/GenImage real=$(ls $DATA/GenImage/train/real 2>/dev/null|wc -l) $(date)"
$CONDA run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29663 \
  main.py trainer@TrainerConfig=image/pgc model@ModelConfig=image/pgc ModelConfig.backbone_name=facebook/dinov2-large \
  dataset@DatasetConfig=image/genimage DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_256,random_crop_224,random_horizontal_flip,to_dtype,normalize_dinov2]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_224,to_dtype,normalize_dinov2]" \
  DatasetConfig.train_batch_size=32 DatasetConfig.val_batch_size=64 DatasetConfig.test_batch_size=64 DatasetConfig.num_workers=10 \
  ++TrainerConfig.epochs=25 ++TrainerConfig.save=true ++TrainerConfig.save_policy=max ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=pgc_long 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== PGC LONG RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
