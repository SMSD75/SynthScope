#!/bin/bash
#SBATCH --job-name=vib_rt --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=2-00:00:00 --cpus-per-task=10 --mem=40G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_vib_%j.log
# VIB-Net FLAVA retrain on cn029 (1.3T free /ssdstore; cn025 was full). Stages GenImage train, writes
# DATA_READY_cn029 for the AIDE-GenImage job, trains frozen-FLAVA + VIB head. FLAVA from shared /ivi cache.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
DATA=/ssdstore/ssalehi/AGID; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/vib_net; mkdir -p "$SAVE" "$DATA"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface
export TORCH_HOME=$DATA/.cache/torch TRITON_CACHE_DIR=$DATA/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
df -h /ssdstore | tail -1
RS=(rsync -aL -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
[ -d "$DATA/GenImage/train/real" ] || { echo "[stage] GenImage (~91G)"; "${RS[@]}" ivi-cn023:/hddstore/ssalehi/AGID/GenImage "$DATA/" 2>&1 | tail -1; }
echo "ssdstore" > /ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/DATA_READY_cn029
echo "node=$(hostname) data=$DATA/GenImage real=$(ls $DATA/GenImage/train/real 2>/dev/null|wc -l) $(date)"
$CONDA run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29642 \
  main.py trainer@TrainerConfig=image/vib_net model@ModelConfig=image/vib_net dataset@DatasetConfig=image/genimage DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_224,to_dtype,normalize_clip]" \
  DatasetConfig.train_batch_size=64 DatasetConfig.val_batch_size=128 DatasetConfig.test_batch_size=128 DatasetConfig.num_workers=10 \
  ++TrainerConfig.epochs=10 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=vib_retrain_flava 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== VIB-NET (FLAVA) RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
