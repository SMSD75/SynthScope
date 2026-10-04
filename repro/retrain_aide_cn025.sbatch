#!/bin/bash
#SBATCH --job-name=aide_rt --partition=cees --account=ceesusers --nodelist=ivi-cn025
#SBATCH --time=2-00:00:00 --cpus-per-task=10 --mem=48G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_aide_%j.log
# AIDE gap-closing retrain on GenImage SDv1.4 (faithful ConvNeXt-XXL + 5-view AIDE class). Conservative
# batch for the heavy 5-view forward so an unattended run won't OOM. Saves to /ivi shared.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
SCR=$(for d in /nvmestore /ssdstore /hddstore /sddstore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done)
DATA=$SCR/ssalehi/AGID; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/aide; mkdir -p "$SAVE"
export HF_HOME=$SCR/ssalehi/.cache/huggingface TORCH_HOME=$SCR/ssalehi/.cache/torch TRITON_CACHE_DIR=$SCR/ssalehi/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
for i in $(seq 1 240); do [ -f /ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/DATA_READY_cn025 ] && [ -d "$DATA/GenImage/train/real" ] && break; echo "[wait] data not ready ($i)"; sleep 30; done
echo "node=$(hostname) scratch=$SCR data=$DATA/GenImage $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29651 \
  main.py trainer@TrainerConfig=image/aide model@ModelConfig=image/aide dataset@DatasetConfig=image/genimage DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_256,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_224,to_dtype,normalize_clip]" \
  DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 DatasetConfig.num_workers=10 \
  ++TrainerConfig.epochs=5 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=aide_retrain 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== AIDE RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
