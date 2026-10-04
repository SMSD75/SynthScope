#!/bin/bash
#SBATCH --job-name=cospy_full --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=1-00:00:00 --cpus-per-task=24 --mem=120G --gres=gpu:4
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_cospy_full_%j.log
# Co-Spy retrain on the FULL GenImage SD-v1.4 (162k, vs the 12k subset that already hit CoSpyBench 95.88)
# to push GenImage-8 88.85 -> ~92. 4-GPU DeepSpeed ZeRO-2, raw-384 (model normalizes internally).
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
DATA=/ssdstore/ssalehi/AGID; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/cospy_full; mkdir -p "$SAVE"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface
export TORCH_HOME=$DATA/.cache/torch TRITON_CACHE_DIR=$DATA/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
[ -d "$DATA/GenImage/train/real" ] || { echo "GenImage not staged on $(hostname)"; exit 1; }
echo "node=$(hostname) data=$DATA/GenImage real=$(ls $DATA/GenImage/train/real|wc -l) $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml \
  --num_processes 4 --main_process_port 29671 \
  main.py trainer@TrainerConfig=image/co_spy model@ModelConfig=image/co_spy dataset@DatasetConfig=image/genimage \
  DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_384,gaussian_blur,jpeg_compression,random_crop_384,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_384,center_crop_384,to_dtype]" \
  DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 DatasetConfig.num_workers=12 \
  ++TrainerConfig.epochs=4 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=cospy_full \
  2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== COSPY FULL RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
