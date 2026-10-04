#!/bin/bash
#SBATCH --job-name=aide_nat --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=2-00:00:00 --cpus-per-task=24 --mem=200G --gres=gpu:4
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/retrain_aide_native_%j.log
# AIDE Phase 2: from-scratch retrain with the FAITHFUL native-resolution 5-view preprocessing
# (the genuine arch; DCT patch selection at native res via the aide_five_view transform, NOT a
# resize-256-before-DCT). Proves the repo can regenerate the AIDE checkpoint. 4-GPU on GenImage
# SD-v1.4 (already staged on cn029 /ssdstore). convnext-xxl trunk is frozen (laion2b soup).
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
DATA=/ssdstore/ssalehi/SynthScope; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/aide_native_v2; mkdir -p "$SAVE"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface   # shared convnext (4.8G) cache
export TORCH_HOME=$DATA/.cache/torch TRITON_CACHE_DIR=$DATA/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
[ -d "$DATA/GenImage/train/real" ] || { echo "GenImage not staged on $(hostname)"; exit 1; }
echo "node=$(hostname) data=$DATA/GenImage real=$(ls $DATA/GenImage/train/real|wc -l) $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml \
  --num_processes 4 --main_process_port 29673 \
  main.py trainer@TrainerConfig=image/aide model@ModelConfig=image/aide dataset@DatasetConfig=image/genimage \
  DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[random_horizontal_flip,aide_five_view_train]" \
  DatasetConfig.eval_transform_list="[aide_five_view_train]" \
  DatasetConfig.train_batch_size=8 DatasetConfig.val_batch_size=16 DatasetConfig.test_batch_size=16 DatasetConfig.num_workers=5 \
  ++TrainerConfig.epochs=10 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=aide_native_v2 \
  2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== AIDE NATIVE-RES RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
