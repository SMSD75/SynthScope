#!/bin/bash
#SBATCH --job-name=aideG_rt --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=1-12:00:00 --cpus-per-task=24 --mem=90G --gres=gpu:3
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_aideG_%j.log
# AIDE GenImage gap-closer, 3-GPU DeepSpeed (per-GPU bs16 -> eff 48). Closes AIDE's GenImage table
# (SynthScope 55 OOD -> official ~95). Saves to checkpoints_retrain/aide_genimage (distinct from ProGAN run).
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
DATA=/ssdstore/ssalehi/AGID; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/aide_genimage; mkdir -p "$SAVE"
export HF_HOME=$DATA/.cache/huggingface TORCH_HOME=$DATA/.cache/torch TRITON_CACHE_DIR=$DATA/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
for i in $(seq 1 240); do [ -f repro/eval_logs/DATA_READY_cn029 ] && [ -d "$DATA/GenImage/train/real" ] && break; echo "[wait] data ($i)"; sleep 30; done
echo "node=$(hostname) gpus=3 data=$DATA/GenImage real=$(ls $DATA/GenImage/train/real|wc -l) $(date)"
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 3 --main_process_port 29653 \
  main.py trainer@TrainerConfig=image/aide model@ModelConfig=image/aide dataset@DatasetConfig=image/genimage DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_256,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_224,to_dtype,normalize_clip]" \
  DatasetConfig.train_batch_size=16 DatasetConfig.val_batch_size=32 DatasetConfig.test_batch_size=32 DatasetConfig.num_workers=6 \
  ++TrainerConfig.epochs=5 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=aide_genimage_mgpu 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== AIDE-GENIMAGE 3GPU DONE (exit ${PIPESTATUS[0]}) $(date) ==="
