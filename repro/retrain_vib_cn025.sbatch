#!/bin/bash
#SBATCH --job-name=vib_rt --partition=cees --account=ceesusers --nodelist=ivi-cn025
#SBATCH --time=2-00:00:00 --cpus-per-task=10 --mem=40G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/retrain_vib_%j.log
# VIB-Net gap-closing retrain with the FAITHFUL FLAVA-full backbone (the paper's headline arch; the old
# SynthScope class used CLIP). This job ALSO stages the shared data onto cn025 scratch (GenImage + cospy_subset)
# and writes a DATA_READY marker that the AIDE/Co-Spy jobs on this node gate on. FLAVA loads from the
# shared /ivi HF cache (already downloaded) so no internet is needed. Saves to /ivi shared.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
SCR=$(for d in /nvmestore /ssdstore /hddstore /sddstore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done)
DATA=$SCR/ssalehi/SynthScope; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/vib_net; mkdir -p "$SAVE" "$DATA/spai_train"
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface
export TORCH_HOME=$SCR/ssalehi/.cache/torch TRITON_CACHE_DIR=$SCR/ssalehi/.cache/triton
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
# --- stage data from cn023 (idempotent) ---
RS=(rsync -aL -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
echo "[stage] GenImage (~91G) -> $DATA ..."; "${RS[@]}" ivi-cn023:/hddstore/ssalehi/SynthScope/GenImage "$DATA/" 2>&1 | tail -1
echo "[stage] cospy_subset ...";            "${RS[@]}" ivi-cn023:/hddstore/ssalehi/SynthScope/spai_train/cospy_subset "$DATA/spai_train/" 2>&1 | tail -1
echo "$SCR" > /ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/DATA_READY_cn025
echo "node=$(hostname) scratch=$SCR data=$DATA/GenImage flava=$HF_HOME genimage_real=$(ls $DATA/GenImage/train/real 2>/dev/null|wc -l) $(date)"
# --- train ---
$CONDA run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29641 \
  main.py trainer@TrainerConfig=image/vib_net model@ModelConfig=image/vib_net dataset@DatasetConfig=image/genimage DatasetConfig.root_dir=$DATA/GenImage \
  DatasetConfig.train_transform_list="[resize_short_256,gaussian_blur,jpeg_compression,random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]" \
  DatasetConfig.eval_transform_list="[resize_short_256,center_crop_224,to_dtype,normalize_clip]" \
  DatasetConfig.train_batch_size=64 DatasetConfig.val_batch_size=128 DatasetConfig.test_batch_size=128 DatasetConfig.num_workers=10 \
  ++TrainerConfig.epochs=10 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=vib_retrain_flava 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== VIB-NET (FLAVA) RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
