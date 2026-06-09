#!/bin/bash
#SBATCH --job-name=spai_full --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=1-00:00:00 --cpus-per-task=24 --mem=120G --gres=gpu:4
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_spai_full_%j.log
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
DATA=/ssdstore/ssalehi/AGID; SC=/ssdstore/ssalehi/.cache; CONDA=/home/ssalehi/miniconda3/bin/conda
SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/spai_full; mkdir -p "$SAVE" "$SC"/{huggingface,torch,triton} "$DATA/spai_train"
export HF_HOME=$SC/huggingface TORCH_HOME=$SC/torch TRITON_CACHE_DIR=$SC/triton OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AGID_MFM_CKPT=$DATA/mfm/mfm_pretrain_vit_base.pth
RS=(rsync -aL -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
[ -d "$DATA/spai_train/dmid_real/train" ] || "${RS[@]}" ivi-cn023:/hddstore/ssalehi/AGID/spai_train/dmid_real "$DATA/spai_train/" 2>&1 | tail -1
[ -f "$AGID_MFM_CKPT" ] || "${RS[@]}" ivi-cn023:/hddstore/ssalehi/AGID/mfm "$DATA/" 2>&1 | tail -1
echo "node=$(hostname) reals=$(ls $DATA/spai_train/dmid_real/train/ldm/0_real 2>/dev/null|wc -l) $(date)"
$CONDA run --no-capture-output -n AGID accelerate launch --config_file accelerate_config.yaml --num_processes 4 --main_process_port 29672 \
  main.py trainer@TrainerConfig=image/spai model@ModelConfig=image/spai dataset@DatasetConfig=image/spai_ldm DatasetConfig.root_dir=$DATA/spai_train/dmid_real \
  DatasetConfig.train_transform_list="[resize_short_448,gaussian_blur,jpeg_compression,random_crop_448,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_448,center_crop_448,to_dtype]" \
  DatasetConfig.train_batch_size=8 DatasetConfig.val_batch_size=8 DatasetConfig.test_batch_size=8 DatasetConfig.num_workers=12 \
  ++TrainerConfig.epochs=8 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=spai_full 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== SPAI FULL RETRAIN DONE (exit ${PIPESTATUS[0]}) $(date) ==="
