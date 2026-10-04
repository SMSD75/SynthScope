#!/bin/bash
#SBATCH --job-name=spai_rt --partition=cees --account=ceesusers
#SBATCH --time=7-00:00:00 --cpus-per-task=14 --mem=48G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/retrain_spai_%j.log
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
SCR=$(for d in /nvmestore /hddstore /ssdstore /sddstore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done); DATA=$SCR/ssalehi/AGID; SC=$SCR/ssalehi/.cache
CONDA=/home/ssalehi/miniconda3/bin/conda; SAVE=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/spai; mkdir -p "$SAVE" "$DATA/spai_train" "$SC"/{huggingface,torch,triton}
export HF_HOME=$SC/huggingface TORCH_HOME=$SC/torch TRITON_CACHE_DIR=$SC/triton OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       SYNTHSCOPE_MFM_CKPT=$DATA/mfm/mfm_pretrain_vit_base.pth
echo "node=$(hostname) scratch=$SCR mfm=$SYNTHSCOPE_MFM_CKPT $(date)"
RS=(rsync -aL --info=progress2 -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
"${RS[@]}" ivi-cn023:/hddstore/ssalehi/AGID/spai_train/dmid_real "$DATA/spai_train/" 2>&1 | tail -1
"${RS[@]}" ivi-cn023:/hddstore/ssalehi/AGID/mfm "$DATA/" 2>&1 | tail -1
[ -d "$DATA/GenImage_full/test" ] || "${RS[@]}" ivi-cn023:/hddstore/ssalehi/AGID/GenImage_full "$DATA/" 2>&1 | tail -1
echo "train reals: $(ls $DATA/spai_train/dmid_real/train/ldm/0_real 2>/dev/null|wc -l) | mfm: $(ls -la $SYNTHSCOPE_MFM_CKPT 2>/dev/null|awk '{print $5}')"
"$CONDA" run --no-capture-output -n SynthScope accelerate launch --config_file accelerate_config.yaml --num_processes 1 --main_process_port 29661 \
  main.py trainer@TrainerConfig=image/spai model@ModelConfig=image/spai dataset@DatasetConfig=image/spai_ldm DatasetConfig.root_dir=$DATA/spai_train/dmid_real \
  DatasetConfig.train_transform_list="[resize_short_448,gaussian_blur,jpeg_compression,random_crop_448,random_horizontal_flip,to_dtype]" \
  DatasetConfig.eval_transform_list="[resize_short_448,center_crop_448,to_dtype]" \
  DatasetConfig.train_batch_size=8 DatasetConfig.val_batch_size=8 DatasetConfig.test_batch_size=8 DatasetConfig.num_workers=10 \
  ++TrainerConfig.epochs=10 ++TrainerConfig.save=true ++TrainerConfig.save_policy=min ++TrainerConfig.save_interval=1 ++TrainerConfig.save_path=$SAVE \
  metrics=[real_accuracy,fake_accuracy,AP,AUC] logger.mode=offline logger.name=spai_retrain_v2 2>&1 | grep --line-buffered -vE "libpng (error|warning)"
echo "=== SPAI RETRAIN-V2 DONE (exit ${PIPESTATUS[0]}) $(date) ==="
