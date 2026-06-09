#!/bin/bash
#SBATCH --job-name=safe_ev --partition=cees --account=ceesusers --time=7-00:00:00 --cpus-per-task=8 --mem=24G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/eval_safe_retrain_%j.log
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/AGID
SCR=$(for d in /nvmestore /hddstore /ssdstore /sddstore; do [ -d "$d/ssalehi/AGID" ] && echo "$d" && break; done); DATA=$SCR/ssalehi/AGID; SC=$SCR/ssalehi/.cache
CONDA=/home/ssalehi/miniconda3/bin/conda; CK=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/safe/model_11.pth
export AGID_CKPT_DIR=$DATA/checkpoints HF_HOME=$SC/huggingface TORCH_HOME=$SC/torch OMP_NUM_THREADS=4
echo "node=$(hostname) data=$DATA ckpt=$CK"
$CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py --model safe --ckpt $CK --benchmark forensynths --data $DATA/CNNDetection/test --bs 64 --device cuda --tag safe_retrain_forensynths
$CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py --model safe --ckpt $CK --benchmark pergen --data $DATA/GenImage_full/test --bs 64 --device cuda --tag safe_retrain_genimage8
echo "=== SAFE EVAL DONE $(date) ==="
