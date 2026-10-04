#!/bin/bash
#SBATCH --job-name=aide_rt_ev --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/aide_rt_ev_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch PYTHONUNBUFFERED=1
H=/hddstore/ssalehi/AGID
CK="${CK:-/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/aide_native_v2/model_2.pth}"
echo "NODE=$(hostname) ckpt=$CK $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
echo "##### GenImage-8 (in-domain SDv1.4 setting) #####"
$CONDA run --no-capture-output -n SynthScope python repro/_aide_retrain_eval.py \
  --data "$H/GenImage_full/test" --ckpt "$CK" --benchmark pergen --limit 300 --bs 16 --tag aide_retrain_native_g8
echo "##### ForenSynths-13 (cross-domain GAN, reference) #####"
$CONDA run --no-capture-output -n SynthScope python repro/_aide_retrain_eval.py \
  --data "$H/CNNDetection/test" --ckpt "$CK" --benchmark forensynths --limit 300 --bs 16 --tag aide_retrain_native_fs
echo "=== AIDE RETRAIN EVAL COMPLETE $(date) ==="
