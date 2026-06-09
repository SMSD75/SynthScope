#!/bin/bash
#SBATCH --job-name=spai_ev
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/spai_eval_e3_%j.log
# Eval the data-bug-fixed SPAI retrain (checkpoints_retrain/spai/model_3.pth) on genimage8 (pergen over
# GenImage_full/test). Pinned to cn029 so the GenImage data + MFM backbone on its /ssdstore are local; uses
# a fresh free GPU (cn029 GPU1-7 idle) so it does NOT contend with the SPAI training on GPU0.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
export AGID_MFM_CKPT=/ssdstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth
export AGID_CKPT_DIR=/ssdstore/ssalehi/AGID/checkpoints; mkdir -p "$AGID_CKPT_DIR"
export HF_HOME=/ssdstore/ssalehi/AGID/.cache/hf TORCH_HOME=/ssdstore/ssalehi/AGID/.cache/torch OMP_NUM_THREADS=4
C=/home/ssalehi/miniconda3/bin/conda
CKPT=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/spai/model_3.pth
"$C" run --no-capture-output -n AGID python -c "import torch;assert torch.cuda.is_available();print('CUDA ok',torch.cuda.get_device_name(0))"
"$C" run --no-capture-output -n AGID python benchmark/bench_eval.py \
  --model spai --ckpt "$CKPT" \
  --benchmark pergen --data /ssdstore/ssalehi/AGID/GenImage_full/test \
  --bs 8 --device cuda --limit 300 --tag spai_retrain_e3_genimage8
echo "=== SPAI RETRAIN e3 EVAL DONE (exit $?) $(date) ==="
