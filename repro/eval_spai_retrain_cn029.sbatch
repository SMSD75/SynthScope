#!/bin/bash
#SBATCH --job-name=spai_ev
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/spai_eval_e3_%j.log
# Eval the data-bug-fixed SPAI retrain (checkpoints_retrain/spai/model_3.pth) on genimage8 (pergen over
# GenImage_full/test). Pinned to cn029 so the GenImage data + MFM backbone on its /ssdstore are local; uses
# a fresh free GPU (cn029 GPU1-7 idle) so it does NOT contend with the SPAI training on GPU0.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
export SYNTHSCOPE_MFM_CKPT=/ssdstore/ssalehi/SynthScope/mfm/mfm_pretrain_vit_base.pth
export SYNTHSCOPE_CKPT_DIR=/ssdstore/ssalehi/SynthScope/checkpoints; mkdir -p "$SYNTHSCOPE_CKPT_DIR"
export HF_HOME=/ssdstore/ssalehi/SynthScope/.cache/hf TORCH_HOME=/ssdstore/ssalehi/SynthScope/.cache/torch OMP_NUM_THREADS=4
C=/home/ssalehi/miniconda3/bin/conda
CKPT=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/spai/model_3.pth
"$C" run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA ok',torch.cuda.get_device_name(0))"
"$C" run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
  --model spai --ckpt "$CKPT" \
  --benchmark pergen --data /ssdstore/ssalehi/SynthScope/GenImage_full/test \
  --bs 8 --device cuda --limit 300 --tag spai_retrain_e3_genimage8
echo "=== SPAI RETRAIN e3 EVAL DONE (exit $?) $(date) ==="
