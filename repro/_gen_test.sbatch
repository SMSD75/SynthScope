#!/bin/bash
# Test the isolated local generators (FLUX.2 klein 4B, Qwen-Image-2.1) on one A5000 with a RAM-lean request.
#SBATCH --job-name=gen_test --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=6 --mem=44G --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/gen_test_%j.out
set -u
G=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared
W=${W:-$G/gen_wt}
CONDA=/home/ssalehi/miniconda3/bin/conda
set -a; . /ivi/zfs/s0/original_homes/ssalehi/AGID/.env 2>/dev/null; set +a      # HF_TOKEN
SCR=/ssdstore; [ -w $SCR ] || SCR=/tmp
export HF_HOME=$SCR/ssalehi/AGID/.cache_hf_gen PYTHONUNBUFFERED=1 PYTHONPATH=$G/gen_env/diffusers_main
mkdir -p $HF_HOME
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"; $CONDA run --no-capture-output -n SynthScope python -c "import torch,diffusers;print('CUDA',torch.cuda.get_device_name(0),'| diffusers',diffusers.__version__)"
IMG=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/ideogram_bench/real/001.png
for M in ${MODELS_TO_TEST:-flux2_klein_4b qwen_image_2_1}; do
  echo "================ $M $(date +%T) ================"
  LOCAL_GEN_DEBUG=1 $CONDA run --no-capture-output -n SynthScope python $W/interface/backend/local_gen.py \
     --model $M --image $IMG --prompt "make another image conceptually similar to this" --out $G/gen_test_out/$M.png \
     > $G/gen_test_out/$M.log 2>&1
  grep -aE "saved|\[mem\]|Error|error:|Killed" $G/gen_test_out/$M.log | cut -c1-240 | tail -10
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo "=== GEN TEST DONE $(date) ==="
