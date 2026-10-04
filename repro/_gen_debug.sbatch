#!/bin/bash
# Isolate why Qwen-Image-2.1 returned a degenerate image: T2I vs i2i, VAE tiling on/off, lower resolution.
#SBATCH --job-name=gen_dbg --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=6 --mem=28G --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/gen_dbg_%j.out
set -u
G=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared; W=$G/gen_wt; CONDA=/home/ssalehi/miniconda3/bin/conda
set -a; . /ivi/zfs/s0/original_homes/ssalehi/AGID/.env 2>/dev/null; set +a
export HF_HOME=/ssdstore/ssalehi/AGID/.cache_hf_gen PYTHONUNBUFFERED=1 PYTHONPATH=$G/gen_env/diffusers_main LOCAL_GEN_DEBUG=1
IMG=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/ideogram_bench/real/001.png
P="make another image conceptually similar to this"
run(){ name=$1; shift; echo "== $name $(date +%T)"
  $CONDA run --no-capture-output -n AGID python $W/interface/backend/local_gen.py --model qwen_image_2_1 --image $IMG --out $G/gen_test_out/qwen_$name.png "$@" > $G/gen_test_out/qwen_$name.log 2>&1
  grep -aE "saved|Error|OutOfMemory" $G/gen_test_out/qwen_$name.log | cut -c1-200 | tail -3; }
run t2i          --prompt "a photograph of a black vulture perched on a bare tree branch, winter, natural light" --text-only
run i2i_notile   --prompt "$P" --no-tiling --res 768
run i2i_tile768  --prompt "$P" --res 768
run i2i_edit     --prompt "turn this photo into a watercolor painting" --res 768
echo "=== GEN DEBUG DONE $(date) ==="
