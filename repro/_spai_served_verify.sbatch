#!/bin/bash
# Verify the SPAI SERVED registry path (native-res, official-remapped spai.pth) reproduces the paper.
# bench_eval --model spai builds via the registry (tf_native_cap + native_res bs=1 + spai.pth=official_synthscope).
#SBATCH --job-name=spai_verify
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/spai_verify_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
CONDA=/home/ssalehi/miniconda3/bin/conda
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/SynthScope/checkpoints
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/.cache_shared/hf PYTHONUNBUFFERED=1
H=/hddstore/ssalehi/SynthScope
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)  spai.pth -> $(readlink -f $SYNTHSCOPE_CKPT_DIR/spai.pth)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
run(){ local tag="$1" data="$2" lim="$3"; [ -d "$data" ] || { echo "[skip] $tag no data $data"; return; }
  echo "===== $tag ====="
  local full=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/${tag}_${SLURM_JOB_ID}.full.log
  $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model spai \
    --benchmark pergen --data "$data" --limit "$lim" --device cuda --tag "$tag" > "$full" 2>&1
  local rc=$?
  grep -aE "RESULT|: acc=|=== bench_eval" "$full" | tail -16
  [ $rc -ne 0 ] && { echo "[FAIL rc=$rc]"; tail -25 "$full"; }; }
# Synthbuster (smaller) first as the headline any-res check, then GenImage-8.
run spai_served_synthbuster "$H/Synthbuster/test"   200
run spai_served_genimage8   "$H/GenImage_full/test" 300
echo "=== SPAI SERVED VERIFY COMPLETE $(date) ==="
