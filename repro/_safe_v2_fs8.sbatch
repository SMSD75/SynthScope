#!/bin/bash
# Clean ForenSynths-8 (paper protocol) eval of the retrained SAFE model_38, matching the exact
# 8-generator subset the official checkpoint was scored on (biggan,cyclegan,deepfake,gaugan,
# progan,stargan,stylegan,stylegan2). Produces bench_results/safe_v2_forensynths8.json.
#SBATCH --job-name=safe_v2_fs8
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/safe_v2_fs8_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
CK=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/safe_full_v2/model_38.pth
H=/hddstore/ssalehi/AGID
echo "NODE=$(hostname) CK=$CK $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n AGID python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
full=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/safe_v2_fs8_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py --model safe --ckpt "$CK" \
  --benchmark forensynths --data "$H/CNNDetection/test" \
  --select biggan,cyclegan,deepfake,gaugan,progan,stargan,stylegan,stylegan2 \
  --bs 64 --device cuda --tag safe_v2_forensynths8 > "$full" 2>&1
rc=$?
grep -aE "RESULT|: acc=|=== bench_eval" "$full" | tail -16
[ $rc -ne 0 ] && { echo "[FAIL rc=$rc]"; tail -25 "$full"; }
echo "=== SAFE v2 FS8 EVAL COMPLETE $(date) ==="
