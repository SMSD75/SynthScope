#!/bin/bash
# Headline eval: the reproduced PGC step-900 checkpoint (early-stop, hard-3 AUROC 94.56) on the FULL
# ForenSynths test set, vs official 96.25 mAP. cn023 (CNNDetection/test is local there).
#SBATCH --job-name=pgc400_fs --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc400_fs_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf PYTHONUNBUFFERED=1
H=/hddstore/ssalehi/AGID
CK=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_400step/step900_acc66_ap94.pth
echo "NODE=$(hostname) ckpt=$CK $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
full=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc400_fs_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model pgc --ckpt "$CK" \
  --benchmark forensynths --data "$H/CNNDetection/test" --bs 32 --device cuda --tag pgc_400step_forensynths > "$full" 2>&1
rc=$?
grep -aE "RESULT|: acc=|=== bench_eval" "$full" | tail -18
[ $rc -ne 0 ] && { echo "[FAIL rc=$rc]"; tail -25 "$full"; }
echo "=== PGC 400-STEP FORENSYNTHS EVAL DONE $(date) ==="
