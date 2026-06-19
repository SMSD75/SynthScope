#!/bin/bash
#SBATCH --job-name=pgc_evfix29 --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_evfix29_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf PYTHONUNBUFFERED=1
H=/ssdstore/ssalehi/AGID                       # CNNDetection test data is node-local on cn029
OFFICIAL=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval/PGC_train_progan_ckpt.pth
STEP900=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_400step/step900_acc66_ap94.pth
echo "NODE=$(hostname) $(date)"
ls -la "$H/CNNDetection/test" | head -3
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n AGID python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
run(){ local tag="$1"; shift; local full=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/${tag}_${SLURM_JOB_ID}.full.log
  echo "===== $tag ====="
  $CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py --model pgc \
    --benchmark forensynths --data "$H/CNNDetection/test" --bs 32 --device cuda --tag "$tag" "$@" > "$full" 2>&1
  grep -aE "RESULT|: acc=|=== bench_eval|mAP|mean" "$full" | tail -24; }
# served official weights (the registry default) with the FIXED tf_crop(224) transform
run pgc_evalfix_official --ckpt "$OFFICIAL"
# our from-scratch step900 reproduction with the same fixed transform
run pgc_evalfix_step900 --ckpt "$STEP900"
echo "=== PGC EVAL-FIX FORENSYNTHS (cn029) DONE $(date) ==="
