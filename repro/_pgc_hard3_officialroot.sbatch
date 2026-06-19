#!/bin/bash
#SBATCH --job-name=pgc_h3root --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=00:40:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_h3root_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/AGID/.cache_shared/hf PYTHONUNBUFFERED=1
ROOT=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval/test   # official's OWN hard-3 test root
OFFICIAL=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval/PGC_train_progan_ckpt.pth
echo "NODE=$(hostname) $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n AGID python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
# bench_eval (AGID harness, fixed tf_crop(224) transform) on the EXACT images the official evaluator used.
# Official-evaluator targets on this same root: deepfake 96.48 / san 99.06 / seeingdark 87.49 AUROC.
$CONDA run --no-capture-output -n AGID python benchmark/bench_eval.py --model pgc \
  --benchmark forensynths --data "$ROOT" --bs 32 --device cuda --tag pgc_h3_officialroot \
  --ckpt "$OFFICIAL" 2>&1 | grep -aE ": acc=|RESULT|=== bench_eval|real \+|fake" | grep -av "Loading weights"
echo "=== PGC HARD3 ON OFFICIAL ROOT DONE $(date) ==="
