#!/bin/bash
# Deploy + TEST the official PGC and SAFE weights as the default inference checkpoints on cn023.
# Tests them on ForenSynths first (PGC official ProGAN ckpt should reach ~96 mAP / hard-3 ~94 AUROC;
# SAFE official ~98), backs up the existing SynthScope-trained pgc.pth/safe.pth, then swaps in the official.
#SBATCH --job-name=deploy_off
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/deploy_official_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
CKDIR=/hddstore/ssalehi/AGID/checkpoints
SHARED=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval
TEST=/hddstore/ssalehi/AGID/CNNDetection/test
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"

# --- CUDA precheck (cn023 old kernel flakily fails CUDA init) ---
ok=0
for i in 1 2 3 4 5; do
  $CONDA run --no-capture-output -n SynthScope python -c \
    "import torch; assert torch.cuda.is_available(); (torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum().item(); print('CUDA OK')" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = 1 ] || { echo "CUDA NEVER CAME UP"; exit 1; }

# --- locate official checkpoints ---
PGC_OFF="$SHARED/PGC_train_progan_ckpt.pth"
[ -f "$PGC_OFF" ] || { echo "PGC official ckpt missing: $PGC_OFF"; exit 1; }
SAFE_OFF=""
for c in /hddstore/ssalehi/AGID/repro/official_ckpts/safe/checkpoint-best.pth \
         /hddstore/ssalehi/AGID/official_ckpts/safe/checkpoint-best.pth \
         /ivi/zfs/s0/original_homes/ssalehi/AGID/repro/official_ckpts/safe/checkpoint-best.pth; do
  [ -f "$c" ] && SAFE_OFF="$c" && break
done
echo "[ckpt] PGC_OFF=$PGC_OFF"
echo "[ckpt] SAFE_OFF=${SAFE_OFF:-NOT FOUND}"
echo "[data] TEST=$TEST  exists=$( [ -d $TEST ] && echo yes || echo NO )"

run_eval () {  # model  ckpt  tag
  echo "[eval] $1 <- $2"
  $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
    --model "$1" --benchmark forensynths --data "$TEST" --bs 32 --ckpt "$2" --tag "$3" 2>&1 | tail -25
}

echo "===== TEST official PGC on ForenSynths (expect ~96 mAP, hard-3 ~94 AUROC) ====="
run_eval pgc "$PGC_OFF" pgc_official_deploycheck
if [ -n "$SAFE_OFF" ]; then
  echo "===== TEST official SAFE on ForenSynths (expect ~98 mAP) ====="
  run_eval safe "$SAFE_OFF" safe_official_deploycheck
fi

# --- DEPLOY: back up existing SynthScope-trained ckpts, then swap in official ---
mkdir -p "$CKDIR"
ts=$(date +%Y%m%d)
for f in pgc safe; do
  if [ -f "$CKDIR/$f.pth" ] && [ ! -e "$CKDIR/$f.synthscopetrained.$ts.bak" ]; then
    cp -a "$CKDIR/$f.pth" "$CKDIR/$f.synthscopetrained.$ts.bak" && echo "[backup] $f.pth -> $f.synthscopetrained.$ts.bak"
  fi
done
cp -f "$PGC_OFF" "$CKDIR/pgc.pth" && echo "[deploy] pgc.pth <- official ProGAN ckpt"
[ -n "$SAFE_OFF" ] && cp -f "$SAFE_OFF" "$CKDIR/safe.pth" && echo "[deploy] safe.pth <- official SAFE ckpt"
echo "=== DEPLOY+TEST COMPLETE $(date) ==="
