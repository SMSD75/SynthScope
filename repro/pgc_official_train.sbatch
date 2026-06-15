#!/bin/bash
# GOLD CONTROL: run the OFFICIAL PGC repo's train.py VERBATIM (xiaoyu6868/PGC @0f3e97d) on the same
# 720k ProGAN data, to test whether their released checkpoint is reproducible from their own code.
# Single GPU (their script is single-device, fp32, bs32 + accum 4 = eff 128). On a 24GB card bs32
# fp32 may OOM -> auto-fallback to bs16 + accum 8 (same eff batch; their LR schedule becomes even
# more constant: 1/8 cosine instead of 1/4 -- final LR 4.81e-5 vs 4.28e-5, both ~constant).
# Their train.py only saves checkpoints at --eval_every_steps (default 200 = impractical); we set it
# to every 2 epochs with the hard-3 test root, so their OWN evaluator prints deepfake/san/seeingdark
# every 2 epochs into this log = the trajectory signal.
#SBATCH --job-name=pgc_official
#SBATCH --partition=all
#SBATCH --account=linuxusers
#SBATCH --gres=gpu:1
#SBATCH --exclude=ivi-cn001,ivi-cn003,ivi-cn005,ivi-cn006
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=6-23:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_official_train_%j.out
set -u
AGID=/ivi/zfs/s0/original_homes/ssalehi/AGID
SHARED=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval
DEST=$AGID/checkpoints_retrain/pgc_official_script        # shared-FS archive of milestone ckpts
CONDA=/home/ssalehi/miniconda3/bin/conda
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"

# --- CUDA precheck ---
ok=0
for i in 1 2 3 4 5; do
  $CONDA run --no-capture-output -n AGID python -c \
    "import torch; assert torch.cuda.is_available(); (torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum().item(); print('CUDA OK')" && { ok=1; break; }
  echo "CUDA init failed ($i)"; sleep 5
done
[ "$ok" = 1 ] || { echo "CUDA NEVER CAME UP"; exit 1; }

# --- local scratch with >=110GB free ---
SCR=""
for d in /ssdstore /nvmestore /hddstore /scratch /local /tmp; do
  [ -d "$d" ] && [ -w "$d" ] || continue
  free_gb=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ "${free_gb:-0}" -ge 110 ] && SCR="$d/ssalehi/pgc_official_${SLURM_JOB_ID}" && break
done
[ -n "$SCR" ] || { echo "NO SCRATCH WITH 110GB FREE"; df -h /ssdstore /scratch /local /tmp 2>/dev/null; exit 1; }
mkdir -p "$SCR"; echo "[scratch] $SCR"

# --- official repo: clone the pristine shared copy (no network needed), pin 0f3e97d ---
git clone -q "$SHARED/repo" "$SCR/repo" && cd "$SCR/repo" && git checkout -q 0f3e97d
echo "[repo] $(git log --oneline -1)"

# --- stage 720k ProGAN train data from cn029 (we hold 185056 there -> ssh allowed) ---
if [ ! -d "$SCR/data/train" ]; then
  echo "[data] tar-streaming 72G train from cn029  $(date)"
  mkdir -p "$SCR/data"; T0=$(date +%s)
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn029 \
    'tar -C /ssdstore/ssalehi/AGID/CNNDetection -cf - train' | tar -C "$SCR/data" -xf - \
    || { echo "STREAM FAILED"; exit 1; }
  echo "[data] staged in $((($(date +%s)-T0)/60)) min"
fi
# build the README's UniversalFakeDetect layout (single 0_real/1_fake roots) via hardlinks (instant)
if [ ! -d "$SCR/data/ufd/0_real" ]; then
  mkdir -p "$SCR/data/ufd/0_real" "$SCR/data/ufd/1_fake"
  for c in "$SCR/data/train"/*/; do
    cls=$(basename "$c")
    cp -al "$c/0_real" "$SCR/data/ufd/0_real/$cls"
    cp -al "$c/1_fake" "$SCR/data/ufd/1_fake/$cls"
  done
fi
echo "[data] ufd layout: real=$(find $SCR/data/ufd/0_real -type f | wc -l) fake=$(find $SCR/data/ufd/1_fake -type f | wc -l)"

mkdir -p "$DEST" "$SCR/ckpts"
export HF_HOME=$AGID/.cache_shared/hf
export PYTHONUNBUFFERED=1

# --- background pruner/archiver: their script saves step{N}_acc_ap.pth every eval; keep best.pth +
#     every 10th-epoch ckpt (archived to shared FS) + the 2 newest locally; delete the rest. ---
pruner() {
  local keepmod=$1 ckdir="$SCR/ckpts/pgc_train_progan"
  while true; do
    sleep 600
    [ -d "$ckdir" ] || continue
    cp -un "$ckdir"/best.pth "$DEST/" 2>/dev/null
    ls -t "$ckdir"/step*.pth 2>/dev/null | tail -n +3 | while read -r f; do
      step=$(basename "$f" | grep -oP 'step\K[0-9]+')
      if [ $((step % keepmod)) -eq 0 ]; then cp -un "$f" "$DEST/" 2>/dev/null; else rm -f "$f"; fi
    done
  done
}

# --- run their train.py VERBATIM (README 3.2 args; only paths + eval cadence set) ---
run_train() {
  local bs=$1 acc=$2 ees=$3
  echo "[train] OFFICIAL train.py  bs=$bs accum=$acc eval_every=$ees  $(date)"
  cd "$SCR/repo"
  $CONDA run --no-capture-output -n AGID python train.py \
    --name pgc_train_progan \
    --checkpoints_dir "$SCR/ckpts" \
    --devices 0 \
    --batch_size $bs \
    --dino_variant dinov2-large \
    --dino_pretrained_root "$SHARED/pretrained_dino" \
    --lora_rank 8 --lora_alpha 1.0 --lora_dropout 0.1 \
    --real_image_dir "$SCR/data/ufd/0_real" \
    --fake_image_dir "$SCR/data/ufd/1_fake" \
    --test_root "$SHARED/test" \
    --cropSize 224 --niter 100 --lr 5e-5 --optim adam \
    --accumulation_steps $acc --weight_decay 0.05 --label_smoothing 0.1 \
    --tau_rgb 0.5 --tau_res 0.5 \
    --eval_batch_size 32 --eval_num_threads 8 --num_threads 8 \
    --eval_every_steps $ees
}

# bs32/accum4 (verbatim README): micro/epoch=22503, eval every 2 ep = 45006 -> keep every 10 ep = 225030
pruner 225030 & PRUNER=$!
run_train 32 4 45006 2>&1 | tee "$SCR/train_bs32.log"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ] && grep -qa "CUDA out of memory" "$SCR/train_bs32.log"; then
  echo "[train] bs32 OOM on this card -> fallback bs16/accum8 (same eff batch 128)"
  kill $PRUNER 2>/dev/null
  # bs16: micro/epoch=45007, eval every 2 ep = 90014 -> keep every 10 ep = 450070
  pruner 450070 & PRUNER=$!
  run_train 16 8 90014
  rc=$?
fi
kill $PRUNER 2>/dev/null
# final archive sweep
cp -un "$SCR/ckpts/pgc_train_progan"/best.pth "$DEST/" 2>/dev/null
ls "$SCR/ckpts/pgc_train_progan"/step*.pth 2>/dev/null | while read -r f; do cp -un "$f" "$DEST/"; done
echo "=== PGC OFFICIAL-SCRIPT TRAIN DONE rc=$rc $(date) ==="
