#!/bin/bash
# PGC authors' guidance (issue reply): the released ProGAN checkpoint is ~400 steps (<1 epoch);
# training longer overfits ProGAN and destroys hard-generator transfer. Re-run their OFFICIAL
# train.py VERBATIM but with a FINE eval/save cadence (every 100 micro-steps) so we capture the
# early peak. Keep --niter 100 so the LR-at-step matches their schedule. Their own evaluator prints
# the hard-3 (deepfake/san/seeingdark) mean ACC/AP at each step into the log = the trajectory.
# cn029 (cees): ProGAN train data is local on /ssdstore (no 72G staging).
#SBATCH --job-name=pgc_400 --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=08:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/pgc_400step_%j.out
set -u
AGID=/ivi/zfs/s0/original_homes/ssalehi/AGID
SHARED=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval
DEST=$AGID/checkpoints_retrain/pgc_400step; mkdir -p "$DEST"
CONDA=/home/ssalehi/miniconda3/bin/conda
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n AGID python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }

DATA=/ssdstore/ssalehi/AGID/CNNDetection
[ -d "$DATA/train" ] || { echo "ProGAN train NOT on cn029 /ssdstore ($DATA/train) — re-stage needed"; exit 1; }
echo "[data] train classes=$(ls $DATA/train | wc -l)"
SCR=/ssdstore/ssalehi/pgc_400_${SLURM_JOB_ID}; mkdir -p "$SCR"
# README UniversalFakeDetect layout (single 0_real/1_fake roots) via hardlinks (instant, same FS)
if [ ! -d "$SCR/ufd/0_real" ]; then
  mkdir -p "$SCR/ufd/0_real" "$SCR/ufd/1_fake"
  for c in "$DATA/train"/*/; do cls=$(basename "$c"); cp -al "$c/0_real" "$SCR/ufd/0_real/$cls"; cp -al "$c/1_fake" "$SCR/ufd/1_fake/$cls"; done
fi
echo "[data] ufd real=$(find $SCR/ufd/0_real -type f|wc -l) fake=$(find $SCR/ufd/1_fake -type f|wc -l)"

# clone pristine official repo @0f3e97d (no network)
git clone -q "$SHARED/repo" "$SCR/repo" && cd "$SCR/repo" && git checkout -q 0f3e97d
echo "[repo] $(git log --oneline -1)"
export HF_HOME=$AGID/.cache_shared/hf PYTHONUNBUFFERED=1
mkdir -p "$SCR/ckpts"

# archive every early step ckpt (step<=3000) to shared FS so they survive + can be evaluated
archiver() { local ck="$SCR/ckpts/pgc_train_progan"; while true; do sleep 120; [ -d "$ck" ] || continue
  cp -un "$ck"/best.pth "$DEST/" 2>/dev/null
  for f in "$ck"/step*.pth; do [ -e "$f" ] || continue; s=$(basename "$f"|grep -oP 'step\K[0-9]+'); [ "${s:-99999}" -le 3000 ] && cp -un "$f" "$DEST/" 2>/dev/null; done; done; }
archiver & ARCH=$!

# their train.py VERBATIM (README 3.2 args), eval/save every 100 micro-steps; niter=100 keeps their LR schedule
$CONDA run --no-capture-output -n AGID python train.py \
  --name pgc_train_progan --checkpoints_dir "$SCR/ckpts" --devices 0 \
  --batch_size 32 --dino_variant dinov2-large --dino_pretrained_root "$SHARED/pretrained_dino" \
  --lora_rank 8 --lora_alpha 1.0 --lora_dropout 0.1 \
  --real_image_dir "$SCR/ufd/0_real" --fake_image_dir "$SCR/ufd/1_fake" --test_root "$SHARED/test" \
  --cropSize 224 --niter 100 --lr 5e-5 --optim adam \
  --accumulation_steps 4 --weight_decay 0.05 --label_smoothing 0.1 --tau_rgb 0.5 --tau_res 0.5 \
  --eval_batch_size 32 --eval_num_threads 8 --num_threads 8 --eval_every_steps 100 2>&1 | tee "$SCR/train.log"
kill $ARCH 2>/dev/null
for f in "$SCR/ckpts/pgc_train_progan"/step*.pth; do [ -e "$f" ] || continue; s=$(basename "$f"|grep -oP 'step\K[0-9]+'); [ "${s:-99999}" -le 3000 ] && cp -un "$f" "$DEST/"; done
echo "=== PGC 400-STEP FINEGRAIN DONE $(date) ==="
