#!/bin/bash
#SBATCH --job-name=ck_eval --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=12:00:00 --cpus-per-task=8 --mem=44G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/eval_ck_%x_%j.log
# Generic eval on cn029: pass MODEL (bench_eval --model), CKDIR (checkpoint dir), TAGP (tag prefix),
# BENCHES (csv of genimage8,forensynths,ufd). Picks latest CKDIR/model_*.pth.
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
: "${MODEL:?}"; : "${CKDIR:?}"; : "${TAGP:?}"; BENCHES="${BENCHES:-genimage8,forensynths,ufd}"
DATA=/ssdstore/ssalehi/SynthScope; CONDA=/home/ssalehi/miniconda3/bin/conda
export SYNTHSCOPE_CKPT_DIR=$DATA/checkpoints SYNTHSCOPE_MFM_CKPT=$DATA/mfm/mfm_pretrain_vit_base.pth \
       HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface TORCH_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/torch \
       OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$SYNTHSCOPE_CKPT_DIR"
RS=(rsync -aL -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
[ -d "$DATA/universialfakedetection" ] || "${RS[@]}" ivi-cn023:/hddstore/ssalehi/SynthScope/universialfakedetection "$DATA/" 2>&1|tail -1
CK=$(ls -1v "$CKDIR"/model_*.pth 2>/dev/null | tail -1); [ -z "$CK" ] && { echo "no ckpt in $CKDIR"; exit 1; }
BS=64; case "$MODEL" in spai) BS=8 ;; aide) BS=32 ;; esac
echo "=== eval $MODEL ckpt=$CK tagp=$TAGP benches=$BENCHES $(date) ==="
for i in $(seq 1 60); do "$CONDA" run -n SynthScope python -c "import torch;assert torch.cuda.is_available();torch.randn(1,device='cuda')" 2>/dev/null && break; sleep 20; done
run(){ echo ">>> $1"; "$CONDA" run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model "$MODEL" --ckpt "$CK" --device cuda "${@:2}" || echo "FAILED $1"; }
for b in ${BENCHES//,/ }; do case "$b" in
  genimage8)   run g8 --benchmark pergen      --data $DATA/GenImage_full/test      --bs $BS --limit 300 --tag ${TAGP}_genimage8 ;;
  forensynths) run fs --benchmark forensynths --data $DATA/CNNDetection/test       --bs $BS             --tag ${TAGP}_forensynths ;;
  ufd)         run ud --benchmark ufd         --data $DATA/universialfakedetection --bs $BS             --tag ${TAGP}_ufd ;;
esac; done
echo "=== EVAL $TAGP DONE $(date) ==="
