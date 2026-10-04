#!/bin/bash
#SBATCH --job-name=rt_eval --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=1-00:00:00 --cpus-per-task=8 --mem=44G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/eval_rt_%x_%j.log
# Eval a gap-closing retrained checkpoint on cn029 (free GPUs; GenImage_full + CNNDetection already on its
# /ssdstore). Self-stages UFD (small) and, for co_spy, CoSpyBench if present. MODEL via --export.
# Picks the latest checkpoints_retrain/<model>/model_*.pth (on shared /ivi). Appends to MASTER_RESULTS.jsonl.
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/AGID
: "${MODEL:?pass MODEL=vib_net|aide|co_spy|spai}"
DATA=/ssdstore/ssalehi/AGID; CONDA=/home/ssalehi/miniconda3/bin/conda
export SYNTHSCOPE_CKPT_DIR=$DATA/checkpoints SYNTHSCOPE_MFM_CKPT=$DATA/mfm/mfm_pretrain_vit_base.pth \
       HF_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/huggingface \
       TORCH_HOME=/ivi/zfs/s0/original_homes/ssalehi/.cache/torch OMP_NUM_THREADS=4 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$SYNTHSCOPE_CKPT_DIR"
RS=(rsync -aL -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
[ -d "$DATA/universialfakedetection" ] || { echo "[stage] ufd"; "${RS[@]}" ivi-cn023:/hddstore/ssalehi/AGID/universialfakedetection "$DATA/" 2>&1|tail -1; }
CKDIR=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/$MODEL
CK=$(ls -1v "$CKDIR"/model_*.pth 2>/dev/null | tail -1)
[ -z "$CK" ] && { echo "no checkpoint in $CKDIR"; exit 1; }
BS=64; case "$MODEL" in spai) BS=8 ;; aide) BS=32 ;; esac
echo "=== eval retrained $MODEL ckpt=$CK bs=$BS node=$(hostname) $(date) ==="
for i in $(seq 1 60); do "$CONDA" run -n SynthScope python -c "import torch;assert torch.cuda.is_available();torch.randn(1,device='cuda')" 2>/dev/null && break; echo "[cuda-gate] $i"; sleep 20; done
run(){ echo ">>> $1"; "$CONDA" run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model "$MODEL" --ckpt "$CK" --device cuda "${@:2}" || echo "FAILED: $1"; }
run genimage8   --benchmark pergen      --data $DATA/GenImage_full/test     --bs $BS --limit 300 --tag ${MODEL}_retrain_genimage8
run forensynths --benchmark forensynths --data $DATA/CNNDetection/test      --bs $BS             --tag ${MODEL}_retrain_forensynths
run ufd         --benchmark ufd         --data $DATA/universialfakedetection --bs $BS            --tag ${MODEL}_retrain_ufd
if [ "$MODEL" = "co_spy" ] && [ -d "$DATA/CoSpyBench/test" ]; then
  run cospybench --benchmark pergen --data $DATA/CoSpyBench/test --bs $BS --limit 200 --tag ${MODEL}_retrain_cospybench
fi
echo "=== RETRAIN EVAL $MODEL DONE $(date) ==="
