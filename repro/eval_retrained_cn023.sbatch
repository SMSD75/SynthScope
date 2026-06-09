#!/bin/bash
#SBATCH --job-name=rt_eval --partition=quva --nodelist=ivi-cn023
#SBATCH --time=1-00:00:00 --cpus-per-task=8 --mem=40G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/eval_rt_%x_%j.log
# Eval a gap-closing retrained checkpoint across ALL of that paper's benchmarks present on cn023
# /hddstore. MODEL passed via --export=MODEL=<key>. Picks the latest checkpoints_retrain/<model>/model_*.pth.
# Runs on cn023 (every test set is local on /hddstore). Appends to repro/MASTER_RESULTS.jsonl.
set -uo pipefail; cd /ivi/zfs/s0/original_homes/ssalehi/AGID
: "${MODEL:?pass MODEL=vib_net|aide|co_spy|spai}"
DATA=/hddstore/ssalehi/AGID; SC=/hddstore/ssalehi/.cache; CONDA=/home/ssalehi/miniconda3/bin/conda
export AGID_CKPT_DIR=$DATA/checkpoints HF_HOME=$SC/huggingface TORCH_HOME=$SC/torch OMP_NUM_THREADS=4 \
       AGID_MFM_CKPT=$DATA/mfm/mfm_pretrain_vit_base.pth PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKDIR=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/$MODEL
CK=$(ls -1v "$CKDIR"/model_*.pth 2>/dev/null | tail -1)
[ -z "$CK" ] && { echo "no checkpoint in $CKDIR"; exit 1; }
BS=64; case "$MODEL" in spai) BS=8 ;; aide) BS=32 ;; esac
echo "=== eval retrained $MODEL ckpt=$CK bs=$BS node=$(hostname) $(date) ==="
# wait for CUDA (cn023 flaky init)
for i in $(seq 1 60); do "$CONDA" run -n AGID python -c "import torch;assert torch.cuda.is_available();torch.randn(1,device='cuda')" 2>/dev/null && break; echo "[cuda-gate] wait $i"; sleep 20; done
run(){ echo ">>> $1"; "$CONDA" run --no-capture-output -n AGID python benchmark/bench_eval.py --model "$MODEL" --ckpt "$CK" --device cuda "${@:2}" || echo "FAILED: $1"; }
run genimage8   --benchmark pergen     --data $DATA/GenImage_full/test          --bs $BS --limit 300 --tag ${MODEL}_retrain_genimage8
run forensynths --benchmark forensynths --data $DATA/CNNDetection/test           --bs $BS            --tag ${MODEL}_retrain_forensynths
run ufd         --benchmark ufd        --data $DATA/universialfakedetection      --bs $BS            --tag ${MODEL}_retrain_ufd
if [ "$MODEL" = "co_spy" ]; then
  run cospybench --benchmark pergen --data $DATA/CoSpyBench/test --bs $BS --limit 200 --tag ${MODEL}_retrain_cospybench
fi
echo "=== RETRAIN EVAL $MODEL DONE $(date) ==="
