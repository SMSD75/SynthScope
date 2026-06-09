#!/bin/bash
# Detached watcher (run via setsid inside the cn023 alloc). Waits for the two cn023 gap-closing retrains
# to finish, then evals each on its paper's benchmarks using the SAME freed GPU (our reserved 4 & 7).
# Co-Spy + AIDE evals MUST run here: CoSpyBench (440G) + AIGCDetect + their backbones are only on cn023.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
LOG=repro/eval_logs/watch_evals_cn023.log
exec >>"$LOG" 2>&1
echo "=== watcher start $(date) pids: cospy=2923983 aide=2959891 ==="
CONDA=/home/ssalehi/miniconda3/bin/conda
export AGID_CKPT_DIR=/hddstore/ssalehi/AGID/checkpoints HF_HOME=/hddstore/ssalehi/.cache/huggingface \
       TORCH_HOME=/hddstore/ssalehi/.cache/torch AGID_MFM_CKPT=/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth \
       OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/hddstore/ssalehi/AGID

cuda_gate(){ for i in $(seq 1 90); do CUDA_VISIBLE_DEVICES=$1 "$CONDA" run -n AGID python -c "import torch;assert torch.cuda.is_available();torch.randn(1,device='cuda')" 2>/dev/null && return 0; echo "[cuda-gate gpu$1] wait $i"; sleep 20; done; }
run(){ local gpu=$1 tag=$2; shift 2; echo ">>> [$tag] gpu=$gpu $(date)"; CUDA_VISIBLE_DEVICES=$gpu "$CONDA" run --no-capture-output -n AGID python benchmark/bench_eval.py --device cuda --tag "$tag" "$@" || echo "FAILED $tag"; }

wait_pid(){ while kill -0 "$1" 2>/dev/null; do sleep 120; done; echo "[pid $1 exited] $(date)"; sleep 20; }

# ---------- Co-Spy (GPU 4) ----------
wait_pid 2923983
CK=$(ls -1v $D/checkpoints/_train/cospy_retrain/model_*.pth 2>/dev/null | tail -1)
echo "=== eval co_spy ckpt=$CK ==="
if [ -n "$CK" ]; then
  cuda_gate 4
  run 4 co_spy_retrain_cospybench  --model co_spy --ckpt "$CK" --benchmark pergen      --data $D/CoSpyBench/test            --bs 32 --limit 200 || true
  run 4 co_spy_retrain_genimage8   --model co_spy --ckpt "$CK" --benchmark pergen      --data $D/GenImage_full/test         --bs 32 --limit 300 || true
  run 4 co_spy_retrain_forensynths --model co_spy --ckpt "$CK" --benchmark forensynths --data $D/CNNDetection/test           --bs 32 || true
  run 4 co_spy_retrain_aigcdetect  --model co_spy --ckpt "$CK" --benchmark pergen      --data $D/AIGCDetectBenchmark/test    --bs 32 --limit 200 || true
fi

# ---------- AIDE-ProGAN (GPU 7) ----------
wait_pid 2959891
CK=$(ls -1v $D/checkpoints/_train/aide_progan/model_*.pth 2>/dev/null | tail -1)
echo "=== eval aide(progan) ckpt=$CK ==="
if [ -n "$CK" ]; then
  cuda_gate 7
  run 7 aide_progan_forensynths --model aide --ckpt "$CK" --benchmark forensynths --data $D/CNNDetection/test        --bs 32 || true
  run 7 aide_progan_ufd         --model aide --ckpt "$CK" --benchmark ufd         --data $D/universialfakedetection  --bs 32 || true
  run 7 aide_progan_aigcdetect  --model aide --ckpt "$CK" --benchmark pergen      --data $D/AIGCDetectBenchmark/test --bs 32 --limit 200 || true
fi

echo "=== watcher DONE $(date) ==="
