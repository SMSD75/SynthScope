#!/bin/bash
# Re-run the Co-Spy retrain evals on cn023 GPU4 (the watcher run failed during a flaky-CUDA spell).
# Per-benchmark retry with a CUDA pre-check so a transient spell just delays, never drops a result.
# All data (incl 440G CoSpyBench) + Co-Spy backbones are on cn023 /hddstore.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
LOG=repro/eval_logs/reeval_cospy_cn023.log; exec >>"$LOG" 2>&1
echo "=== reeval cospy start $(date) ==="
CONDA=/home/ssalehi/miniconda3/bin/conda; D=/hddstore/ssalehi/SynthScope; GPU=4
export SYNTHSCOPE_CKPT_DIR=$D/checkpoints HF_HOME=$D/.cache/huggingface TORCH_HOME=$D/.cache/torch \
       SYNTHSCOPE_MFM_CKPT=$D/mfm/mfm_pretrain_vit_base.pth OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$(ls -1v $D/checkpoints/_train/cospy_retrain/model_*.pth 2>/dev/null | tail -1)
echo "ckpt=$CK gpu=$GPU"
run_retry(){ local tag=$1; shift
  for a in $(seq 1 40); do
    if ! CUDA_VISIBLE_DEVICES=$GPU "$CONDA" run -n SynthScope python -c "import torch;torch.randn(1,device='cuda')" 2>/dev/null; then
      echo "[$tag] CUDA down, wait (try $a) $(date)"; sleep 60; continue; fi
    echo ">>> [$tag] start try $a $(date)"
    if CUDA_VISIBLE_DEVICES=$GPU "$CONDA" run --no-capture-output -n SynthScope python benchmark/bench_eval.py \
         --model co_spy --ckpt "$CK" --device cuda --tag "$tag" "$@"; then echo "[$tag] OK"; return 0; fi
    echo "[$tag] failed, retry (try $a)"; sleep 60
  done; echo "[$tag] GAVE UP"; }
run_retry co_spy_retrain_cospybench  --benchmark pergen      --data $D/CoSpyBench/test         --bs 32 --limit 200
run_retry co_spy_retrain_genimage8   --benchmark pergen      --data $D/GenImage_full/test      --bs 32 --limit 300
run_retry co_spy_retrain_forensynths --benchmark forensynths --data $D/CNNDetection/test        --bs 32
run_retry co_spy_retrain_aigcdetect  --benchmark pergen      --data $D/AIGCDetectBenchmark/test --bs 32 --limit 200
echo "=== reeval cospy DONE $(date) ==="
