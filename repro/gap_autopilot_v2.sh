#!/bin/bash
# Auto-eval the overnight multi-GPU retrains as checkpoints land (cn023 GPU4; data is node-local here).
# Watches checkpoints_retrain/{aide_faithful,cospy_full,spai_full,safe_full}; evals newest unseen ckpt of
# each on its paper-relevant benchmark with --limit (fast). Results -> repro/bench_results/<tag>.json.
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
export AGID_CKPT_DIR=/hddstore/ssalehi/AGID/checkpoints
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch
export AGID_MFM_CKPT=/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4
G=/hddstore/ssalehi/AGID/GenImage_full/test_full
FS=/hddstore/ssalehi/AGID/CNNDetection/test
CB=/hddstore/ssalehi/AGID/CoSpyBench/test
EV(){ conda run --no-capture-output -n AGID python benchmark/bench_eval.py "$@" 2>&1 \
      | grep -vE "libpng|Testing:|/375|warn|UserWarning|%\|" | tail -3; }
RES(){ python3 -c "import json;d=json.load(open('repro/bench_results/$1.json'));print('  >> $1 mAP',d['means']['map'],'acc',d['means']['acc'])" 2>/dev/null; }
declare -A SEEN
for round in $(seq 1 80); do
  echo "===== autopilot-v2 round $round $(date +%H:%M) ====="
  a=$(ls -t checkpoints_retrain/aide_faithful/model_*.pth 2>/dev/null | head -1)
  if [ -n "${a:-}" ] && [ "${SEEN[aide]:-}" != "$a" ]; then SEEN[aide]=$a
    echo "[eval] AIDE $a"; EV --model aide --ckpt "$a" --benchmark pergen --data "$G" --limit 1000 --tag aide_faithful_g8; RES aide_faithful_g8
    EV --model aide --ckpt "$a" --benchmark forensynths --data "$FS" --tag aide_faithful_fs; RES aide_faithful_fs; fi
  c=$(ls -t checkpoints_retrain/cospy_full/model_*.pth 2>/dev/null | head -1)
  if [ -n "${c:-}" ] && [ "${SEEN[cospy]:-}" != "$c" ]; then SEEN[cospy]=$c
    echo "[eval] Co-Spy $c"; EV --model co_spy --ckpt "$c" --benchmark pergen --data "$G" --limit 1500 --tag cospy_full_g8; RES cospy_full_g8
    EV --model co_spy --ckpt "$c" --benchmark pergen --data "$CB" --limit 200 --tag cospy_full_cospybench; RES cospy_full_cospybench; fi
  s=$(ls -t checkpoints_retrain/spai_full/model_*.pth 2>/dev/null | head -1)
  if [ -n "${s:-}" ] && [ "${SEEN[spai]:-}" != "$s" ]; then SEEN[spai]=$s
    echo "[eval] SPAI $s"; EV --model spai --ckpt "$s" --benchmark pergen --data "$G" --limit 1500 --tag spai_full_g8; RES spai_full_g8; fi
  sf=$(ls -t checkpoints_retrain/safe_full/model_*.pth 2>/dev/null | head -1)
  if [ -n "${sf:-}" ] && [ "${SEEN[safe]:-}" != "$sf" ]; then SEEN[safe]=$sf
    echo "[eval] SAFE $sf"; EV --model safe --ckpt "$sf" --benchmark pergen --data "$G" --limit 1500 --tag safe_full_g8; RES safe_full_g8; fi
  # stop when all four training jobs are done
  done4=0
  for L in retrain_aide_faithful retrain_cospy_full retrain_spai_full retrain_safe; do
    grep -qiE "RETRAIN DONE|FAITHFUL RETRAIN DONE|FULL RETRAIN DONE" repro/eval_logs/${L}_*.log 2>/dev/null && done4=$((done4+1)); done
  sleep 600
done
echo "[autopilot-v2] DONE $(date)"
