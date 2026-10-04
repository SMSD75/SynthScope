#!/bin/bash
# Autonomous gap-closing eval pipeline (cn023 GPU4). Waits for the full-GenImage batch to free GPU4,
# then repeatedly evals the newest checkpoint of each running retrain (cospy/spai/vib) on its
# paper-relevant benchmark, so SynthScope-vs-official gap-closing is tracked as training progresses.
# Results -> repro/bench_results/<tag>.json (+ MASTER_RESULTS.jsonl). Log: repro/eval_logs/gap_autopilot.log
set -uo pipefail
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/AGID/checkpoints
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch
export SYNTHSCOPE_MFM_CKPT=/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4
FULLG=/hddstore/ssalehi/AGID/GenImage_full/test_full
COSPYB=/hddstore/ssalehi/AGID/CoSpyBench/test
EV(){ conda run --no-capture-output -n SynthScope python benchmark/bench_eval.py "$@" 2>&1 | grep -vE "libpng|UserWarning|warnings.warn" | tail -4; }

echo "[autopilot] waiting for full-genimage GPU4 batch to finish $(date)"
for i in $(seq 1 90); do grep -q "BATCH DONE" repro/eval_logs/full_genimage_gpu4.log 2>/dev/null && break; sleep 60; done
echo "[autopilot] GPU4 free; starting eval loop $(date)"

# One-time: confirm the flickr data-fix recovers the shipped co_spy.pth on the now-clean full CoSpyBench
echo "[autopilot] flickr-fix confirmation: shipped co_spy.pth on clean CoSpyBench"
EV --model co_spy --benchmark pergen --data "$COSPYB" --limit 200 --tag cospy_shipped_cospybench_cleanflickr

declare -A SEEN
for round in $(seq 1 18); do
  echo "===== autopilot round $round $(date) ====="
  # Co-Spy (SD-v1.4 retrain) -> GenImage-8 full
  c=$(ls -t /hddstore/ssalehi/AGID/checkpoints/_train/cospy_genimage/model_*.pth 2>/dev/null | head -1)
  if [ -n "${c:-}" ] && [ "${SEEN[cospy]:-}" != "$c" ]; then
    SEEN[cospy]=$c; echo "[eval] cospy $c"
    EV --model co_spy --ckpt "$c" --benchmark pergen --data "$FULLG" --tag cospy_gi_rt_fullg8
  fi
  # SPAI v2 -> GenImage-8 full
  s=$(ls -t checkpoints_retrain/spai/model_*.pth 2>/dev/null | head -1)
  if [ -n "${s:-}" ] && [ "${SEEN[spai]:-}" != "$s" ]; then
    SEEN[spai]=$s; echo "[eval] spai $s"
    EV --model spai --ckpt "$s" --benchmark pergen --data "$FULLG" --tag spai_v2_fullg8
  fi
  # VIB CLIP -> GenImage-8 full
  v=$(ls -t checkpoints_retrain/vib_net/model_*.pth 2>/dev/null | head -1)
  if [ -n "${v:-}" ] && [ "${SEEN[vib]:-}" != "$v" ]; then
    SEEN[vib]=$v; echo "[eval] vib $v"
    EV --model vib_net --ckpt "$v" --benchmark pergen --data "$FULLG" --tag vib_v2_fullg8
  fi
  # when cospy training is DONE, run the headline CoSpyBench eval once
  if grep -q "COSPY GENIMAGE RETRAIN DONE" repro/eval_logs/cospy_genimage.log 2>/dev/null \
     && [ "${SEEN[cospyb]:-}" != "done" ]; then
    cf=$(ls -t /hddstore/ssalehi/AGID/checkpoints/_train/cospy_genimage/model_*.pth 2>/dev/null | head -1)
    if [ -n "${cf:-}" ]; then SEEN[cospyb]=done; echo "[eval] cospy CoSpyBench $cf"
      EV --model co_spy --ckpt "$cf" --benchmark pergen --data "$COSPYB" --tag cospy_gi_rt_cospybench; fi
  fi
  sleep 600
done
echo "[autopilot] DONE $(date)"
