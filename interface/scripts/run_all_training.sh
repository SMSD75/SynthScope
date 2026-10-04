#!/bin/bash
# Old AGID_* variables still work: map each to its SYNTHSCOPE_* name unless that is already set.
for _v in $(compgen -e | grep '^AGID_' || true); do _n="SYNTHSCOPE_${_v#AGID_}"; [ -n "${!_n:-}" ] || export "$_n=${!_v}"; done
# Unattended orchestrator: finishes the whole "train the missing detectors" pipeline on cn023
# so it completes even if the user disconnects. It does NOT relaunch the jobs already running
# (clip/dinov2/vib_net probes, restrav) -- it waits for them, then trains the rest, promoting
# and verifying each, calibrates D3, and rewrites the manifest. Detach with:
#
#   setsid bash interface/scripts/run_all_training.sh >/dev/null 2>&1 &
#
# Progress: tail -f $CKPT_DIR/_train/orchestrator.log
set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
CONDA="${CONDA_EXE:-/home/ssalehi/miniconda3/bin/conda}"
CKPT_DIR="${SYNTHSCOPE_CKPT_DIR:-/hddstore/ssalehi/SynthScope/checkpoints}"
ORCH_LOG="$CKPT_DIR/_train/orchestrator.log"
mkdir -p "$CKPT_DIR/_train"
TRAIN=interface/scripts/train_model.sh

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$ORCH_LOG"; }
running() { pgrep -af "main.py" | grep -q "train_$1\b"; }
wait_for() {  # wait until none of the named jobs are running
  local names=("$@")
  while :; do
    local any=0
    for n in "${names[@]}"; do running "$n" && any=1; done
    [ "$any" -eq 0 ] && break
    sleep 30
  done
}
promote() { "$CONDA" run --no-capture-output -n SynthScope python interface/scripts/promote_checkpoint.py "$1" 2>&1 | tee -a "$ORCH_LOG"; }
verify()  { "$CONDA" run --no-capture-output -n SynthScope python interface/scripts/verify_model.py "$1" --n 240 2>&1 | grep -v libpng | tee -a "$ORCH_LOG"; }
collect() { "$CONDA" run --no-capture-output -n SynthScope python interface/scripts/collect_checkpoints.py 2>&1 | tee -a "$ORCH_LOG"; }
have_ckpt() { ls "$CKPT_DIR/_train/$1"/model_*.pth >/dev/null 2>&1; }
finish() {  # promote+verify if a checkpoint was produced
  if have_ckpt "$1"; then log "promote+verify $1"; promote "$1"; verify "$1"
  else log "WARN: $1 produced no checkpoint (see _train/$1/train.log)"; fi
}

log "=== orchestrator start ==="

# Phase 1: the already-running probes.
log "waiting for probes: clip dinov2 vib_net ..."
wait_for clip dinov2 vib_net
sleep 15
for m in clip dinov2 vib_net; do finish "$m"; done

# Phase 2: heavy wave A on the freed GPUs 0-5 (restrav keeps 6,7 until done).
log "launching wave A: adapted_dino(0,1) cnn(2,3) safe(4,5)"
setsid bash "$TRAIN" adapted_dino 0,1 29561 2 >/dev/null 2>&1 &
setsid bash "$TRAIN" cnn          2,3 29562 3 >/dev/null 2>&1 &
setsid bash "$TRAIN" safe         4,5 29563 10 >/dev/null 2>&1 &
sleep 20
wait_for adapted_dino cnn safe
sleep 15
for m in adapted_dino cnn safe; do finish "$m"; done

# Phase 3: heavy wave B on GPUs 0-3.
log "launching wave B: spai(0,1) co_spy(2,3)"
setsid bash "$TRAIN" spai   0,1 29564 5 >/dev/null 2>&1 &
setsid bash "$TRAIN" co_spy 2,3 29565 3 >/dev/null 2>&1 &
sleep 20
wait_for spai co_spy
sleep 15
for m in spai co_spy; do finish "$m"; done

# Phase 4: restrav (started earlier on 6,7).
log "waiting for restrav ..."
wait_for restrav
sleep 10
if have_ckpt restrav; then
  log "promote restrav + full GenVideo-Val eval"
  promote restrav
  "$CONDA" run --no-capture-output -n SynthScope python interface/scripts/eval_restrav_full.py --n 600 2>&1 | grep -v libpng | tee -a "$ORCH_LOG"
else
  log "WARN: restrav produced no checkpoint"
fi

# Phase 5: D3 calibration (GPUs free now).
log "calibrating D3 ..."
CUDA_VISIBLE_DEVICES=0 "$CONDA" run --no-capture-output -n SynthScope python interface/scripts/calibrate_d3.py --n 200 2>&1 | grep -v libpng | tee -a "$ORCH_LOG"

# Phase 6: refresh manifest + summary.
log "refreshing manifest ..."
collect
log "=== orchestrator done ==="
log "checkpoints present:"
ls -lL "$CKPT_DIR"/*.pth 2>/dev/null | awk '{print "  "$NF}' | tee -a "$ORCH_LOG"
cat "$CKPT_DIR/MANIFEST.md" 2>/dev/null | tee -a "$ORCH_LOG"
