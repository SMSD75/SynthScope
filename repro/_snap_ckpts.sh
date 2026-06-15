#!/bin/bash
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
LOG=repro/logs/pgc_8gpu_184771.out
CK=checkpoints_retrain/pgc_8gpu/model.pth
for E in 10 25 50; do
  until grep -qa "\[epoch $E/100\]" "$LOG" 2>/dev/null; do
    grep -qa "\[done\]" "$LOG" 2>/dev/null && exit 0
    sleep 120
  done
  cp "$CK" "checkpoints_retrain/pgc_8gpu/model_ep${E}.pth" && echo "snapped epoch $E"
done
