#!/bin/bash
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
# wait until hold job runs on cn031
until squeue -j 184775 -h -o %t 2>/dev/null | grep -q R; do sleep 20; done
echo "[push] cn031 hold active; pushing GenImage/train (88G)"
T0=$(date +%s)
tar -C /hddstore/ssalehi/SynthScope/GenImage -cf - train | \
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn031 'mkdir -p /nvmestore/ssalehi/SynthScope/GenImage && tar -C /nvmestore/ssalehi/SynthScope/GenImage -xf -' \
  && echo "[push] cn031 DONE in $(( ($(date +%s)-T0)/60 )) min" || echo "[push] cn031 FAILED"
scancel 184775
echo "[push] now cn029 (for the joint run; training job 184771 keeps ssh open)"
T1=$(date +%s)
tar -C /hddstore/ssalehi/SynthScope/GenImage -cf - train | \
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn029 'mkdir -p /ssdstore/ssalehi/SynthScope/GenImage && tar -C /ssdstore/ssalehi/SynthScope/GenImage -xf -' \
  && echo "[push] cn029 DONE in $(( ($(date +%s)-T1)/60 )) min" || echo "[push] cn029 FAILED"
echo "[push] ALL DONE"
