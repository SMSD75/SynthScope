#!/bin/bash
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
until squeue -j 184777 -h -o %t 2>/dev/null | grep -q R; do sleep 15; done
TGT=$(getent hosts ivi-cn031-netgbe0 | awk '{print $1}'); TGT=${TGT:-10.4.1.31}
echo "[push2] pushing GenImage to cn031 via $TGT"
T0=$(date +%s)
tar -C /hddstore/ssalehi/SynthScope/GenImage -cf - train |   ssh -o BatchMode=yes -o StrictHostKeyChecking=no $TGT 'mkdir -p /nvmestore/ssalehi/SynthScope/GenImage && tar -C /nvmestore/ssalehi/SynthScope/GenImage -xf -'   && echo "[push2] cn031 DONE in $(( ($(date +%s)-T0)/60 )) min" || echo "[push2] cn031 FAILED AGAIN"
scancel 184777
