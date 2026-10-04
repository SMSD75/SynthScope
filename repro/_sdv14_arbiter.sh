#!/bin/bash
cd /ivi/zfs/s0/original_homes/ssalehi/SynthScope
HS=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/sdv14_handshake
A5=184795; C31=184776
while :; do
  # handshake: a node requested data -> push GenImage to it
  if [ -f "$HS/NODE" ] && [ ! -f "$HS/DATA_READY" ] && [ ! -f "$HS/PUSHING" ]; then
    N=$(cat "$HS/NODE"); IP=$(getent hosts "$N" | awk '{print $1}' | head -1)
    touch "$HS/PUSHING"
    echo "[arbiter] pushing GenImage to $N ($IP)"
    D=""; for d in /ssdstore /nvmestore /scratch /local; do
      ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$IP" "[ -w $d ]" 2>/dev/null && D="$d" && break; done
    tar -C /hddstore/ssalehi/SynthScope/GenImage -cf - train |       ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$IP" "mkdir -p $D/ssalehi/SynthScope/GenImage && tar -C $D/ssalehi/SynthScope/GenImage -xf -"       && { touch "$HS/DATA_READY"; echo "[arbiter] push done"; } || echo "[arbiter] push FAILED"
    rm -f "$HS/PUSHING"
  fi
  # arbitration: cancel the loser when one SDv14 job starts training
  A5S=$(squeue -j $A5 -h -o %t 2>/dev/null); C31S=$(squeue -j $C31 -h -o %t 2>/dev/null)
  if [ "$A5S" = "R" ] && [ -n "$C31S" ]; then scancel $C31; echo "[arbiter] A5000 won; cancelled cn031 job"; fi
  if [ "$C31S" = "R" ] && [ -n "$A5S" ]; then scancel $A5; echo "[arbiter] cn031 won; cancelled A5000 job"; fi
  [ -z "$A5S" ] && [ -z "$C31S" ] && { echo "[arbiter] both jobs left queue; exit"; break; }
  sleep 60
done
