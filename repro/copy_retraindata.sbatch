#!/bin/bash
#SBATCH --time=4-00:00:00 --cpus-per-task=4 --mem=16G
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/copyrt_%x_%j.log
set -uo pipefail
SCR=$(for d in /nvmestore /hddstore /ssdstore /sddstore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done); DEST=$SCR/ssalehi/SynthScope; mkdir -p "$DEST"
echo "node=$(hostname) dest=$DEST free=$(df -h $SCR|tail -1|awk '{print $4}')"
for ds in $DSETS; do echo "=== $ds $(date) ==="; rsync -a --bwlimit=250m --info=progress2 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=no' "ivi-cn023:/hddstore/ssalehi/SynthScope/$ds" "$DEST/" 2>&1 | tail -1; echo "-> $(du -sh $DEST/$(basename $ds) 2>/dev/null|awk '{print $1}')"; done
echo "=== RETRAINDATA COPY DONE $(date) ==="
