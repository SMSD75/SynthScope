#!/bin/bash
#SBATCH --job-name=stage_genimage
#SBATCH --partition=cees6000
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn030
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/stage_ivi-cn030.out
echo "ON: $(hostname)"
DEST=""
for d in /nvmestore /ssdstore /scratch /local; do [ -d "$d" ] && [ -w "$d" ] && DEST="$d/ssalehi/SynthScope/GenImage" && break; done
[ -z "$DEST" ] && { echo "no writable scratch"; exit 1; }
if [ -d "$DEST/train/real" ] && [ $(ls "$DEST/train/fake" 2>/dev/null | wc -l) -gt 100000 ]; then
  echo "ALREADY STAGED: $DEST ($(du -sh $DEST | cut -f1))"; exit 0
fi
mkdir -p "$DEST"
echo "streaming GenImage/train (88G) from cn023 -> $DEST"
T0=$(date +%s)
ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn023   'tar -C /hddstore/ssalehi/SynthScope/GenImage -cf - train' | tar -C "$DEST" -xf -   && echo "STAGED in $(( ($(date +%s)-T0)/60 )) min: $(du -sh $DEST | cut -f1)"   || echo "STREAM FAILED"
