#!/bin/bash
#SBATCH --job-name=stage_gi29
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/stage_cn029_gi.out
echo "ON: $(hostname)"
DEST=/ssdstore/ssalehi/AGID/GenImage
if [ -d "$DEST/train/real" ] && [ $(ls "$DEST/train/fake" 2>/dev/null | wc -l) -gt 100000 ]; then
  echo "ALREADY STAGED: $DEST"; exit 0
fi
mkdir -p "$DEST"
echo "streaming GenImage/train from cn023 -> $DEST"
T0=$(date +%s)
ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn023 \
  'tar -C /hddstore/ssalehi/AGID/GenImage -cf - train' | tar -C "$DEST" -xf - \
  && echo "STAGED in $(( ($(date +%s)-T0)/60 )) min" || echo "STREAM FAILED"
