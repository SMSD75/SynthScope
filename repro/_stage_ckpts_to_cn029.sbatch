#!/bin/bash
# Stage the SERVED detector checkpoints + DINOv3 backbone from cn023 /hddstore (node-local) to
# cn029 /ssdstore so the web UI can run on a FREE cees GPU (cn023's GPUs are booked 4+ days).
# Excludes the big _train/ subdir (training-only); ~17G of served weights.
#SBATCH --job-name=stage_ckpts --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:0 --cpus-per-task=4 --mem=8G --time=00:40:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/stage_ckpts_%j.out
set -u
DST=ivi-cn029
echo "NODE=$(hostname) -> $DST  $(date)"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no $DST 'mkdir -p /ssdstore/ssalehi/AGID/checkpoints /ssdstore/ssalehi/AGID/dinov3_checkpoints' || { echo "ssh/mkdir failed"; exit 1; }
echo "=== rsync served checkpoints (top-level files only, exclude _train) ==="
rsync -a --info=progress2 --exclude='*/' \
  /hddstore/ssalehi/AGID/checkpoints/ $DST:/ssdstore/ssalehi/AGID/checkpoints/
echo "=== rsync DINOv3 backbone ==="
rsync -a --info=progress2 \
  /hddstore/ssalehi/AGID/dinov3_checkpoints/ $DST:/ssdstore/ssalehi/AGID/dinov3_checkpoints/
echo "=== verify on $DST ==="
ssh -o BatchMode=yes $DST 'echo "checkpoints:"; ls /ssdstore/ssalehi/AGID/checkpoints/*.pth | wc -l; du -sh /ssdstore/ssalehi/AGID/checkpoints; echo "dinov3:"; ls /ssdstore/ssalehi/AGID/dinov3_checkpoints/; echo "mfm:"; ls /ssdstore/ssalehi/AGID/mfm/ 2>/dev/null'
echo "=== STAGE DONE $(date) ==="
