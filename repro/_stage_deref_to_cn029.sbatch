#!/bin/bash
# The served checkpoints on cn023 are SYMLINKS into node-local /hddstore (_train/, repro/ckpts/,
# repro/official_ckpts/). The first pull copied the dangling symlinks, so most models show off on
# cn029. Re-push with -L (dereference) so the real file CONTENT lands on cn029 /ssdstore. Allowed:
# we hold the GPU-UI job 188136 on cn029, so cn023->cn029 ssh passes pam_slurm_adopt.
#SBATCH --job-name=stage_deref --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:0 --cpus-per-task=4 --mem=8G --time=00:40:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/stage_deref_%j.out
set -u
DST=ivi-cn029
echo "NODE=$(hostname) -> $DST  $(date)"
echo "=== re-push served checkpoints, FOLLOWING symlinks (-L) so targets materialize ==="
rsync -aL --info=progress2 --exclude='*/' \
  /hddstore/ssalehi/AGID/checkpoints/ $DST:/ssdstore/ssalehi/AGID/checkpoints/ || { echo "rsync FAILED"; exit 1; }
echo "=== verify: all should now be regular files, not dangling links ==="
ssh -o BatchMode=yes $DST 'cd /ssdstore/ssalehi/AGID/checkpoints; for f in *.pth; do if [ -f "$f" ]; then printf "  OK  %-26s %s\n" "$f" "$(du -h "$f" | cut -f1)"; else printf "  BAD %s (missing target)\n" "$f"; fi; done'
echo "=== STAGE-DEREF DONE $(date) ==="
