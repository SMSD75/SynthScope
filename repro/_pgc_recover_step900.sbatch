#!/bin/bash
#SBATCH --job-name=pgc_recov --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:10:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/pgc_recov_%j.out
SCR=/ssdstore/ssalehi/pgc_400_187220/ckpts/pgc_train_progan
DEST=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/checkpoints_retrain/pgc_400step
echo "NODE=$(hostname) $(date)"
ls -la "$SCR"/step*.pth "$SCR"/best.pth 2>&1 | head
mkdir -p "$DEST"
# recover the peak (step900) only — 1.25G, fits the /ivi quota headroom
if [ -f "$SCR/step900_acc66_ap94.pth" ]; then
  cp -f "$SCR/step900_acc66_ap94.pth" "$DEST/step900_acc66_ap94.pth" && echo "RECOVERED step900 -> $DEST"
else
  echo "step900 NOT in scratch; available:"; ls "$SCR" 2>&1 | head
fi
