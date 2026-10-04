#!/bin/bash
# cn023 CPU job: (1) write the subset file lists, (2) stay alive so a cees GPU job can rsync from this node
# (pam_slurm_adopt only allows ssh to a node where we hold an active job).
#SBATCH --job-name=cure_lists --partition=quva --account=quvausers --nodelist=ivi-cn023
#SBATCH --gres=gpu:0 --cpus-per-task=1 --mem=3G --time=00:30:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/cure_lists_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
echo "NODE=$(hostname) $(date)"
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python repro/_cure_make_lists.py 800 300 2>&1 | grep -avE "Warning|warn|nvml|raw_cnt"
touch /ivi/zfs/s0/original_homes/ssalehi/AGID_shared/cure_ref/lists/READY
echo "LISTS READY $(date); holding node for rsync pulls"
sleep 1500
