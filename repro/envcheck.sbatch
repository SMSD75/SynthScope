#!/bin/bash
#SBATCH --job-name=envchk --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=7-00:00:00 --cpus-per-task=2 --mem=8G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/envchk_%j.log
set -uo pipefail
echo "node=$(hostname)"
SCRATCH=$(for d in /hddstore /ssdstore /sddstore; do [ -d "$d/ssalehi" ] && echo "$d" && break; done)
echo "scratch=$SCRATCH; data: $(ls $SCRATCH/ssalehi/AGID 2>/dev/null | tr '\n' ' ')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
CONDA=/home/ssalehi/miniconda3/bin/conda
$CONDA run -n AGID python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); import sys; sys.path.insert(0,'/ivi/zfs/s0/original_homes/ssalehi/AGID'); import agid; print('agid import OK')" 2>&1 | tail -3
echo "DONE"
