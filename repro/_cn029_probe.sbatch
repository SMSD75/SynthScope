#!/bin/bash
#SBATCH --job-name=cn029_probe
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/cn029_probe.out
echo "ON: $(hostname)"
echo "--- /ssdstore/ssalehi contents ---"
ls /ssdstore/ssalehi/ 2>/dev/null | head -15
echo "--- CNNDetection there? ---"
du -sh /ssdstore/ssalehi/*CNN* /ssdstore/ssalehi/*cnndet* /ssdstore/ssalehi/AGID/CNNDetection 2>/dev/null | head -3 || echo "(no CNNDetection)"
echo "--- free space ---"; df -h /ssdstore | tail -1
echo "--- GPUs free on this node right now ---"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | awk -F, '$2+0<100{c++} END{print c" GPUs free of 8"}'
