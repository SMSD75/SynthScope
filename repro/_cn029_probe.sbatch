#!/bin/bash
#SBATCH --job-name=cn029_probe --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=00:05:00 --cpus-per-task=2 --mem=8G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/cn029_probe_%j.out
echo "node=$(hostname) $(date)"
echo "=== /ssdstore free ==="; df -h /ssdstore 2>&1 | tail -2
echo "=== GenImage staged on cn029? ==="
ls -d /ssdstore/ssalehi/SynthScope/GenImage/train/real /ssdstore/ssalehi/SynthScope/GenImage/train/fake 2>&1
for d in /ssdstore/ssalehi/SynthScope/GenImage/train/real /ssdstore/ssalehi/SynthScope/GenImage/train/fake; do
  [ -d "$d" ] && echo "$d: $(ls "$d" 2>/dev/null | wc -l) files"
done
echo "=== CUDA ==="; /home/ssalehi/miniconda3/bin/conda run -n SynthScope python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | tail -1
