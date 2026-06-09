#!/bin/bash
#SBATCH --job-name=chk30 --partition=cees6000 --account=ceesusers --constraint=6000 --nodelist=ivi-cn030
#SBATCH --time=4-00:00:00 --cpus-per-task=2 --mem=8G --gres=gpu:1
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/chk30_%j.log
SCR=$(for d in /nvmestore /hddstore /ssdstore /sddstore; do [ -d "$d/ssalehi" ] && echo "$d" && break; done)
echo "node=$(hostname) scratch=$SCR"
echo "existing: $(ls $SCR/ssalehi/AGID 2>/dev/null | tr '\n' ' ')"
echo "mfm: $(ls $SCR/ssalehi/AGID/mfm 2>/dev/null|head -1) | GenImage_full: $(ls $SCR/ssalehi/AGID/GenImage_full/test 2>/dev/null|wc -l) gens"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader|head -1
echo DONE
