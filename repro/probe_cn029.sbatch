#!/bin/bash
#SBATCH --job-name=probe --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --time=00:02:00 --cpus-per-task=2 --mem=4G
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/probe_cn029.log
echo "node=$(hostname) date=$(date)"
for d in /ssdstore /nvmestore /hddstore; do echo "== $d =="; du -sh $d/ssalehi/SynthScope/GenImage $d/ssalehi/SynthScope/spai_train/dmid_real 2>/dev/null; df -h $d 2>/dev/null | tail -1; done
