#!/bin/bash
#SBATCH --job-name=cpdata
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/copy_%x_%j.log
set -uo pipefail
echo "node=$(hostname) start=$(date)"
SCRATCH=$(for d in /hddstore /ssdstore /sddstore /nvmestore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done)
DEST="$SCRATCH/ssalehi/SynthScope"; mkdir -p "$DEST"
echo "scratch=$SCRATCH dest=$DEST free=$(df -h $SCRATCH|tail -1|awk '{print $4}')"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn023 'echo SSH_BACK_OK from $(hostname)' || { echo SSH_FAIL; exit 1; }
for ds in repro/benchmark_specs mfm GenImage_full CNNDetection AIGIBench AIGCDetectBenchmark spai_train GenImage; do
  echo "=== rsync $ds  $(date) ==="
  rsync -a --bwlimit=100m --info=progress2 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=no' \
        "ivi-cn023:/hddstore/ssalehi/SynthScope/$ds" "$DEST/" 2>&1 | tail -1
  echo "  -> $(du -sh $DEST/$(basename $ds) 2>/dev/null | awk '{print $1}')"
done
echo "=== COPY DONE $(date) ==="
