#!/bin/bash
#SBATCH --job-name=cptest
#SBATCH --time=7-0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/eval_logs/copy_test_%j.log
set -uo pipefail
echo "node=$(hostname) date=$(date)"
echo "=== detect local scratch ==="
for d in /hddstore /ssdstore /sddstore /nvmestore; do
  [ -d "$d" ] && echo "$d exists, free=$(df -h $d 2>/dev/null|tail -1|awk '{print $4}'), writable=$([ -w "$d" ] && echo yes || echo no)"
done
echo "=== test SSH back to cn023 (I have job there) ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no ivi-cn023 'echo SSH_OK from $(hostname); ls /hddstore/ssalehi/AGID | head -3' 2>&1 | head -5
echo "=== test small rsync FROM cn023 ==="
SCRATCH=$(for d in /hddstore /ssdstore /sddstore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done)
echo "chosen scratch: $SCRATCH"
mkdir -p "$SCRATCH/ssalehi/AGID" 2>/dev/null
timeout 120 rsync -a --info=progress2 ivi-cn023:/hddstore/ssalehi/AGID/repro/benchmark_specs "$SCRATCH/ssalehi/AGID/" 2>&1 | tail -3
echo "copied specs -> $(ls $SCRATCH/ssalehi/AGID/benchmark_specs 2>/dev/null | wc -l) files"
echo "=== DONE ==="
