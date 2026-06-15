#!/bin/bash
#SBATCH --job-name=ssh_probe2
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:04:00
#SBATCH --exclude=ivi-cn020
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/ssh_probe2.out
echo "ON: $(hostname)"
echo "--- name resolution + route ---"
getent hosts ivi-cn023 || echo "DNS FAIL"
timeout 5 bash -c 'echo > /dev/tcp/ivi-cn023/22' && echo "port 22 reachable" || echo "port 22 BLOCKED"
echo "--- ssh attempt (verbose, key auth) ---"
timeout 25 ssh -vv -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/id_rsa ivi-cn023 'echo SSH_OK from $(hostname)' 2>&1 | grep -E "OK|debug1: (Connecting|Connection|Authent|Offering|Server accepts|Permission)|denied|closed|timed out" | head -12
