#!/bin/bash
#SBATCH --job-name=hold_cn031
#SBATCH --partition=cees6000
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn031
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/hold_cn031.out
echo "holding $(hostname) for data push"; mkdir -p /nvmestore/ssalehi/SynthScope
sleep 14400
