#!/bin/bash
#SBATCH --job-name=stage25 --partition=cees --account=ceesusers --nodelist=ivi-cn025
#SBATCH --time=08:00:00 --cpus-per-task=8 --mem=16G
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/stage_cn025_%j.log
# Stage the GenImage (SDv1.4) train set + cospy_subset from cn023 onto cn025's local scratch so the
# VIB-Net / AIDE / Co-Spy retrains can run there in parallel. Writes a DATA_READY marker on /ivi (shared)
# that the dependent training jobs gate on. Idempotent (rsync skips already-present files).
set -uo pipefail
SCR=$(for d in /nvmestore /ssdstore /hddstore /sddstore; do [ -d "$d" ] && [ -w "$d" ] && echo "$d" && break; done)
DATA=$SCR/ssalehi/SynthScope
mkdir -p "$DATA/spai_train"
echo "node=$(hostname) scratch=$SCR data=$DATA $(date)"
RS=(rsync -aL --info=progress2 -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
echo "[stage] GenImage (train+test, ~91G) ..."
"${RS[@]}" ivi-cn023:/hddstore/ssalehi/SynthScope/GenImage "$DATA/" 2>&1 | tail -2
echo "[stage] cospy_subset ..."
"${RS[@]}" ivi-cn023:/hddstore/ssalehi/SynthScope/spai_train/cospy_subset "$DATA/spai_train/" 2>&1 | tail -2
echo "[stage] counts: GenImage/train/real=$(ls "$DATA"/GenImage/train/real 2>/dev/null|wc -l) " \
     "GenImage/train/fake=$(ls "$DATA"/GenImage/train/fake 2>/dev/null|wc -l) " \
     "cospy/train=$(find "$DATA"/spai_train/cospy_subset/train -type f 2>/dev/null|wc -l)"
# Gate marker for dependents (records which scratch dir on cn025 holds the data).
MARK=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/eval_logs/DATA_READY_cn025
echo "$SCR" > "$MARK"
echo "=== STAGE cn025 DONE (scratch=$SCR exit 0) $(date) ==="
