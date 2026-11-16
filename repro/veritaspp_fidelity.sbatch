#!/bin/bash
#SBATCH --job-name=vppfid
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/vlm_wt/repro/logs/veritaspp_fidelity_%j.out
# Veritas++ fidelity (repro/veritaspp_fidelity.py): weights + data are node-local on ivi-cn029.
set -uo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SYNTHSCOPE_CKPT_DIR=/ssdstore/ssalehi/SynthScope/vlm_ckpts
export HF_HOME=/ssdstore/ssalehi/SynthScope/vlm_ckpts/_hf_home TMPDIR=/ssdstore/ssalehi/SynthScope/vlm_ckpts/_tmp
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python     # the SynthScope env (symlinked as envs/SynthScope)
WT=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/vlm_wt
echo "node=$(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== sha256 of the released files (SHA256SUMS written at download, HF EricTan7/VeritasPP @ 22ba6c6b) ==="
(cd $SYNTHSCOPE_CKPT_DIR/veritaspp && sha256sum -c SHA256SUMS)
cd $WT && $PY repro/veritaspp_fidelity.py --out $WT/repro/veritaspp_fidelity.json "$@"
echo "rc=$? $(date)"
