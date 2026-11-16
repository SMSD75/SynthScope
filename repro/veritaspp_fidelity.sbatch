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
# Veritas++ fidelity (repro/veritaspp_fidelity.py): weights + data are node-local on ivi-cn029. Any failed check stops
# the job (set -e): the file checksums, or a local folder that is not the pinned revision.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SYNTHSCOPE_CKPT_DIR=/ssdstore/ssalehi/SynthScope/vlm_ckpts
export HF_HOME=/ssdstore/ssalehi/SynthScope/vlm_ckpts/_hf_home TMPDIR=/ssdstore/ssalehi/SynthScope/vlm_ckpts/_tmp
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python     # the SynthScope env (symlinked as envs/SynthScope)
WT=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/vlm_wt
W=$SYNTHSCOPE_CKPT_DIR/veritaspp
echo "node=$(hostname) $(date) commit=$(git -C $WT rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== revision: every file of the folder was downloaded from the pinned commit (hf download metadata) ==="
cd $WT && $PY - <<'EOF'
import sys
from pathlib import Path
sys.path[:0] = [".", "interface"]
from models.image.veritaspp.veritaspp import VERITASPP
folder = Path("/ssdstore/ssalehi/SynthScope/vlm_ckpts/veritaspp")
sums = dict(line.split()[::-1] for line in (folder / "SHA256SUMS").read_text().splitlines())
meta = sorted((folder / ".cache/huggingface/download").glob("*.metadata"))
assert meta, "no hf download metadata in the folder"
for m in meta:
    commit, etag = m.read_text().splitlines()[:2]
    name = m.name[:-len(".metadata")]
    assert commit == VERITASPP.hf_revision, (name, commit, VERITASPP.hf_revision)
    if name.endswith(".safetensors"):        # an LFS file's etag is its sha256
        assert etag == sums[name], (name, etag, sums[name])
print(f"{len(meta)} files from {VERITASPP.hf_repo} @ {VERITASPP.hf_revision}; shard etags == SHA256SUMS")
EOF
echo "=== sha256 of the released files (SHA256SUMS written at download) ==="
(cd $W && sha256sum -c SHA256SUMS)
cd $WT && $PY repro/veritaspp_fidelity.py --out $WT/repro/veritaspp_fidelity.json "$@"
echo "rc=$? $(date)"
