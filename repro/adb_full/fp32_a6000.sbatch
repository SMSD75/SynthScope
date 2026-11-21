#!/bin/bash
#SBATCH --job-name=adbfull_fp32
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=repro/logs/adbfull_fp32_%j.out     # relative to the submit directory: submit from the repo root
# Align, Don't Blend: the float32 spot check. The adb_vlm VLM branch in float32 + eager attention (the official README's
# inference command) on the FULL WildRF test split, on one 48 GB RTX A6000 (float32 needs about 38 GB):
#   sbatch --nodelist=ivi-cn023 repro/adb_full/fp32_a6000.sbatch
# WildRF/test is copied from ivi-cn029 and its sha1 list compared with the original (the same files as the bf16 run and the
# dda per-image run). Result: repro/adb_full/vlm/wildrf/adb_vlm__fullfp32.json (run_units unit vlm:wildrf:fullfp32: bench_eval
# --vlm-dtype float32 --vlm-attn eager, its own cache folder). Fused with the same dda records by repro/adb_full/fuse.py.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SRC=/ssdstore/ssalehi/SynthScope                         # ivi-cn029: weights + WildRF originals
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/adb_full/fuse.py" ] || { echo "$WT is not the SynthScope repo root"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
S=""   # the node-local folder that holds the SynthScope checkpoints (ivi-cn023: /hddstore; its /ssdstore is nearly full)
for d in /ssdstore /nvmestore /hddstore; do
  [ -d $d/ssalehi/SynthScope/checkpoints ] || [ -d $d/ssalehi/SynthScope/zoo_ckpts ] && { S=$d/ssalehi/SynthScope; break; }
done
: "${S:?no SynthScope checkpoint folder on $(hostname -s)}"
V=$S/vpp
SHA=$(git -C $WT rev-parse HEAD)
W=$V/code_$SLURM_JOB_ID
rm -rf $W && mkdir -p $W $V/cache $V/locks $V/full && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd $W
export PYTHONPATH=$W

echo "=== weights $(date +%T) ==="
mkdir -p $S/vlm_ckpts
for f in align_dont_blend_vlm qwen2.5-vl-7b-instruct; do
  ( flock -w 7200 9; rsync -a --partial --exclude=_hf_home --exclude=_tmp ssalehi@ivi-cn029:$SRC/vlm_ckpts/$f $S/vlm_ckpts/ ) \
    9>$S/vlm_ckpts/.stage.lock
done
mkdir -p $S/vlm_ckpts/_hf_home $S/vlm_ckpts/_tmp
export HF_HOME=$S/vlm_ckpts/_hf_home TMPDIR=$S/vlm_ckpts/_tmp
CK=$V/ckpt_$SLURM_JOB_ID
rm -rf $CK && mkdir -p $CK
ln -sfn $S/vlm_ckpts/align_dont_blend_vlm $CK/adb_vlm
ln -sfn $S/vlm_ckpts/qwen2.5-vl-7b-instruct $CK/qwen2.5-vl-7b-instruct
export SYNTHSCOPE_CKPT_DIR=$CK
$PY - <<'EOF'
import os
import sys
from pathlib import Path
sys.path[:0] = [".", "interface"]
from models.image.adb_vlm.adb_vlm import ADB_VLM
root = Path(os.environ["SYNTHSCOPE_CKPT_DIR"])
for name, repo, rev in (("adb_vlm", ADB_VLM.hf_repo, ADB_VLM.hf_revision),
                        (ADB_VLM.base_model, ADB_VLM.base_hf_repo, ADB_VLM.base_hf_revision)):
    folder = root / name
    sums = dict(line.split()[::-1] for line in (folder / "SHA256SUMS").read_text().splitlines())
    meta = sorted((folder / ".cache/huggingface/download").glob("*.metadata"))
    assert meta, f"no hf download metadata in {folder}"
    for m in meta:
        commit, etag = m.read_text().splitlines()[:2]
        file = m.name[:-len(".metadata")]
        assert commit == rev, (file, commit, rev)
        if file.endswith(".safetensors"):
            assert etag == sums[file], (file, etag, sums[file])
    print(f"{name}: {len(meta)} files from {repo} @ {rev}; safetensors etags == SHA256SUMS")
EOF
for d in adb_vlm qwen2.5-vl-7b-instruct; do (cd -P $CK/$d && sha256sum -c --quiet SHA256SUMS) || { echo "$d sha256 FAILED"; exit 4; }; echo "$d sha256 OK"; done

echo "=== WildRF full $(date +%T) ==="
stage_dir() {   # name (for the lock / sha1 files), ssh host, source folder, destination folder
  local name=$1 host=$2 folder=$3 dst=$4
  exec 8>$V/locks/full_$name.lock; flock -w 7200 8
  mkdir -p $dst
  rsync -a $host:$folder/ $dst/
  ssh $host "cd $folder && find . -type f -print0 | sort -z | xargs -0 sha1sum" > $V/full_$name.src.sha1
  (cd $dst && find . -type f -print0 | sort -z | xargs -0 sha1sum) > $V/full_$name.dst.sha1
  cmp $V/full_$name.src.sha1 $V/full_$name.dst.sha1 || { echo "full $name differs from $host:$folder"; exit 4; }
  echo "full $name: $(wc -l < $V/full_$name.dst.sha1) files sha1-identical to $host:$folder"
  flock -u 8
}
stage_dir wildrf ssalehi@ivi-cn029 $SRC/WildRF/test $V/full/wildrf
FULL=$V/full_data_$SLURM_JOB_ID.json
echo "{\"wildrf\": \"$V/full/wildrf\"}" > $FULL

echo "=== VLM float32 + eager $(date +%T) ==="
set +e
$PY repro/vlm_bench/run_units.py --units vlm:wildrf:fullfp32 --subsets-root $V/subsets --cache-root $V/cache \
    --vlm-model adb_vlm --out-root $WT/repro/adb_full/vlm --lock-dir $V/locks --python $PY --full-data $FULL
rc=$?
echo "=== run_units rc=$rc $(date)"
cd / && rm -rf $W $CK
exit $rc
