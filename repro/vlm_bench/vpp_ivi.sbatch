#!/bin/bash
#SBATCH --job-name=vpp_ivi
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/vpp_ivi_%j.out
# Veritas++ benchmark units on an IVI cees node (repro/vlm_bench/run_units.py), weights and data node-local:
#   UNITS=vlm:genimage_val100k,vlm:aigcdetect17 sbatch --nodelist=ivi-cn029 repro/vlm_bench/vpp_ivi.sbatch
# On ivi-cn029 the weights and originals are used in place. On another node the job picks its local scratch (/ssdstore,
# /nvmestore or /hddstore) and copies the weights folder and the subset files from ivi-cn029 over ssh (needs one of our jobs
# there, e.g. keep029); the checks below then run on the copies.
# The worktree's clean HEAD is copied to /ssdstore first (later edits of the worktree do not change a running job). Subsets: the
# manifest's files (repro/vlm_bench/subsets/<bench>.json) copied from the cn029 originals and sha1-verified; only cn029-site
# benchmarks. Per-image VLM caches, locks and staged subsets stay under /ssdstore/ssalehi/SynthScope/vpp; result JSONs go to the
# worktree's repro/vlm_bench/results/. Any failed check (weights revision / sha256, subset verification) stops the job.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SRC=/ssdstore/ssalehi/SynthScope                        # ivi-cn029: weights + cn029-site originals
ONSRC=0
if [ "$(hostname -s)" = ivi-cn029 ]; then S=$SRC; ONSRC=1
else
  for d in /ssdstore /nvmestore /hddstore; do [ -d $d ] && { S=$d/ssalehi/SynthScope; break; }; done
  : "${S:?no local scratch on $(hostname -s)}"
  mkdir -p $S/vlm_ckpts
  echo "staging the weights from ivi-cn029 to $S/vlm_ckpts/veritaspp $(date +%T)"
  ( flock -w 7200 9; rsync -a --partial --exclude=_hf_home --exclude=_tmp ssalehi@ivi-cn029:$SRC/vlm_ckpts/veritaspp $S/vlm_ckpts/ ) \
    9>$S/vlm_ckpts/.stage.lock
  echo "staged $(du -sh $S/vlm_ckpts/veritaspp | cut -f1) $(date +%T)"
fi
mkdir -p $S/vlm_ckpts/_hf_home $S/vlm_ckpts/_tmp
export SYNTHSCOPE_CKPT_DIR=$S/vlm_ckpts HF_HOME=$S/vlm_ckpts/_hf_home TMPDIR=$S/vlm_ckpts/_tmp
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python     # the SynthScope env (symlinked as envs/SynthScope)
WT=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/vlm_wt
V=$S/vpp
: "${UNITS:?UNITS is required}"
SHA=$(git -C $WT rev-parse HEAD)
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
W=$V/code_$SLURM_JOB_ID
rm -rf $W && mkdir -p $W $V/subsets $V/cache $V/locks && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "UNITS=$UNITS"
cd $W
export PYTHONPATH=$W     # the env has the main SynthScope checkout installed in editable mode: import this copy instead

echo "=== Veritas++ weights: pinned revision (hf download metadata) + sha256 ==="
$PY - <<'EOF'
import os
import sys
from pathlib import Path
sys.path[:0] = [".", "interface"]
from models.image.veritaspp.veritaspp import VERITASPP
folder = Path(os.environ["SYNTHSCOPE_CKPT_DIR"]) / "veritaspp"
sums = dict(line.split()[::-1] for line in (folder / "SHA256SUMS").read_text().splitlines())
meta = sorted((folder / ".cache/huggingface/download").glob("*.metadata"))
assert meta, "no hf download metadata in the folder"
for m in meta:
    commit, etag = m.read_text().splitlines()[:2]
    name = m.name[:-len(".metadata")]
    assert commit == VERITASPP.hf_revision, (name, commit, VERITASPP.hf_revision)
    if name.endswith(".safetensors"):
        assert etag == sums[name], (name, etag, sums[name])
print(f"{len(meta)} files from {VERITASPP.hf_repo} @ {VERITASPP.hf_revision}; shard etags == SHA256SUMS")
EOF
(cd $SYNTHSCOPE_CKPT_DIR/veritaspp && sha256sum -c --quiet SHA256SUMS) && echo "sha256 OK"

echo "=== subsets ==="
for b in $($PY -c "
import sys; print(' '.join(sorted({u.split(':')[1] for u in sys.argv[1].split(',') if u.startswith('vlm:')})))" "$UNITS"); do
  read -r site data reals < <($PY -c "
import json; b=[x for x in json.load(open('repro/vlm_bench/spec.json'))['benchmarks'] if x['key']=='$b'][0]
print(b['site'], b['data'], b.get('reals', '-'))")
  [ "$site" = 029 ] || { echo "$b is not a cn029 benchmark"; exit 4; }
  d=$V/subsets/$b
  exec 8>$V/locks/subset_$b.lock; flock -w 7200 8
  if [ -f $d/.verified ] && [ "$(cat $d/.verified)" = "$(sha1sum repro/vlm_bench/subsets/$b.json | cut -c1-40)" ]; then
    echo "subset $b: verified earlier"; flock -u 8; continue; fi
  mkdir -p $d/data
  $PY repro/vlm_bench/make_subsets.py files $b data > $d/files_data.txt
  if [ $ONSRC = 1 ]; then rsync -a --files-from=$d/files_data.txt $SRC/$data/ $d/data/
  else rsync -a --files-from=$d/files_data.txt ssalehi@ivi-cn029:$SRC/$data/ $d/data/; fi
  if [ "$reals" != - ]; then
    mkdir -p $d/reals
    $PY repro/vlm_bench/make_subsets.py files $b reals > $d/files_reals.txt
    if [ $ONSRC = 1 ]; then rsync -a --files-from=$d/files_reals.txt $SRC/$reals/ $d/reals/
    else rsync -a --files-from=$d/files_reals.txt ssalehi@ivi-cn029:$SRC/$reals/ $d/reals/; fi
  fi
  $PY repro/vlm_bench/make_subsets.py verify $b $d 2>&1 | grep -v -i warning
  sha1sum repro/vlm_bench/subsets/$b.json | cut -c1-40 > $d/.verified
  flock -u 8
done

echo "=== run_units $(date +%T)"
set +e
$PY repro/vlm_bench/run_units.py --units "$UNITS" --subsets-root $V/subsets --cache-root $V/cache \
    --out-root $WT/repro/vlm_bench/results --lock-dir $V/locks --python $PY --full-data repro/vlm_bench/full_data_cn029.json
rc=$?
echo "=== run_units rc=$rc $(date)"
cd / && rm -rf $W
exit $rc
