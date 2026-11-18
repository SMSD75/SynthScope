#!/bin/bash
#SBATCH --job-name=vpp_ivi
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/vpp_ivi_%j.out
# VLM benchmark units (Veritas++, or VLM_MODEL=<registry key>, e.g. adb_vlm) on an IVI cees node (repro/vlm_bench/run_units.py),
# weights and data node-local:
#   UNITS=vlm:genimage_val100k,vlm:aigcdetect17 sbatch --nodelist=ivi-cn029 repro/vlm_bench/vpp_ivi.sbatch
#   VLM_MODEL=adb_vlm UNITS=... sbatch --job-name=adb_ivi --output=<logs>/adb_ivi_%j.out --nodelist=ivi-cn026 repro/vlm_bench/vpp_ivi.sbatch
# Weights: the model's folder (veritaspp; adb_vlm = the LoRA adapter folder align_dont_blend_vlm) and, for a LoRA, its base
# folder (qwen2.5-vl-7b-instruct), linked into a per-job checkpoint folder under the registry names.
# On ivi-cn029 the weights and originals are used in place. On another node the job picks its local scratch (/ssdstore,
# /nvmestore or /hddstore) and copies the weights folders and the subset files from ivi-cn029 over ssh (needs one of our jobs
# there, e.g. keep029); the checks below then run on the copies. A :full unit's full benchmark folder (full_data_cn029.json)
# is copied the same way and its sha1 list compared with the original's.
# Submit from the repo root (or set SYNTHSCOPE_REPO). The checkout's clean HEAD is copied to /ssdstore first (later edits of the worktree do not change a running job). Subsets: the
# manifest's files (repro/vlm_bench/subsets/<bench>.json) copied from the originals (cn029, or cn023 for its sets) and
# sha1-verified. Per-image VLM caches, locks and staged subsets stay under /ssdstore/ssalehi/SynthScope/vpp; result JSONs go to the
# worktree's repro/vlm_bench/results/ (or $VLM_OUT_ROOT, e.g. a folder outside the checkout for a rerun of units whose committed
# JSONs must stay in place). Any failed check (weights revision / sha256, subset verification) stops the job.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SRC=/ssdstore/ssalehi/SynthScope                        # ivi-cn029: weights + cn029-site originals
VLM_MODEL=${VLM_MODEL:-veritaspp}
case $VLM_MODEL in                                      # <registry name>:<folder under vlm_ckpts on ivi-cn029>
  veritaspp) WDIRS="veritaspp:veritaspp" ;;
  adb_vlm)   WDIRS="adb_vlm:align_dont_blend_vlm qwen2.5-vl-7b-instruct:qwen2.5-vl-7b-instruct" ;;
  *) echo "VLM_MODEL=$VLM_MODEL: no weight folders known"; exit 4 ;;
esac
ONSRC=0
if [ "$(hostname -s)" = ivi-cn029 ]; then S=$SRC; ONSRC=1
else
  for d in /ssdstore /nvmestore /hddstore; do [ -d $d ] && { S=$d/ssalehi/SynthScope; break; }; done
  : "${S:?no local scratch on $(hostname -s)}"
  mkdir -p $S/vlm_ckpts
  for wd in $WDIRS; do
    f=${wd#*:}
    echo "staging the weights from ivi-cn029 to $S/vlm_ckpts/$f $(date +%T)"
    ( flock -w 7200 9; rsync -a --partial --exclude=_hf_home --exclude=_tmp ssalehi@ivi-cn029:$SRC/vlm_ckpts/$f $S/vlm_ckpts/ ) \
      9>$S/vlm_ckpts/.stage.lock
    echo "staged $(du -sh $S/vlm_ckpts/$f | cut -f1) $(date +%T)"
  done
fi
mkdir -p $S/vlm_ckpts/_hf_home $S/vlm_ckpts/_tmp
export HF_HOME=$S/vlm_ckpts/_hf_home TMPDIR=$S/vlm_ckpts/_tmp
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python     # the SynthScope env (symlinked as envs/SynthScope)
# The repo root: $SYNTHSCOPE_REPO, else the directory sbatch was run from (sbatch copies this script to a spool dir, so
# its own location is lost), else the checkout this file lives in (run with bash). Not tied to any worktree path.
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/vlm_bench/run_units.py" ] || { echo "$WT is not the SynthScope repo root (submit from it, or set SYNTHSCOPE_REPO)"; exit 4; }
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

CK=$V/ckpt_$SLURM_JOB_ID                 # the registry names -> the node's weight folders
rm -rf $CK && mkdir -p $CK
for wd in $WDIRS; do ln -sfn $S/vlm_ckpts/${wd#*:} $CK/${wd%%:*}; done
export SYNTHSCOPE_CKPT_DIR=$CK
echo "=== $VLM_MODEL weights: pinned revisions (hf download metadata) + sha256 ==="
$PY - $VLM_MODEL <<'EOF'
import os
import sys
from pathlib import Path
sys.path[:0] = [".", "interface"]
from interface.backend.registry import MODELS
spec = MODELS[sys.argv[1]]
cfg = spec.vlm
root = Path(os.environ["SYNTHSCOPE_CKPT_DIR"])
folders = [(spec.checkpoint, cfg.hf_repo, cfg.hf_revision)]
if cfg.base_model:                               # a LoRA: its base folder too
    folders.append((cfg.base_model, cfg.base_hf_repo, cfg.base_hf_revision))
for name, repo, rev in folders:
    folder = root / name
    sums = dict(line.split()[::-1] for line in (folder / "SHA256SUMS").read_text().splitlines())
    meta = sorted((folder / ".cache/huggingface/download").glob("*.metadata"))
    assert meta, f"no hf download metadata in {folder}"
    for m in meta:
        commit, etag = m.read_text().splitlines()[:2]
        file = m.name[:-len(".metadata")]
        assert commit == rev, (file, commit, rev)
        if file.endswith(".safetensors"):        # an LFS file's etag is its sha256
            assert etag == sums[file], (file, etag, sums[file])
    print(f"{name}: {len(meta)} files from {repo} @ {rev}; safetensors etags == SHA256SUMS")
EOF
for wd in $WDIRS; do   # a failing check inside an && list would not trigger set -e: stop explicitly
  (cd -P $CK/${wd%%:*} && sha256sum -c --quiet SHA256SUMS) || { echo "${wd%%:*} sha256 check FAILED"; exit 4; }
  echo "${wd%%:*} sha256 OK"
done

echo "=== subsets ==="
for b in $($PY -c "
import sys; print(' '.join(sorted({u.split(':')[1] for u in sys.argv[1].split(',') if u.startswith('vlm:')})))" "$UNITS"); do
  read -r site data reals < <($PY -c "
import json; b=[x for x in json.load(open('repro/vlm_bench/spec.json'))['benchmarks'] if x['key']=='$b'][0]
print(b['site'], b['data'], b.get('reals', '-'))")
  if [ "$site" = 023 ]; then SRCH=ssalehi@ivi-cn023:/hddstore/ssalehi/SynthScope; LOC=0     # cn023 originals (needs keep023)
  else SRCH=ssalehi@ivi-cn029:$SRC; LOC=$ONSRC; fi
  d=$V/subsets/$b
  exec 8>$V/locks/subset_$b.lock; flock -w 7200 8
  if [ -f $d/.verified ] && [ "$(cat $d/.verified)" = "$(sha1sum repro/vlm_bench/subsets/$b.json | cut -c1-40)" ]; then
    echo "subset $b: verified earlier"; flock -u 8; continue; fi
  mkdir -p $d/data
  $PY repro/vlm_bench/make_subsets.py files $b data > $d/files_data.txt
  if [ $LOC = 1 ]; then rsync -a --files-from=$d/files_data.txt $SRC/$data/ $d/data/
  else rsync -a --files-from=$d/files_data.txt $SRCH/$data/ $d/data/; fi
  if [ "$reals" != - ]; then
    mkdir -p $d/reals
    $PY repro/vlm_bench/make_subsets.py files $b reals > $d/files_reals.txt
    if [ $LOC = 1 ]; then rsync -a --files-from=$d/files_reals.txt $SRC/$reals/ $d/reals/
    else rsync -a --files-from=$d/files_reals.txt $SRCH/$reals/ $d/reals/; fi
  fi
  $PY repro/vlm_bench/make_subsets.py verify $b $d 2>&1 | grep -v -i warning
  sha1sum repro/vlm_bench/subsets/$b.json | cut -c1-40 > $d/.verified
  flock -u 8
done

FULL=repro/vlm_bench/full_data_cn029.json
FULLB=$($PY -c "
import sys; print(' '.join(sorted({u.split(':')[1] for u in sys.argv[1].split(',') if u.startswith('vlm:') and u.split(':')[2:3] == ['full']})))" "$UNITS")
if [ $ONSRC = 0 ] && [ -n "$FULLB" ]; then
  echo "=== full benchmark folders from ivi-cn029: $FULLB ==="
  FULL=$V/full_data_$SLURM_JOB_ID.json
  echo '{"_comment": "node-local copies of full_data_cn029.json folders, sha1 lists equal to the originals"}' > $FULL
  for b in $FULLB; do
    src=$($PY -c "import json; print(json.load(open('repro/vlm_bench/full_data_cn029.json'))['$b'])")
    dst=$V/full/$b
    exec 8>$V/locks/full_$b.lock; flock -w 7200 8
    if [ -f $V/full/.verified_$b ]; then echo "full $b: verified earlier ($(cat $V/full/.verified_$b))"
    else
      mkdir -p $dst
      rsync -a ssalehi@ivi-cn029:$src/ $dst/
      ssh ssalehi@ivi-cn029 "cd $src && find . -type f -print0 | sort -z | xargs -0 sha1sum" > $V/full_$b.src.sha1
      (cd $dst && find . -type f -print0 | sort -z | xargs -0 sha1sum) > $V/full_$b.dst.sha1
      cmp $V/full_$b.src.sha1 $V/full_$b.dst.sha1 || { echo "full $b: sha1 list differs from ivi-cn029:$src"; exit 4; }
      echo "full $b: $(wc -l < $V/full_$b.dst.sha1) files sha1-identical to ivi-cn029:$src"
      date > $V/full/.verified_$b
    fi
    flock -u 8
    $PY -c "import json; p='$FULL'; m=json.load(open(p)); m['$b']='$dst'; json.dump(m, open(p, 'w'))"
  done
fi

echo "=== run_units $(date +%T)"
set +e
$PY repro/vlm_bench/run_units.py --units "$UNITS" --subsets-root $V/subsets --cache-root $V/cache --vlm-model $VLM_MODEL \
    --out-root ${VLM_OUT_ROOT:-$WT/repro/vlm_bench/results} --lock-dir $V/locks --python $PY --full-data $FULL
rc=$?
echo "=== run_units rc=$rc $(date)"
cd / && rm -rf $W $CK
exit $rc
