#!/bin/bash
#SBATCH --job-name=adbfull_bench
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=repro/logs/adbfull_bench_%j.out     # relative to the submit directory: submit from the repo root
# Align, Don't Blend full method (adb_full) on the paper's benchmarks, on two RTX A5000 of an IVI cees node (the GPU model of
# the committed adb_vlm runs):
#   sbatch --nodelist=ivi-cn027 repro/adb_full/bench_a5000.sbatch
# 1. stage: the adb_vlm weights from ivi-cn029 (checked: pinned revisions + sha256), dda.pth (sha256 = HF DDA_ckpt.pth), the
#    vlm_bench subsets (genimage_val100k, aigcdetect17, aigibench25, bfree_online; sha1-verified against their manifests) and
#    the full folders WildRF/test + BFreeOnline/test (ivi-cn029) and AIGIBench/test/{CommunityAI,SocialRF} (ivi-cn023), each
#    sha1-list-compared with its original (needs keep029 / keep023);
# 2. migrate_vlm_cache.py --model adb_vlm on this node's caches (rule "unchanged": main e5749ae -> the current code version), so
#    images scored before are read back, not scored again;
# 3. GPU 0: dda per image (bench_eval --per-image) on every row's images; then a second VLM worker;
#    GPU 1: the VLM units (run_units.py, bf16 greedy): aigibench25 SocialRF full, CommunityAI first 1,500 + 1,500, BFree-Online
#    full. Results: repro/adb_full/{dda,vlm}/<bench>/; logs: this file and adbfull_bench_<job>_w2.out (the second worker).
# Then on a login node: python repro/adb_full/fuse.py | tee repro/logs/adb_full_fuse.out
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SRC=/ssdstore/ssalehi/SynthScope                         # ivi-cn029: weights + cn029-site originals
SRC23=/hddstore/ssalehi/SynthScope                       # ivi-cn023: AIGIBench originals, dda.pth
DDA_SHA=b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/adb_full/fuse.py" ] || { echo "$WT is not the SynthScope repo root"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
S=""   # the node-local folder of the earlier vlm_bench jobs (its caches), else the first scratch disk
for d in /ssdstore /nvmestore /hddstore; do [ -d $d/ssalehi/SynthScope/vpp ] && { S=$d/ssalehi/SynthScope; break; }; done
if [ -z "$S" ]; then for d in /ssdstore /nvmestore /hddstore; do [ -d $d ] && { S=$d/ssalehi/SynthScope; break; }; done; fi
: "${S:?no local scratch on $(hostname -s)}"
mkdir -p $S/vpp
V=$S/vpp
df -h $S | tail -1
[ "$(df --output=avail -BG $S | tail -1 | tr -dc 0-9)" -gt 60 ] || { echo "less than 60 GB free on $S"; exit 4; }
SHA=$(git -C $WT rev-parse HEAD)
W=$V/code_$SLURM_JOB_ID
rm -rf $W && mkdir -p $W $V/subsets $V/cache $V/locks $V/full && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })      # the job's two GPUs, as slurm names them
[ ${#GPUS[@]} -ge 2 ] || { echo "needs two GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"; exit 4; }
G0=${GPUS[0]}; G1=${GPUS[1]}
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
if [ -f $S/checkpoints/dda.pth ]; then ln -sfn $S/checkpoints/dda.pth $CK/dda.pth
else rsync -a ssalehi@ivi-cn029:$SRC/checkpoints/dda.pth $V/dda.pth && ln -sfn $V/dda.pth $CK/dda.pth; fi
echo "$DDA_SHA  $CK/dda.pth" | sha256sum -c --quiet && echo "dda.pth sha256 == HF DDA_ckpt.pth"
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

echo "=== subsets $(date +%T) ==="
for b in genimage_val100k aigcdetect17 aigibench25 bfree_online; do
  read -r site data < <($PY -c "
import json; b=[x for x in json.load(open('repro/vlm_bench/spec.json'))['benchmarks'] if x['key']=='$b'][0]; print(b['site'], b['data'])")
  if [ "$site" = 023 ]; then SRCH=ssalehi@ivi-cn023:$SRC23; else SRCH=ssalehi@ivi-cn029:$SRC; fi
  d=$V/subsets/$b
  exec 8>$V/locks/subset_$b.lock; flock -w 7200 8
  if [ -f $d/.verified ] && [ "$(cat $d/.verified)" = "$(sha1sum repro/vlm_bench/subsets/$b.json | cut -c1-40)" ]; then
    echo "subset $b: verified earlier"
  else
    mkdir -p $d/data
    $PY repro/vlm_bench/make_subsets.py files $b data > $d/files_data.txt
    rsync -a --files-from=$d/files_data.txt $SRCH/$data/ $d/data/
    $PY repro/vlm_bench/make_subsets.py verify $b $d 2>&1 | grep -v -i warning
    sha1sum repro/vlm_bench/subsets/$b.json | cut -c1-40 > $d/.verified
  fi
  flock -u 8
done

echo "=== full folders $(date +%T) ==="
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
stage_dir bfree_online ssalehi@ivi-cn029 $SRC/BFreeOnline/test $V/full/bfree_online
for g in CommunityAI SocialRF; do
  stage_dir aigibench25_$g ssalehi@ivi-cn023 $SRC23/AIGIBench/test/$g $V/full/aigibench25/$g
done
FULL=$V/full_data_$SLURM_JOB_ID.json
echo "{\"wildrf\": \"$V/full/wildrf\", \"bfree_online\": \"$V/full/bfree_online\", \"aigibench25\": \"$V/full/aigibench25\"}" > $FULL
cat $FULL

echo "=== migrate the adb_vlm caches to the current code version $(date +%T) ==="
$PY repro/vlm_bench/migrate_vlm_cache.py --cache-root $V/cache --model adb_vlm 2>&1 | grep -v -i warning

OUT=$WT/repro/adb_full
UNITS1=vlm:aigibench25:full:SocialRF,vlm:aigibench25:limit1500:CommunityAI,vlm:bfree_online:full
UNITS2=vlm:aigibench25:limit1500:CommunityAI,vlm:bfree_online:full,vlm:aigibench25:full:SocialRF
echo "=== VLM worker 1 (GPU 1) $(date +%T): $UNITS1 ==="
CUDA_VISIBLE_DEVICES=$G1 $PY repro/vlm_bench/run_units.py --units $UNITS1 --subsets-root $V/subsets --cache-root $V/cache \
    --vlm-model adb_vlm --out-root $OUT/vlm --lock-dir $V/locks --python $PY --full-data $FULL &
W1=$!

echo "=== dda per image (GPU 0) $(date +%T) ==="
dda() {   # bench, stem, data, extra bench_eval args...
  local bench=$1 stem=$2 data=$3; shift 3
  mkdir -p $OUT/dda/$bench
  [ -f $OUT/dda/$bench/$stem.json ] && { echo "dda $bench/$stem: exists"; return; }
  CUDA_VISIBLE_DEVICES=$G0 $PY benchmark/bench_eval.py --model dda --benchmark pergen --data $data --per-image --bs 32 \
    --tag adbfull_dda_${bench}_$stem --out $OUT/dda/$bench/$stem.json --results-log $OUT/dda/RESULTS.jsonl "$@" 2>&1 \
    | grep -v -i "warn\|it/s\]"
}
dda wildrf dda__full $V/full/wildrf
dda genimage_val100k dda__jpegfakes96 $V/subsets/genimage_val100k/data --limit 63 --jpeg-fakes 96
dda aigcdetect17 dda__jpegfakes96 $V/subsets/aigcdetect17/data --limit 30 --jpeg-fakes 96
dda aigibench25 dda $V/subsets/aigibench25/data --limit 20 --select CommunityAI,SocialRF
dda aigibench25 dda__limit1500__CommunityAI $V/full/aigibench25 --limit 1500 --select CommunityAI
dda aigibench25 dda__full__SocialRF $V/full/aigibench25 --select SocialRF
dda bfree_online dda $V/subsets/bfree_online/data --limit 500
dda bfree_online dda__full $V/full/bfree_online

echo "=== VLM worker 2 (GPU 0) $(date +%T): $UNITS2 ==="
set +e
CUDA_VISIBLE_DEVICES=$G0 $PY repro/vlm_bench/run_units.py --units $UNITS2 --subsets-root $V/subsets --cache-root $V/cache \
    --vlm-model adb_vlm --out-root $OUT/vlm --lock-dir $V/locks --python $PY --full-data $FULL \
    > $WT/repro/logs/adbfull_bench_${SLURM_JOB_ID}_w2.out 2>&1
rc2=$?
wait $W1
rc1=$?
echo "=== run_units rc: worker 1 $rc1, worker 2 $rc2 $(date)"
cd / && rm -rf $W $CK
[ $rc1 = 0 ] || [ $rc2 = 0 ]
