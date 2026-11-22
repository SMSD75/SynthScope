#!/bin/bash
#SBATCH --job-name=adbfull_gi10k
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=34G
#SBATCH --time=12:00:00
#SBATCH --output=repro/logs/adbfull_gi10k_%j.out     # relative to the submit directory: submit from the repo root
# Align, Don't Blend full method (adb_full) on a 10x larger GenImage sample: the first 625 real + 625 fake per generator
# (sorted; bench_eval --limit 625) of GenImage val, fakes JPEG q96, on two RTX A5000 of ivi-cn029, where GenImage_val_full and
# the adb_vlm weights are local:
#   sbatch --nodelist=ivi-cn029 repro/adb_full/genimage10k_a5000.sbatch
# 1. weights: pinned revisions + sha256; dda.pth sha256 = HF DDA_ckpt.pth;
# 2. the committed 63-per-class subset run (results/genimage_val100k/adb_vlm__jpegfakes96.json, a strict prefix of the 625)
#    seeds this node's VLM cache (repro/adb_full/seed_vlm_cache.py: every file's sha1 checked against the subset manifest);
# 3. GPU 0: dda per image (bench_eval --per-image --limit 625 --jpeg-fakes 96), then a VLM worker; GPU 1: a VLM worker. The
#    VLM units are one per generator (run_units vlm:genimage_val100k:limit625_jpegfakes96:<gen>, bf16 greedy), shared by the
#    two workers through their locks. Results: repro/adb_full/{dda,vlm}/genimage_val100k/; logs: this file and
#    adbfull_gi10k_<job>_w2.out (the second worker). Then: python repro/adb_full/fuse.py | tee repro/logs/adb_full_fuse.out
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
S=/ssdstore/ssalehi/SynthScope
DDA_SHA=b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python
DATA=$S/GenImage_val_full/test
[ "$(hostname -s)" = ivi-cn029 ] || { echo "run on ivi-cn029 (GenImage_val_full and the weights are local there)"; exit 4; }
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/adb_full/fuse.py" ] || { echo "$WT is not the SynthScope repo root"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
V=$S/vpp
SHA=$(git -C $WT rev-parse HEAD)
W=$V/code_$SLURM_JOB_ID
rm -rf $W && mkdir -p $W $V/cache $V/locks && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
[ ${#GPUS[@]} -ge 2 ] || { echo "needs two GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"; exit 4; }
G0=${GPUS[0]}; G1=${GPUS[1]}
cd $W
export PYTHONPATH=$W
mkdir -p $S/vlm_ckpts/_hf_home $S/vlm_ckpts/_tmp
export HF_HOME=$S/vlm_ckpts/_hf_home TMPDIR=$S/vlm_ckpts/_tmp

echo "=== weights $(date +%T) ==="
CK=$V/ckpt_$SLURM_JOB_ID
rm -rf $CK && mkdir -p $CK
ln -sfn $S/vlm_ckpts/align_dont_blend_vlm $CK/adb_vlm
ln -sfn $S/vlm_ckpts/qwen2.5-vl-7b-instruct $CK/qwen2.5-vl-7b-instruct
ln -sfn $S/checkpoints/dda.pth $CK/dda.pth
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

echo "=== seed the VLM cache with the committed 63-per-class subset run $(date +%T) ==="
$PY repro/adb_full/seed_vlm_cache.py --result repro/vlm_bench/results/genimage_val100k/adb_vlm__jpegfakes96.json \
    --manifest genimage_val100k --data $DATA --cache-dir $V/cache/genimage_val100k 2>&1 | grep -v -i warning
FULL=$V/full_data_$SLURM_JOB_ID.json
echo "{\"genimage_val100k\": \"$DATA\"}" > $FULL

OUT=$WT/repro/adb_full
GENS="adm biggan glide midjourney sdv1.4 sdv1.5 vqdm wukong"
UNITS=$(for g in $GENS; do printf 'vlm:genimage_val100k:limit625_jpegfakes96:%s,' $g; done); UNITS=${UNITS%,}
UNITS_REV=$(for g in $(echo $GENS | tr ' ' '\n' | tac); do printf 'vlm:genimage_val100k:limit625_jpegfakes96:%s,' $g; done); UNITS_REV=${UNITS_REV%,}
echo "=== VLM worker 1 (GPU $G1) $(date +%T): $UNITS ==="
CUDA_VISIBLE_DEVICES=$G1 $PY repro/vlm_bench/run_units.py --units $UNITS --subsets-root $V/subsets --cache-root $V/cache \
    --vlm-model adb_vlm --out-root $OUT/vlm --lock-dir $V/locks --python $PY --full-data $FULL &
W1=$!

echo "=== dda per image (GPU $G0) $(date +%T) ==="
mkdir -p $OUT/dda/genimage_val100k
if [ ! -f $OUT/dda/genimage_val100k/dda__limit625_jpegfakes96.json ]; then
  CUDA_VISIBLE_DEVICES=$G0 $PY benchmark/bench_eval.py --model dda --benchmark pergen --data $DATA --per-image --bs 32 \
    --limit 625 --jpeg-fakes 96 --tag adbfull_dda_genimage_val100k_dda__limit625_jpegfakes96 \
    --out $OUT/dda/genimage_val100k/dda__limit625_jpegfakes96.json --results-log $OUT/dda/RESULTS.jsonl 2>&1 \
    | grep -v -i "warn\|it/s\]"
fi

echo "=== VLM worker 2 (GPU $G0) $(date +%T): $UNITS_REV ==="
set +e
CUDA_VISIBLE_DEVICES=$G0 $PY repro/vlm_bench/run_units.py --units $UNITS_REV --subsets-root $V/subsets --cache-root $V/cache \
    --vlm-model adb_vlm --out-root $OUT/vlm --lock-dir $V/locks --python $PY --full-data $FULL \
    > $WT/repro/logs/adbfull_gi10k_${SLURM_JOB_ID}_w2.out 2>&1
rc2=$?
wait $W1
rc1=$?
echo "=== run_units rc: worker 1 $rc1, worker 2 $rc2 $(date)"
cd / && rm -rf $W $CK
[ $rc1 = 0 ] || [ $rc2 = 0 ]
