#!/bin/bash
#SBATCH --job-name=adbsync
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:3
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=repro/logs/adbsync_%j.out     # relative to the submit directory: submit from the repo root
# Align, Don't Blend full method (adb_full) re-measured after the upstream sync to GzZning/Align-Dont-Blend @ cf168db (the
# DPO user prompt, max_pixels 1,210,000, repetition penalty 1.0, unparsed answers Real), on three RTX A5000 of ivi-cn029 (the
# GPU model of every earlier adb_full row), where all the data and the adb_vlm weights are local:
#   sbatch --nodelist=ivi-cn029 repro/adb_full/upstream_sync_a5000.sbatch
# Only the VLM branch changed: the DDA records of the earlier rows are reused (same files, sha1-checked by fuse.py); new DDA
# records only for the two new samples (AIGCDetect 100 + 100 per generator, AIGI-Bench CommunityAI full).
# 1. weights: pinned revisions + sha256; dda.pth sha256 = HF DDA_ckpt.pth;
# 2. GPU 0: dda per image for the new samples, then a VLM worker; GPUs 1, 2: VLM workers. The VLM units (run_units.py, bf16
#    greedy, the registry config) are shared by the three workers through their locks: BFree-Online full, WildRF full (per
#    platform), GenImage 625 + 625 per generator (fakes JPEG q96; the same sample as before), AIGCDetect 100 + 100 per
#    generator (fakes JPEG q96; the earlier 30-per-class subset is a prefix), AIGI-Bench SocialRF full (worker 2 first) and
#    CommunityAI first 1,500 + 1,500 then full (worker 3 only; the 1,500 + 1,500 come back from the cache).
#    Results: repro/adb_full/vlm_cf168db/<bench>/; logs: this file and adbsync_<job>_w{2,3}.out.
#    Then: python repro/adb_full/fuse.py | tee repro/logs/adb_full_fuse.out
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
S=/ssdstore/ssalehi/SynthScope
DDA_SHA=b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python
[ "$(hostname -s)" = ivi-cn029 ] || { echo "run on ivi-cn029 (the data and the weights are local there)"; exit 4; }
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
[ ${#GPUS[@]} -ge 3 ] || { echo "needs three GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"; exit 4; }
G0=${GPUS[0]}; G1=${GPUS[1]}; G2=${GPUS[2]}
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
sys.path[:0] = [".", "interface", "benchmark"]
from models.image.adb_vlm.adb_vlm import ADB_VLM
import bench_eval
from interface.backend.registry import get_spec
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
print("adb_vlm cache identity:", bench_eval.vlm_identity(get_spec("adb_vlm"), "greedy"))
EOF
for d in adb_vlm qwen2.5-vl-7b-instruct; do (cd -P $CK/$d && sha256sum -c --quiet SHA256SUMS) || { echo "$d sha256 FAILED"; exit 4; }; echo "$d sha256 OK"; done

FULL=$V/full_data_$SLURM_JOB_ID.json
cat > $FULL <<EOF
{"wildrf": "$S/WildRF/test", "bfree_online": "$S/BFreeOnline/test", "aigibench25": "$S/AIGIBench/test",
 "genimage_val100k": "$S/GenImage_val_full/test", "aigcdetect17": "$S/AIGCDetect/test"}
EOF
cat $FULL

OUT=$WT/repro/adb_full
GI="adm biggan glide midjourney sdv1.4 sdv1.5 vqdm wukong"
AD="ADM DALLE2 Glide Midjourney VQDM biggan cyclegan gaugan progan sd_xl stable_diffusion_v_1_4 stable_diffusion_v_1_5 stargan stylegan stylegan2 whichfaceisreal wukong"
GENERAL="vlm:bfree_online:full,vlm:wildrf:full:facebook,vlm:wildrf:full:reddit,vlm:wildrf:full:twitter"
for g in $GI; do GENERAL="$GENERAL,vlm:genimage_val100k:limit625_jpegfakes96:$g"; done
for g in $AD; do GENERAL="$GENERAL,vlm:aigcdetect17:limit100_jpegfakes96:$g"; done
SOCIAL=vlm:aigibench25:full:SocialRF
COMM1500=vlm:aigibench25:limit1500:CommunityAI
COMMFULL=vlm:aigibench25:full:CommunityAI   # only worker 3: its first 1,500 + 1,500 are COMM1500's (cached), never scored twice
# per-worker order (a unit is run by whoever takes its lock first): the two long AIGI-Bench units start at once on workers 2
# and 3, the rest goes to worker 1 and to whoever is free
U1="$GENERAL,$SOCIAL,$COMM1500"
U2="$SOCIAL,$GENERAL,$COMM1500"
U3="$COMM1500,$COMMFULL,$GENERAL,$SOCIAL"
run_worker() {   # gpu, units, log (or "-" for this file)
  local gpu=$1 units=$2 log=$3
  if [ "$log" = - ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY repro/vlm_bench/run_units.py --units $units --subsets-root $V/subsets --cache-root $V/cache \
        --vlm-model adb_vlm --out-root $OUT/vlm_cf168db --lock-dir $V/locks --python $PY --full-data $FULL
  else
    CUDA_VISIBLE_DEVICES=$gpu $PY repro/vlm_bench/run_units.py --units $units --subsets-root $V/subsets --cache-root $V/cache \
        --vlm-model adb_vlm --out-root $OUT/vlm_cf168db --lock-dir $V/locks --python $PY --full-data $FULL > $log 2>&1
  fi
}
echo "=== VLM worker 2 (GPU $G1): $U2"
echo "=== VLM worker 3 (GPU $G2): $U3 $(date +%T)"
set +e
run_worker $G1 $U2 $WT/repro/logs/adbsync_${SLURM_JOB_ID}_w2.out &
W2=$!
sleep 120                                       # stagger the model loads
run_worker $G2 $U3 $WT/repro/logs/adbsync_${SLURM_JOB_ID}_w3.out &
W3=$!
set -e

echo "=== dda per image, the new samples (GPU $G0) $(date +%T) ==="
dda() {   # bench, stem, data, extra bench_eval args...
  local bench=$1 stem=$2 data=$3; shift 3
  mkdir -p $OUT/dda/$bench
  [ -f $OUT/dda/$bench/$stem.json ] && { echo "dda $bench/$stem: exists"; return; }
  CUDA_VISIBLE_DEVICES=$G0 $PY benchmark/bench_eval.py --model dda --benchmark pergen --data $data --per-image --bs 32 \
    --tag adbfull_dda_${bench}_$stem --out $OUT/dda/$bench/$stem.json --results-log $OUT/dda/RESULTS.jsonl "$@" 2>&1 \
    | grep -v -i "warn\|it/s\]"
}
dda aigcdetect17 dda__limit100_jpegfakes96 $S/AIGCDetect/test --limit 100 --jpeg-fakes 96
dda aigibench25 dda__full__CommunityAI $S/AIGIBench/test --select CommunityAI

echo "=== VLM worker 1 (GPU $G0) $(date +%T): $U1 ==="
set +e
run_worker $G0 $U1 -
rc1=$?
wait $W2; rc2=$?
wait $W3; rc3=$?
echo "=== run_units rc: worker 1 $rc1, worker 2 $rc2, worker 3 $rc3 $(date)"
cd / && rm -rf $W $CK
[ $rc1 = 0 ] || [ $rc2 = 0 ] || [ $rc3 = 0 ]
