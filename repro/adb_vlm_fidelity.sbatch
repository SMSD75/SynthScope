#!/bin/bash
#SBATCH --job-name=adbfid
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=repro/logs/adb_vlm_fidelity_%j.out     # relative to the submit directory: submit from the repo root
# Align, Don't Blend VLM fidelity (repro/adb_vlm_fidelity.py) on one cees A5000:
#   sbatch --nodelist=ivi-cn026 repro/adb_vlm_fidelity.sbatch
# The base model, the adapter and the 16 images live on ivi-cn029 (/ssdstore/ssalehi/SynthScope). On ivi-cn029 they are
# used in place; on another node they are copied to its local scratch over ssh (needs one of our jobs on ivi-cn029, e.g.
# keep029). Then: the pinned revisions + sha256 of both folders, the official predict.py (ms-swift 4.4.2, pt backend), the
# same swift command with --logprobs, and the comparison (reference transformers + peft call, engine). Any failed check
# stops the job (set -e). The checkout's clean HEAD is copied to scratch first; the JSON is written into the checkout.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SRC=/ssdstore/ssalehi/SynthScope                        # ivi-cn029: weights + images
G=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared
SWIFT_BIN=$G/vlm_gate/bin/swift                         # ms-swift 4.4.2 entry point (pip --target in vlm_pydeps/swift442)
REF=$G/vlm_ref/Align-Dont-Blend                         # official code, commit f4b4794
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python        # the SynthScope env (symlinked as envs/SynthScope)
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/adb_vlm_fidelity.py" ] || { echo "$WT is not the SynthScope repo root (submit from it, or set SYNTHSCOPE_REPO)"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
[ "$(git -C $REF rev-parse --short=7 HEAD)" = f4b4794 ] || { echo "official clone is not at f4b4794"; exit 5; }
ONSRC=0
if [ "$(hostname -s)" = ivi-cn029 ]; then S=$SRC; ONSRC=1
else
  for d in /ssdstore /nvmestore /hddstore; do [ -d $d ] && { S=$d/ssalehi/SynthScope; break; }; done
  : "${S:?no local scratch on $(hostname -s)}"
fi
CK=$S/vlm_ckpts
J=$S/adb_fid/$SLURM_JOB_ID
SHA=$(git -C $WT rev-parse HEAD)
W=$J/code
mkdir -p $CK $W $J/ckpt $J/data $J/cache $J/tmp && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ $ONSRC = 0 ]; then
  for d in qwen2.5-vl-7b-instruct align_dont_blend_vlm; do
    echo "staging $d from ivi-cn029 $(date +%T)"
    ( flock -w 7200 9; rsync -a --partial --exclude=_hf_home --exclude=_tmp ssalehi@ivi-cn029:$SRC/vlm_ckpts/$d $CK/ ) 9>$CK/.stage.lock
  done
fi
ln -sfn $CK/align_dont_blend_vlm $J/ckpt/adb_vlm
ln -sfn $CK/qwen2.5-vl-7b-instruct $J/ckpt/qwen2.5-vl-7b-instruct
export SYNTHSCOPE_CKPT_DIR=$J/ckpt HF_HOME=$J/cache/hf TMPDIR=$J/tmp XDG_CACHE_HOME=$J/cache MODELSCOPE_CACHE=$J/cache/modelscope
cd $W
export PYTHONPATH=$W     # the env has the main SynthScope checkout installed in editable mode: import this copy instead

echo "=== weights: pinned revisions (hf download metadata) + sha256 ==="
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
        if file.endswith(".safetensors"):        # an LFS file's etag is its sha256
            assert etag == sums[file], (file, etag, sums[file])
    print(f"{name}: {len(meta)} files from {repo} @ {rev}; safetensors etags == SHA256SUMS")
EOF
for d in adb_vlm qwen2.5-vl-7b-instruct; do (cd -P $J/ckpt/$d && sha256sum -c --quiet SHA256SUMS) && echo "$d sha256 OK"; done

echo "=== images ==="
$PY -c "import sys; sys.path[:0]=['.','interface']; from repro.adb_vlm_fidelity import IMAGES; print('\n'.join(r for r, _ in IMAGES))" > $J/files.txt
if [ $ONSRC = 1 ]; then rsync -a --files-from=$J/files.txt $SRC/ $J/data/
else rsync -a --files-from=$J/files.txt ssalehi@ivi-cn029:$SRC/ $J/data/; fi
sed "s|^|$J/data/|" $J/files.txt > $J/inputs.txt
(cd $J/data && sha1sum $(cat $J/files.txt))
$PY -c "
from PIL import Image
for p in open('$J/inputs.txt').read().split():
    im = Image.open(p); print(im.size, im.mode, p.split('/data/', 1)[1])"

echo "=== official: predict.py (unmodified), ms-swift 4.4.2 pt backend $(date +%T) ==="
t0=$(date +%s)
(cd $REF && $PY predict.py --input $J/inputs.txt --output $J/official.jsonl --vlm-adapter $J/ckpt/adb_vlm \
   --vlm-base-model $J/ckpt/qwen2.5-vl-7b-instruct --infer-backend pt --swift-bin $SWIFT_BIN)
SWIFT_S=$(( $(date +%s) - t0 ))
echo "predict.py: ${SWIFT_S}s (model load included)"
echo "=== official command again with --logprobs true $(date +%T) ==="
$PY repro/adb_vlm_fidelity.py swift-logprobs --inputs $J/inputs.txt --out-dir $J/swift_lp --swift-bin $SWIFT_BIN \
   --official-dir $REF
echo "=== reference + engine comparison $(date +%T) ==="
$PY repro/adb_vlm_fidelity.py compare --data $J/data --swift-official $J/official.jsonl \
   --swift-logprobs $J/swift_lp/vlm_result.jsonl --swift-official-seconds $SWIFT_S --official-dir $REF \
   --out $WT/repro/adb_vlm_fidelity.json
echo "=== smoke test with the released files ([B] rendered request, [G] strict load + one image) $(date +%T) ==="
$PY tests/test_adb_vlm_smoke.py 2>&1 | grep -v -i "warn\|nvml"
echo "done $(date)"
cd / && rm -rf $W $J/tmp $J/cache && echo "removed the code copy and caches (kept $J/data, official.jsonl, swift_lp)"
