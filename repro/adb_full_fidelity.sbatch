#!/bin/bash
#SBATCH --job-name=adbfull
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=repro/logs/adb_full_fidelity_%j.out     # relative to the submit directory: submit from the repo root
# Align, Don't Blend FULL method (VLM OR DDA) fidelity (repro/adb_full_fidelity.py) on one 48 GB RTX A6000 (the README's
# float32 command needs about 33 GB):
#   sbatch --nodelist=ivi-cn023 repro/adb_full_fidelity.sbatch
# The base model, the adapter and the images live on ivi-cn029 (/ssdstore/ssalehi/SynthScope): copied to this node's
# scratch over ssh (needs one of our jobs on ivi-cn029, e.g. keep029). dda.pth (= the official DDA_ckpt.pth) is read from
# the node's zoo / checkpoint folder and its sha256 checked. Steps: weights (pinned revisions + sha256), the pool of
# disagreement candidates scored by SynthScope's dda (select), the official predict.py (unmodified; ms-swift 4.4.2, pt
# backend, bf16) with the README's expert command, the official expert alone (probabilities), the official predict.py
# with the README's full float32 / eager command, then the engine comparison. Any failed step stops the job (set -e).
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SRC=/ssdstore/ssalehi/SynthScope                        # ivi-cn029: weights + images
G=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared
SWIFT_BIN=$G/vlm_gate/bin/swift                         # ms-swift 4.4.2 entry point (pip --target in vlm_pydeps/swift442)
REF=$G/vlm_ref/Align-Dont-Blend-cf168db                 # official code, commit cf168db (the 5d83e40 run: job 296802)
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python        # the SynthScope env (symlinked as envs/SynthScope)
DDA_SHA=b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e   # HF Junwei-Xi/Dual-Data-Alignment DDA_ckpt.pth (LFS)
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/adb_full_fidelity.py" ] || { echo "$WT is not the SynthScope repo root (submit from it, or set SYNTHSCOPE_REPO)"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
[ "$(git -C $REF rev-parse --short=7 HEAD)" = cf168db ] || { echo "official clone is not at cf168db"; exit 5; }
[ -z "$(git -C $REF status --short)" ] || { echo "official clone is modified"; exit 5; }
ONSRC=0
if [ "$(hostname -s)" = ivi-cn029 ]; then S=$SRC; ONSRC=1
else
  # the node-local SynthScope folder that already holds checkpoints (ivi-cn023: /hddstore; its /ssdstore is nearly full)
  for d in /ssdstore /nvmestore /hddstore; do
    [ -d $d/ssalehi/SynthScope/checkpoints ] || [ -d $d/ssalehi/SynthScope/zoo_ckpts ] && { S=$d/ssalehi/SynthScope; break; }
  done
  : "${S:?no local scratch on $(hostname -s)}"
fi
CK=$S/vlm_ckpts
J=$S/adbfull_fid/$SLURM_JOB_ID
SHA=$(git -C $WT rev-parse HEAD)
W=$J/code
mkdir -p $CK $W $J/ckpt $J/data $J/cache $J/tmp && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
echo "official=$(git -C $REF log --oneline -1 | cut -c1-80)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ $ONSRC = 0 ]; then
  for d in qwen2.5-vl-7b-instruct align_dont_blend_vlm; do
    echo "staging $d from ivi-cn029 $(date +%T)"
    ( flock -w 7200 9; rsync -a --partial --exclude=_hf_home --exclude=_tmp ssalehi@ivi-cn029:$SRC/vlm_ckpts/$d $CK/ ) 9>$CK/.stage.lock
  done
fi
DDA=""
for f in $S/zoo_ckpts/dda.pth $S/checkpoints/dda.pth; do [ -f $f ] && { DDA=$f; break; }; done
: "${DDA:?no dda.pth on $(hostname -s)}"
echo "$DDA_SHA  $DDA" | sha256sum -c --quiet && echo "dda.pth sha256 == HF DDA_ckpt.pth ($DDA)"
ln -sfn $CK/align_dont_blend_vlm $J/ckpt/adb_vlm
ln -sfn $CK/qwen2.5-vl-7b-instruct $J/ckpt/qwen2.5-vl-7b-instruct
ln -sfn $DDA $J/ckpt/dda.pth
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

echo "=== images: the 16 adb_vlm fidelity images + the GenImage disagreement pool ==="
$PY -c "
import sys; sys.path[:0] = ['.', 'interface']
from repro.adb_full_fidelity import IMAGES, pool
print('\n'.join([r for r, _ in IMAGES] + [c['image'] for c in pool()]))" > $J/files.txt
if [ $ONSRC = 1 ]; then rsync -a --files-from=$J/files.txt $SRC/ $J/data/
else rsync -a --files-from=$J/files.txt ssalehi@ivi-cn029:$SRC/ $J/data/; fi
(cd $J/data && sha1sum $(cat $J/files.txt))
$PY repro/adb_full_fidelity.py select --data $J/data --out $J/selection.json --inputs $J/inputs.txt
$PY -c "import json,sys; a,b=(json.load(open(f))['selected'] for f in sys.argv[1:]); assert a==b, (a,b); print('the 8 selected images == the committed selection')" \
   $J/selection.json $WT/repro/adb_full_fidelity_official/selection.json
$PY -c "
from PIL import Image
for p in open('$J/inputs.txt').read().split():
    im = Image.open(p); print(im.size, im.mode, p.split('/data/', 1)[1])"

export PATH=$(dirname $PY):$PATH            # "python" in the README's expert command = the SynthScope env's interpreter
EXPERT="python -m detector.expert --checkpoint $J/ckpt/dda.pth"
echo "=== official: predict.py (unmodified), ms-swift 4.4.2 pt backend (bf16) + the expert command $(date +%T) ==="
t0=$(date +%s)
(cd $REF && $PY predict.py --input $J/inputs.txt --output $J/official_bf16.jsonl --vlm-adapter $J/ckpt/adb_vlm \
   --vlm-base-model $J/ckpt/qwen2.5-vl-7b-instruct --infer-backend pt --swift-bin $SWIFT_BIN --expert-command "$EXPERT")
T_BF16=$(( $(date +%s) - t0 ))
echo "=== official expert alone (probabilities) $(date +%T) ==="
(cd $REF && $PY -m detector.expert --checkpoint $J/ckpt/dda.pth --input $J/inputs.txt --output $J/official_expert.jsonl)
echo "=== official: predict.py with the README command (float32, eager, device_map auto) $(date +%T) ==="
t0=$(date +%s)
(cd $REF && $PY predict.py --input $J/inputs.txt --output $J/official_fp32.jsonl --vlm-adapter $J/ckpt/adb_vlm \
   --vlm-base-model $J/ckpt/qwen2.5-vl-7b-instruct --infer-backend pt --torch-dtype float32 --attn-impl eager \
   --device-map auto --swift-bin $SWIFT_BIN --expert-command "$EXPERT")
T_FP32=$(( $(date +%s) - t0 ))
echo "predict.py wall seconds (model loads included): bf16 ${T_BF16}s, fp32 ${T_FP32}s"
mkdir -p $WT/repro/adb_full_fidelity_official
cp $J/official_bf16.jsonl $J/official_fp32.jsonl $J/official_expert.jsonl $J/selection.json $WT/repro/adb_full_fidelity_official/
echo "=== engine comparison $(date +%T) ==="
$PY repro/adb_full_fidelity.py compare --data $J/data --selection $J/selection.json \
   --official-bf16 $J/official_bf16.jsonl --official-fp32 $J/official_fp32.jsonl --expert-probs $J/official_expert.jsonl \
   --official-seconds "{\"bf16\": $T_BF16, \"fp32\": $T_FP32}" --out $WT/repro/adb_full_fidelity.json
echo "done $(date)"
cd / && rm -rf $W $J/tmp $J/cache && echo "removed the code copy and caches (kept $J/data and the official outputs)"
