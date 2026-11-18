#!/bin/bash
#SBATCH --job-name=fvlmfid
#SBATCH --partition=cees
#SBATCH --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=repro/logs/fakevlm_fidelity_%j.out     # relative to the submit directory: submit from the repo root
# FakeVLM fidelity (repro/fakevlm_fidelity.py) on one cees A5000:
#   sbatch --nodelist=ivi-cn026 repro/fakevlm_fidelity.sbatch
# The weights, the llava-hf processor files and the images live on ivi-cn029 (/ssdstore/ssalehi/SynthScope). On ivi-cn029
# they are used in place; on another node they are copied to its local scratch over ssh (needs one of our jobs on
# ivi-cn029, e.g. keep029). Then: the pinned revisions + sha256 of both folders, the official eval.py under transformers
# 4.45.2 (the authors' pin, from $DEPS/transformers4452) and under the env's 5.5.4, the official 4.45.2 processor inputs,
# and the comparison (reference transformers call, engine teacher-forced and generated, extended prefix set). Any failed
# check stops the job (set -e). The checkout's clean HEAD is copied to scratch first; the JSON is written into the checkout.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
SRC=/ssdstore/ssalehi/SynthScope                        # ivi-cn029: weights + images
G=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared
DEPS=$G/vlm_pydeps                                      # transformers4452: transformers 4.45.2, tokenizers 0.20.3, hub 0.25.2
REF=$G/vlm_ref/FakeVLM                                  # official code, commit fdcdcf1
PY=/home/ssalehi/miniconda3/envs/AGID/bin/python        # the SynthScope env (symlinked as envs/SynthScope)
PROC=llava-1.5-7b-hf_a272c74_processor
WT=${SYNTHSCOPE_REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}
[ -f "$WT/repro/fakevlm_fidelity.py" ] || { echo "$WT is not the SynthScope repo root (submit from it, or set SYNTHSCOPE_REPO)"; exit 4; }
[ "$(git -C $WT status --short --untracked-files=no | wc -l)" = 0 ] || { echo "worktree has uncommitted tracked files"; exit 5; }
[ "$(git -C $REF rev-parse --short=7 HEAD)" = fdcdcf1 ] || { echo "official clone is not at fdcdcf1"; exit 5; }
ONSRC=0
if [ "$(hostname -s)" = ivi-cn029 ]; then S=$SRC; ONSRC=1
else
  for d in /ssdstore /nvmestore /hddstore; do [ -d $d ] && { S=$d/ssalehi/SynthScope; break; }; done
  : "${S:?no local scratch on $(hostname -s)}"
fi
CK=$S/vlm_ckpts
J=$S/fakevlm_fid/$SLURM_JOB_ID
SHA=$(git -C $WT rev-parse HEAD)
W=$J/code
mkdir -p $CK $W $J/ckpt $J/data $J/cache $J/tmp && git -C $WT archive $SHA | tar -x -C $W
echo "node=$(hostname) job=$SLURM_JOB_ID $(date) code=$(git -C $WT log --oneline -1 $SHA | cut -c1-100) ($SHA, copied to $W)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ $ONSRC = 0 ]; then
  for d in fakevlm $PROC; do
    echo "staging $d from ivi-cn029 $(date +%T)"
    ( flock -w 7200 9; rsync -a --partial --exclude=_hf_home --exclude=_tmp ssalehi@ivi-cn029:$SRC/vlm_ckpts/$d $CK/ ) 9>$CK/.stage.lock
  done
fi
ln -sfn $CK/fakevlm $J/ckpt/fakevlm
ln -sfn $CK/$PROC $J/ckpt/$PROC
export SYNTHSCOPE_CKPT_DIR=$J/ckpt HF_HOME=$J/cache/hf TMPDIR=$J/tmp XDG_CACHE_HOME=$J/cache
cd $W
export PYTHONPATH=$W     # the env has the main SynthScope checkout installed in editable mode: import this copy instead

echo "=== weights + processor: pinned revisions (hf download metadata) + sha256 ==="
$PY - <<'EOF'
import os
import sys
from pathlib import Path
sys.path[:0] = [".", "interface"]
from models.image.fakevlm.fakevlm import FAKEVLM
root = Path(os.environ["SYNTHSCOPE_CKPT_DIR"])
for name, repo, rev in (("fakevlm", FAKEVLM.hf_repo, FAKEVLM.hf_revision),
                        (FAKEVLM.processor, FAKEVLM.processor_hf_repo, FAKEVLM.processor_hf_revision)):
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
for d in fakevlm $PROC; do (cd -P $J/ckpt/$d && sha256sum -c --quiet SHA256SUMS) && echo "$d sha256 OK"; done

echo "=== images: the 16 fidelity images + the extended set (8 per class folder, sorted positions 21-28) ==="
$PY -c "import sys; sys.path[:0]=['.','interface']; from repro.fakevlm_fidelity import IMAGES; print('\n'.join(r for r, _ in IMAGES))" > $J/files.txt
LIST='cd '$SRC' && for d in GenImage_val_full/test/*/* SynthWildX/test/*/* WildRF/test/*/*; do ls $d | sort | sed -n 21,28p | sed "s|^|$d/|"; done'
if [ $ONSRC = 1 ]; then bash -c "$LIST" > $J/extended.txt; else ssh ssalehi@ivi-cn029 "$LIST" > $J/extended.txt; fi
cat $J/files.txt $J/extended.txt > $J/all.txt
if [ $ONSRC = 1 ]; then rsync -a --files-from=$J/all.txt $SRC/ $J/data/
else rsync -a --files-from=$J/all.txt ssalehi@ivi-cn029:$SRC/ $J/data/; fi
echo "$(wc -l < $J/files.txt) fidelity images, $(wc -l < $J/extended.txt) extended images"
(cd $J/data && sha1sum $(cat $J/files.txt))
$PY -c "
from PIL import Image
for p in open('$J/files.txt').read().split():
    im = Image.open('$J/data/' + p); print(im.size, im.mode, p)"

echo "=== official eval.py (unmodified; batches of 4) under transformers 4.45.2 $(date +%T) ==="
$PY repro/fakevlm_fidelity.py --write-test-json $J/test.json
PYTHONPATH=$DEPS/transformers4452 $PY repro/fakevlm_official.py eval --model $J/ckpt/fakevlm --processor $J/ckpt/$PROC \
   --data $J/data --test-json $J/test.json --out $J/official_4452.json --bs 4 --official-dir $REF
echo "=== official eval.py under transformers 5.5.4 (the env) $(date +%T) ==="
$PY repro/fakevlm_official.py eval --model $J/ckpt/fakevlm --processor $J/ckpt/$PROC \
   --data $J/data --test-json $J/test.json --out $J/official_554.json --bs 4 --official-dir $REF
echo "=== official processor inputs under transformers 4.45.2 $(date +%T) ==="
PYTHONPATH=$DEPS/transformers4452 $PY repro/fakevlm_official.py inputs --processor $J/ckpt/$PROC --data $J/data \
   --list $J/files.txt --out $J/official_inputs_4452.npz
echo "=== reference + engine comparison $(date +%T) ==="
$PY repro/fakevlm_fidelity.py --data $J/data --official-4452 $J/official_4452.json --official-554 $J/official_554.json \
   --official-inputs $J/official_inputs_4452.npz --extended $J/extended.txt --out $WT/repro/fakevlm_fidelity.json
echo "=== smoke test with the released files ([B] tokenizer, [F] strict load + one image) $(date +%T) ==="
$PY tests/test_fakevlm_smoke.py 2>&1 | grep -v -i "warn\|nvml\|it/s\]"
echo "done $(date)"
cd / && rm -rf $W $J/tmp $J/cache && echo "removed the code copy and caches (kept $J/data and the official outputs)"
