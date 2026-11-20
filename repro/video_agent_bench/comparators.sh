#!/bin/bash
# SynthScope's video detectors on the video_agent_bench subsets (the same staged files the agent scores), on ivi-cn023.
# Called by wevis_worker.sbatch with CMP=1 (or on its own inside a cn023 GPU job, from the code copy: W=<copy> R=...).
# One bench_eval_video run per (detector, benchmark): --per-video, a resumable score cache $R/cache/<det>/<bench>, result
# $R/out/<bench>/<det>.json (skipped when it exists). Weights / aux files / ffmpeg builds are the ones of the merged
# video-matrix cells (video_matrix_fill.sbatch, eval_video.sbatch); each result JSON records ckpt_sha256 / aux / ffmpeg.
#   DETS="d3 split ..." BENCHES="genvideo ..." limit the cells (default: spec.json's detectors x benchmarks).
set -u
G=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared
R=${R:-/hddstore/ssalehi/SynthScope/video_agent_bench}
W=${W:?the code copy}
PY=/home/ssalehi/miniconda3/envs/SynthScope/bin/python
S=/hddstore/ssalehi/SynthScope
H=/home/ssalehi/.cache/torch/hub
SPEC=$W/repro/video_agent_bench/spec.json
DETS=${DETS:-$($PY -c "import json;print(' '.join(json.load(open('$SPEC'))['detectors']))")}
BENCHES=${BENCHES:-$($PY -c "import json;print(' '.join(b['key'] for b in json.load(open('$SPEC'))['benchmarks']))")}
T=$R/cmp_env; mkdir -p $T/torch/hub/checkpoints $T/vina_ckpt $T/tmp
cp -anr $H/facebookresearch_dinov2_main $H/facebookresearch_dinov3 $H/trusted_list $T/torch/hub/ 2>/dev/null
cp -an $H/checkpoints/dinov2_vits14_pretrain.pth $T/torch/hub/checkpoints/ 2>/dev/null
[ -s $T/vina_ckpt/vina.pth ] || cp $G/ckpts/vina/serve/vina.pth $T/vina_ckpt/vina.pth
echo "comparators on $(hostname) $(date): DETS=[$DETS] BENCHES=[$BENCHES]"

det_env() {   # the environment of detector $1 (as in the merged video-matrix jobs, cn023 paths)
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=$T/tmp TORCH_HOME=$T/torch
  export HF_HOME=/home/ssalehi/.cache/huggingface PYTHONPATH=$W
  unset SYNTHSCOPE_FFMPEG SYNTHSCOPE_FFMPEG_NSG_VD
  case $1 in
    d3|split) export SYNTHSCOPE_CKPT_DIR=$S/checkpoints ;;                      # training-free; X-CLIP-B/16 from HF_HOME
    restrav)  export SYNTHSCOPE_CKPT_DIR=$S/checkpoints ;;                      # restrav.pth + DINOv2 ViT-S/14 (torch hub copy)
    stall)    export SYNTHSCOPE_CKPT_DIR=$S/checkpoints DINO_V3_REPO_DIR=$T/torch/hub/facebookresearch_dinov3 \
                     DINO_V3_WEIGHTS=$S/dinov3_checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth ;;
    vina_video) export SYNTHSCOPE_CKPT_DIR=$T/vina_ckpt SYNTHSCOPE_FFMPEG=$G/vina_parity_media/bin/ffmpeg ;;
    nsg_vd)   C=$G/ckpts/nsg_vd
              ( cd $C && grep -E "serve/(nsg_vd\.pth|nsg_vd_reference_msrvtt_val100\.safetensors)$" SHA256SUMS | sha256sum -c --quiet - ) \
                || { echo "REFUSE: nsg_vd files do not match SHA256SUMS"; return 3; }
              export SYNTHSCOPE_CKPT_DIR=$C/serve SYNTHSCOPE_FFMPEG_NSG_VD=$G/ffmpeg_pkg_nsgvd/ffmpeg-linux64-v4.2.2 ;;
    waverep|waverep_g4) export SYNTHSCOPE_CKPT_DIR=$G/ckpts/waverep HF_HOME=$G/waverep_ref_hfcache ;;
    *) echo "unknown detector $1"; return 2 ;;
  esac
}

cd $W
for b in $BENCHES; do
  ARGS=$($PY - "$SPEC" "$b" "$R/subsets" <<'EOF'
import json, sys
spec, key, sub = sys.argv[1:]
b = next(x for x in json.load(open(spec))["benchmarks"] if x["key"] == key)
a = ["--benchmark", b["profile"], "--data", f"{sub}/{key}/{b['data']}"]
a += ["--reals", f"{sub}/{key}/{b['reals']}"] if b.get("reals") else []
a += ["--limit", str(b["limit"])] if "limit" in b else []
print(" ".join(a))
EOF
)
  DATA=$(echo $ARGS | awk '{print $4}')
  [ -d "$DATA" ] || { echo "=== $b: no staged subset at $DATA, skip"; continue; }
  for d in $DETS; do
    OUT=$R/out/$b/$d.json
    [ -s $OUT ] && { echo "=== $d $b exists, skip"; continue; }
    mkdir -p $R/out/$b
    echo "=== unit cmp:$d:$b START $(date '+%F %T') on $(hostname): bench_eval_video -m $d $ARGS"
    ( det_env $d && $PY -W ignore benchmark/bench_eval_video.py -m $d $ARGS --tag vab_${b}_$d --per-video \
        --cache-dir $R/cache/$d/$b --out $OUT.partial --results-log $R/out/RESULTS.cmp.$(hostname -s).$$.jsonl 2>&1 \
      | grep --line-buffered -v -i -E "it/s\]|warn|UNEXPECTED|LOAD REPORT|Loading weights|^\s*$|can be ignored|^Notes|^Key |^---" ) \
      && [ -s $OUT.partial ] && mv $OUT.partial $OUT && echo "=== unit cmp:$d:$b DONE $(date '+%F %T')" \
      || echo "=== unit cmp:$d:$b FAILED $(date '+%F %T')"
  done
done
echo "comparators done $(date)"
