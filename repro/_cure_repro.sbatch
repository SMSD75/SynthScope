#!/bin/bash
# CuRe reproduction: stage subsets from cn023 (needs the cure_lists hold job running there), then run
# bench_eval with the OFFICIAL adapter on UFD (full), AIGCDetect + AIGIBench (capped per class).
#SBATCH --job-name=cure_repro --partition=cees --account=ceesusers
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=8 --mem=24G --time=04:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/cure_repro_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
REF=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/cure_ref
SRC=ivi-cn023:/hddstore/ssalehi/AGID
SCR=/ssdstore; [ -w $SCR ] || SCR=/tmp
D=$SCR/$USER/cure_data; mkdir -p $D $SCR/$USER/cure_hf $SCR/$USER/cure_ckpt
export HF_HOME=$SCR/$USER/cure_hf PYTHONUNBUFFERED=1 SYNTHSCOPE_CKPT_DIR=$SCR/$USER/cure_ckpt
cp $REF/CuRe/weights/cure_adapter.pt $SYNTHSCOPE_CKPT_DIR/cure.pth      # the OFFICIAL adapter, unchanged
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) | data=$D"
$CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK',torch.cuda.get_device_name(0))" || exit 1
for i in $(seq 1 120); do [ -f $REF/lists/READY ] && break; sleep 15; done
[ -f $REF/lists/READY ] || { echo "lists never became READY (cn023 hold job not running?)"; exit 1; }
declare -A OK
for b in ufd aigcdetect aigibench; do
  echo "=== stage $b: $(wc -l < $REF/lists/$b.txt) files ==="
  if rsync -a --files-from=$REF/lists/$b.txt $SRC/ $D/; then OK[$b]=1; else echo "rsync $b FAILED -> will skip its eval"; fi
done
du -sh $D
run(){ local tag="$1" bench="$2" root="$3" limit="$4" key="$5"
  [ "${OK[$key]:-0}" = 1 ] || { echo "===== $tag SKIPPED (data not fully staged) ====="; return; }
  echo "===== $tag ====="
  local full=repro/logs/${tag}_${SLURM_JOB_ID}.full.log
  $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model cure --benchmark $bench \
    --data "$root" --bs 16 --device cuda --tag "$tag" ${limit:+--limit $limit} > "$full" 2>&1
  grep -aE "RESULT|: acc=|=== bench_eval" "$full" | tail -30; }
run cure_repro_ufd        ufd    $D/universialfakedetection/test "" ufd
run cure_repro_aigcdetect pergen $D/AIGCDetectBenchmark/test 800 aigcdetect
run cure_repro_aigibench  pergen $D/AIGIBench/test 300 aigibench
echo "=== CURE REPRO DONE $(date) ==="
