#!/bin/bash
# Resume of _cure_aigibench_cn029_full.sbatch (download already complete). Fixes: (1) unzip's zip-bomb heuristic
# false-positives on these large archives -> UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE (archives come from the public
# HF dataset; counts are verified below); (2) run from a dedicated git worktree of trending/cure so that switching
# branches in the shared checkout cannot change what the job executes.
#SBATCH --job-name=cure_aigi29b --partition=cees --account=ceesusers --nodelist=ivi-cn029
#SBATCH --gres=gpu:rtx_a5000:1 --cpus-per-task=8 --mem=24G --time=05:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/cure_aigi29b_%j.out
set -u
W=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/cure_wt
cd $W
CONDA=/home/ssalehi/miniconda3/bin/conda
S=/ssdstore/ssalehi/AGID; R=$S/AIGIBench/test
export HF_HOME=$S/.cache_hf_cure PYTHONUNBUFFERED=1 SYNTHSCOPE_CKPT_DIR=$S/cure_ckpt PYTHONPATH=$W
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) | code=$(git -C $W log --oneline -1)"
$CONDA run --no-capture-output -n SynthScope python -c "import synthscope,interface.backend.registry as r;assert 'cure' in r.MODELS;print('cure registered from',r.__file__)" || exit 1
echo "=== extract all zips (parallel x6) $(date) ==="
cd $R
ls *.zip | xargs -P 6 -I{} sh -c 'unzip -q -n "{}" && echo "ok {}" || echo "UNZIP FAILED {}"'
echo "=== per-generator file counts ==="
bad=0
for d in $(ls -d */ | tr -d /); do
  r=$(find $d -ipath "*0_real*" -type f | wc -l); f=$(find $d -ipath "*1_fake*" -type f | wc -l)
  printf "  %-14s real=%-6s fake=%-6s\n" $d $r $f
  [ "$r" -gt 0 ] && [ "$f" -gt 0 ] || { echo "  ^^^ INCOMPLETE: $d"; bad=1; }
done
[ $bad = 0 ] || { echo "extraction incomplete -> NOT evaluating"; exit 2; }
echo "=== evaluate $(date) ==="
cd $W
tag=cure_repro_aigibench; full=repro/logs/${tag}_cn029_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model cure --benchmark pergen \
  --data $R --bs 16 --device cuda --limit 1500 --tag $tag > $full 2>&1
grep -aE "RESULT|: acc=|=== bench_eval|skip " $full | tail -45
echo "=== CURE AIGIBENCH RESUME DONE $(date) ==="
