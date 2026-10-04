#!/bin/bash
# Evaluate the retrained SAFE (safe_full_v2) across ALL SAFE benchmarks with the corrected crop-only
# transform (bench_eval uses the interface registry's SAFE transform = tf_crop(256), no resize/norm).
# Runs on cn023 where all the benchmark datasets are local (/hddstore). Reads the checkpoint from
# shared /ivi/zfs. Pass CK=<path> to override the auto-picked best epoch.
#SBATCH --job-name=safe_v2_ev
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/safe_v2_allbench_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
H=/hddstore/ssalehi/AGID
SV=/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/safe_full_v2
CK="${CK:-}"
# pick highest-epoch checkpoint if not overridden
[ -z "$CK" ] && CK=$(ls -1 "$SV"/model_*.pth 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | awk '{print $2}')
echo "NODE=$(hostname) CK=$CK $(date)"
[ -n "$CK" ] && [ -f "$CK" ] || { echo "no safe_full_v2 checkpoint yet"; exit 2; }
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }

run () {  # tag  benchmark  data  [extra...]
  local tag="$1" bench="$2" data="$3"; shift 3
  [ -d "$data" ] || { echo "[skip] $tag: no data $data"; return; }
  echo "===== $tag ($bench) ====="
  $CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model safe --ckpt "$CK" \
    --benchmark "$bench" --data "$data" --bs 64 --device cuda --tag "$tag" "$@" 2>&1 | grep -aE "RESULT|: acc=" | tail -16
}

run safe_v2_forensynths forensynths "$H/CNNDetection/test"
run safe_v2_ufd          ufd         "$H/universialfakedetection/test"
run safe_v2_genimage8    pergen      "$H/GenImage_full/test"
run safe_v2_gangen       pergen      "$H/GANGen-Detection"
run safe_v2_difforensics pergen      "$H/DiffusionForensics/test"
run safe_v2_ditfake      pergen      "$H/DiTFake/test" --limit 2000
echo "=== SAFE v2 ALL-BENCH EVAL COMPLETE $(date) ==="
