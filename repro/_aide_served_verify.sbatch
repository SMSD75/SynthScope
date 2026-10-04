#!/bin/bash
# Verify the SERVED AIDE registry path (genuine 5-view arch + native-res DCT + official sd14 weights)
# loads strict and reproduces on GenImage-8.
#SBATCH --job-name=aide_verify
#SBATCH --partition=quva
#SBATCH --account=quvausers
#SBATCH --nodelist=ivi-cn023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/aide_verify_%j.out
set -u
cd /ivi/zfs/s0/original_homes/ssalehi/AGID
CONDA=/home/ssalehi/miniconda3/bin/conda
export SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/AGID/checkpoints
export HF_HOME=/hddstore/ssalehi/.cache/huggingface TORCH_HOME=/hddstore/ssalehi/.cache/torch PYTHONUNBUFFERED=1
H=/hddstore/ssalehi/AGID
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES aide.pth -> $(readlink -f $SYNTHSCOPE_CKPT_DIR/aide.pth) $(date)"
ok=0; for i in 1 2 3 4 5; do $CONDA run --no-capture-output -n SynthScope python -c "import torch;assert torch.cuda.is_available();print('CUDA OK')" && { ok=1;break; }; sleep 5; done
[ "$ok" = 1 ] || { echo "CUDA NEVER"; exit 1; }
echo "===== load-check (registry path, strict official load) ====="
$CONDA run --no-capture-output -n SynthScope python - <<'PY'
import glob, torch, synthscope
det = synthscope.load("aide", device="cuda")
m = det._engine._get_model(det.spec)
print("LOAD STATUS:", det._engine._status.get("aide"))
fk = (sorted(glob.glob(f"/hddstore/ssalehi/AGID/GenImage_full/test/sdv1.4/1_fake/*"))[:1])
rl = (sorted(glob.glob(f"/hddstore/ssalehi/AGID/GenImage_full/test/sdv1.4/0_real/*"))[:1])
for tag,p in [("FAKE",fk[0]),("REAL",rl[0])]:
    r=det.predict(p); print(tag,"->",r.get("verdict"),"p_fake=%.4f"%r.get("fake_probability"))
PY
echo "===== bench_eval --model aide GenImage-8 (limit 300) ====="
full=/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/logs/aide_served_genimage8_${SLURM_JOB_ID}.full.log
$CONDA run --no-capture-output -n SynthScope python benchmark/bench_eval.py --model aide \
  --benchmark pergen --data "$H/GenImage_full/test" --limit 300 --bs 16 --device cuda \
  --tag aide_served_genimage8 > "$full" 2>&1
rc=$?
grep -aE "RESULT|: acc=|=== bench_eval" "$full" | tail -12
[ $rc -ne 0 ] && { echo "[FAIL rc=$rc]"; tail -25 "$full"; }
echo "=== AIDE SERVED VERIFY COMPLETE $(date) ==="
