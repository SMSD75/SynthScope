"""CPU cost of one NSG-VD clip on the engine path (`nsg_vd`, fp32 UNet on CPU), at fixed thread counts.

For every thread count a fresh process builds the engine, loads the model (reference discriminator features computed at the first
score), scores the same clip once cold and `--repeats` times warm, and records wall-clock seconds and the raw statistic (`logit`).
The parent process collects the records into one JSON (host, CPU model, torch / ffmpeg versions, SLURM job).

    SYNTHSCOPE_CKPT_DIR=<dir with nsg_vd.pth, 256x256_diffusion_uncond.pt, nsg_vd_reference_msrvtt_val100.safetensors> \\
    SYNTHSCOPE_FFMPEG_NSG_VD=<ffmpeg 4.2 binary> CUDA_VISIBLE_DEVICES= \\
    python repro/nsg_vd_cpu_timing.py --clip clip.mp4 --threads 2 8 16 --out repro/nsg_vd_cpu_timing.json

Without `--clip` a 4 s, 256x256, 24 fps `testsrc2` clip is written with the pinned ffmpeg. Job: see the JSON (`slurm_job`).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface")]

from backend.engine import Engine, spec_ffmpeg_exe  # noqa: E402
from backend.registry import get_spec  # noqa: E402


def run_one(clip: str, threads: int, repeats: int) -> dict:
    """Score `clip` in this process with `threads` torch threads (called in a child process)."""
    torch.set_num_threads(threads)
    spec = get_spec("nsg_vd")
    engine = Engine(device="cpu")
    t0 = time.perf_counter()
    first = engine.predict_video_clip(spec, clip)
    cold = time.perf_counter() - t0
    warm, logits = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = engine.predict_video_clip(spec, clip)
        warm.append(time.perf_counter() - t0)
        logits.append(res["logit"])
    return {"threads": threads, "torch_threads_effective": torch.get_num_threads(), "first_call_s_incl_model_load": round(cold, 2),
            "first_logit": first["logit"], "warm_s": [round(w, 2) for w in warm], "warm_logits": logits,
            "verdict": first["verdict"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 8, 16])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--child", type=int, default=None, help="internal: run one thread count and print its record")
    args = ap.parse_args()
    if args.child is not None:
        print("RECORD " + json.dumps(run_one(args.clip, args.child, args.repeats)))
        return
    exe = spec_ffmpeg_exe(get_spec("nsg_vd"))
    clip = args.clip
    if clip is None:
        clip = os.path.join(tempfile.mkdtemp(), "clip.mp4")
        subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=24", "-t", "4", "-pix_fmt",
                        "yuv420p", clip], check=True)
    records = []
    for n in args.threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n), CUDA_VISIBLE_DEVICES="")
        out = subprocess.run([sys.executable, "-W", "ignore", __file__, "--clip", clip, "--child", str(n), "--repeats",
                              str(args.repeats)], env=env, check=True, capture_output=True, text=True).stdout
        records.append(json.loads(next(l for l in out.splitlines() if l.startswith("RECORD "))[7:]))
        print(records[-1], flush=True)
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), "?")
    res = {"what": "engine.predict_video_clip, key nsg_vd, fp32 UNet on CPU, one clip, fresh process per thread count",
           "clip": os.path.basename(clip), "host": platform.node(), "cpu": cpu, "cpus_allocated": len(os.sched_getaffinity(0)),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__, "ffmpeg": os.path.basename(exe), "records": records}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
