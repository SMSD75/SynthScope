"""Diagnose why D3 calls everything fake on GenVidBench but works on EvalCrafter.

D3 score = std of 2nd-order frozen-CLIP trajectory features; calibration says real>thr(5.68)>fake.
We print the raw score, calibrated P(fake), and the clip duration/native-fps for a few videos in
each group, to see whether GenVidBench reals score low (flat) and whether it's a short-clip artifact.
"""
import glob, random, sys
from pathlib import Path
import numpy as np, torch, decord

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.engine import get_engine          # noqa: E402
from backend.registry import get_spec          # noqa: E402
from backend import media                       # noqa: E402

GVB = "/hddstore/ssalehi/AGID/GenVidBench/test"
EVC = "/hddstore/ssalehi/AGID/EvalCrafter/test"
GROUPS = {
    "GVB real (vript+hd_vg)": sorted(glob.glob(f"{GVB}/0_real/*/*.mp4")),
    "GVB fake (8 gens)":      sorted(glob.glob(f"{GVB}/1_fake/*/*.mp4")),
    "EvalCrafter fake":       sorted(glob.glob(f"{EVC}/1_fake/*/*.mp4")),
}


def native_dur(path):
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    n = len(vr); fps = float(vr.get_avg_fps()) or 30.0
    return n, fps, n / fps


@torch.no_grad()
def main():
    eng = get_engine(); spec = get_spec("d3"); model = eng._get_model(spec)
    mean = torch.tensor(spec.video_norm[0]).view(1, 3, 1, 1)
    std = torch.tensor(spec.video_norm[1]).view(1, 3, 1, 1)
    for gname, paths in GROUPS.items():
        random.Random(0).shuffle(paths)
        scores, probs, durs, nfr = [], [], [], []
        for p in paths[:20]:
            try:
                n, fps, dur = native_dur(p)
                clip = media.video_clip_tensor(p, spec.num_frames, spec.fps, spec.frame_size,
                                               window_center=spec.window_center, interp=spec.resize_interp)
                clip = (clip - mean) / std
                s = float(model.score(clip.unsqueeze(0).to(eng.device)).item())
                pf = float(eng._video_fake_prob(model, spec, clip).item())
                scores.append(s); probs.append(pf); durs.append(dur); nfr.append(n)
            except Exception as e:
                print(f"  skip {Path(p).name}: {e}")
        scores, probs, durs = map(np.array, (scores, probs, durs))
        print(f"\n{gname}: n={len(scores)}")
        print(f"  D3 raw score  mean={scores.mean():.2f}  (real>5.68>fake; min={scores.min():.2f} max={scores.max():.2f})")
        print(f"  P(fake)       mean={probs.mean():.3f}  -> called fake: {(probs>0.5).mean()*100:.0f}%")
        print(f"  native dur(s) mean={durs.mean():.2f}  min={durs.min():.2f} max={durs.max():.2f}   nframes mean={np.mean(nfr):.0f}")


if __name__ == "__main__":
    main()
