"""NSG-VD frame extraction across ffmpeg builds, on the 75 GenVideo-Val parity videos (job 286253's list).

For each build (static 4.2.2 = the parity build, 9.0.2, 7.0.2): the moviepy-rule frame count, the number of frames
the official ffmpeg command writes (ffmpeg_select_frames' command, counted before the 8-frame check), and, against
4.2.2, the mean / max absolute difference of the decoded JPEG frames (0-255) and of the model inputs after the
official LANCZOS-224 transform (in grey levels, x255). Writes one JSON (per-video + summary).

Producer of repro/nsg_vd_ffmpeg_builds_genvideo75.json (job 286412 on ivi-cn029, log repro/logs/nsgvdff_286412.out;
the code below is the file that job ran, only this docstring paragraph was added when it was committed):

    python repro/nsg_vd_ffmpeg_builds.py <worktree> <parity JSON with a "videos" list (id, generator, video path)> <out.json>

The input list is the official parity run's JSON (SynthScope_shared/nsgvd_parity_out/
official_ffmpeg-linux64-v4.2.2_286253.pt.json, 75 GenVideo-Val videos). ``BUILDS`` points at the static binaries in
SynthScope_shared (SHA256SUMS next to them): 4.2.2 = imageio-ffmpeg 0.4.2, 9.0.2 = imageio-ffmpeg 0.8.1, 7.0.2 = the
repo's VINA / D3 build; edit the paths to rerun elsewhere."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

W = sys.argv[1]
sys.path[:0] = [W, W + "/interface"]
from backend import media                                       # noqa: E402
from models.video.nsg_vd.preprocess import NSGVDFrame            # noqa: E402

G = "/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared"
BUILDS = {"4.2.2": G + "/ffmpeg_pkg_nsgvd/ffmpeg-linux64-v4.2.2",
          "9.0.2": G + "/ffmpeg_pkg_nsgvd/ffmpeg-linux-x86_64-v9.0.2",
          "7.0.2": G + "/ffmpeg_pkg/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"}
tf = NSGVDFrame()


def extract(video, exe):
    """The official command (as media.ffmpeg_select_frames): (frame count, written frames as uint8 arrays)."""
    n = media.ffmpeg_frame_count(video, exe=exe)
    k = max(1, n // 8)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([exe, "-i", video, "-vf", f"select=not(mod(n\\,{k})),setpts=N/FRAME_RATE/TB", "-vframes", "8",
                        os.path.join(tmp, "frame%d.jpg")], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
        frames = []
        for f in sorted(Path(tmp).glob("*")):
            with Image.open(f) as im:
                im.load()
                frames.append(im.copy())
    return n, frames


def main():
    videos = json.load(open(sys.argv[2]))["videos"]
    recs = []
    for v in videos:
        r = {"id": v["id"], "generator": v["generator"], "video": v["video"]}
        out = {}
        for name, exe in BUILDS.items():
            n, frames = extract(v["video"], exe)
            out[name] = frames
            r[name] = {"frame_count": n, "frames_written": len(frames)}
        base = out["4.2.2"]
        for name in ("9.0.2", "7.0.2"):
            if len(out[name]) == 8 and len(base) == 8:
                raw = np.stack([np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) for a, b in zip(out[name], base)])
                inp = np.stack([np.abs(tf(a).numpy().astype(np.float64) - tf(b).numpy()) * 255 for a, b in zip(out[name], base)])
                r[name].update({"jpeg_mean_abs_diff_vs_4.2.2": float(raw.mean()), "jpeg_max_abs_diff_vs_4.2.2": float(raw.max()),
                                "input_mean_abs_diff_grey_vs_4.2.2": float(inp.mean()),
                                "identical_to_4.2.2": bool(raw.max() == 0)})
        recs.append(r)
        print(r["id"], {b: r[b]["frames_written"] for b in BUILDS}, flush=True)
    summary = {}
    for name, exe in BUILDS.items():
        s = {"ffmpeg_version": media.ffmpeg_version(exe),
             "videos_with_8_frames": sum(r[name]["frames_written"] == 8 for r in recs),
             "videos_short": sorted({(r["generator"], r["id"], r[name]["frames_written"]) for r in recs
                                     if r[name]["frames_written"] != 8}),
             "frame_count_equal_to_4.2.2": sum(r[name]["frame_count"] == r["4.2.2"]["frame_count"] for r in recs)}
        if name != "4.2.2":
            d = [r[name]["input_mean_abs_diff_grey_vs_4.2.2"] for r in recs if "input_mean_abs_diff_grey_vs_4.2.2" in r[name]]
            j = [r[name]["jpeg_mean_abs_diff_vs_4.2.2"] for r in recs if "jpeg_mean_abs_diff_vs_4.2.2" in r[name]]
            s.update({"compared_videos": len(d), "identical_videos": sum(r[name].get("identical_to_4.2.2", False) for r in recs),
                      "input_mean_abs_diff_grey_mean": float(np.mean(d)), "input_mean_abs_diff_grey_max": float(np.max(d)),
                      "jpeg_mean_abs_diff_mean": float(np.mean(j)), "jpeg_mean_abs_diff_max": float(np.max(j))})
        summary[name] = s
    res = {"source": sys.argv[2], "n_videos": len(recs), "builds": BUILDS, "summary": summary, "per_video": recs}
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
