"""Regression test: animated GIFs decode to their real frames, not a black clip.

decord cannot open GIFs ("broken metadata"), and ``sample_video_frames`` used to return an all-zero clip for them, so
every video detector saw GIFs (e.g. 282 of GenVideo's WildScrape fakes) as black, motionless video. The PIL fallback
must return the GIF's own frames at the requested sampling rate.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sample_video_frames  # noqa: E402


def make_gif(path: str, n: int = 24, duration_ms: int = 100) -> None:
    """``n`` frames at 10 fps (GIF stores durations in 10 ms steps); frame k is a solid gray of level 10 * k."""
    frames = [Image.fromarray(np.full((64, 96, 3), 10 * k, dtype=np.uint8)) for k in range(n)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def test_gif_frames() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.gif")
        make_gif(path)
        clip = sample_video_frames(path, num_frames=16, fps=10.0, size=32, fps_nearest=True)
        assert clip.shape == (16, 3, 32, 32), clip.shape
        assert float(clip.max()) > 0, "GIF decoded as a black clip"
        levels = (clip[:, 0, 0, 0] * 255).round().tolist()
        assert levels == [10.0 * k for k in range(16)], levels      # source fps = target fps: frames 0..15 in order


if __name__ == "__main__":
    test_gif_frames()
    print("GIF decode test passed")
