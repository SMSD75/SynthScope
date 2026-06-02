"""Image and video decoding helpers.

Reuses the AGID repo's own decoders so the interface sees pixels exactly the way the
training/eval pipeline does:
  * ``dataset._decode_rgb``         -> robust still-image decode (torchvision, PIL fallback)
  * ``dataset.sample_video_frames`` -> evenly-sampled clip [T,3,size,size] float in [0,1]
"""
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

# Some user uploads are truncated/progressive; let PIL load what it can.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Faithful repo decoders.
from dataset import sample_video_frames  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def media_kind(filename: str, content_type: str | None = None) -> str:
    """Return ``"image"`` or ``"video"`` from the filename/extension (content_type as hint)."""
    suffix = Path(filename).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if content_type:
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("image/"):
            return "image"
    raise ValueError(f"Unsupported file type: {filename!r} (content_type={content_type})")


def decode_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def sample_video_pil(path: str | Path, num_frames: int) -> list[Image.Image]:
    """Sample ``num_frames`` frames spread across the whole video, at native resolution.

    Used when running an *image* detector on a video: each frame is treated exactly like an
    uploaded still (the detector's own image transform handles resize/crop/normalize).
    """
    import decord  # local import: heavy, only needed for video

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    n_src = len(reader)
    if n_src == 0:
        raise ValueError("Could not decode any frames from the video.")
    n = max(1, min(num_frames, n_src))
    indices = np.linspace(0, n_src - 1, num=n)
    indices = np.clip(np.rint(indices).astype(np.int64), 0, n_src - 1).tolist()
    frames = []
    for i in indices:
        try:
            arr = reader[i].asnumpy()  # [H, W, 3] uint8
            frames.append(Image.fromarray(arr).convert("RGB"))
        except Exception:
            continue
    if not frames:
        raise ValueError("Could not decode any frames from the video.")
    return frames


def video_clip_tensor(
    path: str | Path,
    num_frames: int,
    fps: float | None,
    frame_size: int,
    window_seconds: float = 2.0,
) -> torch.Tensor:
    """Decode a clip for a *video* detector: [T, 3, frame_size, frame_size] float in [0, 1].

    Delegates to the repo's ``sample_video_frames`` so sampling matches D3 / ReStraV exactly.
    """
    clip = sample_video_frames(
        path,
        num_frames=num_frames,
        fps=fps,
        window_seconds=window_seconds,
        size=frame_size,
    )
    return clip
