"""Image and video decoding helpers.

Reuses the SynthScope repo's own decoders so the interface sees pixels exactly the way the
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
from models.image.cure.preprocess import SRC_FORMAT_KEY  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
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
    source = Image.open(io.BytesIO(data))
    source_format = source.format
    image = source.convert("RGB")
    image.info[SRC_FORMAT_KEY] = source_format  # lets format-aware transforms (CuRe) see PNG vs JPEG
    return image


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


def thumbnail_bytes(path: str | Path, modality: str, size: int = 256) -> bytes:
    """A small JPEG thumbnail for the dataset explorer grid.

    Image -> downscaled still; video -> its middle frame (decord), downscaled. Returns raw
    JPEG bytes so the endpoint can stream them without writing temp files.
    """
    if modality == "video":
        import decord  # heavy; only needed for video
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
        n = len(reader)
        if n == 0:
            raise ValueError("Could not decode any frames from the video.")
        arr = reader[n // 2].asnumpy()             # [H,W,3] uint8, middle frame
        img = Image.fromarray(arr).convert("RGB")
    else:
        img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def video_clip_tensor(
    path: str | Path,
    num_frames: int,
    fps: float | None,
    frame_size: int,
    window_seconds: float = 2.0,
    window_center: bool = False,
    interp: str = "bilinear",
    edge_crop: float = 0.0,
) -> torch.Tensor:
    """Decode a clip for a *video* detector: [T, 3, frame_size, frame_size] float in [0, 1].

    Delegates to the repo's ``sample_video_frames`` so inference sampling matches each detector's
    training decode (e.g. ReStraV uses ``window_center=True, interp="bicubic"`` per the official repo;
    D3 uses ``edge_crop=0.1`` to trim borders per the official D3 preprocessing).
    """
    return sample_video_frames(
        path,
        num_frames=num_frames,
        fps=fps,
        window_seconds=window_seconds,
        size=frame_size,
        window_center=window_center,
        interp=interp,
        edge_crop=edge_crop,
    )
