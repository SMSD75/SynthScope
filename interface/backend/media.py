"""Image and video decoding helpers.

Reuses the SynthScope repo's own decoders so the interface sees pixels exactly the way the
training/eval pipeline does:
  * ``dataset._decode_rgb``         -> robust still-image decode (torchvision, PIL fallback)
  * ``dataset.sample_video_frames`` -> evenly-sampled clip [T,3,size,size] float in [0,1]
"""
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile

try:                                   # optional: a static ffmpeg for the ffmpeg-based video protocols (VINA)
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

# Some user uploads are truncated/progressive; let PIL load what it can.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Faithful repo decoders.
from dataset import sample_video_frames  # noqa: E402
from models.image.cure.preprocess import SRC_FORMAT_KEY  # noqa: E402

class VideoDecodeError(ValueError):
    """The video file yields no decodable frame (corrupt / empty). A ValueError, so every existing handler still catches
    it; benchmark/bench_eval_video.py counts and caches only these as failed-to-decode videos."""


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


def to_rgb(source: Image.Image) -> Image.Image:
    """``source.convert("RGB")`` that records the source file format (PNG, JPEG, ...) in ``info[SRC_FORMAT_KEY]``, so
    format-aware eval transforms (CuRe, MIRROR: PNG sources get a JPEG round trip) see what the official loaders see.
    A hint already present (an image that was converted before) is kept; the caller's image is not modified."""
    source_format = source.info.get(SRC_FORMAT_KEY) or source.format
    image = source.convert("RGB")
    if source_format:
        image.info[SRC_FORMAT_KEY] = source_format
    return image


def decode_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image (with the source-format hint, see ``to_rgb``)."""
    return to_rgb(Image.open(io.BytesIO(data)))


def open_rgb(path: str | Path) -> Image.Image:
    """Open an image file as RGB (with the source-format hint, see ``to_rgb``)."""
    with Image.open(path) as source:
        return to_rgb(source)


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
        raise VideoDecodeError("Could not decode any frames from the video.")
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
        raise VideoDecodeError("Could not decode any frames from the video.")
    return frames


def ffmpeg_available() -> bool:
    try:
        ffmpeg_exe()
    except (FileNotFoundError, RuntimeError, OSError):   # RuntimeError: imageio-ffmpeg found no binary
        return False
    return True


def ffmpeg_exe() -> str:
    """The ffmpeg binary: ``$SYNTHSCOPE_FFMPEG``, else ``ffmpeg`` on the PATH, else imageio-ffmpeg's static build."""
    env = os.environ.get("SYNTHSCOPE_FFMPEG")
    if env:
        if not os.path.isfile(env) and shutil.which(env) is None:
            raise FileNotFoundError(f"SYNTHSCOPE_FFMPEG={env!r} is not an executable file")
        return env
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError as e:
            raise FileNotFoundError(f"imageio-ffmpeg found no ffmpeg binary ({e})") from e
    raise FileNotFoundError("ffmpeg not found: install it, set SYNTHSCOPE_FFMPEG, or pip install imageio-ffmpeg")


def ffmpeg_fps_frames(path: str | Path, fps: float, num_frames: int, timeout: float = 300.0) -> list[Image.Image]:
    """VINA's official video protocol: ffmpeg writes the video as JPEG frames at ``fps`` with its default JPEG settings
    (preprocess/video2frame.py: ``ffmpeg -loglevel quiet -i <video> -vf fps=2 <dir>/%d.jpg``), and the central window
    of ``num_frames`` of them is read back as RGB, the last frame repeated when there are fewer (the test phase of
    dataset/dataloader.py ``_load_frames_dir``).

    The whole clip is extracted, not just the frames that are read: ffmpeg's mjpeg rate control runs across frames, so
    only the full extraction writes the same JPEG bytes as the official preprocessing. Parity holds for the ffmpeg
    version used (checked with the static ffmpeg 7.0.2); other versions may encode slightly differently. As the
    official ``process_video``, a failing ffmpeg run (a truncated clip) keeps the frames it wrote; only a clip with no
    frame at all is an error. ``timeout`` (seconds) bounds a malformed upload."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [ffmpeg_exe(), "-loglevel", "quiet", "-i", str(path), "-vf", f"fps={fps:g}", os.path.join(tmp, "%d.jpg")]
        try:
            subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ValueError(f"ffmpeg did not finish within {timeout:g} s for this video.") from e
        ids = sorted(int(p.stem) for p in Path(tmp).glob("*.jpg") if p.stem.isdigit())
        if not ids:
            raise VideoDecodeError("ffmpeg wrote no frames for this video.")
        if len(ids) >= num_frames:
            start = (len(ids) - num_frames) // 2
            ids = ids[start:start + num_frames]
        else:
            ids += [ids[-1]] * (num_frames - len(ids))
        return [open_rgb(Path(tmp) / f"{i}.jpg") for i in ids]


def _moviepy_seconds(stamp: str) -> float:
    """moviepy.tools.convert_to_seconds for an ``HH:MM:SS.ss`` string (same float operations, same order)."""
    return sum(mult * part for mult, part in zip((1, 60, 3600), reversed([float(p.replace(",", ".")) for p in stamp.split(":")])))


def ffmpeg_frame_count(path: str | Path, timeout: float = 60.0) -> int:
    """The frame count NSG-VD's official code computes for a video (data/video_frames_dataset.py
    ``get_video_frame_count``: ``int(VideoFileClip(path).fps * VideoFileClip(path).duration)``, moviepy 2.x), without
    moviepy: moviepy runs ``ffmpeg -hide_banner -i <video>`` and parses its stderr (video/io/ffmpeg_reader.py
    ``FFmpegInfosParser``); this follows the same rules:
      * duration: the container's ``  Duration: HH:MM:SS.ss`` line (``convert_to_seconds``);
      * fps: of the default video stream (the first ``Stream ... Video:`` line ending in ``(default)``, else the first
        video stream): ``<x> fps``, else ``<x> tbr`` (``k`` = x1000); a value within 0.01 of n*1000/1001 for n in
        23, 24, 25, 30, 50 becomes exactly n*1000/1001.
    A file with no duration or no video stream raises ``VideoDecodeError`` (moviepy fails on it and the official code
    skips the video)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-i", str(path)]
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"ffmpeg did not finish within {timeout:g} s for this video.") from e
    duration, video_lines = None, []
    for line in proc.stderr.decode("utf8", errors="ignore").splitlines()[1:]:
        if not line or line[0] != " ":
            continue
        if line.startswith("  Duration:") and duration is None:
            stamp = re.search(r"([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line.split("Duration: ")[-1])
            if stamp is None:
                raise VideoDecodeError("The video has no duration (moviepy, as the official code, cannot read it).")
            duration = _moviepy_seconds(stamp.group(1))
        elif re.search(r"^Stream\s#(\d+):(\d+)(?:\[\w+\])?\(?(\w+)?\)?:\sVideo:", line.lstrip()):
            video_lines.append(line)
    if duration is None or not video_lines:
        raise VideoDecodeError("ffmpeg found no video stream / duration in this file.")
    line = next((v for v in video_lines if v.endswith("(default)")), video_lines[0])
    fps = re.search(r" (\d+.?\d*) fps", line)
    if fps is not None:
        fps = float(fps.group(1))
    else:
        tbr = re.search(r" (\d+.?\d*k?) tbr", line)
        if tbr is None:
            raise VideoDecodeError("ffmpeg reports no frame rate for this video.")
        tbr = tbr.group(1)
        fps = float(tbr[:-1]) * 1000 if tbr[-1] == "k" else float(tbr)
    coef = 1000.0 / 1001.0
    for x in [23, 24, 25, 30, 50]:
        if (fps != x) and abs(fps - x * coef) < 0.01:
            fps = x * coef
    return int(fps * duration)


def ffmpeg_select_frames(path: str | Path, num_frames: int = 8, timeout: float = 300.0) -> list[Image.Image]:
    """NSG-VD's official frame extraction (data/video_frames_dataset.py ``process_video``): with ``n`` the moviepy frame
    count (``ffmpeg_frame_count``) and ``k = max(1, n // num_frames)``, ffmpeg writes the first ``num_frames`` frames
    whose index is a multiple of ``k`` as JPEGs with its default settings::

        ffmpeg -i <video> -vf select='not(mod(n\\,k))',setpts=N/FRAME_RATE/TB -vframes <num_frames> <dir>/frame%d.jpg

    and they are read back in sorted file-name order (``glob`` + ``sort``). The official dataset drops a video that does
    not give exactly ``num_frames`` frames; here that is a ``VideoDecodeError``. The result depends on the ffmpeg build:
    its JPEG encoder (parity with the official code checked with the static ffmpeg 4.2.2 of imageio-ffmpeg 0.4.2; 9.0.2
    encodes the same frames ~0.8 grey levels apart on average, which moves NSG-VD scores by 0.04-0.19 on average), and
    ffmpeg 7.0.x drops selected frames on some 50 / 30 fps clips (MoonValley, Sora), so fewer than 8 are written. Set
    ``$SYNTHSCOPE_FFMPEG`` to the build to use. The frames are returned as decoded (the official code opens
    each JPEG and resizes it as is); a non-RGB JPEG (grayscale) is converted to RGB, where the official code would fail."""
    k = max(1, ffmpeg_frame_count(path) // num_frames)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [ffmpeg_exe(), "-i", str(path), "-vf", f"select=not(mod(n\\,{k})),setpts=N/FRAME_RATE/TB",
               "-vframes", str(num_frames), os.path.join(tmp, "frame%d.jpg")]
        try:
            subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ValueError(f"ffmpeg did not finish within {timeout:g} s for this video.") from e
        files = sorted(str(p) for p in Path(tmp).glob("*"))
        if len(files) != num_frames:
            raise VideoDecodeError(f"ffmpeg wrote {len(files)} frames for this video; NSG-VD needs {num_frames} "
                                   "(the official code skips such videos).")
        frames = []
        for f in files:
            with Image.open(f) as im:
                im.load()
                frames.append(im.copy() if im.mode == "RGB" else im.convert("RGB"))
        return frames


def cv2_frames(path: str | Path, limit: int | None = None):
    """WaveRep's official frame reader (demo/utils.py ``ReadVideoIteratorCV``, vendored in
    models/video/waverep/official_utils.py): every frame OpenCV's ``VideoCapture`` decodes, in order, converted from BGR
    to RGB, as PIL images; reading stops at the first failed read, and after ``limit`` frames when it is set (the demo's
    ``--limit``). A generator, so a long video is never held in memory at once.

    Errors (raised when the generator is first advanced): a path that does not exist is a ``FileNotFoundError`` and a
    file OpenCV cannot open an ``OSError``; both are I/O problems, not a verdict on the video, so callers do not cache
    them as decode failures. An existing but empty (0-byte) file is a ``VideoDecodeError``, as is a file that opens but
    decodes no frame (raised by the caller, which sees the empty iterator)."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    if os.path.getsize(path) == 0:
        raise VideoDecodeError("The video file is empty (0 bytes).")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise OSError(f"OpenCV could not open the video file: {path}")
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx = idx + 1
            if limit and idx >= limit:
                break
    finally:
        cap.release()


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
            raise VideoDecodeError("Could not decode any frames from the video.")
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
    edge_crop_long_side: bool = False,
    short_clip_frames: int = 0,
    fps_nearest: bool = False,
    bgr: bool = False,
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
        edge_crop_long_side=edge_crop_long_side,
        short_clip_frames=short_clip_frames,
        fps_nearest=fps_nearest,
        bgr=bgr,
    )
