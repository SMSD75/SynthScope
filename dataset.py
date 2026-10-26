from torch.utils.data import Dataset
from typing import Literal
from pathlib import Path
import hashlib
import os
import re
import zipfile
import tarfile
import shutil
import subprocess
import csv
import time
import urllib.error
import urllib.parse
import urllib.request
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import DataLoader
from transformations import AIGITransformations
from torchvision.io import decode_image, ImageReadMode
from PIL import Image
from config_utils import compose_config


def _decode_rgb(path) -> torch.Tensor:
    """Decode an image to a uint8 [3, H, W] RGB tensor.

    Uses torchvision's fast native decoder, falling back to PIL for the rare files
    its backend rejects (e.g. CMYK / progressive / slightly-malformed JPEGs that
    raise "Internal error"). Keeps the uint8 CHW contract the transform pipeline expects.
    """
    try:
        return decode_image(str(path), mode=ImageReadMode.RGB)
    except Exception:
        pass
    try:
        img = Image.open(str(path)).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()
    except Exception:
        # Corrupt / truncated / empty file (some GenImage PNGs are 0-byte): don't kill the
        # whole run — return a black 256x256 image so this sample is just uninformative.
        return torch.zeros(3, 256, 256, dtype=torch.uint8)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".jfif", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".gif")

def _iter_images(directory: Path):
    return (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _has_images(directory: Path) -> bool:
    return directory.is_dir() and any(_iter_images(directory))


def _iter_videos(directory: Path):
    return (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def _has_videos(directory: Path) -> bool:
    return directory.is_dir() and any(_iter_videos(directory))


def sample_video_frames(
    path: str | Path,
    num_frames: int,
    fps: float | None = None,
    window_seconds: float = 2.0,
    size: int = 224,
    window_center: bool = False,
    interp: str = "bilinear",
    edge_crop: float = 0.0,
    edge_crop_long_side: bool = False,
    short_clip_frames: int = 0,
    fps_nearest: bool = False,
    bgr: bool = False,
) -> torch.Tensor:
    """Decode a clip of exactly ``num_frames`` frames and return ``[T, 3, size, size]`` float in ``[0, 1]``.

    Frames are sampled over a ``window_seconds`` window. By default the window starts at the
    beginning of the video; set ``window_center=True`` to center it at the video midpoint (this is
    what the official ReStraV pipeline does). When ``fps`` is given (e.g. D3 samples at 8 fps), the
    window is shortened to ``num_frames / fps`` seconds so the effective sampling rate matches
    ``fps``; when ``fps`` is ``None`` the frames are spread evenly across ``window_seconds``. The
    resize interpolation is ``interp`` (``"bilinear"`` default; ``"bicubic"`` for ReStraV, with
    antialiasing, matching the official preprocessing). The output frame count is always
    ``num_frames`` so the default collate can stack clips into ``[B, T, 3, size, size]``.

    Two options reproduce the official D3 preprocessing: ``edge_crop_long_side`` trims ``edge_crop`` from the two ends
    of the LONGER side only (D3's ``crop_center_by_percentage``), and ``short_clip_frames`` (D3: 8) returns that many
    frames, instead of repeating frames up to ``num_frames``, when the video holds fewer than ``num_frames`` frames at
    ``fps`` (D3 reads 16 frames, or 8 when fewer than 16 exist). With it the frame count can be below ``num_frames``.
    ``fps_nearest`` takes the source frame nearest to each time ``k / fps`` (what ffmpeg's ``fps`` filter writes) instead
    of spreading frames evenly over the window, and ``bgr`` returns the channels in BGR order (D3 reads its frames with
    ``cv2.imread`` and never converts them to RGB, so its model sees BGR).
    """
    try:
        reader = VideoReader(str(path), ctx=cpu(0))
        num_source = len(reader)
    except Exception:
        return torch.zeros(num_frames, 3, size, size)
    if num_source == 0:
        return torch.zeros(num_frames, 3, size, size)

    avg_fps = float(reader.get_avg_fps()) or 30.0
    duration = num_source / avg_fps
    window = min(window_seconds, duration)
    if fps and fps > 0:
        if short_clip_frames and int(duration * fps) < num_frames:     # too short for num_frames at this fps
            num_frames = short_clip_frames
        window = min(window, num_frames / float(fps))

    n_window = max(int(round(window * avg_fps)), 1)        # frames spanning the window
    start_index = 0
    if window_center:                                      # center the window at the midpoint
        start_index = int(round((duration / 2.0 - window / 2.0) * avg_fps))
    start_index = min(max(start_index, 0), num_source - 1)
    end_index = min(start_index + n_window - 1, num_source - 1)
    if fps_nearest and fps and fps > 0:
        indices = start_index + np.rint(np.arange(num_frames) * avg_fps / float(fps))
    else:
        indices = np.linspace(start_index, end_index, num=num_frames)
    indices = np.clip(np.rint(indices).astype(np.int64), 0, num_source - 1)

    # decord's batched get_batch raises a size-mismatch DECORDError on videos whose frames
    # vary in resolution (or are partly corrupt); fall back to per-frame decode+resize, and
    # finally to a black clip, so one bad video never crashes the whole DataLoader worker.
    def _resize(t):  # t: [T,3,H,W] float
        if edge_crop and edge_crop > 0.0:
            # Crop the central (1 - 2*edge_crop) region off each frame BEFORE resizing — the official
            # D3 preprocessing trims borders/letterboxing that otherwise dominate the frozen-feature
            # trajectory. Default 0.0 leaves D3-without-crop / ReStraV / STALL untouched.
            h, w = t.shape[-2], t.shape[-1]
            ch, cw = int(round(h * edge_crop)), int(round(w * edge_crop))
            if edge_crop_long_side:                        # D3: crop only along the longer side
                ch, cw = (0, int(w * edge_crop)) if w > h else (int(h * edge_crop), 0)
            if h - 2 * ch >= 8 and w - 2 * cw >= 8:
                t = t[..., ch:h - ch, cw:w - cw]
        if t.shape[-2:] != (size, size):
            # Default ("bilinear") preserves the original no-antialias behavior (D3/STALL unchanged);
            # "bicubic" enables antialiasing to match the official ReStraV resize.
            t = F.interpolate(t, size=(size, size), mode=interp, align_corners=False,
                              antialias=(interp == "bicubic"))
        return t

    try:
        frames = reader.get_batch(indices.tolist()).asnumpy()  # [T, H, W, 3] uint8
        clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        clip = _resize(clip)
    except Exception:
        per_frame = []
        for i in indices.tolist():
            try:
                f = reader[i].asnumpy()  # [H, W, 3]
                ft = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                per_frame.append(_resize(ft))
            except Exception:
                per_frame.append(torch.zeros(1, 3, size, size))
        clip = torch.cat(per_frame, dim=0) if per_frame else torch.zeros(num_frames, 3, size, size)
    # Coerce to exactly 3 channels: some clips decode to 1 (grayscale), 4 (RGBA) or more
    # (e.g. packed/stereo streams) channels, which breaks the 3-channel Normalize downstream.
    c = clip.shape[1]
    if c == 1:
        clip = clip.repeat(1, 3, 1, 1)
    elif c != 3:
        clip = clip[:, :3]
    if bgr:
        clip = clip[:, [2, 1, 0]]
    return clip.clamp_(0.0, 1.0)


def _safe_extract_zip(zip_path: str | Path, destination_dir: str | Path) -> None:
    """Extract a zip file without allowing path traversal outside destination_dir."""
    zip_path = Path(zip_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_root = destination_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (destination_dir / member.filename).resolve()
            if not member_path.is_relative_to(destination_root):
                raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
        zf.extractall(destination_dir)


def _safe_extract_tar(tar_path: str | Path, destination_dir: str | Path) -> None:
    """Extract a tar file without allowing path traversal outside destination_dir."""
    tar_path = Path(tar_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_root = destination_dir.resolve()

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            member_path = (destination_dir / member.name).resolve()
            if not member_path.is_relative_to(destination_root):
                raise RuntimeError(f"Refusing to extract unsafe tar member: {member.name}")
        tf.extractall(destination_dir)


def _download_hf_file(
    repo_id: str,
    filename: str,
    download_dir: str | Path,
    repo_type: str = "dataset",
) -> Path:
    """Download one file from the Hugging Face Hub into download_dir and return its local path."""
    from huggingface_hub import hf_hub_download

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=download_dir,
        )
    )


class CoSpyDataset(Dataset):
    """CO-SPY-Bench test-only dataset (Cheng et al., CVPR 2025).

    The benchmark lives in two HuggingFace dataset repos:

    * Synthetic images: ``ruojiruoli/Co-Spy-Bench`` as
      ``{source_dataset}/{model}/images_with_metadata.tar.gz`` archives,
      with five real-image source datasets and 22 generator models.
    * Real images: ``ruojiruoli/Co-Spy-Misc`` as a single
      ``real_image_examples.zip`` archive (one subfolder per source).

    After download, ``prepare_split`` restructures the data into the exact
    layout used by ``CNNDetectionDataset`` (``cnn_synth_testset``)::

        <root>/test/
        ├── <source>__<model>/
        │   ├── 0_real/*.png   (hardlinks into a single shared real pool)
        │   └── 1_fake/*.png   (moved from the extracted tarball)
        ├── ...
        └── (5 * 22 = 110 class folders)

    The class name uses ``<source>__<model>`` (separator ``__``), which gives
    up to 110 unique ``folder_name``s matching the granularity of Table 2 in
    the paper. Real images are physically stored once per source under
    ``<root>/test/_real_pool/<source>/`` and exposed inside each per-model
    ``0_real/`` directory via ``os.link`` (zero extra disk usage); a
    ``shutil.copy2`` fallback is used on filesystems that do not support
    hardlinks. With this layout, label and folder extraction reuse the
    same convention as ``CNNDetectionDataset``: the leaf folder name
    (``0_real``/``1_fake``) drives the label and the grandparent folder
    (``<source>__<model>``) drives ``folder_name``.

    Note: ``ruojiruoli/Co-Spy-Misc/real_image_examples.zip`` is a *partial*
    release that ships reals for ``CC3M``/``SBU``/``TextCaps`` only (with
    that exact casing). Reals for ``mscoco`` and ``flickr`` must be obtained
    from their original sources and placed under
    ``<root>/test/_real_pool/<source>/``; class folders without reals will
    contain only ``1_fake/``.
    """

    repo_id = "ruojiruoli/Co-Spy-Bench"
    real_repo_id = "ruojiruoli/Co-Spy-Misc"
    real_filename = "real_image_examples.zip"

    default_source_datasets: tuple[str, ...] = (
        "mscoco",
        "flickr",
        "cc3m",
        "textcaps",
        "sbu",
    )
    default_generator_models: tuple[str, ...] = (
        "CompVis@ldm-text2im-large-256",
        "CompVis@stable-diffusion-v1-4",
        "runwayml@stable-diffusion-v1-5",
        "segmind@SSD-1B",
        "segmind@tiny-sd",
        "segmind@SegMoE-SD-4x2-v0",
        "segmind@small-sd",
        "stabilityai@stable-diffusion-2-1",
        "stabilityai@stable-diffusion-3-medium-diffusers",
        "stabilityai@sdxl-turbo",
        "stabilityai@stable-diffusion-2",
        "stabilityai@stable-diffusion-xl-base-1.0",
        "playgroundai@playground-v2.5-1024px-aesthetic",
        "playgroundai@playground-v2-1024px-aesthetic",
        "playgroundai@playground-v2-512px-base",
        "playgroundai@playground-v2-256px-base",
        "PixArt-alpha@PixArt-XL-2-1024-MS",
        "PixArt-alpha@PixArt-XL-2-512x512",
        "latent-consistency@lcm-lora-sdxl",
        "latent-consistency@lcm-lora-sdv1-5",
        "black-forest-labs@FLUX.1-schnell",
        "black-forest-labs@FLUX.1-dev",
    )

    CLASS_SEPARATOR = "__"
    REAL_LABEL_DIR = "0_real"
    FAKE_LABEL_DIR = "1_fake"
    REAL_POOL_DIR = "_real_pool"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        source_datasets: tuple[str, ...] | list[str] | None = None,
        generator_models: tuple[str, ...] | list[str] | None = None,
        include_real: bool = True,
        repo_id: str | None = None,
        real_repo_id: str | None = None,
        real_filename: str | None = None,
    ):
        super().__init__()
        if split != "test":
            raise ValueError("CoSpyDataset only supports the 'test' split.")

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.repo_id = repo_id or self.repo_id
        self.real_repo_id = real_repo_id or self.real_repo_id
        self.real_filename = real_filename or self.real_filename
        self.source_datasets = tuple(source_datasets) if source_datasets else self.default_source_datasets
        self.generator_models = tuple(generator_models) if generator_models else self.default_generator_models
        self.include_real = include_real
        self.destination_dir = self.root_dir / self.split

        if prefetch and not _has_images(self.destination_dir):
            print("Downloading CO-SPY-Bench test split from Hugging Face...")
            self._download_data()

        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        source_datasets: tuple[str, ...] | list[str] | None = None,
        generator_models: tuple[str, ...] | list[str] | None = None,
        include_real: bool = True,
        repo_id: str = "ruojiruoli/Co-Spy-Bench",
        real_repo_id: str = "ruojiruoli/Co-Spy-Misc",
        real_filename: str = "real_image_examples.zip",
        force: bool = False,
        keep_downloads: bool = False,
    ) -> None:
        """Download CO-SPY-Bench and lay it out in CNNDetection-style ``test/`` tree."""
        if split != "test":
            raise ValueError("CoSpyDataset only supports the 'test' split.")

        from huggingface_hub import hf_hub_download, snapshot_download

        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_images(destination_dir):
            return
        if force:
            shutil.rmtree(destination_dir, ignore_errors=True)
        destination_dir.mkdir(parents=True, exist_ok=True)

        sources = tuple(source_datasets) if source_datasets else cls.default_source_datasets
        models = tuple(generator_models) if generator_models else cls.default_generator_models

        bench_download_dir = root_dir / "_downloads" / "Co-Spy-Bench"
        patterns = [
            f"{ds}/{model}/images_with_metadata.tar.gz"
            for ds in sources
            for model in models
        ]
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=bench_download_dir,
            allow_patterns=patterns,
        )

        for tar_path in sorted(bench_download_dir.rglob("images_with_metadata.tar.gz")):
            rel = tar_path.relative_to(bench_download_dir)
            if len(rel.parts) < 2:
                continue
            source_name = rel.parts[0]
            extract_target = bench_download_dir / source_name
            _safe_extract_tar(tar_path, extract_target)

        real_extracted_root: Path | None = None
        if include_real:
            real_download_dir = root_dir / "_downloads" / "Co-Spy-Misc"
            real_download_dir.mkdir(parents=True, exist_ok=True)
            real_path = hf_hub_download(
                repo_id=real_repo_id,
                filename=real_filename,
                repo_type="dataset",
                local_dir=real_download_dir,
            )
            real_extract_dir = real_download_dir / "extracted"
            _safe_extract_zip(real_path, real_extract_dir)
            real_extracted_root = cls._find_real_root(real_extract_dir, sources)

        cls._restructure_to_cnndet_layout(
            synthetic_root=bench_download_dir,
            real_root=real_extracted_root,
            destination_dir=destination_dir,
            sources=sources,
            models=models,
        )

        if not _has_images(destination_dir):
            raise RuntimeError(
                "CO-SPY-Bench download did not produce any recognizable images. "
                f"Check that the HuggingFace repo {repo_id!r} is reachable and "
                "that at least one (source_dataset, model) pair has been requested."
            )

        if not keep_downloads:
            shutil.rmtree(root_dir / "_downloads", ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        source_datasets: tuple[str, ...] | list[str] | None = None,
        generator_models: tuple[str, ...] | list[str] | None = None,
        include_real: bool = True,
        repo_id: str = "ruojiruoli/Co-Spy-Bench",
        real_repo_id: str = "ruojiruoli/Co-Spy-Misc",
        real_filename: str = "real_image_examples.zip",
        force: bool = False,
        keep_downloads: bool = False,
    ) -> None:
        """Download CO-SPY-Bench as a test-only dataset under ``root_dir/test``."""
        cls.prepare_split(
            root_dir=root_dir,
            split="test",
            source_datasets=source_datasets,
            generator_models=generator_models,
            include_real=include_real,
            repo_id=repo_id,
            real_repo_id=real_repo_id,
            real_filename=real_filename,
            force=force,
            keep_downloads=keep_downloads,
        )

    @staticmethod
    def _case_insensitive_subdir(parent: Path, name: str) -> Path | None:
        """Return ``parent/<entry>`` whose lowercase name equals ``name.lower()``."""
        if not parent.is_dir():
            return None
        target = name.lower()
        for entry in parent.iterdir():
            if entry.is_dir() and entry.name.lower() == target:
                return entry
        return None

    @classmethod
    def _find_real_root(cls, real_extract_dir: Path, sources: tuple[str, ...]) -> Path | None:
        """Locate the directory that holds per-source real-image subfolders.

        The upstream ``real_image_examples.zip`` only ships reals for a subset
        of the five Co-Spy sources (currently ``CC3M``/``SBU``/``TextCaps``)
        and uses an inconsistent casing for the per-source folder names, so
        the search is case-insensitive and accepts a partial source set.
        """
        candidates: list[Path] = [
            real_extract_dir / "real_image_examples",
            real_extract_dir,
        ]
        candidates.extend(p for p in real_extract_dir.rglob("real_image_examples") if p.is_dir())
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            if any(cls._case_insensitive_subdir(candidate, src) is not None for src in sources):
                return candidate
        for candidate in real_extract_dir.rglob("*"):
            if not candidate.is_dir():
                continue
            if any(cls._case_insensitive_subdir(candidate, src) is not None for src in sources):
                return candidate
        return None

    @classmethod
    def _restructure_to_cnndet_layout(
        cls,
        synthetic_root: Path,
        real_root: Path | None,
        destination_dir: Path,
        sources: tuple[str, ...],
        models: tuple[str, ...],
    ) -> None:
        """Move synthetic + hardlink real images into ``<source>__<model>/{0_real,1_fake}``."""
        destination_dir.mkdir(parents=True, exist_ok=True)

        real_pool_dir = destination_dir / cls.REAL_POOL_DIR
        if real_root is not None:
            for source in sources:
                source_real_dir = cls._case_insensitive_subdir(real_root, source)
                if source_real_dir is None:
                    continue
                pool_target = real_pool_dir / source
                pool_target.mkdir(parents=True, exist_ok=True)
                for img in _iter_images(source_real_dir):
                    target = cls._dedup_name(pool_target / img.name, img)
                    if target.exists():
                        continue
                    shutil.move(str(img), str(target))

        report: list[tuple[str, str, int, int]] = []
        for source in sources:
            for model in models:
                model_synthetic_dir = synthetic_root / source / model
                model_images = (
                    list(_iter_images(model_synthetic_dir))
                    if model_synthetic_dir.is_dir()
                    else []
                )
                source_real_pool = real_pool_dir / source
                pool_images = (
                    list(_iter_images(source_real_pool))
                    if source_real_pool.is_dir()
                    else []
                )
                if not model_images and not pool_images:
                    continue

                class_dir = destination_dir / f"{source}{cls.CLASS_SEPARATOR}{model}"
                if model_images:
                    fake_dir = class_dir / cls.FAKE_LABEL_DIR
                    fake_dir.mkdir(parents=True, exist_ok=True)
                    for img in model_images:
                        target = cls._dedup_name(fake_dir / img.name, img)
                        if target.exists():
                            continue
                        shutil.move(str(img), str(target))
                if pool_images:
                    real_dir = class_dir / cls.REAL_LABEL_DIR
                    real_dir.mkdir(parents=True, exist_ok=True)
                    for img in pool_images:
                        target = cls._dedup_name(real_dir / img.name, img)
                        if target.exists():
                            continue
                        cls._hardlink_or_copy(img, target)

                report.append((source, model, len(model_images), len(pool_images)))

        if report:
            sources_missing_reals = sorted({src for src, _, _, n_real in report if n_real == 0})
            sources_with_reals = sorted({src for src, _, _, n_real in report if n_real > 0})
            print(f"[CoSpyDataset] reals available for sources : {sources_with_reals}")
            if sources_missing_reals:
                print(
                    f"[CoSpyDataset] reals MISSING for sources    : {sources_missing_reals}  "
                    f"(class folders will have only '{cls.FAKE_LABEL_DIR}/'; provide reals "
                    f"under '{real_pool_dir}/<source>/' to include them)"
                )

    @staticmethod
    def _hardlink_or_copy(src: Path, dst: Path) -> None:
        """Create ``dst`` as a hardlink to ``src``; fall back to a copy on failure."""
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    @staticmethod
    def _dedup_name(target: Path, source: Path) -> Path:
        """Append a short SHA-1 prefix to ``target`` when its name already exists."""
        if not target.exists():
            return target
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
        return target.with_name(f"{target.stem}_{digest}{target.suffix}")

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            source_datasets=self.source_datasets,
            generator_models=self.generator_models,
            include_real=self.include_real,
            repo_id=self.repo_id,
            real_repo_id=self.real_repo_id,
            real_filename=self.real_filename,
            force=force,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")

    def _load_data(self):
        images, labels, folder_names = [], [], []
        split_dir = self.destination_dir
        pool_marker = self.REAL_POOL_DIR
        for filepath in _iter_images(split_dir):
            if pool_marker in filepath.parts:
                continue
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(
                filepath.parent.parent.name
                if filepath.parent.parent != split_dir.parent
                else self.split
            )
        return images, labels, folder_names


class GenImageDataset(Dataset):
    google_drive_folder_url = "https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
    hf_repo_id = None
    hf_subdir = None
    split_aliases = {
        "train": "train",
        "val": "test",
        "test": "test",
    }
    source_split_markers = {
        "train": ("train",),
        "test": ("val", "validation", "test"),
    }
    class_dir_to_label = {
        "nature": "real",
        "real": "real",
        "ai": "fake",
        "fake": "fake",
        "false": "fake",
    }

    def __init__(self, root_dir: str | Path, split: Literal["train", "val", "test"], transform: AIGITransformations | None = None, prefetch: bool = False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.requested_split = split
        self.split = self.split_aliases[split]
        self.transform = transform
        self.prefetch = prefetch
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print(f"Downloading {self.split} split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @staticmethod
    def _tokens(path: Path) -> list[str]:
        tokens: list[str] = []
        for part in path.parts:
            tokens.extend(token for token in re.split(r"[^a-z0-9]+", part.lower()) if token)
        return tokens

    @classmethod
    def download_files(
        cls,
        download_dir: str | Path,
        url: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download the shared GenImage Google Drive folder into download_dir."""
        download_dir = Path(download_dir)
        if download_dir.exists() and any(download_dir.iterdir()) and not force:
            return download_dir

        if force:
            shutil.rmtree(download_dir, ignore_errors=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        if cls.hf_repo_id:
            from huggingface_hub import hf_hub_download, list_repo_files

            repo_files = [f for f in list_repo_files(cls.hf_repo_id, repo_type="dataset") if not f.endswith("/")]
            if cls.hf_subdir:
                prefix = str(cls.hf_subdir).rstrip("/") + "/"
                repo_files = [f for f in repo_files if f.startswith(prefix)]
            if not repo_files:
                raise RuntimeError(
                    f"No files found in HF dataset {cls.hf_repo_id!r} under subdir {cls.hf_subdir!r}."
                )
            for filename in sorted(repo_files):
                hf_hub_download(
                    repo_id=cls.hf_repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(download_dir),
                )
            return download_dir

        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "Downloading the GenImage Google Drive folder requires `gdown`. "
                "Install it with `pip install gdown`."
            ) from exc

        gdown.download_folder(
            url=url or cls.google_drive_folder_url,
            output=str(download_dir),
            quiet=False,
            use_cookies=False,
        )
        return download_dir

    @classmethod
    def _extract_downloaded_archives(cls, download_dir: Path) -> None:
        for zip_path in sorted(download_dir.rglob("*.zip")):
            extract_dir = zip_path.with_suffix("")
            if _has_images(extract_dir):
                continue
            if zip_path.with_suffix(".z01").exists():
                cls._extract_spanned_zip(zip_path, extract_dir)
            else:
                _safe_extract_zip(zip_path, extract_dir)

    @staticmethod
    def _extract_spanned_zip(zip_path: Path, destination_dir: Path) -> None:
        """Extract a multi-part (spanned) zip (.z01..+.zip) using 7z."""
        seven_zip = shutil.which("7z") or shutil.which("7za")
        if seven_zip is None:
            raise RuntimeError(
                "Extracting multi-part (spanned) zip archives requires the `7z` command. Install p7zip-full."
            )
        destination_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [seven_zip, "x", str(zip_path), f"-o{destination_dir}", "-y"],
            cwd=zip_path.parent,
            check=True,
        )

    @classmethod
    def _source_split_for_path(cls, filepath: Path, destination_split: Literal["train", "test"]) -> bool:
        tokens = cls._tokens(filepath.parent)
        return any(marker in tokens for marker in cls.source_split_markers[destination_split])

    @classmethod
    def _label_dir_for_path(cls, filepath: Path) -> str | None:
        tokens = cls._tokens(filepath.parent)
        for source_name, label_dir in cls.class_dir_to_label.items():
            if source_name in tokens:
                return label_dir
        return None

    @classmethod
    def _relative_image_path(cls, filepath: Path) -> Path:
        parts = list(filepath.parts)
        lower_parts = [part.lower() for part in parts]
        marker_index = -1
        for idx, part in enumerate(lower_parts[:-1]):
            tokens = cls._tokens(Path(part))
            if any(token in cls.source_split_markers["train"] + cls.source_split_markers["test"] for token in tokens):
                marker_index = idx
            if any(token in cls.class_dir_to_label for token in tokens):
                marker_index = idx
        relative_parts = parts[marker_index + 1 :] if marker_index >= 0 else [filepath.name]
        return Path(*relative_parts)

    @staticmethod
    def _deduplicated_destination(destination_path: Path, source_path: Path) -> Path:
        if not destination_path.exists():
            return destination_path

        digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
        return destination_path.with_name(f"{destination_path.stem}_{digest}{destination_path.suffix}")

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        url: str | None = None,
        force: bool = False,
        keep_downloads: bool = True,
    ) -> None:
        """Download GenImage and normalize it to root/train and root/test."""
        root_dir = Path(root_dir)
        train_dir = root_dir / "train"
        test_dir = root_dir / "test"
        if not force and _has_images(train_dir) and _has_images(test_dir):
            return

        if force:
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(test_dir, ignore_errors=True)

        download_dir = root_dir / "_downloads" / "GenImage"
        cls.download_files(download_dir, url=url, force=force)
        cls._extract_downloaded_archives(download_dir)

        copied = {"train": 0, "test": 0}
        for destination_split in ("train", "test"):
            destination_split_dir = root_dir / destination_split
            destination_split_dir.mkdir(parents=True, exist_ok=True)

            for source_path in _iter_images(download_dir):
                source_relative_path = source_path.relative_to(download_dir)
                if not cls._source_split_for_path(source_relative_path, destination_split):
                    continue

                label_dir = cls._label_dir_for_path(source_relative_path)
                if label_dir is None:
                    continue

                relative_path = cls._relative_image_path(source_relative_path)
                destination_path = cls._deduplicated_destination(
                    destination_split_dir / label_dir / relative_path,
                    source_path,
                )
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                copied[destination_split] += 1

        missing = [split for split, count in copied.items() if count == 0]
        if missing:
            raise RuntimeError(
                "GenImage download did not contain recognizable images for "
                f"{', '.join(missing)}. Expected train/val split folders and ai/nature class folders."
            )

        if not keep_downloads:
            shutil.rmtree(root_dir / "_downloads", ignore_errors=True)

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        url: str | None = None,
        force: bool = False,
        keep_downloads: bool = True,
    ) -> None:
        """Download and normalize GenImage; val is stored as the test split."""
        destination_split = cls.split_aliases[split]
        destination_dir = Path(root_dir) / destination_split
        if not force and _has_images(destination_dir):
            return
        cls.prepare_dataset(root_dir, url=url, force=force, keep_downloads=keep_downloads)

    def _download_data(self, force: bool = False) -> None:
        """Download and normalize this dataset split from Google Drive."""
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.requested_split,
            url=self.google_drive_folder_url,
            force=force,
        )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])
    
    def _iter_images(self, directory: Path):
        return _iter_images(directory)
    
    def _load_data(self):
        images, labels, folder_names = [], [], []
        split_dir = self.destination_dir
        for filepath in self._iter_images(split_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(self.split)
        return images, labels, folder_names

    @staticmethod
    def _get_label(filepath: Path) -> int:
        names = [part.lower() for part in filepath.parent.parts]
        if "real" in names:
            return 0
        if "fake" in names or "false" in names:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")


class CNNDetectionDataset(Dataset):
    google_drive_folder_url = "https://drive.google.com/drive/u/0/folders/1RwCSaraEUctIwFgoQXWMKFvW07gM80_3"
    archive_name_markers = {
        "train": ("progan_train", "train"),
        "val": ("progan_val", "val"),
        "test": ("cnn_synth_testset", "test"),
    }

    def __init__(self, root_dir: str | Path, split: Literal["train", "val", "test"], transform: AIGITransformations | None = None, prefetch: bool = False,
                 classes: list[str] | None = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.prefetch = prefetch
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print(f"Downloading {self.split} split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        if classes:   # e.g. the 4-class ProGAN setting (car, cat, chair, horse) of CNNDetection/NPR/FatFormer
            keep = [i for i, f in enumerate(self.folder_names) if f in set(classes)]
            if not keep:
                raise ValueError(f"CNNDetection {split}: none of the classes {list(classes)} found")
            self.images = [self.images[i] for i in keep]; self.labels = [self.labels[i] for i in keep]
            self.folder_names = [self.folder_names[i] for i in keep]
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @classmethod
    def download_archives(
        cls,
        archive_dir: str | Path,
        url: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download the shared CNNDetection Google Drive folder into archive_dir."""
        archive_dir = Path(archive_dir)
        if archive_dir.exists() and any(archive_dir.iterdir()) and not force:
            return archive_dir

        archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "Downloading a Google Drive folder requires `gdown`. "
                "Install it with `pip install gdown`."
            ) from exc

        gdown.download_folder(
            url=url or cls.google_drive_folder_url,
            output=str(archive_dir),
            quiet=False,
            use_cookies=False,
        )
        return archive_dir

    @classmethod
    def _find_split_archives(
        cls,
        archive_dir: Path,
        split: Literal["train", "val", "test"],
    ) -> list[Path]:
        markers = cls.archive_name_markers[split]
        archive_suffixes = (".zip", ".7z", ".001")
        archives = [
            p
            for p in archive_dir.rglob("*")
            if p.is_file()
            and p.name.lower().endswith(archive_suffixes)
            and any(marker in p.name.lower() for marker in markers)
        ]

        if split == "train":
            zip_archives = [p for p in archives if p.suffix.lower() == ".zip"]
            if zip_archives:
                return sorted(zip_archives)
            first_7z_parts = [p for p in archives if p.name.lower().endswith(".7z.001")]
            if first_7z_parts:
                return sorted(first_7z_parts)

        return sorted(archives)

    @staticmethod
    def _extract_7z(archive_path: Path, destination_dir: Path) -> None:
        seven_zip = shutil.which("7z") or shutil.which("7za")
        if seven_zip is None:
            raise RuntimeError("Extracting .7z archives requires the `7z` command. Install p7zip-full.")

        destination_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [seven_zip, "x", str(archive_path), f"-o{destination_dir}", "-y"],
            cwd=archive_path.parent,
            check=True,
        )

    @classmethod
    def _extract_archive(cls, archive_path: Path, destination_dir: Path) -> None:
        if archive_path.suffix.lower() == ".zip":
            _safe_extract_zip(archive_path, destination_dir)
            return

        if archive_path.name.lower().endswith(".7z") or archive_path.name.lower().endswith(".7z.001"):
            cls._extract_7z(archive_path, destination_dir)
            for zip_path in destination_dir.rglob("*.zip"):
                _safe_extract_zip(zip_path, destination_dir)
            return

        raise RuntimeError(f"Unsupported archive type: {archive_path}")

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download and extract one CNNDetection split under root_dir/split."""
        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_images(destination_dir):
            return

        archive_dir = root_dir / "_archives" / "CNNDetection"
        cls.download_archives(archive_dir)
        archive_paths = cls._find_split_archives(archive_dir, split)
        if not archive_paths:
            raise RuntimeError(f"No archive found for CNNDetection {split!r} split in {archive_dir}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        for archive_path in archive_paths:
            cls._extract_archive(archive_path, destination_dir)

        if not keep_archives:
            shutil.rmtree(archive_dir, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        splits: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test"),
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download and extract the requested CNNDetection splits."""
        for split in splits:
            cls.prepare_split(root_dir, split, force=force, keep_archives=keep_archives)

    def _download_data(self, force: bool = False) -> None:
        """Download and extract this dataset split under root_dir/split."""
        self.prepare_split(self.root_dir, self.split, force=force)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")
    
    def _iter_images(self, directory: Path):
        return _iter_images(directory)

    def _load_data(self):
        images, labels, folder_names = [], [], []
        split_dir = self.destination_dir
        for filepath in self._iter_images(split_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name if filepath.parent.parent != split_dir.parent else self.split)
        return images, labels, folder_names


class CNNDetectionDatasetModule:
    def __init__(self, config):
        self.config = config
        train_list = list(config.DatasetConfig.train_transform_list)
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        classes = config.DatasetConfig.get("train_classes", None)   # ProGAN classes for train/val (None: all 20)
        classes = list(classes) if classes else None
        self.train_dataset = CNNDetectionDataset(config.DatasetConfig.root_dir, "train", self.train_transform, prefetch=prefetch, classes=classes)
        self.val_dataset = CNNDetectionDataset(config.DatasetConfig.root_dir, "val", self.eval_transform, prefetch=prefetch, classes=classes)
        self.test_dataset = CNNDetectionDataset(config.DatasetConfig.root_dir, "test", self.eval_transform, prefetch=prefetch)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class GenImageDatasetModule:
    def __init__(self, config):
        self.config = config
        train_list = list(config.DatasetConfig.train_transform_list)
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        source_url = config.DatasetConfig.get("source_url", GenImageDataset.google_drive_folder_url)
        GenImageDataset.google_drive_folder_url = source_url
        GenImageDataset.hf_repo_id = config.DatasetConfig.get("hf_repo_id", None)
        GenImageDataset.hf_subdir = config.DatasetConfig.get("hf_subdir", None)
        self.train_dataset = GenImageDataset(config.DatasetConfig.root_dir, "train", self.train_transform, prefetch=prefetch)
        self.val_dataset = GenImageDataset(config.DatasetConfig.root_dir, "val", self.eval_transform, prefetch=prefetch)
        self.test_dataset = GenImageDataset(config.DatasetConfig.root_dir, "test", self.eval_transform, prefetch=prefetch)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class AIGIBenchDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        setting: Literal["Setting-I", "Setting-II"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.setting = setting
        self.transform = transform
        self.unique_folder_names = ["train"]
        self.destination_dir = self.root_dir / self.split
        self.huggingface_path = "https://huggingface.co/datasets/HorizonTEL/AIGIBench/tree/main"
        self.huggingface_repo_id = "HorizonTEL/AIGIBench"
        if prefetch:
            if _has_images(self.destination_dir):
                print(f"on prefetch_hf mode, {self.destination_dir} already exists")
            else:
                print(f"Downloading {self.split} split from Hugging Face...")
                self._download_data()
        if self.split != "train":
            self.unique_folder_names += [d.name for d in self.destination_dir.iterdir() if d.is_dir()] if self.destination_dir.exists() else []
        self.images, self.labels, self.folder_names = self._load_data()

    @staticmethod
    def _split_has_images(directory: Path) -> bool:
        return _has_images(directory)

    @classmethod
    def prepare_split(
        cls,
        destination_dir: str | Path,
        split: Literal["train", "val", "test"],
        repo_id: str = "HorizonTEL/AIGIBench",
        force: bool = False,
    ) -> None:
        """Download all zip archives for one split from the Hub and extract under root_dir."""
        from huggingface_hub import hf_hub_download, list_repo_files

        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{split}/"
        files = list_repo_files(repo_id, repo_type="dataset")
        zips = sorted(f for f in files if f.startswith(prefix) and f.lower().endswith(".zip"))
        if not zips:
            raise RuntimeError(f"No .zip files under {prefix!r} in dataset {repo_id!r}")

        for rel in zips:
            stem = Path(rel).stem
            target_dir = destination_dir / stem
            if not force and target_dir.is_dir() and cls._split_has_images(target_dir):
                continue

            local_zip = hf_hub_download(
                repo_id=repo_id,
                filename=rel,
                repo_type="dataset",
            )
            _safe_extract_zip(local_zip, destination_dir)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        splits: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test"),
        repo_id: str = "HorizonTEL/AIGIBench",
        force: bool = False,
    ) -> None:
        """Download and extract AIGIBench split archives under root_dir/train, root_dir/val, root_dir/test."""
        root_dir = Path(root_dir)
        for split in splits:
            cls.prepare_split(
                destination_dir=root_dir / split,
                split=split,
                repo_id=repo_id,
                force=force,
            )

    def _download_data(self, force: bool = False) -> None:
        """Download and extract this dataset split from Hugging Face (skips subsets that already contain images)."""
        self.prepare_split(
            destination_dir=self.destination_dir,
            split=self.split,
            repo_id=self.huggingface_repo_id,
            force=force,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path, folder_names: list[str] | None = None) -> int:
        name = filepath.parent.name
        if "real" in name:
            return 0
        elif "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")
    
    def _iter_images(self, directory: Path):
        return _iter_images(directory)

    def _load_data(self):
        images, labels, folder_names = [], [], []
        split_dir = self.destination_dir

        for filepath in self._iter_images(split_dir):
            if self.split == "train" and self.setting == "Setting-I":
                if "sdv1.4" in str(filepath):
                    continue

            images.append(filepath)
            labels.append(self._get_label(filepath, self.unique_folder_names))
            folder_names.append(
                filepath.parent.parent.name if self.split != "train" else "train"
            )

        return images, labels, folder_names


class DiTFakeDataset(Dataset):
    repo_id = "lioooox/DiTFake"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["test"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        repo_id: str | None = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.repo_id = repo_id or self.repo_id
        self.destination_dir = self.root_dir / self.split
        self._normalize_existing_layout()

        if prefetch and not _has_images(self.destination_dir):
            print("Downloading DiTFake test split from Hugging Face...")
            self._download_data()
            self.destination_dir = self.root_dir / self.split
            self._normalize_existing_layout()

        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    def _normalize_existing_layout(self) -> None:
        nested_split_dir = self.root_dir / "DiTFake" / self.split
        if _has_images(nested_split_dir) and not _has_images(self.destination_dir):
            if self.destination_dir.exists():
                shutil.rmtree(self.destination_dir)
            self.destination_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(nested_split_dir), str(self.destination_dir))
        nested_root = self.root_dir / "DiTFake"
        if nested_root.exists() and not any(nested_root.rglob("*")):
            shutil.rmtree(nested_root, ignore_errors=True)

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        repo_id: str = "lioooox/DiTFake",
        force: bool = False,
    ) -> None:
        """Download one DiTFake split subtree into root_dir/split."""
        from huggingface_hub import snapshot_download

        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_images(destination_dir):
            return

        root_dir.mkdir(parents=True, exist_ok=True)
        download_dir = root_dir / "_downloads" / "DiTFake"
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=download_dir,
            allow_patterns=[f"DiTFake/{split}/**"],
        )
        downloaded_split_dir = download_dir / "DiTFake" / split
        if downloaded_split_dir.exists() and downloaded_split_dir != destination_dir:
            if destination_dir.exists() and (force or not _has_images(destination_dir)):
                shutil.rmtree(destination_dir)
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded_split_dir), str(destination_dir))
        shutil.rmtree(root_dir / "_downloads", ignore_errors=True)
        nested_root = root_dir / "DiTFake"
        if nested_root.exists():
            shutil.rmtree(nested_root, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        repo_id: str = "lioooox/DiTFake",
        force: bool = False,
    ) -> None:
        """Download DiTFake test split under root_dir."""
        cls.prepare_split(root_dir=root_dir, split="test", repo_id=repo_id, force=force)

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            repo_id=self.repo_id,
            force=force,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")

    def _load_data(self):
        images, labels, folder_names = [], [], []
        for filepath in _iter_images(self.destination_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name)
        return images, labels, folder_names



class FreqNetDetectionDataset(Dataset):
    google_drive_folder_url = "https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["test"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        source_url: str | None = None,
    ):
        super().__init__()
        if split != "test":
            raise ValueError("FreqNetDetectionDataset only supports the test split.")
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.prefetch = prefetch
        self.source_url = source_url or self.google_drive_folder_url
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print("Downloading FreqNetDetection test split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @classmethod
    def download_archives(
        cls,
        archive_dir: str | Path,
        url: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download the shared FreqNetDetection Google Drive folder into archive_dir."""
        archive_dir = Path(archive_dir)
        if archive_dir.exists() and any(archive_dir.iterdir()) and not force:
            return archive_dir

        if force:
            shutil.rmtree(archive_dir, ignore_errors=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "Downloading the FreqNetDetection Google Drive folder requires `gdown`. "
                "Install it with `pip install gdown`."
            ) from exc

        gdown.download_folder(
            url=url or cls.google_drive_folder_url,
            output=str(archive_dir),
            quiet=False,
            use_cookies=False,
        )
        return archive_dir

    @staticmethod
    def _archive_paths(archive_dir: Path) -> list[Path]:
        archive_suffixes = (".tar.gz", ".tgz", ".tar")
        return sorted(
            p
            for p in archive_dir.rglob("*")
            if p.is_file() and p.name.lower().endswith(archive_suffixes)
        )

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        source_url: str | None = None,
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download FreqNetDetection generator archives and extract them under root_dir/test."""
        if split != "test":
            raise ValueError("FreqNetDetectionDataset only supports the test split.")

        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if _has_images(destination_dir) and not force:
            return
        if force:
            shutil.rmtree(destination_dir, ignore_errors=True)

        archive_dir = root_dir / "_archives" / "FreqNetDetection"
        cls.download_archives(archive_dir, url=source_url, force=force)
        archive_paths = cls._archive_paths(archive_dir)
        if not archive_paths:
            raise RuntimeError(f"No .tar archives found for FreqNetDetection in {archive_dir}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        for archive_path in archive_paths:
            _safe_extract_tar(archive_path, destination_dir)

        if not _has_images(destination_dir):
            raise RuntimeError("FreqNetDetection archives did not contain recognizable images.")
        if not keep_archives:
            shutil.rmtree(archive_dir, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        source_url: str | None = None,
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download FreqNetDetection as a test-only dataset under root_dir/test."""
        cls.prepare_split(
            root_dir=root_dir,
            split="test",
            source_url=source_url,
            force=force,
            keep_archives=keep_archives,
        )

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            source_url=self.source_url,
            force=force,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")

    def _load_data(self):
        images, labels, folder_names = [], [], []
        for filepath in _iter_images(self.destination_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name)
        return images, labels, folder_names


class UniversialFakeDetectionDataset(Dataset):
    source_url = "https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["test"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        source_url: str | None = None,
    ):
        super().__init__()
        if split != "test":
            raise ValueError("UniversialFakeDetectionDataset only supports the test split.")
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.prefetch = prefetch
        self.source_url = source_url or self.source_url
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print(f"Downloading {self.split} split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _flatten_extracted_tree(extract_dir: Path, destination_dir: Path) -> None:
        source_dir = extract_dir / "diffusion_models"
        if not source_dir.exists():
            nested = [p for p in extract_dir.rglob("diffusion_models") if p.is_dir()]
            source_dir = nested[0] if nested else extract_dir
        container_names = {"diffusion_datasets", "diffusion_dataset", "datasets", "data"}
        child_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
        if len(child_dirs) == 1 and child_dirs[0].name.lower() in container_names:
            source_dir = child_dirs[0]

        destination_dir.mkdir(parents=True, exist_ok=True)
        for child in source_dir.iterdir():
            target = destination_dir / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        source_url: str | None = None,
        force: bool = False,
        keep_downloads: bool = False,
    ) -> None:
        """Download the Universal Fake Detection zip and flatten diffusion_models under root_dir/test."""
        if split != "test":
            raise ValueError("UniversialFakeDetectionDataset only supports the test split.")

        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if _has_images(destination_dir) and not force:
            return
        if force:
            shutil.rmtree(destination_dir, ignore_errors=True)

        import gdown

        download_dir = root_dir / "_downloads" / "UniversialFakeDetection"
        extract_dir = download_dir / "extracted"
        zip_path = download_dir / "universial_fake_detection.zip"
        shutil.rmtree(download_dir, ignore_errors=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        download_source = source_url or cls.source_url
        file_id_match = re.search(r"/d/([^/]+)", download_source) or re.search(r"[?&]id=([^&]+)", download_source)
        download_kwargs = {"id": file_id_match.group(1)} if file_id_match else {"url": download_source}
        gdown.download(
            output=str(zip_path),
            quiet=False,
            **download_kwargs,
        )
        _safe_extract_zip(zip_path, extract_dir)
        cls._flatten_extracted_tree(extract_dir, destination_dir)

        if not _has_images(destination_dir):
            raise RuntimeError(
                "Universal Fake Detection download did not contain images after "
                "flattening diffusion_models into the test split."
            )
        if not keep_downloads:
            shutil.rmtree(download_dir, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        source_url: str | None = None,
        force: bool = False,
        keep_downloads: bool = False,
    ) -> None:
        """Download Universal Fake Detection as a test-only dataset under root_dir/test."""
        cls.prepare_split(
            root_dir=root_dir,
            split="test",
            source_url=source_url,
            force=force,
            keep_downloads=keep_downloads,
        )

    def _load_data(self):
        images, labels, folder_names = [], [], []
        for filepath in _iter_images(self.destination_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name)
        return images, labels, folder_names

    def _get_label(self, filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            source_url=self.source_url,
            force=force,
        )

class AIGIBenchDatasetModule:
    def __init__(self, config):
        self.config = config
        train_list = list(config.DatasetConfig.train_transform_list)
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        self.train_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "train", config.DatasetConfig.setting, self.train_transform, prefetch=prefetch)
        self.val_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "val", config.DatasetConfig.setting, self.eval_transform, prefetch=prefetch)
        self.test_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "test", config.DatasetConfig.setting, self.eval_transform, prefetch=prefetch)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class DiTFakeDatasetModule:
    def __init__(self, config):
        self.config = config
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        repo_id = config.DatasetConfig.get("repo_id", DiTFakeDataset.repo_id)
        self.test_dataset = DiTFakeDataset(config.DatasetConfig.root_dir, "test", self.eval_transform, prefetch=prefetch, repo_id=repo_id)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class FreqNetDetectionDatasetModule:
    def __init__(self, config):
        self.config = config
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        source_url = config.DatasetConfig.get("source_url", FreqNetDetectionDataset.google_drive_folder_url)
        self.test_dataset = FreqNetDetectionDataset(
            config.DatasetConfig.root_dir,
            "test",
            self.eval_transform,
            prefetch=prefetch,
            source_url=source_url,
        )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class UniversialFakeDetectionDatasetModule:
    def __init__(self, config):
        self.config = config
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        source_url = config.DatasetConfig.get("source_url", UniversialFakeDetectionDataset.source_url)
        self.test_dataset = UniversialFakeDetectionDataset(
            config.DatasetConfig.root_dir,
            "test",
            self.eval_transform,
            prefetch=prefetch,
            source_url=source_url,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None


class CoSpyDatasetModule:
    def __init__(self, config):
        self.config = config
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        source_datasets = config.DatasetConfig.get("source_datasets", None)
        generator_models = config.DatasetConfig.get("generator_models", None)
        include_real = bool(config.DatasetConfig.get("include_real", True))
        repo_id = config.DatasetConfig.get("repo_id", CoSpyDataset.repo_id)
        real_repo_id = config.DatasetConfig.get("real_repo_id", CoSpyDataset.real_repo_id)
        real_filename = config.DatasetConfig.get("real_filename", CoSpyDataset.real_filename)
        self.test_dataset = CoSpyDataset(
            config.DatasetConfig.root_dir,
            "test",
            self.eval_transform,
            prefetch=prefetch,
            source_datasets=list(source_datasets) if source_datasets else None,
            generator_models=list(generator_models) if generator_models else None,
            include_real=include_real,
            repo_id=repo_id,
            real_repo_id=real_repo_id,
            real_filename=real_filename,
        )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class VideoFolderDataset(Dataset):
    """Generic AI-generated-video benchmark backed by a folder of video files.

    Expected layout under ``root_dir/<split>`` (mirrors ``CNNDetectionDataset``)::

        <root>/<split>/
        ├── <generator_a>/
        │   ├── 0_real/*.mp4   (or real/)
        │   └── 1_fake/*.mp4   (or fake/)
        ├── <generator_b>/
        │   └── ...
        └── ...

    A flat ``<root>/<split>/{0_real,1_fake}`` layout (no per-generator subdir) is also
    supported. The label is read from the leaf directory name (``real`` -> 0,
    ``fake``/``false`` -> 1) and the per-generator ``folder_name`` from the grandparent
    directory, exactly like the image datasets, so the per-generator AP/AUROC metrics keep
    working. ``__getitem__`` returns ``(frames[T, 3, H, W] float in [0, 1], label, folder_id_index)``.

    Downloading the (large) video benchmarks is out of scope: ``prefetch`` defaults to
    ``False``. If ``hf_repo_id`` is configured the download stub fetches the split subtree
    from the Hugging Face Hub; otherwise it raises ``NotImplementedError`` with guidance.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        num_frames: int = 16,
        fps: float | None = None,
        window_seconds: float = 2.0,
        frame_size: int = 224,
        hf_repo_id: str | None = None,
        window_center: bool = False,
        resize_interp: str = "bilinear",
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_frames = int(num_frames)
        self.fps = fps if fps else None
        self.window_seconds = float(window_seconds)
        self.frame_size = int(frame_size)
        self.hf_repo_id = hf_repo_id
        self.window_center = bool(window_center)
        self.resize_interp = str(resize_interp)
        self.destination_dir = self.root_dir / self.split

        if prefetch and not _has_videos(self.destination_dir):
            print(f"Downloading {self.split} video split into {self.destination_dir}...")
            self._download_data()

        self.videos, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        hf_repo_id: str | None = None,
        force: bool = False,
    ) -> None:
        """Download one video split into ``root_dir/split`` (HF Hub only; stub otherwise)."""
        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_videos(destination_dir):
            return
        if hf_repo_id:
            from huggingface_hub import snapshot_download

            destination_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=str(destination_dir),
                allow_patterns=[f"{split}/**", f"{split}/*"],
            )
            return
        raise NotImplementedError(
            f"No automatic downloader is configured for video split {destination_dir!s}. "
            "Place the benchmark videos under '<root>/<split>/<generator>/{0_real,1_fake}/' "
            "manually, or set 'hf_repo_id' in the dataset config to a Hugging Face dataset repo."
        )

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            hf_repo_id=self.hf_repo_id,
            force=force,
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames = sample_video_frames(
            self.videos[idx],
            num_frames=self.num_frames,
            fps=self.fps,
            window_seconds=self.window_seconds,
            size=self.frame_size,
            window_center=self.window_center,
            interp=self.resize_interp,
        )
        label = torch.tensor(self.labels[idx])
        if self.transform:
            frames = self.transform(frames)
        return frames, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from path: {filepath}")

    def _load_data(self):
        videos, labels, folder_names = [], [], []
        split_dir = self.destination_dir
        for filepath in _iter_videos(split_dir):
            videos.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(
                filepath.parent.parent.name
                if filepath.parent.parent != split_dir.parent
                else self.split
            )
        return videos, labels, folder_names


class VideoFolderDatasetModule:
    def __init__(self, config):
        self.config = config
        cfg = config.DatasetConfig
        train_list = list(cfg.get("train_transform_list", []))
        eval_list = list(cfg.get("eval_transform_list", []))
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = bool(cfg.get("prefetch", False))
        common = dict(
            num_frames=cfg.get("num_frames", 16),
            fps=cfg.get("fps", None),
            window_seconds=cfg.get("window_seconds", 2.0),
            frame_size=cfg.get("frame_size", 224),
            hf_repo_id=cfg.get("hf_repo_id", None),
            window_center=cfg.get("window_center", False),
            resize_interp=cfg.get("resize_interp", "bilinear"),
        )
        self.train_dataset = VideoFolderDataset(cfg.root_dir, "train", self.train_transform, prefetch=prefetch, **common)
        self.val_dataset = VideoFolderDataset(cfg.root_dir, "val", self.eval_transform, prefetch=prefetch, **common)
        self.test_dataset = VideoFolderDataset(cfg.root_dir, "test", self.eval_transform, prefetch=prefetch, **common)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class AIGIDatasetModule:
    def __init__(self, config):
        self.delegate = SynthScopeModuleDict[config.DatasetConfig.name](config)

    def train_dataloader(self):
        return self.delegate.train_dataloader()

    def val_dataloader(self):
        return self.delegate.val_dataloader()

    def test_dataloader(self):
        return self.delegate.test_dataloader()

class EvalCrafterDataset(VideoFolderDataset):
    """EvalCrafter T2V benchmark (Liu et al. 2024) — used by D3.

    All clips are AI-generated (no real videos), so every generator is placed under
    ``<root>/test/1_fake/<generator>/*.mp4``. ``prepare_split`` snapshots the per-generator
    ``*.tar.gz`` from the Hugging Face Hub and safe-extracts each into the ``1_fake`` tree.
    """

    hf_repo_id = "RaphaelLiu/EvalCrafter_T2V_Dataset"

    @classmethod
    def prepare_split(cls, root_dir, split="test", hf_repo_id=None, force=False):
        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_videos(destination_dir):
            return
        from huggingface_hub import snapshot_download

        download_dir = root_dir / "_downloads"
        snapshot_download(
            repo_id=hf_repo_id or cls.hf_repo_id, repo_type="dataset",
            local_dir=str(download_dir), allow_patterns=["*.tar.gz"],
        )
        fake_dir = destination_dir / "1_fake"
        fake_dir.mkdir(parents=True, exist_ok=True)
        for tar in sorted(download_dir.glob("*.tar.gz")):
            _safe_extract_tar(tar, fake_dir)          # -> 1_fake/<generator>/*.mp4
        shutil.rmtree(download_dir, ignore_errors=True)

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(self.root_dir, self.split, self.hf_repo_id, force)


class GenVidBenchDataset(VideoFolderDataset):
    """GenVidBench (Ni et al. 2025) — used by ReStraV.

    Real sources (Vript, HD-VG-130M) -> ``<root>/test/0_real/<src>/``; the 8 generators
    (ms, pika, t2vz, vc2, cogvideo, mora, musev, svd) -> ``<root>/test/1_fake/<gen>/``.
    ``prepare_split`` snapshots the per-source ``.rar``/multi-part ``.7z`` archives and
    extracts them (needs the ``unrar`` binary; ``7z`` for the HD-VG archive).
    """

    hf_repo_id = "bitmind/GenVidBench"
    REAL_SOURCES = {"vript", "hd_vg_130m"}

    @classmethod
    def prepare_split(cls, root_dir, split="test", hf_repo_id=None, force=False):
        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_videos(destination_dir):
            return
        from huggingface_hub import snapshot_download

        download_dir = root_dir / "_downloads"
        snapshot_download(repo_id=hf_repo_id or cls.hf_repo_id, repo_type="dataset", local_dir=str(download_dir))
        for rar in sorted(download_dir.glob("**/*.rar")):
            label = "0_real" if rar.stem.lower() in cls.REAL_SOURCES else "1_fake"
            out = destination_dir / label
            out.mkdir(parents=True, exist_ok=True)
            cls._extract_rar(rar, out)
        seven = sorted(download_dir.glob("**/hd_vg_130m.7z.001"))   # multi-part 7z real source
        if seven:
            out = destination_dir / "0_real"
            out.mkdir(parents=True, exist_ok=True)
            cls._extract_7z(seven[0], out)
        shutil.rmtree(download_dir, ignore_errors=True)

    @staticmethod
    def _extract_rar(archive, out):
        unrar = shutil.which("unrar") or "unrar"
        subprocess.run([unrar, "x", "-o+", str(archive), str(out) + "/"], check=True, capture_output=True)

    @staticmethod
    def _extract_7z(archive, out):
        seven = shutil.which("7z") or "7z"
        subprocess.run([seven, "x", "-y", f"-o{out}", str(archive)], check=True, capture_output=True)
        for nested in Path(out).glob("*.7z"):
            subprocess.run([seven, "x", "-y", f"-o{out}", str(nested)], check=True, capture_output=True)
            nested.unlink(missing_ok=True)

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(self.root_dir, self.split, self.hf_repo_id, force)


class _TestOnlyVideoModule:
    """Shared loader for the eval-only video benchmarks (test split only)."""

    DATASET_CLASS = VideoFolderDataset

    def __init__(self, config):
        self.config = config
        cfg = config.DatasetConfig
        eval_list = list(cfg.get("eval_transform_list", []))
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        self.test_dataset = self.DATASET_CLASS(
            cfg.root_dir, "test", self.eval_transform, prefetch=bool(cfg.get("prefetch", False)),
            num_frames=cfg.get("num_frames", 16), fps=cfg.get("fps", None),
            window_seconds=cfg.get("window_seconds", 2.0), frame_size=cfg.get("frame_size", 224),
            hf_repo_id=cfg.get("hf_repo_id", self.DATASET_CLASS.hf_repo_id),
            window_center=cfg.get("window_center", False), resize_interp=cfg.get("resize_interp", "bilinear"),
        )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        cfg = self.config.DatasetConfig
        return DataLoader(self.test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
                          num_workers=cfg.num_workers, persistent_workers=cfg.persistent_workers, pin_memory=cfg.pin_memory)


class EvalCrafterDatasetModule(_TestOnlyVideoModule):
    DATASET_CLASS = EvalCrafterDataset


class GenVidBenchDatasetModule(_TestOnlyVideoModule):
    DATASET_CLASS = GenVidBenchDataset


# ---------------------------------------------------------------------------
# Eval-only cross-generator benchmark datasets (reproduction suite). Image benchmarks share a
# small test-only base + module (mirroring the video VideoFolderDataset / _TestOnlyVideoModule
# pattern); each subclass only declares its download source and restructures the download into
# the common ``<root>/test/<generator>/{0_real,1_fake}/`` layout used by the per-generator metrics.
# ---------------------------------------------------------------------------

def _extract_archives_into(download_dir, staging_dir):
    """Extract every .zip/.tar(.gz) under ``download_dir`` into ``staging_dir`` and return the dir
    that holds the extracted generator tree (``staging_dir``; or ``download_dir`` if already flat)."""
    download_dir, staging_dir = Path(download_dir), Path(staging_dir)
    archives = [p for p in download_dir.rglob("*")
                if p.is_file() and (p.suffix.lower() == ".zip"
                                    or p.name.lower().endswith((".tar.gz", ".tgz", ".tar")))]
    if not archives:
        return download_dir
    staging_dir.mkdir(parents=True, exist_ok=True)
    for a in archives:
        if a.suffix.lower() == ".zip":
            _safe_extract_zip(a, staging_dir)
        else:
            _safe_extract_tar(a, staging_dir)
    return staging_dir


def _restructure_image_pergen(src_root, dest_split, cap=None, rename=None):
    """Lay a downloaded tree out as ``<dest_split>/<generator>/{0_real,1_fake}/``.

    For each immediate subdirectory of ``src_root`` (one generator), images found under any
    real-token dir (0_real/real/nature) become 0_real and fake-token dir (1_fake/fake/ai) become
    1_fake — searched recursively so per-class source-domain sublevels (e.g. DIRE's
    ``<subset>/<domain>/{0_real,1_fake}``) are merged. ``cap`` limits images/class; ``rename`` maps
    the generator dir name (e.g. strip GenImage's ``_imagenet`` suffix)."""
    src_root, dest_split = Path(src_root), Path(dest_split)
    real_toks = ("0_real", "real", "reals", "nature", "natural")
    fake_toks = ("1_fake", "fake", "fakes", "ai", "generated")
    for gen_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        name = rename(gen_dir.name) if rename else gen_dir.name
        for label, toks in (("0_real", real_toks), ("1_fake", fake_toks)):
            imgs = []
            for tok in toks:
                for sub in gen_dir.rglob(tok):
                    if sub.is_dir():
                        imgs += [p for p in sub.rglob("*")
                                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            imgs = sorted(set(imgs))
            if cap:
                imgs = imgs[:cap]
            if not imgs:
                continue
            out = dest_split / name / label
            out.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(imgs):
                tgt = out / f"{i:06d}_{p.name}"
                if tgt.exists():
                    continue
                try:
                    os.link(p, tgt)
                except OSError:
                    shutil.copy2(p, tgt)


def _restructure_video_pergen(src_root, dest_split, cap=None):
    """Video analogue of ``_restructure_image_pergen`` → ``<dest_split>/<gen>/{0_real,1_fake}/``."""
    src_root, dest_split = Path(src_root), Path(dest_split)
    real_toks = ("0_real", "real", "reals")
    fake_toks = ("1_fake", "fake", "fakes")
    for gen_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for label, toks in (("0_real", real_toks), ("1_fake", fake_toks)):
            vids = []
            for tok in toks:
                for sub in gen_dir.rglob(tok):
                    if sub.is_dir():
                        vids += [p for p in sub.rglob("*")
                                 if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
            vids = sorted(set(vids))
            if cap:
                vids = vids[:cap]
            if not vids:
                continue
            out = dest_split / gen_dir.name / label
            out.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(vids):
                tgt = out / f"{i:06d}_{p.name}"
                if tgt.exists():
                    continue
                try:
                    os.link(p, tgt)
                except OSError:
                    shutil.copy2(p, tgt)


def _iter_images_follow(directory):
    """Like ``_iter_images`` but follows symlinked directories (``os.walk`` with followlinks),
    so generators whose ``0_real``/``1_fake`` are directory symlinks (e.g. GenImage's SD-v1.4 slice
    reused from the single-pool dataset) are still discovered."""
    for dirpath, _dirnames, filenames in os.walk(Path(directory), followlinks=True):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTENSIONS:
                yield Path(dirpath) / fn


class _TestOnlyImageDataset(Dataset):
    """Base for eval-only image benchmarks laid out as ``<root>/test/<generator>/{0_real,1_fake}/``.

    Subclasses declare a download source and override ``prepare_split``; everything else (label
    from the leaf dir name, per-generator ``folder_name`` from the grandparent dir, ``__getitem__``
    returning ``(image, label, folder_id)``) matches the other image datasets so the per-generator
    AP/AUROC metrics keep working."""

    def __init__(self, root_dir, split: Literal["test"] = "test",
                 transform: AIGITransformations | None = None, prefetch: bool = False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print(f"Downloading {type(self).__name__} {split} split...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @classmethod
    def prepare_split(cls, root_dir, split: str = "test", force: bool = False) -> None:
        raise NotImplementedError(
            f"No downloader configured for {cls.__name__}; place the benchmark under "
            f"'{Path(root_dir) / split}/<generator>/{{0_real,1_fake}}/' manually.")

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(self.root_dir, self.split, force=force)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _decode_rgb(self.images[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name or name in ("nature", "natural"):
            return 0
        if "fake" in name or "false" in name or name == "ai":
            return 1
        raise ValueError(f"Cannot determine label from path: {filepath}")

    def _load_data(self):
        images, labels, folder_names = [], [], []
        for filepath in _iter_images_follow(self.destination_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name)
        return images, labels, folder_names


class DiffusionForensicsDataset(_TestOnlyImageDataset):
    """DiffusionForensics / DIRE diffusion test set (Wang et al. 2023) — used by NPR (Table 3) and
    FatFormer's diffusion table. 8 subsets (adm, ddpm, iddpm, ldm, pndm, sdv1, sdv2, vqdiffusion),
    each with its own paired reals (LSUN/ImageNet/CelebA-HQ)."""
    gdrive_folder_id = "1tKsOU-6FDdstrrKLPYuZ7RpQwtOSHxUD"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        import gdown
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(id=cls.gdrive_folder_id, output=str(dl), quiet=False, use_cookies=False)
        staging = _extract_archives_into(dl, root_dir / "_staging")
        _restructure_image_pergen(staging, dest, cap=1000)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)


class Diffusion1kStepDataset(_TestOnlyImageDataset):
    """Diffusion-1kStep self-synthesis test set — used by NPR (Table 5). 5 subsets (DALLE, ddpm,
    guided-diffusion=ADM, improved-diffusion=IDDPM, midjourney)."""
    gdrive_folder_id = "14f0vApTLiukiPvIHukHDzLujrvJpDpRq"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        import gdown
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(id=cls.gdrive_folder_id, output=str(dl), quiet=False, use_cookies=False)
        staging = _extract_archives_into(dl, root_dir / "_staging")
        _restructure_image_pergen(staging, dest, cap=1000)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)


class AIGCDetectBenchmarkDataset(_TestOnlyImageDataset):
    """AIGCDetectBenchmark test set (16-17 GAN and diffusion generators) — used by Co-Spy (ProGAN/AIGCDetect table),
    AIDE and CuRe. Sourced from the ModelScope dataset ``aemilia/AIGCDetectionBenchmark`` (test_set.zip). The full
    test data is kept (the papers evaluate all of it, about 76k real / 76k fake)."""
    modelscope_url = ("https://modelscope.cn/api/v1/datasets/aemilia/AIGCDetectionBenchmark/"
                      "repo?Revision=master&FilePath=AIGCDetectionBenchMark%2Ftest_set.zip")

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        zip_path = dl / "test_set.zip"
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(zip_path), cls.modelscope_url],
                       check=True)
        staging = _extract_archives_into(dl, root_dir / "_staging")
        inner = staging
        while len(subdirs := [p for p in inner.iterdir() if p.is_dir()]) == 1:   # step into wrapper folders
            inner = subdirs[0]
        _restructure_image_pergen(inner, dest)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)


class CommGen15Dataset(_TestOnlyImageDataset):
    """CommGen15 — 15 commercial-generator test set introduced by PGC (Table 3). Sourced from the
    Hugging Face dataset ``xiaoyuzhou68/CommGen15`` (CommGen15.zip; com_plat_v3/<gen>/{0_real,1_fake})."""
    hf_repo_id = "xiaoyuzhou68/CommGen15"
    hf_filename = "CommGen15.zip"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        from huggingface_hub import hf_hub_download
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        z = hf_hub_download(repo_id=cls.hf_repo_id, filename=cls.hf_filename,
                            repo_type="dataset", local_dir=str(dl))
        staging = _extract_archives_into(Path(z).parent, root_dir / "_staging")
        inner = next((p for p in staging.rglob("com_plat_v3") if p.is_dir()), staging)
        _restructure_image_pergen(inner, dest, cap=1000)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)


class GenImageFullDataset(_TestOnlyImageDataset):
    """Full 8-generator GenImage test set (Midjourney, SD-v1.4/1.5, ADM, GLIDE, Wukong, VQDM,
    BigGAN) — used by AIDE/VIB-Net/SAFE/PGC/Co-Spy for the cross-generator GenImage table. (The
    single-pool SD-v1.4 slice is the separate ``genimage`` dataset.) Sourced from the HF mirror
    ``jzousz/GenImage`` (genimage_test.zip: test/<gen>_imagenet/{nature,ai})."""
    hf_repo_id = "jzousz/GenImage"
    hf_filename = "genimage_test.zip"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        from huggingface_hub import hf_hub_download
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        z = hf_hub_download(repo_id=cls.hf_repo_id, filename=cls.hf_filename,
                            repo_type="dataset", local_dir=str(dl))
        staging = _extract_archives_into(Path(z).parent, root_dir / "_staging")
        inner = next((p for p in staging.rglob("test") if p.is_dir()), staging)
        _restructure_image_pergen(inner, dest, cap=1500, rename=lambda n: n.replace("_imagenet", ""))
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)


class SynthbusterDataset(_TestOnlyImageDataset):
    """Synthbuster (Bammey, 2023) — 9 generators (DALL-E 2/3, Firefly, GLIDE, Midjourney-v5, SD 1.3/1.4/2/XL),
    1000 fakes each. Used by SPAI (Table 1) and other detectors as a high-quality diffusion test set.
    Sourced from Zenodo record 10066460 (synthbuster.zip = <gen>/*.png, fakes only). Synthbuster ships
    NO reals (the originals are RAISE-1k, which require a separate request); this loader pairs each
    generator's fakes against a shared clean-real pool (GenImage nature photos) as a documented proxy —
    the detector keys on generative artifacts, so proxy reals give a valid per-generator AP/AUROC."""
    zenodo_record = "10066460"
    zenodo_file = "synthbuster.zip"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False, real_pool=None, cap=1000):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        import urllib.request
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        zpath = dl / cls.zenodo_file
        if not zpath.exists():
            url = f"https://zenodo.org/records/{cls.zenodo_record}/files/{cls.zenodo_file}?download=1"
            print(f"Downloading Synthbuster from {url} ...")
            urllib.request.urlretrieve(url, zpath)
        staging = _extract_archives_into(dl, root_dir / "_staging")
        inner = next((p for p in staging.rglob("synthbuster") if p.is_dir()), staging)
        # proxy reals: a clean real-photo pool (GenImage nature by default)
        pool = real_pool or "/hddstore/ssalehi/SynthScope/GenImage/test/real"
        reals = sorted(p for p in Path(pool).glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)[:cap]
        for gen_dir in sorted(p for p in inner.iterdir() if p.is_dir()):
            fakes = sorted(p for p in gen_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)[:cap]
            for label, imgs in (("1_fake", fakes), ("0_real", reals)):
                out = dest / gen_dir.name / label; out.mkdir(parents=True, exist_ok=True)
                for i, p in enumerate(imgs):
                    tgt = out / f"{i:05d}_{p.name}"
                    if not tgt.exists():
                        try: os.link(p, tgt)
                        except OSError: shutil.copy2(p, tgt)
        shutil.rmtree(root_dir / "_staging", ignore_errors=True)


def _link_images(imgs, out_dir):
    """Hard-link (copy as fallback) ``imgs`` into ``out_dir`` as ``<i>_<name>``, keeping the order."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(imgs):
        tgt = out_dir / f"{i:05d}_{p.name}"
        if not tgt.exists():
            try: os.link(p, tgt)
            except OSError: shutil.copy2(p, tgt)


def _first_int(path):
    """Sort key: the first integer in the file stem (``Flux1_100_person.jpg`` -> 100), then the name."""
    m = re.search(r"\d+", Path(path).stem)
    return (int(m.group()) if m else -1, Path(path).name)


class RealChainDataset(_TestOnlyImageDataset):
    """RealChain (REM paper; used by CuRe Table 1) — 7 generators (Flux1, SDv3.5, QwenImage, Hunyuan3,
    NanoBanana, Seedream4, i2i), 1000 fakes each, plus one pool of 7000 reals. Sourced from the HF dataset
    ``handsomerich/RealChain`` (CC BY-NC 4.0: <gen>.zip, Real.tar). The paper describes a clean (ND) and a
    chain-degraded (CD) version, but only one image set is released; every image in it (real and fake) is
    JPEG at the same quality (luma quant step 46.1, i.e. libjpeg q60). The real pool is not paired with
    the generators, so it is split into 7 disjoint slices in file-index order, one per generator: the mean
    of the per-generator balanced accuracies then equals the pooled 7000-vs-7000 balanced accuracy."""
    hf_repo_id = "handsomerich/RealChain"
    generators = ("Flux1", "SDv3.5", "QwenImage", "Hunyuan3", "NanoBanana", "Seedream4", "i2i")

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        from huggingface_hub import snapshot_download
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=cls.hf_repo_id, repo_type="dataset", local_dir=str(dl),
                          allow_patterns=[f"{g}.zip" for g in cls.generators] + ["Real.tar"])
        staging = root_dir / "_staging"
        for g in cls.generators:                     # the zips are flat, so each gets its own folder
            _safe_extract_zip(dl / f"{g}.zip", staging / g)
        _safe_extract_tar(dl / "Real.tar", staging)
        reals = sorted((p for p in (staging / "Real").rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
                       key=_first_int)
        per_gen = len(reals) // len(cls.generators)
        for i, g in enumerate(cls.generators):
            fakes = sorted((p for p in (staging / g).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
                           key=_first_int)
            _link_images(fakes, dest / g / "1_fake")
            _link_images(reals[i * per_gen:(i + 1) * per_gen], dest / g / "0_real")
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(staging, ignore_errors=True)


class EvalGENDataset(_TestOnlyImageDataset):
    """EvalGEN (DDA, NeurIPS 2025; used by DDA and CuRe) — 5 generators (Flux, GoT, Infinity, OmniGen,
    NOVA) on GenEval prompts, about 11,060 fakes each, JPEG q96. Sourced from the HF dataset
    ``Junwei-Xi/EvalGEN`` (Apache-2.0: EvalGEN.zip = GenEval-JPEG/<gen>/*.jpg). EvalGEN ships no reals:
    CuRe pairs it with an unspecified "shared MSCOCO real pool", and DDA reports accuracy on the fakes
    only. This loader pairs every generator with the same MSCOCO val2017 pool (5000 images, downloaded
    from cocodataset.org), so ``fake_acc`` reproduces the DDA protocol and ``acc`` the balanced one."""
    hf_repo_id = "Junwei-Xi/EvalGEN"
    hf_filename = "EvalGEN.zip"
    coco_url = "http://images.cocodataset.org/zips/val2017.zip"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        from huggingface_hub import hf_hub_download
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        z = hf_hub_download(repo_id=cls.hf_repo_id, filename=cls.hf_filename,
                            repo_type="dataset", local_dir=str(dl))
        coco = dl / "val2017.zip"
        if not coco.exists():
            subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(coco), cls.coco_url], check=True)
        staging = root_dir / "_staging"
        _safe_extract_zip(z, staging)
        _safe_extract_zip(coco, staging)
        reals = sorted(p for p in (staging / "val2017").glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        inner = next((p for p in staging.rglob("GenEval-JPEG") if p.is_dir()), staging)
        for gen_dir in sorted(p for p in inner.iterdir() if p.is_dir()):
            fakes = sorted(p for p in gen_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
            _link_images(fakes, dest / gen_dir.name / "1_fake")
            _link_images(reals, dest / gen_dir.name / "0_real")
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(staging, ignore_errors=True)


def _read_csv_url(url, headers):
    """Rows of the CSV file at ``url`` as dicts."""
    with urllib.request.urlopen(urllib.request.Request(url, None, headers), timeout=60) as resp:
        return list(csv.DictReader(resp.read().decode().splitlines()))


def _fetch_url_list(rows, out_dir, headers, name, attempts=3):
    """Download each ``row["url"]`` to ``out_dir / row["filename"]`` (datasets released as URL lists). A link that
    answers with an HTTP error, keeps failing, or (when the row has an ``md5``) returns different bytes, e.g. a
    placeholder for a deleted image, is skipped. Returns the filenames that could not be fetched."""
    missing = []
    for i, row in enumerate(rows):
        fpath = Path(out_dir) / row["filename"]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(attempts):
            if fpath.is_file():
                break
            try:
                with urllib.request.urlopen(urllib.request.Request(row["url"], None, headers), timeout=30) as resp:
                    data = resp.read()
            except urllib.error.HTTPError:          # deleted post or page: the official scripts also skip it
                break
            except (OSError, ValueError):           # network hiccup or bad URL: back off and retry
                time.sleep(2 ** attempt)
                continue
            if row.get("md5") and hashlib.md5(data).hexdigest() != row["md5"]:
                break
            fpath.write_bytes(data)
        if not fpath.is_file():
            missing.append(row["filename"])
        if (i + 1) % 200 == 0:
            print(f"  {name}: {i + 1}/{len(rows)} fetched, {len(missing)} missing", flush=True)
    return missing


class SynthWildXDataset(_TestOnlyImageDataset):
    """SynthWildX (Cozzolino et al., CVPRW 2024; used by DDA and CuRe) — images posted on X: 500 real and 500 each
    from DALL-E 3, Firefly and Midjourney v5. Only the image URLs are released (``data/synthwildx/list.csv`` in
    grip-unina/ClipBased-SyntheticImageDetection), so, like the official script, this loader fetches each image
    from X. Posts deleted since release return 404 and are skipped (listed in ``<root>/missing.txt``), so the set
    on disk can be smaller than the paper's. Each generator is paired with all 500 reals."""
    list_url = ("https://raw.githubusercontent.com/grip-unina/ClipBased-SyntheticImageDetection/"
                "main/data/synthwildx/list.csv")
    headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"}

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        rows = _read_csv_url(cls.list_url, cls.headers)
        missing = _fetch_url_list(rows, dl, cls.headers, "SynthWildX")
        (root_dir / "missing.txt").write_text("".join(f"{m}\n" for m in missing))
        reals = sorted((dl / "real").glob("*"))
        for gen_dir in sorted(p for p in dl.iterdir() if p.is_dir() and p.name != "real"):
            _link_images(sorted(gen_dir.glob("*")), dest / gen_dir.name / "1_fake")
            _link_images(reals, dest / gen_dir.name / "0_real")
        print(f"SynthWildX: {len(rows) - len(missing)}/{len(rows)} images available ({len(missing)} deleted from X)")
        shutil.rmtree(dl, ignore_errors=True)


class WildRFDataset(_TestOnlyImageDataset):
    """WildRF (Cavia et al. 2024, the LaDeDa paper; used by DDA and CuRe) — real and fake images collected from
    social platforms. Its test split has one folder per platform (facebook, reddit, twitter), each with real/fake;
    those become the "generators" here. Sourced from the authors' Google Drive zip (barcavia/
    RealTime-DeepfakeDetection-in-the-RealWorld); only the test split is laid out."""
    gdrive_file_id = "1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        import gdown
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        gdown.download(id=cls.gdrive_file_id, output=str(dl / "WildRF.zip"), quiet=False, use_cookies=False)
        staging = _extract_archives_into(dl, root_dir / "_staging")
        inner = next(p for p in staging.rglob("test") if p.is_dir())
        _restructure_image_pergen(inner, dest)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)


class BFreeOnlineDataset(_TestOnlyImageDataset):
    """BFree-Online — the B-Free paper's viral-images set (Guillaro et al., CVPR 2025; used by DDA and CuRe): real
    and fake images that went viral, with several web-scraped copies of each (about 1400 in the paper). Only URLs are
    released (``viral_images_dataset/BFree_viral_images.csv`` in grip-unina/B-Free, 1111 rows, nonprofit use only);
    the loader fetches them and keeps an image only when its md5 matches the list, so dead links and placeholders are
    skipped (listed in ``<root>/missing.txt``). The 3 Facebook links need a browser to resolve and are skipped too.
    Laid out as a single group, ``test/viral/{0_real,1_fake}``."""
    list_url = "https://raw.githubusercontent.com/grip-unina/B-Free/main/viral_images_dataset/BFree_viral_images.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/128.0.0.0 Safari/537.36"}

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        rows = _read_csv_url(cls.list_url, cls.headers)
        missing = _fetch_url_list(rows, dl, cls.headers, "BFree-Online")
        (root_dir / "missing.txt").write_text("".join(f"{m}\n" for m in missing))
        for label, folder in (("REAL", "0_real"), ("FAKE", "1_fake")):
            _link_images(sorted((dl / label).rglob("*.*")), dest / "viral" / folder)
        print(f"BFree-Online: {len(rows) - len(missing)}/{len(rows)} images available ({len(missing)} links dead)")
        shutil.rmtree(dl, ignore_errors=True)


class RRDataset(_TestOnlyImageDataset):
    """RRDataset (ICCV 2025, "Bridging the Gap Between Ideal and Real-world Evaluation"; used by CuRe) — real and
    AI-generated images in three scenarios: ``original``, ``transfer`` (after sharing over online platforms) and
    ``redigital`` (re-captured / re-digitized), about 8500 real and 8500 fake each. The scenarios become the
    "generators" here. Sourced from Zenodo record 14963880 (RRDataset_test.tar.gz, 20 GB, CC BY 4.0); the separate
    train/val archive is not used."""
    zenodo_record = "14963880"
    zenodo_file = "RRDataset_test.tar.gz"

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        archive = dl / cls.zenodo_file
        if not archive.exists():
            url = f"https://zenodo.org/api/records/{cls.zenodo_record}/files/{cls.zenodo_file}/content"
            subprocess.run(["curl", "-L", "--fail", "-C", "-", "--retry", "5", "-o", str(archive), url], check=True)
        staging = _extract_archives_into(dl, root_dir / "_staging")
        inner = next((p for p in staging.rglob("RRDataset_final") if p.is_dir()), staging)
        _restructure_image_pergen(inner, dest)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)


class ARForensicsDataset(_TestOnlyImageDataset):
    """ARForensics (D3QE, ICCV 2025) — images from 7 visual autoregressive generators (LlamaGen, VAR, Infinity,
    Janus-Pro, RAR, Switti, Open-MAGVIT2), 6000 fakes and 6000 ImageNet reals each, laid out as
    ``ARForensics/test/<generator>/{0_real,1_fake}``. Sourced from the HF dataset ``Yanran21/ARForensics`` (MIT): the
    corrected test release (MAR replaced by Switti) in four independent zips, ARForensics_updated_test_part00{1..4}.zip
    (21 GB); a generator can span two zips, so all four are extracted into one folder. D3QE's training data is in the
    same repo: ``train`` (100k LlamaGen fakes + 100k ImageNet reals; one zip split into 7 parts, 35 GB) and ``val``
    (10k + 10k, 3.7 GB), each laid out as ``ARForensics/<split>/{0_real,1_fake}``."""
    hf_repo_id = "Yanran21/ARForensics"
    hf_patterns = {"test": "ARForensics_updated_test_part*.zip", "val": "val/ARForensics_val.zip",
                   "train": "train/ARForensics_train_split.zip.a?"}

    @classmethod
    def prepare_split(cls, root_dir, split="test", force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_images(dest):
            return
        from huggingface_hub import snapshot_download
        pattern = cls.hf_patterns[split]
        dl = root_dir / f"_downloads_{split}"; dl.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=cls.hf_repo_id, repo_type="dataset", local_dir=str(dl), allow_patterns=[pattern])
        archives = sorted(dl.glob(pattern))
        if split == "train":                          # a single zip split into parts: join them first
            joined = dl / "train_joined.zip"
            with open(joined, "wb") as out:
                for part in archives:
                    with open(part, "rb") as f:
                        shutil.copyfileobj(f, out, 1 << 24)
                    part.unlink()
            archives = [joined]
        staging = root_dir / f"_staging_{split}"
        for archive in archives:
            _safe_extract_zip(archive, staging)
        inner = staging / "ARForensics" / split
        if not inner.is_dir():
            raise RuntimeError(f"ARForensics: expected ARForensics/{split}/ inside the archives, found {list(staging.iterdir())}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(inner), str(dest))
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(staging, ignore_errors=True)


class _TestOnlyImageModule:
    """Shared loader for the eval-only image benchmarks (test split only)."""
    DATASET_CLASS = _TestOnlyImageDataset

    def __init__(self, config):
        self.config = config
        cfg = config.DatasetConfig
        eval_list = list(cfg.get("eval_transform_list", []))
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        self.test_dataset = self.DATASET_CLASS(
            cfg.root_dir, "test", self.eval_transform, prefetch=bool(cfg.get("prefetch", False)))

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        cfg = self.config.DatasetConfig
        return DataLoader(self.test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
                          num_workers=cfg.num_workers, persistent_workers=cfg.persistent_workers,
                          pin_memory=cfg.pin_memory)


class DiffusionForensicsDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = DiffusionForensicsDataset


class Diffusion1kStepDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = Diffusion1kStepDataset


class AIGCDetectBenchmarkDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = AIGCDetectBenchmarkDataset


class CommGen15DatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = CommGen15Dataset


class GenImageFullDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = GenImageFullDataset


class SynthbusterDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = SynthbusterDataset


class RealChainDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = RealChainDataset


class EvalGENDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = EvalGENDataset


class SynthWildXDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = SynthWildXDataset


class WildRFDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = WildRFDataset


class BFreeOnlineDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = BFreeOnlineDataset


class RRDatasetModule(_TestOnlyImageModule):
    DATASET_CLASS = RRDataset


class ARForensicsDatasetModule(_TestOnlyImageModule):
    """The test split for benchmarking; with a ``train_transform_list`` in the config also the train and val splits
    (D3QE's training data), validated with ``val_transform_list`` (default: ``eval_transform_list``)."""
    DATASET_CLASS = ARForensicsDataset

    def __init__(self, config):
        super().__init__(config)
        cfg = config.DatasetConfig
        train_list = list(cfg.get("train_transform_list", []))
        self.train_dataset = self.val_dataset = None
        if train_list:
            prefetch = bool(cfg.get("prefetch", False))
            val_list = list(cfg.get("val_transform_list", cfg.get("eval_transform_list", [])))
            self.train_dataset = ARForensicsDataset(cfg.root_dir, "train", AIGITransformations(train_list), prefetch=prefetch)
            self.val_dataset = ARForensicsDataset(cfg.root_dir, "val", AIGITransformations(val_list), prefetch=prefetch)

    def _loader(self, dataset, batch_size, shuffle):
        cfg = self.config.DatasetConfig
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                          persistent_workers=cfg.persistent_workers, pin_memory=cfg.pin_memory)

    def train_dataloader(self):
        return None if self.train_dataset is None else self._loader(self.train_dataset, self.config.DatasetConfig.train_batch_size, True)

    def val_dataloader(self):
        return None if self.val_dataset is None else self._loader(self.val_dataset, self.config.DatasetConfig.val_batch_size, False)


class VideoFeedbackDataset(VideoFolderDataset):
    """VideoFeedback (TIGER-Lab) — used by STALL. 11 generators vs DiDeMo+Panda70M real, laid out
    as ``<root>/test/<generator>/{0_real,1_fake}/*.mp4``."""
    hf_repo_id = "TIGER-Lab/VideoFeedback"

    @classmethod
    def prepare_split(cls, root_dir, split="test", hf_repo_id=None, force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_videos(dest):
            return
        from huggingface_hub import snapshot_download
        dl = root_dir / "_downloads"
        snapshot_download(repo_id=hf_repo_id or cls.hf_repo_id, repo_type="dataset", local_dir=str(dl))
        staging = _extract_archives_into(dl, root_dir / "_staging")
        _restructure_video_pergen(staging, dest, cap=150)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)

    def _download_data(self, force=False):
        self.prepare_split(self.root_dir, self.split, self.hf_repo_id, force)


class ComGenVidDataset(VideoFolderDataset):
    """ComGenVid (STALL's new benchmark) — Sora + VEO3 fakes vs MSVD real, laid out as
    ``<root>/test/<generator>/{0_real,1_fake}/*.mp4``. The HF repo ``OmerXYZ/comgenvid`` also ships
    precomputed embeddings for STALL's official embedding-based eval."""
    hf_repo_id = "OmerXYZ/comgenvid"

    @classmethod
    def prepare_split(cls, root_dir, split="test", hf_repo_id=None, force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_videos(dest):
            return
        from huggingface_hub import snapshot_download
        dl = root_dir / "_downloads"
        snapshot_download(repo_id=hf_repo_id or cls.hf_repo_id, repo_type="dataset",
                          local_dir=str(dl), allow_patterns=["*.mp4", "**/*.mp4"])
        staging = _extract_archives_into(dl, root_dir / "_staging")
        _restructure_video_pergen(staging, dest, cap=150)
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(root_dir / "_staging", ignore_errors=True)

    def _download_data(self, force=False):
        self.prepare_split(self.root_dir, self.split, self.hf_repo_id, force)


GENVIDEO_VAL_URL = "https://modelscope.cn/datasets/cccnju/Gen-Video/resolve/master/GenVideo-Val.zip"


def _genvideo_val_zip(dl_dir):
    """GenVideo-Val.zip (DeMamba's validation set, 14 GB) from ModelScope, downloaded into ``dl_dir`` unless present."""
    zip_path = Path(dl_dir) / "GenVideo-Val.zip"
    if not zip_path.is_file():
        Path(dl_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "--retry", "5", "-o", str(zip_path), GENVIDEO_VAL_URL],
                       check=True)
    return zip_path


def _msrvtt_reals(dl_dir):
    """The 10,000 MSR-VTT real videos of GenVideo-Val (``Real/msrvtt_<n>.mp4``), extracted into ``dl_dir/msrvtt`` and
    returned in numeric order. D3 pairs every fake-only video benchmark (EvalCrafter, VideoPhy, VidProM) with them."""
    out = Path(dl_dir) / "msrvtt"
    if not _has_videos(out):
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_genvideo_val_zip(dl_dir)) as zf:
            for m in zf.infolist():
                name = Path(m.filename).name
                if ("/Real/" in m.filename and "__MACOSX" not in m.filename and not name.startswith("._")
                        and name.lower().endswith(VIDEO_EXTENSIONS)):           # skip macOS resource-fork stubs
                    with zf.open(m) as src, open(out / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    return sorted((p for p in out.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS), key=_first_int)


class GenVideoValDataset(VideoFolderDataset):
    """GenVideo-Val (DeMamba, Chen et al. 2024) — the test set of D3 (Table 1) and STALL: 10 generators (Crafter, Gen2,
    HotShot, Lavie, ModelScope, MoonValley, MorphStudio, Show_1, Sora, WildScrape) against 10,000 MSR-VTT real videos.
    Sourced from ModelScope ``cccnju/Gen-Video`` (GenVideo-Val.zip, 14 GB). Laid out as ``test/<generator>/1_fake`` with
    the shared real pool hard-linked into every ``test/<generator>/0_real`` (numeric order, so ``--limit 1000`` takes
    the first 1000 reals, as D3's official eval does with ``max_len=1000``)."""
    hf_repo_id = None

    @classmethod
    def prepare_split(cls, root_dir, split="test", hf_repo_id=None, force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_videos(dest):
            return
        dl = root_dir / "_downloads"
        staging = root_dir / "_staging"
        _safe_extract_zip(_genvideo_val_zip(dl), staging)
        inner = next(p for p in staging.rglob("Fake") if p.is_dir() and "__MACOSX" not in p.parts).parent
        ok = lambda p: p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith("._")    # noqa: E731
        reals = sorted((p for p in (inner / "Real").iterdir() if ok(p)), key=_first_int)
        for gen_dir in sorted(p for p in (inner / "Fake").iterdir() if p.is_dir()):
            fakes = sorted((p for p in gen_dir.rglob("*") if ok(p)), key=_first_int)
            _link_images(fakes, dest / gen_dir.name / "1_fake")
            _link_images(reals, dest / gen_dir.name / "0_real")
        shutil.rmtree(dl, ignore_errors=True); shutil.rmtree(staging, ignore_errors=True)

    def _download_data(self, force=False):
        self.prepare_split(self.root_dir, self.split, self.hf_repo_id, force)


class VideoPhyDataset(VideoFolderDataset):
    """VideoPhy test set (Bansal et al. 2024; D3 Table 3) — 3360 text-to-video clips from 10 generators, released only as
    URLs (``videophysics/videophy_test_public`` on Hugging Face, MIT). The loader fetches each clip; links that no longer
    work (both CogVideoX sets were offline in October 2026) are skipped and listed in ``<root>/missing.txt``. VideoPhy has
    no real videos, so, as D3 does for its fake-only sets, every generator is paired with GenVideo-Val's MSR-VTT reals."""
    hf_repo_id = None
    csv_url = "https://huggingface.co/datasets/videophysics/videophy_test_public/resolve/main/videophy_test_public.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    source_names = {"lavie": "LaVie", "vc2": "VideoCrafter2", "opensora": "OpenSora", "zeroscope": "ZeroScope",
                    "gen2": "Gen-2", "svd": "SVD", "pika": "Pika", "Dream Machine (LumaAI)": "Dream-Machine",
                    "CogVideoX-2B": "CogVideoX-2B", "CogVideoX-5B": "CogVideoX-5B"}

    @classmethod
    def prepare_split(cls, root_dir, split="test", hf_repo_id=None, force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_videos(dest):
            return
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, row in enumerate(_read_csv_url(cls.csv_url, cls.headers)):
            url = urllib.parse.quote(row["video_url"], safe=":/?=&%")
            gen = cls.source_names.get(row["source"], row["source"])
            rows.append({"url": url, "filename": f"{gen}/{i:05d}_{Path(urllib.parse.urlparse(url).path).name}"})
        missing = _fetch_url_list(rows, dl / "videos", cls.headers, "VideoPhy")
        (root_dir / "missing.txt").write_text("".join(f"{m}\n" for m in missing))
        reals = _msrvtt_reals(dl)
        for gen_dir in sorted(p for p in (dl / "videos").iterdir() if p.is_dir()):
            fakes = sorted(p for p in gen_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
            if fakes:
                _link_images(fakes, dest / gen_dir.name / "1_fake")
                _link_images(reals, dest / gen_dir.name / "0_real")
        print(f"VideoPhy: {len(rows) - len(missing)}/{len(rows)} clips available ({len(missing)} links dead)")
        shutil.rmtree(dl, ignore_errors=True)

    def _download_data(self, force=False):
        self.prepare_split(self.root_dir, self.split, self.hf_repo_id, force)


class VidProMDataset(VideoFolderDataset):
    """VidProM (Wang & Yang, NeurIPS 2024; D3 Table 4, ReStraV) — millions of text-to-video clips from ModelScope, Pika,
    Text2Video-Zero and VideoCrafter2 on Hugging Face ``WenhaoWang/VidProM`` (CC BY-NC 4.0; 30 tar parts per generator,
    0.35-1 TB each). The loader streams only the first ``cap`` clips of part 1 per generator instead of downloading
    terabytes, so results are on a subsample (D3 does not say how it sampled). OpenSora and StreamingT2V, also in D3's
    table, are not in that repository. Each generator is paired with GenVideo-Val's MSR-VTT reals."""
    hf_repo_id = "WenhaoWang/VidProM"
    generators = {"ms": "ModelScope", "pika": "Pika", "t2vz": "Text2Video-Zero", "vc2": "VideoCrafter2"}
    cap = 1000

    @classmethod
    def prepare_split(cls, root_dir, split="test", hf_repo_id=None, force=False):
        root_dir = Path(root_dir); dest = root_dir / split
        if not force and _has_videos(dest):
            return
        dl = root_dir / "_downloads"; dl.mkdir(parents=True, exist_ok=True)
        repo = hf_repo_id or cls.hf_repo_id
        for key, gen in cls.generators.items():
            out = dl / "videos" / gen; out.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{key}_videos/{key}_videos_1.tar"
            kept = len(list(out.glob("*.mp4")))
            if kept < cls.cap:
                with urllib.request.urlopen(urllib.request.Request(url, None, {"User-Agent": "Mozilla/5.0"})) as resp, \
                        tarfile.open(fileobj=resp, mode="r|*") as tf:      # streamed: stops after `cap` clips
                    for m in tf:
                        name = Path(m.name).name
                        if not (m.isfile() and name.lower().endswith(VIDEO_EXTENSIONS)):
                            continue
                        with tf.extractfile(m) as src, open(out / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        kept += 1
                        if kept >= cls.cap:
                            break
            print(f"  VidProM {gen}: {kept} clips", flush=True)
        reals = _msrvtt_reals(dl)
        for gen_dir in sorted(p for p in (dl / "videos").iterdir() if p.is_dir()):
            _link_images(sorted(gen_dir.glob("*.mp4"))[:cls.cap], dest / gen_dir.name / "1_fake")
            _link_images(reals, dest / gen_dir.name / "0_real")
        shutil.rmtree(dl, ignore_errors=True)

    def _download_data(self, force=False):
        self.prepare_split(self.root_dir, self.split, self.hf_repo_id, force)


class VideoFeedbackDatasetModule(_TestOnlyVideoModule):
    DATASET_CLASS = VideoFeedbackDataset


class ComGenVidDatasetModule(_TestOnlyVideoModule):
    DATASET_CLASS = ComGenVidDataset


class GenVideoValDatasetModule(_TestOnlyVideoModule):
    DATASET_CLASS = GenVideoValDataset


class VideoPhyDatasetModule(_TestOnlyVideoModule):
    DATASET_CLASS = VideoPhyDataset


class VidProMDatasetModule(_TestOnlyVideoModule):
    DATASET_CLASS = VidProMDataset


SynthScopeModuleDict = {
    "aigibench": AIGIBenchDatasetModule,
    "cnndetection": CNNDetectionDatasetModule,
    "cospy": CoSpyDatasetModule,
    "ditfake": DiTFakeDatasetModule,
    "freqnet": FreqNetDetectionDatasetModule,
    "freqnetdetection": FreqNetDetectionDatasetModule,
    "genimage": GenImageDatasetModule,
    "universialfakedetection": UniversialFakeDetectionDatasetModule,
    # Cross-generator image benchmarks added for the reproduction suite (test-only).
    "genimage_full": GenImageFullDatasetModule,
    "synthbuster": SynthbusterDatasetModule,
    "diffusionforensics": DiffusionForensicsDatasetModule,
    "diffusion1kstep": Diffusion1kStepDatasetModule,
    "aigcdetect": AIGCDetectBenchmarkDatasetModule,
    "commgen15": CommGen15DatasetModule,
    "realchain": RealChainDatasetModule,
    "evalgen": EvalGENDatasetModule,
    "synthwildx": SynthWildXDatasetModule,
    "wildrf": WildRFDatasetModule,
    "bfree_online": BFreeOnlineDatasetModule,
    "rrdataset": RRDatasetModule,
    "arforensics": ARForensicsDatasetModule,
    # Video benchmarks. EvalCrafter/GenVidBench/VideoFeedback/ComGenVid have dedicated classes that
    # download + extract their HF archives into the real/fake layout; the rest use the generic
    # VideoFolderDatasetModule.
    "genvideo": GenVideoValDatasetModule,
    "videophy": VideoPhyDatasetModule,
    "evalcrafter": EvalCrafterDatasetModule,
    "vidprom": VidProMDatasetModule,
    "genvidbench": GenVidBenchDatasetModule,
    "physics_iq": VideoFolderDatasetModule,
    "videofeedback": VideoFeedbackDatasetModule,
    "comgenvid": ComGenVidDatasetModule,
}


if __name__ == "__main__":
    config = compose_config()
    dataset_module = SynthScopeModuleDict[config.DatasetConfig.name](config)
    train_dataloader = dataset_module.train_dataloader()
    val_dataloader = dataset_module.val_dataloader()
    test_dataloader = dataset_module.test_dataloader()
    for batch in train_dataloader:
        print(batch)
        break
    for batch in val_dataloader:
        print(batch)
        break
    for batch in test_dataloader:
        print(batch)
        plt.imshow(batch[0].permute(1, 2, 0))
        plt.savefig(fname='test_image.png')
        break
