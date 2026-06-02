"""Public API: ``load`` a detector and ``predict`` on an image or video.

This wraps the web-interface backend, which already encapsulates the single source of truth
for every detector: the registry (per-model Hydra config, eval preprocessing, checkpoint +
official-weights remap, and score convention) and the inference engine (lazy build, checkpoint
load, and the forward -> P(fake) math). We reuse it rather than duplicate that logic.
"""
from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import Union

from PIL import Image

# Importing the backend adds the repo root to sys.path and wires cache locations; it pulls
# torch/torchvision and the repo decoders, so keep it at module scope (this is an ML library).
from interface.backend import config as _config
from interface.backend import media as _media
from interface.backend.engine import Engine, get_engine
from interface.backend.registry import ModelSpec, get_spec, list_specs

PathOrImage = Union[str, Path, bytes, Image.Image]

__all__ = ["Detector", "load", "list_models", "available_models"]


def list_models() -> list[dict]:
    """Return one dict per implemented detector, including whether its weights are present.

    Keys: ``key``, ``name``, ``paper``, ``modality`` (``image``/``video``), ``available``
    (checkpoint resolvable, or training-free), and ``notes``.
    """
    eng = get_engine()
    return [
        {
            "key": s.key,
            "name": s.name,
            "paper": s.paper,
            "modality": s.modality,
            "available": eng.is_available(s),
            "notes": s.notes,
        }
        for s in list_specs()
    ]


def available_models() -> list[str]:
    """Keys of detectors that can run right now (weights present, or training-free)."""
    eng = get_engine()
    return [s.key for s in list_specs() if eng.is_available(s)]


def load(
    model: str,
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    auto_download: bool = True,
    repo: str | None = None,
) -> "Detector":
    """Build a :class:`Detector` for ``model`` (e.g. ``"fat_former"``, ``"d3"``).

    Parameters
    ----------
    model:        detector key (see :func:`list_models`).
    checkpoint:   optional path to a ``.pth`` to load instead of the one in ``$AGID_CKPT_DIR``.
    device:       ``"cuda"``/``"cpu"`` (default: CUDA if available). The model is built lazily on
                  the first :meth:`Detector.predict` call.
    auto_download: if the weights are absent locally, fetch them from the Hugging Face model zoo
                  on first use (default True). Ignored when ``checkpoint`` is given.
    repo:         Hub repo to download from (default ``$AGID_HUB_REPO`` / the project default).
    """
    return Detector(get_spec(model), checkpoint=checkpoint, device=device,
                    auto_download=auto_download, repo=repo)


class Detector:
    """A single loaded detector. Build via :func:`agid.load`.

    The underlying model is constructed lazily on the first :meth:`predict` and then kept
    resident for the lifetime of this object.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        auto_download: bool = True,
        repo: str | None = None,
    ):
        self._custom_ckpt = checkpoint is not None
        if checkpoint is not None:
            # An absolute checkpoint path overrides the file resolved under $AGID_CKPT_DIR
            # (CHECKPOINTS_DIR / "<abs>" collapses to the absolute path in pathlib).
            ckpt = str(Path(checkpoint).expanduser().resolve())
            spec = dataclasses.replace(spec, checkpoint=ckpt)
        self.spec = spec
        self._auto_download = auto_download
        self._repo = repo
        # A custom device or checkpoint gets a dedicated engine so it never collides with the
        # shared LRU cache; the common path reuses the process-wide singleton.
        self._engine = Engine(device=device) if (device or checkpoint) else get_engine()

    # -- introspection ------------------------------------------------------
    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def modality(self) -> str:
        return self.spec.modality

    @property
    def device(self) -> str:
        return self._engine.device

    @property
    def available(self) -> bool:
        """Whether this detector's weights are present (always True for training-free models)."""
        return self._engine.is_available(self.spec)

    def __repr__(self) -> str:
        return f"Detector(key={self.spec.key!r}, modality={self.spec.modality!r}, device={self.device!r})"

    # -- prediction ---------------------------------------------------------
    def predict(self, source: PathOrImage, *, num_frames: int | None = None) -> dict:
        """Score ``source`` and return a result dict.

        ``source`` may be a file path, raw ``bytes``, or a PIL ``Image``. Result keys include
        ``fake_probability``, ``real_probability``, ``verdict`` (``"fake"``/``"real"``),
        ``confidence``, and ``elapsed_ms``. For an image detector run on a video, the result
        also carries ``per_frame`` probabilities (mean-aggregated); ``num_frames`` overrides
        how many frames are sampled.
        """
        self._ensure_weights()
        kind, pil, path = self._classify(source)

        if kind == "image":
            if self.spec.modality == "video" and self.spec.transform is None:
                raise ValueError(
                    f"'{self.spec.key}' is a video-only detector and cannot score a still image."
                )
            return self._engine.predict_image(self.spec, pil)

        # kind == "video"
        if self.spec.modality == "video":
            return self._engine.predict_video_clip(self.spec, path)
        # An image detector on a video: sample frames and aggregate.
        n = num_frames or _config.VIDEO_IMAGE_FRAMES
        frames = _media.sample_video_pil(path, num_frames=n)
        return self._engine.predict_video_frames(self.spec, frames)

    def _ensure_weights(self) -> None:
        """Fetch weights from the model zoo on first use if they're not present locally."""
        if self.available or self._custom_ckpt or self.spec.checkpoint is None:
            return  # already present, user-managed, or training-free
        if not self._auto_download:
            raise FileNotFoundError(
                f"No local checkpoint for '{self.spec.key}'. Enable auto_download, place the "
                f"file under $AGID_CKPT_DIR, or pass checkpoint=PATH."
            )
        from . import hub

        print(f"[agid] downloading '{self.spec.key}' weights from the model zoo ...")
        hub.download(self.spec.key, repo_id=self._repo)

    # -- helpers ------------------------------------------------------------
    def _classify(self, source: PathOrImage):
        """Resolve ``source`` to ``(kind, pil_image_or_None, path_or_None)``."""
        if isinstance(source, Image.Image):
            return "image", source.convert("RGB"), None
        if isinstance(source, (bytes, bytearray)):
            return "image", Image.open(io.BytesIO(bytes(source))).convert("RGB"), None
        path = Path(source)
        kind = _media.media_kind(path.name)
        if kind == "image":
            return "image", _media.decode_image(path.read_bytes()), None
        return "video", None, path
