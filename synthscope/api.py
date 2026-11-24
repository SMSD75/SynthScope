"""Public API: ``load`` a detector and ``predict`` on an image or video.

This wraps the web-interface backend, which already encapsulates the single source of truth
for every detector: the registry (per-model Hydra config, eval preprocessing, checkpoint +
official-weights remap, and score convention) and the inference engine (lazy build, checkpoint
load, and the forward -> P(fake) math). We reuse it rather than duplicate that logic.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Union

from PIL import Image

# Importing the backend adds the repo root to sys.path and wires cache locations; it pulls
# torch/torchvision and the repo decoders, so keep it at module scope (this is an ML library).
from interface.backend import config as _config
from interface.backend import media as _media
from interface.backend.engine import Engine, agent_missing, get_engine, vlm_images_only
from interface.backend.registry import ModelSpec, get_spec, list_specs
from models.video.wevis.wevis import download_hint as agent_download_hint
from models.vlm.vlm import download_hint

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
    checkpoint:   optional path to a ``.pth`` to load instead of the one in ``$SYNTHSCOPE_CKPT_DIR``.
    device:       ``"cuda"``/``"cpu"`` (default: CUDA if available). The model is built lazily on
                  the first :meth:`Detector.predict` call.
    auto_download: if the weights are absent locally, fetch them from the Hugging Face model zoo
                  on first use (default True). Ignored when ``checkpoint`` is given.
    repo:         Hub repo to download from (default ``$SYNTHSCOPE_HUB_REPO`` / the project default).
    """
    return Detector(get_spec(model), checkpoint=checkpoint, device=device,
                    auto_download=auto_download, repo=repo)


class Detector:
    """A single loaded detector. Build via :func:`synthscope.load`.

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
            # An absolute checkpoint path overrides the file resolved under $SYNTHSCOPE_CKPT_DIR
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
    def predict(self, source: PathOrImage, *, num_frames: int | None = None, return_input_image: bool = False) -> dict:
        """Score ``source`` and return a result dict.

        ``source`` may be a file path, raw ``bytes``, or a PIL ``Image``. Result keys include
        ``fake_probability``, ``real_probability``, ``verdict`` (``"fake"``/``"real"``),
        ``confidence``, and ``elapsed_ms``. For an image detector run on a video, the result
        also carries ``per_frame`` probabilities (mean-aggregated); ``num_frames`` overrides
        how many frames are sampled. ``input_size``/``orig_size``/``input_note`` describe what the detector saw;
        pass ``return_input_image=True`` to also get that image as a PNG data URL (the web page shows it).
        """
        kind, pil, path = self._classify(source)
        if kind == "image" and self.spec.modality == "video":   # video-only (also the agent), before weights or a load
            raise ValueError(
                f"'{self.spec.key}' is a video-only detector and cannot score a still image."
            )
        self._ensure_weights()

        if kind == "image":
            result = self._engine.predict_image(self.spec, pil)
            if not return_input_image:
                result.pop("input_image", None)      # a large base64 string; only the web page needs it
            return result

        # kind == "video"
        if self.spec.has_vlm:
            raise ValueError(vlm_images_only(self.spec))
        if self.spec.modality == "video":
            return self._engine.predict_video_clip(self.spec, path)
        # An image detector on a video: sample frames and aggregate.
        n = num_frames or _config.VIDEO_IMAGE_FRAMES
        frames = _media.sample_video_pil(path, num_frames=n)
        return self._engine.predict_video_frames(self.spec, frames)

    def _ensure_weights(self) -> None:
        """Fetch weights from the model zoo on first use if they're not present locally."""
        # Extra files a detector reads from the checkpoint folder (NSG-VD: its diffusion model and reference set) are not
        # in the zoo: say which are missing instead of trying to download the checkpoint.
        missing_aux = [f for f in self.spec.aux_files.values() if not (_config.CHECKPOINTS_DIR / f).exists()]
        if missing_aux:
            raise FileNotFoundError(
                f"'{self.spec.key}' needs {', '.join(missing_aux)} in $SYNTHSCOPE_CKPT_DIR ({_config.CHECKPOINTS_DIR}); "
                "they are not in the model zoo; the detector's README (models/<modality>/) says how to get them."
            )
        if self.spec.fusion is not None:            # no weights of its own: each component fetches (or explains) its own
            for key in (self.spec.fusion.vlm, self.spec.fusion.expert):
                Detector(get_spec(key), auto_download=self._auto_download, repo=self._repo)._ensure_weights()
            return
        if self.available or self._custom_ckpt or self.spec.checkpoint is None:
            return  # already present, user-managed, or training-free
        if self._engine.ffmpeg_missing(self.spec):
            raise FileNotFoundError(self._engine.ffmpeg_missing_message(self.spec))
        if self.spec.agent is not None:             # an agent's official release (several folders + code), not in the zoo
            raise FileNotFoundError(f"'{self.spec.key}' cannot run here: {'; '.join(agent_missing(self.spec))}. "
                                    f"Get the official release: {agent_download_hint(self.spec.agent)}")
        if self.spec.vlm is not None:               # a multi-GB Hugging Face folder, not in the model zoo
            vlm = self.spec.vlm
            raise FileNotFoundError(
                f"'{self.spec.key}' loads the official Hugging Face folder {vlm.hf_repo} (not in the model zoo): "
                + download_hint(vlm, _config.CHECKPOINTS_DIR / self.spec.checkpoint, _config.CHECKPOINTS_DIR)
                + ", or pass checkpoint=FOLDER."
            )
        ltf_ckpt_present = self.spec.ckpt_kind == "official_ltf" and (_config.CHECKPOINTS_DIR / self.spec.checkpoint).exists()
        if not self._auto_download and not ltf_ckpt_present:
            raise FileNotFoundError(
                f"No local checkpoint for '{self.spec.key}'. Enable auto_download, place the "
                f"file under $SYNTHSCOPE_CKPT_DIR, or pass checkpoint=PATH."
            )
        if self.spec.ckpt_kind == "official_ltf":   # not in the zoo: the authors' repo at the pinned revision
            from models.image.ltf.ltf import fetch_core, resolve_backbone
            if not ltf_ckpt_present:
                print("[synthscope] downloading 'ltf' weights/epoch_20.pth from MuskAI/LTF-v3.3 ...")
                fetch_core(_config.CHECKPOINTS_DIR)
            resolve_backbone()                      # the pinned CLIP ViT-L/14 (fetched now if it is not cached)
            return
        from . import hub

        print(f"[synthscope] downloading '{self.spec.key}' weights from the model zoo ...")
        hub.download(self.spec.key, repo_id=self._repo)

    # -- helpers ------------------------------------------------------------
    def _classify(self, source: PathOrImage):
        """Resolve ``source`` to ``(kind, pil_image_or_None, path_or_None)``. A VLM detector (or a fusion with one) gets
        the decoded file without an RGB conversion (its own loader applies the official EXIF / transparency handling)."""
        vlm = getattr(self, "spec", None) is not None and self.spec.has_vlm   # (a bare Detector in tests has no spec)
        if isinstance(source, Image.Image):
            return "image", (source if vlm else _media.to_rgb(source)), None
        if isinstance(source, (bytes, bytearray)):
            return "image", (_media.decode_image_raw if vlm else _media.decode_image)(bytes(source)), None
        path = Path(source)
        kind = _media.media_kind(path.name)
        if kind == "image":
            return "image", (_media.decode_image_raw if vlm else _media.decode_image)(path.read_bytes()), None
        return "video", None, path
