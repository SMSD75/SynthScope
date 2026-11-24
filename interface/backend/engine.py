"""Inference engine: build SynthScope models from config, load checkpoints, and predict.

Models are built lazily on first use (backbones are large and slow to download/instantiate)
and kept in a small LRU so a shared GPU is not exhausted. All forward passes happen under
``torch.no_grad()`` with the model in fp32/eval mode.
"""
import base64
import gc
import io
import logging
import math
import os
import shutil
import threading
import time
import warnings
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from models.image.ltf.preprocess import LTFViews
from models.image.probe.preprocess import ProbePad, SlidingPatches
from models.image.adb_full.adb_full import fuse_components
from models.video.wevis import wevis as wevis_agent
from models.vlm.vlm import SynthScopeVLM, download_hint
from . import config
from .hf_compat import translate_hf_keys
from .registry import ModelSpec, get_spec, input_note
from . import media

logger = logging.getLogger(__name__)


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _saliency_rgb(sal: torch.Tensor):
    """2-D saliency -> (jet RGB uint8 HxWx3, normalized intensity HxW in [0,1]).

    Robust-normalizes by the 1st/99th percentile so a few hot pixels don't wash it out.
    """
    s = sal.detach().float().cpu().numpy()
    lo, hi = np.percentile(s, 1.0), np.percentile(s, 99.0)
    s = np.clip((s - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * s - 3), 0, 1)   # jet colormap
    g = np.clip(1.5 - np.abs(4 * s - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * s - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8), s


def _saliency_to_png_b64(sal: torch.Tensor, out_size: int = 288) -> str:
    """Render a saliency tensor as a base64 'jet' heatmap PNG."""
    rgb, _ = _saliency_rgb(sal)
    return _png_b64(Image.fromarray(rgb, "RGB").resize((out_size, out_size), Image.BILINEAR))


def _overlay_to_png_b64(sal: torch.Tensor, base: Image.Image, out_size: int = 288,
                        alpha: float = 0.6) -> str:
    """Blend the jet heatmap over the base image; per-pixel opacity = saliency intensity so cool
    regions show the image and hot regions show the artifact heat."""
    rgb, s = _saliency_rgb(sal)
    heat = Image.fromarray(rgb, "RGB").resize((out_size, out_size), Image.BILINEAR)
    base = base.convert("RGB").resize((out_size, out_size), Image.BILINEAR)
    mask = Image.fromarray((np.clip(s, 0, 1) * alpha * 255).astype(np.uint8), "L").resize(
        (out_size, out_size), Image.BILINEAR)
    return _png_b64(Image.composite(heat, base, mask))


def _geometric_base(spec, pil: Image.Image) -> Image.Image:
    """The model's input region in RGB (apply only the resize/center-crop steps of the eval
    transform, skipping to_dtype/normalize) so the overlay aligns with the saliency. Custom
    transforms (AIDE/SPAI native-res) fall back to the native image."""
    base = pil.convert("RGB")
    for t in spec.transform.geometric_steps():
        base = t(base)
    return base


# ---------------------------------------------------------------------------
# Model construction (mirrors trainer.SynthScopeModelTrainer._build_model dispatch, with lazy
# imports so unused heavy dependencies are never loaded).
# ---------------------------------------------------------------------------
def _build_model(cfg):
    model_type = cfg.ModelConfig.get("model_type", "dino").lower()
    if model_type == "safe":
        from models.image.safe.safe import SynthScopeSAFEModel
        return SynthScopeSAFEModel(cfg)
    if model_type == "fat_former":
        from models.image.fat_former.fat_former import SynthScopeFatFormerModel
        return SynthScopeFatFormerModel(cfg)
    if model_type == "co_spy":
        from models.image.co_spy.co_spy import SynthScopeCoSpyModel
        return SynthScopeCoSpyModel(cfg)
    if model_type == "aide":
        from models.image.aide.aide import SynthScopeAIDEModel
        return SynthScopeAIDEModel(cfg)
    if model_type == "spai":
        from models.image.spai.spai import SPAI
        return SPAI(cfg)
    if model_type == "vib_net":
        from models.image.vib_net.vib_net import SynthScopeVIBNetModel
        return SynthScopeVIBNetModel(cfg)
    if model_type == "npr":
        from models.image.npr.npr import NPR
        return NPR(cfg)
    if model_type == "pgc":
        from models.image.pgc.pgc import SynthScopePGCModel
        return SynthScopePGCModel(cfg)
    if model_type == "cure":
        from models.image.cure.cure import SynthScopeCuReModel
        return SynthScopeCuReModel(cfg)
    if model_type == "dda":
        from models.image.dda.dda import SynthScopeDDAModel
        return SynthScopeDDAModel(cfg)
    if model_type == "omniaid":
        from models.image.omniaid.omniaid import SynthScopeOmniAIDModel
        return SynthScopeOmniAIDModel(cfg)
    if model_type == "effort":
        from models.image.effort.effort import SynthScopeEffortModel
        return SynthScopeEffortModel(cfg)
    if model_type == "moejepa":
        from models.image.moejepa.moejepa import SynthScopeMoEJEPAModel
        return SynthScopeMoEJEPAModel(cfg)
    if model_type == "bfree":
        from models.image.bfree.bfree import SynthScopeBFreeModel
        return SynthScopeBFreeModel(cfg)
    if model_type == "d3qe":
        from models.image.d3qe.d3qe import SynthScopeD3QEModel
        return SynthScopeD3QEModel(cfg)
    if model_type == "ltf":
        from models.image.ltf.ltf import SynthScopeLTFModel
        return SynthScopeLTFModel(cfg)
    if model_type == "mirror":
        from models.image.mirror.mirror import SynthScopeMIRRORModel
        return SynthScopeMIRRORModel(cfg)
    if model_type == "vina":
        from models.image.vina.vina import SynthScopeVINAModel
        return SynthScopeVINAModel(cfg)
    if model_type == "probe":
        from models.image.probe.probe import SynthScopePROBEModel
        return SynthScopePROBEModel(cfg)
    if model_type == "probe_resnet":
        from models.image.probe.probe import SynthScopePROBEResNetModel
        return SynthScopePROBEResNetModel(cfg)
    if model_type == "stall":
        from models.video.stall.stall import SynthScopeSTALLModel
        return SynthScopeSTALLModel(cfg)
    if model_type == "d3":
        from models.video.d3.d3 import SynthScopeD3Model
        return SynthScopeD3Model(cfg)
    if model_type == "split":
        from models.video.split.split import SynthScopeSPLITModel
        return SynthScopeSPLITModel(cfg)
    if model_type == "restrav":
        from models.video.restrav.restrav import SynthScopeReStraVModel
        return SynthScopeReStraVModel(cfg)
    if model_type == "waverep":
        from models.video.waverep.waverep import SynthScopeWaveRepModel
        return SynthScopeWaveRepModel(cfg)
    if model_type == "nsg_vd":
        from models.video.nsg_vd.nsg_vd import SynthScopeNSGVDModel
        return SynthScopeNSGVDModel(cfg)
    if model_type == "cnn":
        from models.image.cnn_detection.cnn_detection import SynthScopeCNNModel
        return SynthScopeCNNModel(cfg)
    if model_type == "adapted_dino":
        from models.image.adapted_dino.adapted_dino import SynthScopeAdaptedDINOModel
        return SynthScopeAdaptedDINOModel(cfg)
    if model_type == "dinov2":
        from models.image.dino_detection.dino_detection import SynthScopeDINOModel
        return SynthScopeDINOModel(cfg)
    if model_type == "dinov3":
        from models.image.dino_detection.dino_detection import SynthScopeDINOv3Model
        return SynthScopeDINOv3Model(cfg)
    if model_type == "clip" or cfg.ModelConfig.backbone_name.startswith("CLIP:"):
        from models.image.clip_detection.clip_detection import SynthScopeCLIPModel
        return SynthScopeCLIPModel(cfg)
    raise ValueError(f"Invalid model_type: {model_type}")


def ffmpeg_missing_message(spec: ModelSpec) -> str:
    """Why ``spec`` has no usable ffmpeg. For a detector with a pinned build and its own variable (NSG-VD) it names that
    variable and the pin, and says so when the variable is set but is not an executable file; the other detectors keep
    the global wording."""
    generic = f"'{spec.key}' needs ffmpeg (set SYNTHSCOPE_FFMPEG or pip install imageio-ffmpeg)."
    if not (spec.ffmpeg_env and spec.ffmpeg_version):
        return generic
    env = os.environ.get(spec.ffmpeg_env)
    pin = (f"'{spec.key}' needs an ffmpeg {spec.ffmpeg_version} build (its scores depend on the build; another one is used "
           f"with a warning and does not reproduce the official results)")
    if env:
        return (f"{pin}: ${spec.ffmpeg_env}={env!r} is set but is not an executable file or a command on the PATH. "
                f"Point ${spec.ffmpeg_env} at an ffmpeg {spec.ffmpeg_version} binary.")
    return (f"{pin}: no ffmpeg was found. Set ${spec.ffmpeg_env} to an ffmpeg {spec.ffmpeg_version} binary.")


def spec_ffmpeg_exe(spec: ModelSpec) -> str:
    """The ffmpeg binary for ``spec``: its own ``ffmpeg_env`` variable when set (NSG-VD: $SYNTHSCOPE_FFMPEG_NSG_VD), else
    the global choice (media.ffmpeg_exe). Raises FileNotFoundError when there is none (for a detector with an
    ``ffmpeg_env`` the message names that variable and the pin; a set but invalid variable is never replaced by the
    global choice)."""
    env = os.environ.get(spec.ffmpeg_env) if spec.ffmpeg_env else None
    if env:
        if not os.path.isfile(env) and shutil.which(env) is None:
            raise FileNotFoundError(ffmpeg_missing_message(spec))
        return env
    try:
        return media.ffmpeg_exe()
    except (FileNotFoundError, RuntimeError, OSError) as e:
        if spec.ffmpeg_env and spec.ffmpeg_version:
            raise FileNotFoundError(ffmpeg_missing_message(spec)) from e
        raise


_FFMPEG_WARNED: set = set()


def spec_ffmpeg(spec: ModelSpec) -> dict:
    """{"path", "version", "required", "match"} of the ffmpeg build ``spec`` will use; warns (once per detector and
    binary, through logging and ``warnings``) when it is not the pinned ``spec.ffmpeg_version``."""
    exe = spec_ffmpeg_exe(spec)
    version = media.ffmpeg_version(exe)
    pin = spec.ffmpeg_version
    match = not pin or version == pin or version.startswith(pin + ".")
    if not match and (spec.key, exe) not in _FFMPEG_WARNED:
        _FFMPEG_WARNED.add((spec.key, exe))
        msg = (f"{spec.key}: ffmpeg {version} ({exe}) is not the pinned {pin}.x: this detector's frames (JPEG encoder, "
               f"frame selection) and so its scores depend on the build; set {spec.ffmpeg_env or 'SYNTHSCOPE_FFMPEG'} "
               f"to an ffmpeg {pin} binary to reproduce the official results.")
        logger.warning(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return {"path": exe, "version": version, "required": pin, "match": match}


def aux_overrides(spec: ModelSpec) -> list[str]:
    """Hydra overrides pointing a model at the extra files it reads from the checkpoint folder (``spec.aux_files``, e.g.
    NSG-VD's diffusion model and reference set)."""
    return [f"{key}={config.CHECKPOINTS_DIR / name}" for key, name in spec.aux_files.items()]


def _load_checkpoint(model, spec: ModelSpec):
    """Load weights for ``spec`` into ``model`` (already constructed). Returns a status string."""
    path = config.CHECKPOINTS_DIR / spec.checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train or place the checkpoint, then retry."
        )
    if path.suffix == ".safetensors":                 # released as safetensors (e.g. MoE-JEPA): a flat state dict
        sd = load_safetensors(str(path))
    else:
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if spec.remap is not None:
        sd = spec.remap(sd)
    # A checkpoint that holds a transformers backbone carries the parameter names of the release that saved it; map them
    # onto this release's names when the rename is exact (hf_compat.HF_RENAMES), else leave them for the checks below.
    sd, hf_note = translate_hf_keys(sd, model.state_dict())
    if hf_note:
        logger.info("%s: transformers parameter names translated (%s)", spec.key, hf_note)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Buffers that are legitimately reconstructed in __init__ / not in official dumps.
    benign = {"clip_mean", "clip_std", "imnet_mean", "imnet_std"}
    # Frozen pretrained parts that a model loads from their own release in __init__ and that its official
    # checkpoint therefore does not contain (e.g. D3QE's CLIP and VQ tokenizer); everything else stays strict.
    pretrained = tuple(getattr(model, "pretrained_prefixes", ()))
    real_missing = [
        k for k in missing
        if not k.startswith("criterion") and not k.startswith("dwt_filter.")
        and k.split(".")[-1] not in benign and not (pretrained and k.startswith(pretrained))
    ]
    if spec.ckpt_kind.startswith("official"):
        # Official-weights loads are verified to be exact; flag any genuine mismatch loudly.
        if real_missing or unexpected:
            raise RuntimeError(
                f"{spec.key}: checkpoint mismatch missing={real_missing[:8]} "
                f"unexpected={unexpected[:8]}"
            )
    elif real_missing or unexpected:
        # Non-official kinds are not raised on (a model may legitimately rebuild some tensors itself, declared via
        # ``pretrained_prefixes``), but a mismatch must never be silent: dropped keys (e.g. a wrapper prefix the
        # checkpoint carries and the class does not, which once left the served PGC without its LoRA weights) or
        # tensors that kept their __init__ values change what the detector computes.
        logger.warning(
            "%s: checkpoint %s does not match the model exactly: %d unexpected key(s) were dropped and %d key(s) kept "
            "their __init__ values (unexpected=%s, missing=%s). Fix the remap or declare the keys in the model's "
            "pretrained_prefixes.", spec.key, path.name, len(unexpected), len(real_missing), unexpected[:4],
            real_missing[:4],
        )
    return (
        f"loaded {len(sd)} tensors from {path.name} "
        f"(missing={len(real_missing)}, unexpected={len(unexpected)})"
        + (f" [transformers key names translated, {hf_note}]" if hf_note else "")
    )


def patch_layout(spec: ModelSpec, pil: Image.Image, num_patches: int):
    """For a detector that scores a bag of non-overlapping patches (PROBE: ``ProbePad`` then ``SlidingPatches``): the
    padded image size ``(width, height)`` and the patch grid ``(rows, cols)`` the bag came from, or ``None`` when the
    transform is not of that kind or the grid does not hold ``num_patches`` patches."""
    steps = spec.transform.transforms if spec.transform is not None else []
    pads = [t for t in steps if isinstance(t, ProbePad)]
    cuts = [t for t in steps if isinstance(t, SlidingPatches)]
    if len(pads) != 1 or len(cuts) != 1:
        return None
    width, height = pads[0](pil).size
    rows, cols = height // cuts[0].size, width // cuts[0].size
    return ((width, height), (rows, cols)) if rows * cols == num_patches else None


def is_view_bag(spec: ModelSpec) -> bool:
    """Whether ``spec`` reads an image as a bag of native-resolution views whose count grows with the image size (LTF's
    tiles, PROBE's patches): the detectors ``SYNTHSCOPE_MAX_VIEWS`` and the web app's batch cap are about."""
    steps = spec.transform.transforms if spec.native_res and spec.transform is not None else []
    return any(isinstance(t, (LTFViews, SlidingPatches)) for t in steps)


class InputTooLarge(ValueError):
    """A request whose view bag is above ``config.MAX_VIEWS`` (the hosted CPU demo's guard; the web API answers 413)."""


def check_views(spec: ModelSpec, x: torch.Tensor) -> None:
    """Refuse an input that a native-resolution view-bag detector (``x`` = [B, views, 3, H, W] whose view count grows
    with the image size: LTF's tiles, PROBE's patches) would read as more than ``config.MAX_VIEWS`` views in all (summed
    over the frames of a video). No-op when the limit is 0 (the default) or for any other detector (AIDE's fixed five
    views are not counted). The input is refused rather than resized, so every score the demo returns is the
    detector's official score."""
    if not config.MAX_VIEWS or x.dim() != 5 or not is_view_bag(spec):
        return
    views = int(x.shape[0]) * int(x.shape[1])
    if views > config.MAX_VIEWS:
        raise InputTooLarge(
            f"{spec.name} reads this input at its own resolution as {views} views of {int(x.shape[-1])} px (tiles plus "
            f"a global view); this server scores at most {config.MAX_VIEWS} per request, because each view takes about "
            f"two seconds on its CPU. The image is not shrunk automatically, since that would change what the "
            f"detector sees: upload a smaller or cropped image, or run SynthScope locally, which has no "
            f"limit.")


def vlm_images_only(spec: ModelSpec) -> str:
    what = "combines a VLM detector, which" if spec.fusion is not None else "is a VLM detector that"
    return (f"'{spec.key}' {what} writes a rationale per image (seconds per image on a GPU): it scores "
            "still images only, not videos.")


def fusion_components(spec: ModelSpec) -> tuple[ModelSpec, ModelSpec]:
    """The (VLM, expert) specs of a ``predict_kind="fusion"`` detector."""
    return get_spec(spec.fusion.vlm), get_spec(spec.fusion.expert)


def vlm_base_folder(spec: ModelSpec):
    """A LoRA VLM detector's local base folder (``vlm.base_model``, relative names under CHECKPOINTS_DIR); else None."""
    base = spec.vlm.base_model if spec.vlm is not None else None
    if not base:
        return None
    return Path(base) if Path(base).is_absolute() else config.CHECKPOINTS_DIR / base


def vlm_processor_folder(spec: ModelSpec):
    """A VLM detector's separate local processor folder (``vlm.processor``, relative names under CHECKPOINTS_DIR, e.g.
    FakeVLM's llava-hf processor); else None."""
    proc = spec.vlm.processor if spec.vlm is not None else None
    if not proc:
        return None
    return Path(proc) if Path(proc).is_absolute() else config.CHECKPOINTS_DIR / proc


def _load_vlm(spec: ModelSpec, path, device: str) -> SynthScopeVLM:
    """Load a VLM detector strictly (``SynthScopeVLM`` raises on any missing / unexpected / mismatched tensor): a
    Hugging Face model folder (``config.json``), or a LoRA adapter folder (``adapter_config.json``) on the local
    ``base_model`` folder. Relative base / processor names are resolved under CHECKPOINTS_DIR; nothing is downloaded."""
    marker = "adapter_config.json" if spec.vlm.base_model else "config.json"
    base = vlm_base_folder(spec)
    missing = [f"no {marker} in {path}"] if path is None or not (path / marker).exists() else []
    if base is not None and not (base / "config.json").exists():
        missing.append(f"no base model (config.json) in {base}")
    proc = vlm_processor_folder(spec)
    if proc is not None and not (proc / "preprocessor_config.json").exists():
        missing.append(f"no processor (preprocessor_config.json) in {proc}")
    if missing:
        hint = download_hint(spec.vlm, path, config.CHECKPOINTS_DIR)
        raise FileNotFoundError(f"{spec.key}: {'; '.join(missing)}." + (f" Download the official weights: {hint}"
                                                                         if hint else ""))
    return SynthScopeVLM(path, spec.vlm, device=device, root=config.CHECKPOINTS_DIR)


def agent_missing(spec: ModelSpec) -> list:
    """Why an agentic video detector (``predict_kind="video_agent"``) cannot run here; empty when it can."""
    return wevis_agent.missing(spec.agent, config.CHECKPOINTS_DIR, config.CACHE_DIR)


def _load_agent(spec: ModelSpec, device: str) -> "wevis_agent.WeVisRunner":
    """Start an agentic video detector's runner process (WeVisForensics: the official code under its own pinned
    dependencies; it verifies the sha256 of every pinned file and loads the adapters strictly; local files only)."""
    if device != "cuda":
        raise RuntimeError(f"'{spec.key}' runs the official agent on a CUDA GPU (about 22 GB); no GPU here.")
    reasons = agent_missing(spec)
    if reasons:
        raise FileNotFoundError(f"{spec.key}: {'; '.join(reasons)}. Get the official release: "
                                + wevis_agent.download_hint(spec.agent))
    runner = wevis_agent.WeVisRunner(spec.agent, config.CHECKPOINTS_DIR, config.CACHE_DIR)
    runner.start()
    return runner


def _external(model) -> bool:
    """True for a detector that runs outside this process (an agent's runner) and must be stopped with ``close()``."""
    return callable(getattr(model, "close", None)) and not isinstance(model, torch.nn.Module)


def _close(model) -> None:
    """Stop what an evicted detector runs outside this process (an agent's runner); no-op for torch modules. An agent's
    ``close()`` first waits for the video in progress (up to ``close_timeout``), so callers run it OUTSIDE
    ``Engine._lock``: the other detectors keep serving meanwhile."""
    if _external(model):
        model.close()


class Engine:
    """Lazily builds, caches, and runs the detectors."""

    def __init__(self, device: str | None = None):
        # ``device`` lets a caller (e.g. the ``synthscope`` Python API) pin a specific device;
        # the web backend constructs it with no argument and auto-detects via config.device().
        self.device = device or config.device()
        self._cache: "OrderedDict[str, torch.nn.Module]" = OrderedDict()
        self._status: dict[str, str] = {}
        self._lock = threading.Lock()
        # Detectors a request is using right now (``pinned``): the LRU never evicts them and ``release_gpu`` keeps them,
        # so concurrent requests (the web app scores in worker threads) cannot drop a model mid-use.
        self._pins: Counter = Counter()

    # -- availability -------------------------------------------------------
    def checkpoint_path(self, spec: ModelSpec):
        if spec.checkpoint is None:
            return None
        return config.CHECKPOINTS_DIR / spec.checkpoint

    @staticmethod
    def ffmpeg_missing(spec: ModelSpec) -> bool:
        """True for a detector whose video protocol shells out to ffmpeg (vina_video, nsg_vd) when none can be found."""
        if spec.video_sampler == "ffmpeg_select_jpeg":
            try:
                spec_ffmpeg_exe(spec)
            except (FileNotFoundError, RuntimeError, OSError):
                return True
            return False
        return spec.video_sampler == "ffmpeg_fps_center" and not media.ffmpeg_available()

    @staticmethod
    def ffmpeg_missing_message(spec: ModelSpec) -> str:
        """The error text for ``ffmpeg_missing(spec)`` (see ``ffmpeg_missing_message``)."""
        return ffmpeg_missing_message(spec)

    def is_available(self, spec: ModelSpec) -> bool:
        if spec.key == "stall":  # training-free, but needs the gated DINOv3 ViT-L/16 weights
            import os
            from .registry import DINOV3_WEIGHTS
            return os.path.exists(os.environ.get("DINO_V3_WEIGHTS") or DINOV3_WEIGHTS)
        if self.ffmpeg_missing(spec):
            return False
        if spec.fusion is not None:                 # no weights of its own: both components must be runnable
            return all(self.is_available(part) for part in fusion_components(spec))
        if not all((config.CHECKPOINTS_DIR / f).exists() for f in spec.aux_files.values()):
            return False
        if spec.ckpt_kind == "none":  # training-free (D3)
            return True
        if spec.predict_kind == "video_agent":     # adapters, base, OCR models, official code and the isolated runtime
            return not agent_missing(spec)
        p = self.checkpoint_path(spec)
        base = vlm_base_folder(spec)                # a LoRA VLM detector also needs its local base model
        proc = vlm_processor_folder(spec)           # ... and a VLM with a separate processor, that folder
        return bool(p and p.exists()) and (base is None or base.exists()) and (proc is None or proc.exists())

    # -- model lifecycle ----------------------------------------------------
    @contextmanager
    def pinned(self, spec: ModelSpec):
        """Mark ``spec`` (and a fusion detector's two components) as in use for the ``with`` block: while pinned, the
        LRU does not evict it and ``release_gpu`` keeps it. A load that finds every resident detector pinned goes over
        ``MAX_RESIDENT_MODELS`` for the moment instead of waiting (no deadlock); the cache is trimmed back to the limit
        when the pins are released. Pins count, so two requests may pin the same detector."""
        keys = [spec.key] + ([part.key for part in fusion_components(spec)] if spec.fusion is not None else [])
        with self._lock:
            self._pins.update(keys)
        try:
            yield
        finally:
            with self._lock:
                self._pins.subtract(keys)
                self._pins += Counter()                 # drop the zero counts
                dropped = []
                while len(self._cache) > config.MAX_RESIDENT_MODELS:
                    victim = self._unpinned_lru()
                    if victim is None:
                        break
                    dropped.append(self._cache.pop(victim))
            self._discard(dropped)

    def _unpinned_lru(self):
        """The least recently used resident detector that no request has pinned, or None; the caller holds the lock."""
        return next((key for key in self._cache if not self._pins[key]), None)

    def _discard(self, dropped: list) -> None:
        """Free evicted detectors OUTSIDE the lock (an agent's ``close()`` waits for its video in progress)."""
        if not dropped:
            return
        for model in dropped:
            _close(model)
        del model
        dropped.clear()                                 # the caller's list: no reference keeps the models alive
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def release_gpu(self) -> None:
        """Drop every resident detector that no request is using (they reload lazily on the next request) so a heavy
        local image generator, which runs in its own process and wants most of the GPU, does not collide with them.
        Pinned detectors stay: the request using them finishes first (the web app also holds ``_GPU_LOCK``)."""
        with self._lock:
            dropped = [self._cache.pop(key) for key in [k for k in self._cache if not self._pins[k]]]
        for model in dropped:                      # outside the lock: an agent finishes its video in progress first
            _close(model)
        del dropped
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _get_model(self, spec: ModelSpec) -> torch.nn.Module:
        while True:
            with self._lock:
                if spec.key in self._cache:
                    self._cache.move_to_end(spec.key)
                    return self._cache[spec.key]

                # Evict LRU entries if at capacity, never one a request has pinned (if all are, load over the limit).
                external = []
                while len(self._cache) >= config.MAX_RESIDENT_MODELS:
                    victim = self._unpinned_lru()
                    if victim is None:
                        break
                    old_model = self._cache.pop(victim)
                    if _external(old_model):
                        external.append(old_model)
                    del old_model
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                if not external:
                    return self._load(spec)
            # An evicted agent's runner holds most of the GPU and may still be scoring a video for another request:
            # stop it OUTSIDE the lock (close() waits for that video), so the other resident detectors keep serving,
            # then take the lock again and load (the loop re-checks the cache, which may have changed meanwhile).
            for model in external:
                _close(model)
            del external
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _load(self, spec: ModelSpec) -> torch.nn.Module:
        """Build ``spec``'s detector and add it to the cache; the caller holds ``self._lock`` and has made room.

        Loading runs UNDER the engine lock, as for every detector: other requests wait for it. For a torch model that is
        seconds; for the agent (``video_agent``) it is the runner's start, about 40 s with warm file caches and 3-4 min
        on a cold node (``ready_timeout`` 3600 s), during which every other detector request on this engine waits."""
        if spec.predict_kind == "fusion":    # its components load and cache under their own keys
            raise ValueError(f"'{spec.key}' combines {spec.fusion.vlm} and {spec.fusion.expert}, which load on their own; "
                             "score it with Engine.predict_image or Engine.fusion_score.")
        if spec.predict_kind == "vlm":       # a released HF folder: strict from_pretrained, kept in its own dtype
            model = _load_vlm(spec, self.checkpoint_path(spec), self.device)
            self._status[spec.key] = (f"loaded {len(model.hf.state_dict())} tensors from {model.path.name} "
                                      f"(missing=0, unexpected=0; {spec.vlm.model_class}, {spec.vlm.dtype}"
                                      + (f"; LoRA adapter unmerged on {spec.vlm.base_model}" if spec.vlm.base_model
                                         else "")
                                      + (f"; processor {spec.vlm.processor}" if spec.vlm.processor else "") + ")")
            self._cache[spec.key] = model
            return model

        if spec.predict_kind == "video_agent":   # the official agent in its own process (models/video/wevis)
            runner = _load_agent(spec, self.device)
            info = runner.info or {}
            rel = info.get("release", {})
            self._status[spec.key] = (f"official code {str(rel.get('code_commit', '?'))[:7]} + adapters "
                                      f"{str(rel.get('adapters_revision', '?'))[:8]} on base "
                                      f"{str(rel.get('base_revision', '?'))[:8]} (sha256 verified, adapters strict: "
                                      f"{sum((info.get('adapter_tensors') or {}).values())} tensors), transformers "
                                      f"{(info.get('versions') or {}).get('transformers')} in a separate process; "
                                      f"audio off")
            self._cache[spec.key] = runner
            return runner

        from config_utils import compose_config

        cfg = compose_config(spec.overrides + aux_overrides(spec))
        model = _build_model(cfg)
        if spec.ckpt_kind != "none":
            self._status[spec.key] = _load_checkpoint(model, spec)
        else:
            self._status[spec.key] = "training-free (no checkpoint)"
        model = model.to(self.device).float().eval()
        self._cache[spec.key] = model
        return model

    # -- prediction helpers -------------------------------------------------
    @torch.no_grad()
    def _fake_prob(self, model, spec: ModelSpec, x: torch.Tensor) -> torch.Tensor:
        """Return P(fake) per sample for an image batch ``x`` [B,3,H,W]."""
        kind = spec.predict_kind
        if kind == "ce":
            return model(x).softmax(dim=1)[:, 1]
        if kind == "bce":
            return torch.sigmoid(model(x).reshape(x.shape[0]))
        if kind == "bce_npr":
            logits = model.forward(model.NPR_transform(x)).reshape(x.shape[0])
            return torch.sigmoid(logits)
        if kind == "vib":
            out = model(x)
            logit = out[1] if isinstance(out, (tuple, list)) else out
            return torch.sigmoid(logit.reshape(x.shape[0]))
        if kind == "vlm":
            raise ValueError(f"'{spec.key}' is a VLM detector: it scores PIL images one at a time (Engine.vlm_score), "
                             "not preprocessed tensors.")
        raise ValueError(f"Unexpected image predict_kind: {kind}")

    def _video_calibration(self, key: str):
        """Load (and cache) a training-free video detector's raw score -> probability calibration, if it has one: a
        ``<key>_calibration.json`` in the checkpoint folder overrides the one shipped in ``models/video/<key>/``."""
        if not hasattr(self, "_video_cal"):
            self._video_cal = {}
        if key not in self._video_cal:
            import json
            p = config.CHECKPOINTS_DIR / f"{key}_calibration.json"
            if not p.exists():
                p = config.REPO_ROOT / "models" / "video" / key / f"{key}_calibration.json"
            self._video_cal[key] = json.loads(p.read_text()) if p.exists() else None
        return self._video_cal[key]

    @torch.no_grad()
    def _video_fake_prob(self, model, spec: ModelSpec, clip: torch.Tensor) -> torch.Tensor:
        """Return P(fake) for a single normalized clip tensor [T,3,H,W]."""
        frames = clip.unsqueeze(0).to(self.device)  # [1,T,3,H,W]
        if spec.key in ("d3", "split"):
            cal = self._video_calibration(spec.key)
            if cal is not None:
                s = model.score(frames)  # [1] raw score (D3 volatility, SPLIT roughness x incoherence)
                if cal.get("log_score"):  # SPLIT's score spans many orders of magnitude: calibrated on its log
                    s = s.clamp_min(1e-30).log()
                z = (s - cal["threshold"]) / cal["scale"]
                if not cal["higher_is_fake"]:
                    z = -z
                return torch.sigmoid(z)
        out = model(frames)  # [1,2]
        if spec.video_prob == "sigmoid":  # one fake logit (VINA: the mean of the frame logits)
            return torch.sigmoid(out.reshape(-1))
        if spec.key == "stall":
            # STALL returns [real_prob, fake_prob] (already calibrated percentiles); use directly.
            return out[:, 1]
        return out.softmax(dim=1)[:, 1]

    # -- public API ---------------------------------------------------------
    @torch.no_grad()
    def _interpret_map(self, model, spec: ModelSpec, x: torch.Tensor, pil=None):
        """Return ``{"map": heatmap_b64, "overlay": overlay_b64}`` for the detector's artifact map,
        or None. ``map`` is the bare jet heatmap; ``overlay`` blends it on the model-input image.

        Best-effort + isolated: any failure returns None and never affects the prediction.
        ``x`` is the already-transformed [1,3,H,W] input; ``pil`` is the native image (some maps,
        e.g. AIDE's native-res DCT energy, are computed from the original resolution).
        """
        kind = getattr(spec, "interpret", "") or ""
        if not kind:
            return None
        try:
            if kind == "safe_hf":          # SAFE: DWT HH high-frequency band magnitude
                hh = model._preprocess_high_frequency(x)        # [1,3,H,W]
                sal = hh.abs().mean(dim=1)[0]                   # [H,W]
            elif kind == "npr_residual":   # NPR: neighboring-pixel residual magnitude
                res = model.NPR_transform(x)                    # [1,3,H,W]
                sal = res.abs().mean(dim=1)[0]                  # [H,W]
            elif kind == "pgc_peak":       # PGC: peak-guided residual fake-evidence map
                sal = model.peak_residual_map(x)               # [Hr,Wr]
            elif kind == "aide_dct":       # AIDE: native-res DCT per-window energy (patch selection)
                if pil is None:
                    return None
                five_view = next((t for t in spec.transform.transforms if hasattr(t, "dct_energy_map")), None)
                if five_view is None:
                    return None
                sal = five_view.dct_energy_map(pil)            # [nH,Wn]
            elif kind == "spai_patch":     # SPAI: per-patch spectral fake-evidence over the 224-grid
                sal = model.patch_fake_map(x)                  # [nH,nW] or None (five-crop/large grid)
                if sal is None:
                    return None
            else:
                return None
            base = _geometric_base(spec, pil) if pil is not None else None
            out = {"map": _saliency_to_png_b64(sal)}
            if base is not None:
                out["overlay"] = _overlay_to_png_b64(sal, base)
            return out
        except Exception:  # noqa: BLE001 — interpretability is optional, never break predict
            return None

    @staticmethod
    def _model_input_data_url(spec: ModelSpec, x: torch.Tensor, max_side: int = 1280, cols: int | None = None) -> str:
        """The image exactly as the network receives it (after this detector's preprocessing), as a PNG data URL.
        Normalization is undone for display; multi-view inputs are laid out side by side (up to 5 views, e.g. AIDE)
        or in a near-square grid (LTF has one view per tile, up to dozens). ``cols`` fixes the number of columns of a
        multi-view layout (PROBE: the patches' real grid)."""
        t = x[0].detach().float().cpu()
        norm = spec.transform.normalize_step()
        if norm is not None:
            t = t * torch.tensor(norm.std)[:, None, None] + torch.tensor(norm.mean)[:, None, None]
        elif spec.key == "cure":                       # CuRe maps pixels to (x - 127.5) / 127.5
            t = t * 0.5 + 0.5
        if t.dim() == 4:
            cols = cols or (len(t) if len(t) <= 5 else math.ceil(math.sqrt(len(t))))
            pad = -len(t) % cols
            if pad:
                t = torch.cat([t, torch.zeros(pad, *t.shape[1:])])
            t = torch.cat([torch.cat(list(row), dim=2) for row in t.view(-1, cols, *t.shape[1:])], dim=1)
        if t.min() < 0 or t.max() > 1:                 # unknown scaling: stretch for display only
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        img = Image.fromarray((t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())
        if max(img.size) > max_side:
            k = max_side / max(img.size)
            img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def vlm_score(self, spec: ModelSpec, pil: Image.Image, decoding: str = "greedy", seed: int | None = None) -> dict:
        """A VLM detector's full output for one image (``SynthScopeVLM.score``): ``fake_probability`` (sigmoid of
        ``logit``, the fake-minus-real logit at the answer token, or at the word the official parser read; 0.5 with
        ``score_source`` "unparsed" (no verdict either) or "unscored" (a parsed verdict) when no verdict token was found), the parsed ``verdict`` ("fake" / "real" / None), the ``rationale`` and ``response`` text,
        the verdict token and step, and the generation length. ``decoding="official"`` uses the authors' sampling
        (seeded with ``seed``) instead of greedy decoding. Pass the decoded file without an RGB conversion
        (``media.decode_image_raw``): the config's ``image_loader`` reproduces the official server's EXIF / alpha
        handling."""
        return self._get_model(spec).score(pil, decoding=decoding, seed=seed)

    @torch.no_grad()
    def expert_probability(self, spec: ModelSpec, pil: Image.Image) -> float:
        """An image detector's P(fake) for one RGB image, unrounded (a fusion's threshold is applied to it)."""
        x = spec.transform(pil).unsqueeze(0).to(self.device)
        return float(self._fake_prob(self._get_model(spec), spec, x).item())

    def fusion_score(self, spec: ModelSpec, pil: Image.Image) -> dict:
        """A ``fusion`` detector's output for one image (Align, Don't Blend's full method, ``models/image/adb_full``): the
        VLM component's full output (``vlm_score``, on the decoded file as a VLM gets it), the expert's unrounded
        P(fake) (on the RGB conversion, ``media.to_rgb``: the official ``Image.open(path).convert("RGB")``), and the
        official decision and SynthScope's score (``fuse_components``)."""
        vspec, espec = fusion_components(spec)
        vlm = self.vlm_score(vspec, pil)
        p_expert = self.expert_probability(espec, media.to_rgb(pil))
        fused = fuse_components(vlm["fake_probability"], vlm["verdict"], p_expert)
        return {**fused, "vlm": vlm, "expert_fake_probability": p_expert}

    def _predict_fusion(self, spec: ModelSpec, pil: Image.Image) -> dict:
        vspec, espec = fusion_components(spec)
        self._get_model(vspec)                      # load first: elapsed covers scoring only
        self._get_model(espec)
        t0 = time.time()
        out = self.fusion_score(spec, pil)
        res = self._result(spec, out["score"], time.time() - t0)
        res["verdict"] = out["verdict"]             # the official rule, not the rounded score (it can round to 0.5)
        res["fused_score"] = out["score"]           # unrounded: >= 0.5 exactly when the verdict is "fake"
        res["decided_by"] = out["decided_by"]       # "vlm" | "expert" | "both" | "neither"
        vlm = out["vlm"]
        res["components"] = {
            vspec.key: {"verdict": out["vlm_verdict"], "fake_probability": vlm["fake_probability"],
                        "logit": round(vlm["logit"], 6), "score_source": vlm["score_source"],
                        "verdict_token": vlm["verdict_token"], "new_tokens": vlm["new_tokens"],
                        "finished": vlm["finished"], "input_size": vlm["input_size"]},
            espec.key: {"verdict": out["expert_verdict"], "fake_probability": out["expert_fake_probability"],
                        "threshold": spec.fusion.expert_threshold},
        }
        res["rationale"], res["response"] = vlm["rationale"], vlm["response"]
        res["orig_size"] = [pil.width, pil.height]
        res["input_note"] = input_note(spec)
        return res

    def _predict_vlm(self, spec: ModelSpec, pil: Image.Image) -> dict:
        self._get_model(spec)                       # load first: elapsed covers scoring only
        t0 = time.time()
        out = self.vlm_score(spec, pil)
        res = self._result(spec, out["fake_probability"], time.time() - t0)
        # The official parser's answer; under greedy decoding it agrees with the sign of the logit, except for an
        # answer naming both words (the parser checks "real" first). No parsable answer: "unparsed".
        res["verdict"] = out["verdict"] or "unparsed"
        res["logit"] = round(out["logit"], 6)
        for key in ("rationale", "response", "verdict_token", "score_source", "new_tokens", "finished", "decoding"):
            res[key] = out[key]
        res["orig_size"] = [pil.width, pil.height]
        res["input_size"] = out["input_size"]
        res["input_note"] = input_note(spec)
        return res

    def predict_image(self, spec: ModelSpec, pil: Image.Image) -> dict:
        if spec.predict_kind == "video_agent":        # refused before the runner is started
            raise ValueError(f"'{spec.key}' is a video-only detector (an agent that investigates a video); it cannot "
                             "score a still image.")
        if spec.predict_kind == "vlm":
            return self._predict_vlm(spec, pil)
        if spec.predict_kind == "fusion":
            return self._predict_fusion(spec, pil)
        x = spec.transform(pil).unsqueeze(0)
        check_views(spec, x)                          # before the model loads: a refused input costs nothing
        model = self._get_model(spec)
        x = x.to(self.device)
        t0 = time.time()
        p = float(self._fake_prob(model, spec, x).item())
        res = self._result(spec, p, time.time() - t0)
        res["orig_size"] = [pil.width, pil.height]
        res["input_size"] = [int(x.shape[-1]), int(x.shape[-2])]     # what the network actually received
        layout = patch_layout(spec, pil, int(x.shape[1])) if x.dim() == 5 else None
        if layout is not None:        # a patch bag: report the padded image and its patch grid, not one patch's size
            (width, height), (rows, cols) = layout
            res["input_size"] = [width, height]
            res["input_grid"] = [rows, cols]
            res["input_patches"] = rows * cols
        res["input_note"] = input_note(spec)
        try:
            res["input_image"] = self._model_input_data_url(spec, x, cols=layout[1][1] if layout else None)
        except Exception:                                           # display-only: never fail a prediction over it
            pass
        amap = self._interpret_map(model, spec, x, pil)
        if amap is not None:
            res["interpret_map"] = amap["map"]
            if amap.get("overlay"):
                res["interpret_overlay"] = amap["overlay"]
            res["interpret_kind"] = spec.interpret
        return res

    def predict_video_frames(self, spec: ModelSpec, frames: list[Image.Image]) -> dict:
        """Run an *image* detector on sampled video frames and aggregate (mean P(fake))."""
        if spec.predict_kind == "video_agent":
            raise ValueError(f"'{spec.key}' scores the video file itself (Engine.predict_video_clip), not sampled frames.")
        if spec.predict_kind in ("vlm", "fusion"):
            raise ValueError(vlm_images_only(spec))
        batch = torch.stack([spec.transform(f) for f in frames])
        check_views(spec, batch)
        model = self._get_model(spec)
        batch = batch.to(self.device)
        t0 = time.time()
        per_frame = self._fake_prob(model, spec, batch).detach().float().cpu().tolist()
        elapsed = time.time() - t0
        mean_p = sum(per_frame) / len(per_frame)
        res = self._result(spec, mean_p, elapsed)
        res["per_frame"] = [round(v, 6) for v in per_frame]
        res["aggregation"] = "mean over sampled frames"
        return res

    def video_clip(self, spec: ModelSpec, path) -> torch.Tensor:
        """Decode and normalize a clip exactly as ``spec`` prescribes: ``[T, 3, H, W]``.

        For video_sampler "cv2_frames" (WaveRep) this materializes every selected frame at 504x504 float (~3 MB each):
        with ``num_frames=0`` (the official demo's default, every frame) a long video needs that much memory per frame;
        scoring goes through ``video_frame_logits`` instead, which streams."""
        if spec.video_sampler == "ffmpeg_fps_center":   # VINA: ffmpeg JPEG frames, central window, image transform
            frames = media.ffmpeg_fps_frames(path, fps=spec.fps, num_frames=spec.num_frames)
            return torch.stack([spec.transform(f) for f in frames])
        if spec.video_sampler == "ffmpeg_select_jpeg":  # NSG-VD: every k-th frame as ffmpeg JPEGs, per-frame transform
            frames = media.ffmpeg_select_frames(path, num_frames=spec.num_frames, exe=spec_ffmpeg(spec)["path"])
            return torch.stack([spec.transform(f) for f in frames])
        if spec.video_sampler == "cv2_frames":          # WaveRep: the first num_frames (0: all) OpenCV frames
            frames = [spec.transform(f) for f in media.cv2_frames(path, limit=spec.num_frames or None)]
            if not frames:
                raise media.VideoDecodeError("Could not decode any frames from the video.")
            return torch.stack(frames)
        clip = media.video_clip_tensor(
            path, num_frames=spec.num_frames, fps=spec.fps, frame_size=spec.frame_size,
            window_center=spec.window_center, interp=spec.resize_interp,
            edge_crop=getattr(spec, "edge_crop", 0.0),
            edge_crop_long_side=getattr(spec, "edge_crop_long_side", False),
            short_clip_frames=getattr(spec, "short_clip_frames", 0),
            fps_nearest=getattr(spec, "fps_nearest", False), bgr=getattr(spec, "bgr", False),
        )
        mean, std = spec.video_norm
        return (clip - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

    @torch.no_grad()
    def video_frame_logits(self, spec: ModelSpec, path) -> torch.Tensor:
        """The per-frame logits [T] of a frame-level video detector (video_sampler "cv2_frames", WaveRep): the frames
        are read, transformed and scored ``frame_batch`` at a time, so a long video is never held in memory at once."""
        model = self._get_model(spec)
        batch, logits = [], []
        for frame in media.cv2_frames(path, limit=spec.num_frames or None):
            batch.append(spec.transform(frame))
            if len(batch) == model.frame_batch:
                logits.append(model.frame_logits(torch.stack(batch).to(self.device)).float().cpu())
                batch = []
        if batch:
            logits.append(model.frame_logits(torch.stack(batch).to(self.device)).float().cpu())
        if not logits:
            raise media.VideoDecodeError("Could not decode any frames from the video.")
        return torch.cat(logits)

    def agent_score(self, spec: ModelSpec, path, verify_direct: bool = False) -> dict:
        """An agentic video detector's full output for one video (``WeVisRunner.score``): ``fake_probability``,
        ``logit``, ``score_source``, the agent's ``verdict`` ("real" / "fake" / None), ``status``, ``trace`` and ``cost``.
        Raises ``WeVisOOMError`` (retry on a larger GPU), ``WeVisToolError`` (the official tool_error) or
        ``WeVisRunnerError``. A runner client evicted (closed) by another request between ``_get_model`` and ``score``
        raises ``WeVisClosedError``: the detector is fetched again (a fresh runner) once."""
        runner = self._get_model(spec)
        try:
            return runner.score(path, verify_direct=verify_direct)
        except wevis_agent.WeVisClosedError:
            again = self._get_model(spec)
            if again is runner:
                raise
            return again.score(path, verify_direct=verify_direct)

    def _predict_agent(self, spec: ModelSpec, path) -> dict:
        self._get_model(spec)                       # load first: elapsed covers the investigation only
        t0 = time.time()
        out = self.agent_score(spec, path)
        res = self._result(spec, out["fake_probability"], time.time() - t0)
        res["verdict"] = out["verdict"] or "unparsed"     # the agent's answer; no valid answer counts as wrong
        res["logit"] = round(out["logit"], 6)
        for key in ("score_source", "status", "verdict_token", "trace", "cost"):
            res[key] = out.get(key)
        res["input_note"] = input_note(spec)
        return res

    def predict_video_clip(self, spec: ModelSpec, path) -> dict:
        """Run a native *video* detector on a decoded clip."""
        if spec.predict_kind == "video_agent":         # WeVisForensics: the official agent, one video at a time
            return self._predict_agent(spec, path)
        if spec.video_sampler == "cv2_frames":          # frame-level: P(fake) = sigmoid(mean frame logit), official
            self._get_model(spec)                       # load first: elapsed covers decoding and scoring only
            t0 = time.time()
            logits = self.video_frame_logits(spec, path)
            logit = logits.double().mean()              # the official mean (pandas, float64) of the float32 logits
            res = self._result(spec, float(torch.sigmoid(logit)), time.time() - t0)   # official: scipy expit
            res["num_frames"] = int(logits.numel())
            res["logit"] = round(float(logit), 6)
            return res
        if spec.video_prob == "mmd":                   # NSG-VD: the official MMD statistic vs its reference set
            model = self._get_model(spec)
            t0 = time.time()
            clip = self.video_clip(spec, path)
            score = model.score(clip.unsqueeze(0).to(self.device)).double()
            res = self._result(spec, float(model.fake_probability(score)), time.time() - t0)
            res["verdict"] = "fake" if bool(model.is_fake(score)) else "real"   # official: fake iff MMD > 1
            res["num_frames"] = int(clip.shape[0])
            res["logit"] = round(float(score), 6)   # the raw statistic (fake if > 1): bench_eval_video ranks by it
            res["ffmpeg"] = spec_ffmpeg(spec)
            if not res["ffmpeg"]["match"]:
                res["warning"] = (f"ffmpeg {res['ffmpeg']['version']} is not the pinned {spec.ffmpeg_version}.x; scores "
                                  "depend on the build (see the detector README).")
            return res
        model = self._get_model(spec)
        clip = self.video_clip(spec, path)
        t0 = time.time()
        p = float(self._video_fake_prob(model, spec, clip).item())
        res = self._result(spec, p, time.time() - t0)
        res["num_frames"] = int(clip.shape[0])
        return res

    @torch.no_grad()
    def video_raw_score(self, spec: ModelSpec, path) -> float:
        """A training-free video detector's raw score for one clip, oriented so that higher means more likely generated
        (for fake recall at a fixed false-positive rate, where saturated probabilities would tie)."""
        model = self._get_model(spec)
        s = float(model.score(self.video_clip(spec, path).unsqueeze(0).to(self.device)).item())
        return s if getattr(model, "higher_is_fake", True) else -s

    def _result(self, spec: ModelSpec, fake_prob: float, elapsed: float) -> dict:
        fake_prob = max(0.0, min(1.0, fake_prob))
        return {
            "model": spec.key,
            "model_name": spec.name,
            "fake_probability": round(fake_prob, 6),
            "real_probability": round(1.0 - fake_prob, 6),
            "verdict": "fake" if fake_prob >= 0.5 else "real",
            "confidence": round(max(fake_prob, 1.0 - fake_prob), 6),
            "elapsed_ms": round(elapsed * 1000, 1),
            "checkpoint_status": self._status.get(spec.key, ""),
        }


_ENGINE: Engine | None = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE
