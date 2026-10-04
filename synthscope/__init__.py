"""SynthScope — a unified library for AI-generated image & video detection.

A thin, dependency-light public API over the SynthScope model registry and inference engine.
Every detector is a binary ``real`` vs ``fake`` classifier; ``predict`` returns the
probability that the input is AI-generated.

Example
-------
    >>> import synthscope
    >>> synthscope.list_models()                 # what's implemented + which weights are present
    >>> det = synthscope.load("fat_former")      # build the detector (downloads backbones on first use)
    >>> det.predict("photo.png")
    {'model': 'fat_former', 'fake_probability': 0.98, 'verdict': 'fake', ...}
    >>> det.predict("clip.mp4")            # image detectors sample frames; video detectors score the clip

Checkpoints are resolved from ``$AGID_CKPT_DIR`` (default ``~/.cache/synthscope/checkpoints``);
pass ``checkpoint=`` to point at your own ``.pth``. Training-free detectors (e.g. D3) need no
weights.
"""
import os
from pathlib import Path

# Portable, XDG-style defaults so the library runs anywhere out of the box. The cluster
# experiment scripts and the web interface still override these via the same env vars.
# These must be set *before* importing the backend, which reads them at import time.
# ``AGID_*`` env vars keep their names (cluster scripts set them); an existing ~/.cache/agid is reused.
_LEGACY_HOME = Path.home() / ".cache" / "agid"
_DEFAULT_HOME = _LEGACY_HOME if _LEGACY_HOME.exists() else Path.home() / ".cache" / "synthscope"
_AGID_HOME = Path(os.environ.get("SYNTHSCOPE_HOME") or os.environ.get("AGID_HOME") or _DEFAULT_HOME).expanduser()
os.environ.setdefault("AGID_CKPT_DIR", str(_AGID_HOME / "checkpoints"))
os.environ.setdefault("AGID_CACHE_DIR", str(_AGID_HOME / "cache"))

from .api import Detector, available_models, list_models, load  # noqa: E402

__version__ = "0.1.0"
__all__ = ["Detector", "load", "list_models", "available_models", "__version__"]
