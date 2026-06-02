"""AGID — a unified library for AI-generated image & video detection.

A thin, dependency-light public API over the AGID model registry and inference engine.
Every detector is a binary ``real`` vs ``fake`` classifier; ``predict`` returns the
probability that the input is AI-generated.

Example
-------
    >>> import agid
    >>> agid.list_models()                 # what's implemented + which weights are present
    >>> det = agid.load("fat_former")      # build the detector (downloads backbones on first use)
    >>> det.predict("photo.png")
    {'model': 'fat_former', 'fake_probability': 0.98, 'verdict': 'fake', ...}
    >>> det.predict("clip.mp4")            # image detectors sample frames; video detectors score the clip

Checkpoints are resolved from ``$AGID_CKPT_DIR`` (default ``~/.cache/agid/checkpoints``);
pass ``checkpoint=`` to point at your own ``.pth``. Training-free detectors (e.g. D3) need no
weights.
"""
import os
from pathlib import Path

# Portable, XDG-style defaults so the library runs anywhere out of the box. The cluster
# experiment scripts and the web interface still override these via the same env vars.
# These must be set *before* importing the backend, which reads them at import time.
_AGID_HOME = Path(os.environ.get("AGID_HOME", Path.home() / ".cache" / "agid")).expanduser()
os.environ.setdefault("AGID_CKPT_DIR", str(_AGID_HOME / "checkpoints"))
os.environ.setdefault("AGID_CACHE_DIR", str(_AGID_HOME / "cache"))

from .api import Detector, available_models, list_models, load  # noqa: E402

__version__ = "0.1.0"
__all__ = ["Detector", "load", "list_models", "available_models", "__version__"]
