"""Runtime configuration and path wiring for the SynthScope web interface.

The interface imports the AGID repo (models, dataset decoders, Hydra config composition)
directly, so we add the repo root to ``sys.path`` and pin a few cache locations that make
sense on the GPU node where the models/checkpoints actually live (cn023 -> /hddstore).
Everything is overridable via environment variables.
"""
import os
import sys
from pathlib import Path

# interface/backend/config.py -> interface/ -> repo root
BACKEND_DIR = Path(__file__).resolve().parent
INTERFACE_DIR = BACKEND_DIR.parent
REPO_ROOT = INTERFACE_DIR.parent

# Make the AGID package importable (models.*, dataset, config_utils, ...).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _strip_env_value(v: str) -> str:
    """Parse a ``.env`` value: honor quotes, drop an inline ``# comment``.

    Tolerates the common mistake of filling a key in next to the template's trailing comment, e.g.
    ``OPENAI_API_KEY=sk-... # GPT Image -> platform.openai.com/api-keys`` — without this, the comment
    becomes part of the key and the API rejects it. A ``#`` only starts a comment when it is at the
    start or preceded by whitespace, so keys/values that legitimately contain ``#`` are preserved.
    """
    v = v.strip()
    if v[:1] in ('"', "'"):                       # quoted: take quoted span, ignore trailing comment
        q = v[0]
        end = v.find(q, 1)
        return v[1:end] if end != -1 else v[1:]
    for i, ch in enumerate(v):                    # unquoted: cut at the first whitespace-led '#'
        if ch == "#" and (i == 0 or v[i - 1] in " \t"):
            v = v[:i]
            break
    return v.strip()


def _load_dotenv() -> None:
    """Load secrets (SOTA-generator API keys, etc.) from a gitignored ``.env`` at the repo root.

    No dependency required: a minimal KEY=VALUE parser. Existing real env vars take precedence
    (``setdefault``), so a shell export or a secrets manager always overrides the file. The ``.env``
    file is gitignored — never commit keys; commit only ``.env.example``.
    """
    for env_path in (REPO_ROOT / ".env", INTERFACE_DIR / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            val = _strip_env_value(v)
            if val:                               # skip empty values (unfilled template keys)
                os.environ.setdefault(k.strip(), val)


_load_dotenv()

# Central checkpoint directory. Defaults to the portable ``~/.cache/agid`` home (matching the
# ``agid`` package + README); cluster jobs override with AGID_CKPT_DIR for node-local scratch.
_AGID_HOME = Path(os.environ.get("AGID_HOME", Path.home() / ".cache" / "agid")).expanduser()
CHECKPOINTS_DIR = Path(
    os.environ.get("AGID_CKPT_DIR", str(_AGID_HOME / "checkpoints"))
).expanduser()

# Where to cache downloaded pretrained backbones (CLIP/DINOv2/SigLIP/ConvNeXt/...). On cn023
# the /nvmestore caches referenced by the experiment scripts are not present, so default to a
# node-local cache. Only set if the caller has not already configured these.
_DEFAULT_CACHE = Path(os.environ.get("AGID_CACHE_DIR", str(_AGID_HOME / "cache"))).expanduser()
os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_DEFAULT_CACHE / "torch"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Upload / decoding limits.
MAX_UPLOAD_BYTES = int(os.environ.get("AGID_MAX_UPLOAD_MB", "200")) * 1024 * 1024
# Number of frames sampled from a video when running an *image* detector frame-by-frame.
VIDEO_IMAGE_FRAMES = int(os.environ.get("AGID_VIDEO_IMAGE_FRAMES", "16"))

# How many models to keep resident at once (LRU). Big backbones (ConvNeXt-XXL, SigLIP+VAE)
# are several GB each, so we evict rather than risk OOM on a shared GPU.
MAX_RESIDENT_MODELS = int(os.environ.get("AGID_MAX_RESIDENT_MODELS", "2"))


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
