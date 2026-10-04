"""SOTA image-generation backends for the live detector-comparison demo.

Each generator takes a source image + a prompt and returns PNG bytes of a freshly generated image. Two modes:
  * "recreate" (default): a captioner describes the source photo and the generator creates a NEW image from that
    description alone (image_png=None), so every pixel is synthesized. This is how detection benchmarks build fakes.
  * "edit": the generator edits the source photo itself (image-conditioned). The result keeps the original's layout.
The result is then scored by the SynthScope detectors, which is then scored by the SynthScope detectors so a
user can see how today's detectors hold up against today's generators. API keys come from env vars;
if a key is absent the generator is reported unavailable (the UI greys it out).
"""
from __future__ import annotations
import base64, io, os, re, subprocess, sys, tempfile, threading
import _synthscope_env  # noqa: F401  (maps legacy AGID_* variables to SYNTHSCOPE_*)
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests
from PIL import Image

DEFAULT_PROMPT = "make another image conceptually similar to this"


@dataclass
class Generator:
    key: str
    name: str
    env_key: str          # API-key env var (hosted generators); HF token for the gated download (local)
    uses_image: bool      # True = image-conditioned (edit); False = text-to-image only
    call: Callable[..., bytes]      # call(image_png | None, prompt, size=None); image_png=None = text-only
    note: str = ""
    local: bool = False   # True = open-weight, runs locally (no paid API key)
    avail: Optional[Callable[[], bool]] = None   # custom availability check (overrides the defaults)
    heavy: bool = False   # True = needs (most of) the GPU: the web engine releases its detectors first
    edit_only: bool = False   # True = an editing model that cannot create an image from text alone

    @property
    def modes(self) -> list[str]:
        if not self.uses_image:
            return ["recreate"]
        return ["edit"] if self.edit_only else ["recreate", "edit"]

    def available(self) -> bool:
        if self.avail is not None:
            return self.avail()
        if self.local:                       # local = available if the inference pipeline is importable
            try:
                from diffusers import Ideogram4Pipeline  # noqa: F401
                return True
            except Exception:
                return False
        return bool(os.environ.get(self.env_key))


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG"); return buf.getvalue()


# --- Image captioning (BLIP) for text-to-image generators ----------------------------------------
# Text-to-image generators (Grok, Ideogram) can't see the source image, so the edit-style default
# prompt ("make another image conceptually similar to this") is meaningless to them and tends to
# produce garbage or trip the model's safety filter. We caption the source image and feed THAT, so
# the result is genuinely "conceptually similar" and the prompt is a normal scene description.
_BLIP = None


def _caption(image_png: bytes) -> Optional[str]:
    """Best-effort caption of the source image. Returns None on any failure (callers fall back)."""
    global _BLIP
    try:
        import torch
        if _BLIP is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            name = "Salesforce/blip-image-captioning-base"
            _BLIP = (BlipProcessor.from_pretrained(name),
                     BlipForConditionalGeneration.from_pretrained(name).eval())
        proc, model = _BLIP
        img = Image.open(io.BytesIO(image_png)).convert("RGB")
        with torch.no_grad():
            out = model.generate(**proc(img, return_tensors="pt"), max_new_tokens=30)
        cap = proc.decode(out[0], skip_special_tokens=True).strip()
        return cap or None
    except Exception:
        return None


STYLE_SUFFIX = ", photorealistic photograph, natural lighting, high detail"


def describe_prompt(image_png: bytes) -> dict:
    """Caption the source photo and turn it into a text-to-image prompt (shown to the user, who can edit it)."""
    cap = _caption(image_png)
    return {"caption": cap or "", "prompt": (cap + STYLE_SUFFIX) if cap else "a photorealistic natural scene, high detail"}


def effective_prompt(gen_key: str, image_png: bytes, prompt: str) -> str:
    """The prompt actually sent to the generator. For text-to-image generators left on the default
    edit instruction, derive a scene description from the source image (caption) instead."""
    g = GENERATORS.get(gen_key)
    if g and not g.uses_image and prompt.strip() == DEFAULT_PROMPT:
        cap = _caption(image_png)
        return f"{cap}, photorealistic, high detail" if cap else "a photorealistic natural scene, high detail"
    return prompt


# --- Google Gemini 2.5 Flash Image ("Nano Banana") — image-conditioned ---------------------------
def _gemini(image_png: Optional[bytes], prompt: str, size=None) -> bytes:
    key = os.environ["GEMINI_API_KEY"]
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash-image:generateContent?key=" + key)
    parts = [{"text": prompt}]
    if image_png is not None:
        parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_png).decode()}})
    body = {"contents": [{"parts": parts}]}
    r = requests.post(url, json=body, timeout=120); r.raise_for_status()
    for part in r.json()["candidates"][0]["content"]["parts"]:
        if "inline_data" in part or "inlineData" in part:
            data = (part.get("inline_data") or part.get("inlineData"))["data"]
            return base64.b64decode(data)
    raise RuntimeError("Gemini returned no image part")


# --- OpenAI gpt-image-1 — image edit --------------------------------------------------------------
def _openai(image_png: Optional[bytes], prompt: str, size=None) -> bytes:
    key = os.environ["OPENAI_API_KEY"]
    if image_png is None:                                   # text-only: plain image generation
        r = requests.post("https://api.openai.com/v1/images/generations",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": "gpt-image-1", "prompt": prompt, "size": "1024x1024"}, timeout=180)
        r.raise_for_status()
        return base64.b64decode(r.json()["data"][0]["b64_json"])
    r = requests.post("https://api.openai.com/v1/images/edits",
                      headers={"Authorization": f"Bearer {key}"},
                      data={"model": "gpt-image-1", "prompt": prompt, "size": "1024x1024"},
                      files={"image": ("src.png", image_png, "image/png")}, timeout=180)
    r.raise_for_status()
    return base64.b64decode(r.json()["data"][0]["b64_json"])


# --- xAI Grok (grok-2-image) — text-to-image (no image input; uses the prompt) --------------------
def _grok(image_png: Optional[bytes], prompt: str, size=None) -> bytes:
    key = os.environ["XAI_API_KEY"]
    r = requests.post("https://api.x.ai/v1/images/generations",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": "grok-2-image", "prompt": prompt, "response_format": "b64_json"},
                      timeout=180)
    r.raise_for_status()
    return base64.b64decode(r.json()["data"][0]["b64_json"])


# --- Black Forest Labs FLUX.1.1 (via fal.ai) — image-to-image -------------------------------------
def _flux(image_png: Optional[bytes], prompt: str, size=None) -> bytes:
    key = os.environ["FAL_KEY"]
    if image_png is None:                                   # text-only: FLUX.1 [dev] text-to-image
        r = requests.post("https://fal.run/fal-ai/flux/dev",
                          headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
                          json={"prompt": prompt, "image_size": "square_hd"}, timeout=180)
        r.raise_for_status()
        return requests.get(r.json()["images"][0]["url"], timeout=120).content
    # fal.ai accepts a data-URI image_url for image-to-image
    data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode()
    r = requests.post("https://fal.run/fal-ai/flux/dev/image-to-image",
                      headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
                      json={"prompt": prompt, "image_url": data_uri, "strength": 0.85}, timeout=180)
    r.raise_for_status()
    img_url = r.json()["images"][0]["url"]
    return requests.get(img_url, timeout=120).content


# --- Ideogram 4.0 — OPEN-WEIGHT, runs LOCALLY (no paid API key) -----------------------------------
# Ideogram 4.0 (June 2026) is open-weight (9.3B DiT; Apache-2.0 inference code). The nf4 checkpoint
# (`ideogram-ai/ideogram-4-nf4-diffusers`) fits on a single 24 GB GPU. We load it via diffusers and
# run inference locally — no Ideogram API key. A free HF token (HF_TOKEN) is only needed ONCE to
# download the license-gated weights; after that it runs offline. Text-to-image (flow-matching DiT),
# so it generates from the prompt rather than conditioning on the source image.
_IDEOGRAM_PIPE = None


def _ideogram_local(image_png: Optional[bytes], prompt: str, size=None) -> bytes:
    global _IDEOGRAM_PIPE
    if _IDEOGRAM_PIPE is None:
        import torch
        from diffusers import Ideogram4Pipeline
        pipe = Ideogram4Pipeline.from_pretrained(
            "ideogram-ai/ideogram-4-nf4-diffusers", torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"))
        _IDEOGRAM_PIPE = pipe.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
    out = _IDEOGRAM_PIPE(prompt=prompt, height=1024, width=1024).images[0]
    return _png_bytes(out)


# --- Open-weight image-conditioned generators (FLUX.2 [klein], Qwen-Image-2.1), run in a SUBPROCESS ------
# These are 20-60 GB repos that need a newer diffusers than the shared SynthScope env and would not fit on the
# web UI's 24 GB GPU next to the detectors, so each image is produced by interface/backend/local_gen.py in a
# separate process (model CPU-offload, memory fully released afterwards). Configure with:
#   SYNTHSCOPE_GEN_PYTHONPATH  dir holding diffusers main + huggingface_hub 1.x (prepended to the child PYTHONPATH)
#   SYNTHSCOPE_GEN_HF_HOME     node-local HF cache for the child (keeps tens of GB off the shared /ivi quota)
_LOCAL_GEN = Path(__file__).with_name("local_gen.py")
_LOCAL_LOCK = threading.Lock()      # one heavy local generation at a time


def _local_gen_available() -> bool:
    extra = os.environ.get("SYNTHSCOPE_GEN_PYTHONPATH", "")
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    return bool(extra) and Path(extra).is_dir() and has_gpu


def _local_size(size) -> Optional[str]:
    """'WxH' for the child: the source's aspect ratio at about one megapixel, sides in multiples of 16."""
    if not size:
        return None
    w, h = size
    k = (1024 * 1024 / (w * h)) ** 0.5
    return f"{max(256, int(w * k) // 16 * 16)}x{max(256, int(h * k) // 16 * 16)}"


def _local_subprocess(model_key: str) -> Callable[..., bytes]:
    def call(image_png: Optional[bytes], prompt: str, size=None) -> bytes:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (os.environ["SYNTHSCOPE_GEN_PYTHONPATH"], env.get("PYTHONPATH", "")) if p)
        if os.environ.get("SYNTHSCOPE_GEN_HF_HOME"):
            env["HF_HOME"] = os.environ["SYNTHSCOPE_GEN_HF_HOME"]
        with _LOCAL_LOCK, tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.png", Path(tmp) / "out.png"
            cmd = [sys.executable, str(_LOCAL_GEN), "--model", model_key, "--prompt", prompt, "--out", str(dst)]
            if image_png is None:
                cmd.append("--text-only")
                if _local_size(size):
                    cmd += ["--size", _local_size(size)]
            else:
                src.write_bytes(image_png)
                cmd += ["--image", str(src)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0 or not dst.exists():
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
                raise RuntimeError(f"{model_key} failed (exit {r.returncode}): " + " | ".join(tail))
            return dst.read_bytes()
    return call


GENERATORS = {
    g.key: g for g in [
        Generator("nanobanana", "Nano Banana (Gemini 2.5 Flash Image)", "GEMINI_API_KEY", True, _gemini,
                  "Google's image-conditioned editor; respects the source image."),
        Generator("gpt_image", "GPT Image (gpt-image-1)", "OPENAI_API_KEY", True, _openai,
                  "OpenAI image edit from the source."),
        Generator("grok", "Grok (grok-2-image)", "XAI_API_KEY", False, _grok,
                  "xAI text-to-image; can't see the image, so it generates from an auto-caption of the source."),
        Generator("flux", "FLUX.1 (img2img, via fal.ai)", "FAL_KEY", True, _flux,
                  "Black Forest Labs FLUX image-to-image."),
        Generator("ideogram", "Ideogram 4.0 (open-weight, local)", "HF_TOKEN", False, _ideogram_local,
                  "Open-weight 9.3B DiT, nf4 on a 24GB GPU — runs locally via diffusers, NO paid API "
                  "key. HF_TOKEN only to download the license-gated weights once. Text-to-image: "
                  "generates from an auto-caption of the source (~4 min/image).", local=True),
        Generator("flux2_klein", "FLUX.2 [klein] 4B (open-weight, local)", "SYNTHSCOPE_GEN_PYTHONPATH", True,
                  _local_subprocess("flux2_klein_4b"),
                  "Black Forest Labs' distilled FLUX.2 [klein] 4B (Apache-2.0), edits the source photo into a "
                  "new image locally on the GPU (4 steps); no API key.", local=True, avail=_local_gen_available,
                  heavy=True),
        Generator("flux2_klein_9b", "FLUX.2 [klein] 9B (open-weight, local)", "SYNTHSCOPE_GEN_PYTHONPATH", True,
                  _local_subprocess("flux2_klein_9b"),
                  "Black Forest Labs' distilled FLUX.2 [klein] 9B (FLUX non-commercial license; gated - accept the "
                  "terms on the Hub), edits the source photo locally on the GPU (4 steps); no API key.", local=True,
                  avail=_local_gen_available, heavy=True),
        Generator("flux1_kontext", "FLUX.1 Kontext [dev] (open-weight, local)", "SYNTHSCOPE_GEN_PYTHONPATH", True,
                  _local_subprocess("flux1_kontext_dev"),
                  "Black Forest Labs' 12B in-context image editor (FLUX non-commercial license; gated - accept the "
                  "terms on the Hub), edits the source photo locally on the GPU (28 steps); no API key. "
                  "Edit-only: it cannot create an image from text.", local=True,
                  avail=_local_gen_available, heavy=True, edit_only=True),
        Generator("qwen_image_2_1", "Qwen-Image-2.1 (open-weight, local)", "SYNTHSCOPE_GEN_PYTHONPATH", True,
                  _local_subprocess("qwen_image_2_1"),
                  "Alibaba Qwen-Image-2.1 (7B unified generation + editing; Qwen Research License - research use "
                  "only), edits the source photo locally on the GPU; no API key.", local=True,
                  avail=_local_gen_available, heavy=True),
    ]
}


def list_generators() -> list[dict]:
    return [{"key": g.key, "name": g.name, "uses_image": g.uses_image, "modes": g.modes,
             "available": g.available(), "env_key": g.env_key, "note": g.note}
            for g in GENERATORS.values()]


# Text-to-image generators with a baked-in content policy (Ideogram) refuse photorealistic PEOPLE and
# return a flat gray "Image blocked by safety filter" placeholder. We pre-empt the obvious case (person
# in the caption) before wasting a full ~4-min generation, and detect the placeholder as a backstop.
_PERSON_RE = re.compile(
    r"\b(person|people|man|men|woman|women|boy|girl|child|children|baby|lady|human|portrait|selfie|crowd|couple)\b",
    re.I)
_STEER = "Pick a non-person image, or use an image-conditioned generator (Nano Banana / FLUX)."


def _looks_blocked(img: Image.Image) -> bool:
    """A safety placeholder is near-uniform (text on a solid background): pixel std ~12 vs ~50 for a real image."""
    try:
        import numpy as np
        return float(np.asarray(img.convert("RGB"), dtype="float32").std()) < 22.0
    except Exception:
        return False


def generate(gen_key: str, image_png: bytes, prompt: str = DEFAULT_PROMPT, mode: str = "recreate") -> bytes:
    """mode "recreate": generate a new image from the text prompt alone; "edit": edit ``image_png`` itself."""
    if gen_key not in GENERATORS:
        raise KeyError(f"unknown generator '{gen_key}'")
    g = GENERATORS[gen_key]
    if not g.available():
        raise RuntimeError(f"{g.name} unavailable: set {g.env_key}")
    if mode not in ("recreate", "edit"):
        raise RuntimeError(f"unknown mode '{mode}'")
    if mode not in g.modes:
        raise RuntimeError(f"{g.name} does not support the '{mode}' mode (it supports: {', '.join(g.modes)}).")
    if not g.uses_image and _PERSON_RE.search(prompt or ""):
        raise RuntimeError(f"{g.name} blocks photorealistic people, and this image reads as a photo of a "
                           f"person (“{prompt[:70]}”). {_STEER}")
    size = Image.open(io.BytesIO(image_png)).size
    out = g.call(image_png if mode == "edit" else None, prompt, size)
    # normalize to PNG, cap size
    img = Image.open(io.BytesIO(out)).convert("RGB")
    if _looks_blocked(img):
        raise RuntimeError(f"{g.name} returned a safety-filter placeholder — it blocked this content. {_STEER}")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    return _png_bytes(img)
