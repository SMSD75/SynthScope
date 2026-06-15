"""SOTA image-generation backends for the live detector-comparison demo.

Each generator takes a source image + a prompt ("make another image conceptually similar to this")
and returns PNG bytes of a freshly generated image, which is then scored by the AGID detectors so a
user can see how today's detectors hold up against today's generators. API keys come from env vars;
if a key is absent the generator is reported unavailable (the UI greys it out).
"""
from __future__ import annotations
import base64, io, os, re
from dataclasses import dataclass
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
    call: Callable[[bytes, str], bytes]
    note: str = ""
    local: bool = False   # True = open-weight, runs locally (no paid API key)

    def available(self) -> bool:
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


def effective_prompt(gen_key: str, image_png: bytes, prompt: str) -> str:
    """The prompt actually sent to the generator. For text-to-image generators left on the default
    edit instruction, derive a scene description from the source image (caption) instead."""
    g = GENERATORS.get(gen_key)
    if g and not g.uses_image and prompt.strip() == DEFAULT_PROMPT:
        cap = _caption(image_png)
        return f"{cap}, photorealistic, high detail" if cap else "a photorealistic natural scene, high detail"
    return prompt


# --- Google Gemini 2.5 Flash Image ("Nano Banana") — image-conditioned ---------------------------
def _gemini(image_png: bytes, prompt: str) -> bytes:
    key = os.environ["GEMINI_API_KEY"]
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash-image:generateContent?key=" + key)
    body = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_png).decode()}},
    ]}]}
    r = requests.post(url, json=body, timeout=120); r.raise_for_status()
    for part in r.json()["candidates"][0]["content"]["parts"]:
        if "inline_data" in part or "inlineData" in part:
            data = (part.get("inline_data") or part.get("inlineData"))["data"]
            return base64.b64decode(data)
    raise RuntimeError("Gemini returned no image part")


# --- OpenAI gpt-image-1 — image edit --------------------------------------------------------------
def _openai(image_png: bytes, prompt: str) -> bytes:
    key = os.environ["OPENAI_API_KEY"]
    r = requests.post("https://api.openai.com/v1/images/edits",
                      headers={"Authorization": f"Bearer {key}"},
                      data={"model": "gpt-image-1", "prompt": prompt, "size": "1024x1024"},
                      files={"image": ("src.png", image_png, "image/png")}, timeout=180)
    r.raise_for_status()
    return base64.b64decode(r.json()["data"][0]["b64_json"])


# --- xAI Grok (grok-2-image) — text-to-image (no image input; uses the prompt) --------------------
def _grok(image_png: bytes, prompt: str) -> bytes:
    key = os.environ["XAI_API_KEY"]
    r = requests.post("https://api.x.ai/v1/images/generations",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": "grok-2-image", "prompt": prompt, "response_format": "b64_json"},
                      timeout=180)
    r.raise_for_status()
    return base64.b64decode(r.json()["data"][0]["b64_json"])


# --- Black Forest Labs FLUX.1.1 (via fal.ai) — image-to-image -------------------------------------
def _flux(image_png: bytes, prompt: str) -> bytes:
    key = os.environ["FAL_KEY"]
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


def _ideogram_local(image_png: bytes, prompt: str) -> bytes:
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
    ]
}


def list_generators() -> list[dict]:
    return [{"key": g.key, "name": g.name, "uses_image": g.uses_image,
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


def generate(gen_key: str, image_png: bytes, prompt: str = DEFAULT_PROMPT) -> bytes:
    if gen_key not in GENERATORS:
        raise KeyError(f"unknown generator '{gen_key}'")
    g = GENERATORS[gen_key]
    if not g.available():
        raise RuntimeError(f"{g.name} unavailable: set {g.env_key}")
    if not g.uses_image and _PERSON_RE.search(prompt or ""):
        raise RuntimeError(f"{g.name} blocks photorealistic people, and this image reads as a photo of a "
                           f"person (“{prompt[:70]}”). {_STEER}")
    out = g.call(image_png, prompt)
    # normalize to PNG, cap size
    img = Image.open(io.BytesIO(out)).convert("RGB")
    if _looks_blocked(img):
        raise RuntimeError(f"{g.name} returned a safety-filter placeholder — it blocked this content. {_STEER}")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    return _png_bytes(img)
