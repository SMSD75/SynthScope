"""SOTA image-generation backends for the live detector-comparison demo.

Each generator takes a source image + a prompt ("make another image conceptually similar to this")
and returns PNG bytes of a freshly generated image, which is then scored by the AGID detectors so a
user can see how today's detectors hold up against today's generators. API keys come from env vars;
if a key is absent the generator is reported unavailable (the UI greys it out).
"""
from __future__ import annotations
import base64, io, os
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


# --- Google Gemini 2.5 Flash Image ("Nano Banana") — image-conditioned ---------------------------
def _gemini(image_png: bytes, prompt: str) -> bytes:
    key = os.environ["GEMINI_API_KEY"]
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash-image-preview:generateContent?key=" + key)
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
                  "xAI text-to-image; generates from the prompt (no image input)."),
        Generator("flux", "FLUX.1 (img2img, via fal.ai)", "FAL_KEY", True, _flux,
                  "Black Forest Labs FLUX image-to-image."),
        Generator("ideogram", "Ideogram 4.0 (open-weight, local)", "HF_TOKEN", False, _ideogram_local,
                  "Open-weight 9.3B DiT, nf4 on a 24GB GPU — runs locally via diffusers, NO paid API "
                  "key. HF_TOKEN only to download the license-gated weights once.", local=True),
    ]
}


def list_generators() -> list[dict]:
    return [{"key": g.key, "name": g.name, "uses_image": g.uses_image,
             "available": g.available(), "env_key": g.env_key, "note": g.note}
            for g in GENERATORS.values()]


def generate(gen_key: str, image_png: bytes, prompt: str = DEFAULT_PROMPT) -> bytes:
    if gen_key not in GENERATORS:
        raise KeyError(f"unknown generator '{gen_key}'")
    g = GENERATORS[gen_key]
    if not g.available():
        raise RuntimeError(f"{g.name} unavailable: set {g.env_key}")
    out = g.call(image_png, prompt)
    # normalize to PNG, cap size
    img = Image.open(io.BytesIO(out)).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    return _png_bytes(img)
