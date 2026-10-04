"""Isolated runner for heavy OPEN-WEIGHT image generators (FLUX.2 [klein] 4B/9B, FLUX.1 Kontext [dev], Qwen-Image-2.1).

``generators.py`` launches this file as a SUBPROCESS, one image per call, so that
  * these models can use a newer ``diffusers`` than the shared AGID env (set ``$AGID_GEN_PYTHONPATH`` to a
    directory holding diffusers main + huggingface_hub 1.x; see ``interface/README.md``), and
  * every byte of GPU/CPU memory is released after the image, so the web UI's detectors keep their 24 GB.

    python interface/backend/local_gen.py --model qwen_image_2_1 --image in.png --prompt "..." --out out.png

Both models are image-CONDITIONED (they edit the source photo into a new image), run in bf16 on a 24 GB GPU
(FLUX models: ``enable_model_cpu_offload``; Qwen-Image-2.1: staged load, see ``load_pipeline_staged``), and need
no paid API key. Licenses: FLUX.2 [klein] 4B is
Apache-2.0; Qwen-Image-2.1 is under the Qwen Research License (research use).
"""
import argparse
import gc
import os
import sys

import torch
import diffusers
import transformers
from PIL import Image

MAX_AREA = 1024 * 1024
MULTIPLE = 16

# key -> (HF repo, diffusers pipeline class, extra call kwargs)
MODELS = {
    "flux2_klein_4b": ("black-forest-labs/FLUX.2-klein-4B", "Flux2KleinPipeline",
                       {"num_inference_steps": 4, "guidance_scale": 1.0}),
    "qwen_image_2_1": ("Qwen/Qwen-Image-2.1", "QwenImage21Pipeline",
                       {"num_inference_steps": 40}),
    # license-gated (accept the terms on the Hub with the account behind $HF_TOKEN); non-commercial licenses
    "flux2_klein_9b": ("black-forest-labs/FLUX.2-klein-9B", "Flux2KleinPipeline",
                       {"num_inference_steps": 4, "guidance_scale": 1.0}),
    "flux1_kontext_dev": ("black-forest-labs/FLUX.1-Kontext-dev", "FluxKontextPipeline",
                          {"num_inference_steps": 28, "guidance_scale": 2.5}),
}
DEVICE = torch.device("cuda")
DEBUG = bool(os.environ.get("LOCAL_GEN_DEBUG"))


def mem(tag: str) -> None:
    """GPU memory trace (set LOCAL_GEN_DEBUG=1)."""
    if DEBUG:
        print(f"[mem] {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
              f"max {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", file=sys.stderr, flush=True)


def fit(img: Image.Image) -> Image.Image:
    """Scale to <= MAX_AREA keeping aspect, with both sides a multiple of MULTIPLE."""
    w, h = img.size
    s = min(1.0, (MAX_AREA / (w * h)) ** 0.5)
    nw, nh = max(MULTIPLE, int(w * s) // MULTIPLE * MULTIPLE), max(MULTIPLE, int(h * s) // MULTIPLE * MULTIPLE)
    return img.resize((nw, nh), Image.LANCZOS)


def load_pipeline_cpu_offload(repo: str, cls_name: str, token):
    """Standard path: whole pipeline in host RAM, modules streamed to the GPU on demand (FLUX.2 [klein])."""
    pipe = getattr(diffusers, cls_name).from_pretrained(repo, torch_dtype=torch.bfloat16, token=token)
    pipe.enable_model_cpu_offload()
    return pipe


def load_pipeline_staged(repo: str, cls_name: str, token):
    """Low-host-RAM path for Qwen-Image-2.1 (33 GB repo, OOMs a 28 GB host with ``enable_model_cpu_offload``).

    The Qwen3-VL text encoder is loaded straight onto the GPU, used once to encode prompt + source image, then
    dropped; only then is the 7B transformer loaded straight onto the GPU. Nothing large ever sits in host RAM.
    The pipeline is created with ``transformer=None`` and the transformer is attached inside ``encode_prompt``
    (which ``__call__`` invokes before it first touches ``self.transformer``).
    """
    pipeline_cls = getattr(diffusers, cls_name)
    text_encoder = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        repo, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda", token=token)
    pipe = pipeline_cls.from_pretrained(repo, text_encoder=text_encoder, transformer=None,
                                        torch_dtype=torch.bfloat16, token=token)
    pipe.vae.to(DEVICE)
    # The 1 MP VAE decode does not fit next to the 14 GB transformer on a 24 GB GPU. VAE tiling is NOT a fix: tiling the
    # ENCODE of the conditioning image corrupts it (output degenerates into texture), and tiled decode leaves visible
    # seams that a detector could key on. Instead the transformer (no longer needed once denoising is done) is moved
    # to host RAM right before the single-pass decode.
    vae_decode = pipe.vae.decode

    def decode_after_freeing_transformer(*args, **kwargs):
        pipe.transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        mem("transformer offloaded before decode")
        return vae_decode(*args, **kwargs)

    pipe.vae.decode = decode_after_freeing_transformer
    mem("text encoder + vae loaded")
    pipeline_cls._execution_device = property(lambda self: DEVICE)   # no offload hooks in this mode
    held = {"text_encoder": text_encoder}      # the only outside reference, so it can be dropped after encoding
    del text_encoder
    encode_prompt = pipe.encode_prompt

    def encode_then_load_transformer(*args, **kwargs):
        out = encode_prompt(*args, **kwargs)
        pipe.text_encoder = None
        held.clear()
        gc.collect()
        torch.cuda.empty_cache()
        mem("text encoder dropped")
        pipe.register_modules(transformer=diffusers.QwenImage21Transformer2DModel.from_pretrained(
            repo, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda", token=token))
        mem("transformer loaded")
        return out

    pipe.encode_prompt = encode_then_load_transformer
    return pipe


LOADERS = {"qwen_image_2_1": load_pipeline_staged}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-only", action="store_true", help="debug: ignore the source image (pure text-to-image)")
    ap.add_argument("--res", type=int, default=1024, help="output resolution (Qwen-Image-2.1 output_resolution)")
    args = ap.parse_args()

    repo, cls_name, kw = MODELS[args.model]
    if not hasattr(diffusers, cls_name):
        print(f"diffusers {diffusers.__version__} has no {cls_name}; point $AGID_GEN_PYTHONPATH at a newer diffusers",
              file=sys.stderr)
        return 3
    pipe = LOADERS.get(args.model, load_pipeline_cpu_offload)(repo, cls_name, os.environ.get("HF_TOKEN"))

    src = fit(Image.open(args.image).convert("RGB"))
    call = dict(kw)
    if args.model.startswith("flux"):
        call.update(height=src.height, width=src.width)
    if args.model.startswith("qwen"):
        call["output_resolution"] = args.res
    cond = {} if args.text_only else {"image": src}
    out = pipe(prompt=args.prompt, generator=torch.Generator("cpu").manual_seed(args.seed), **cond, **call).images[0]
    out.convert("RGB").save(args.out, format="PNG")
    print(f"saved {args.out} {out.size} via {cls_name} ({repo})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
