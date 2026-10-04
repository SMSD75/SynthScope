"""Isolated runner for heavy OPEN-WEIGHT image generators (FLUX.2 [klein] 4B/9B, FLUX.1 Kontext [dev], Qwen-Image-2.1).

``generators.py`` launches this file as a SUBPROCESS, one image per call, so that
  * these models can use a newer ``diffusers`` than the shared SynthScope env (set ``$SYNTHSCOPE_GEN_PYTHONPATH`` to a
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
import json
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """Standard path: whole pipeline in host RAM, modules streamed to the GPU on demand (FLUX.2 [klein]).
    Without a GPU the pipeline simply runs on the CPU in bfloat16 (slow: minutes per image)."""
    pipe = getattr(diffusers, cls_name).from_pretrained(repo, torch_dtype=torch.bfloat16, token=token)
    if DEVICE.type == "cuda":
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


def load_pipeline_group_offload(repo: str, cls_name: str, token):
    """FLUX.1 Kontext [dev]: the 12B transformer (24 GB in bf16) cannot sit whole on a 24 GB GPU, which is what
    ``enable_model_cpu_offload`` needs. Keep the text encoders + VAE (~10 GB) on the GPU and stream the transformer
    through it two blocks at a time."""
    pipe = getattr(diffusers, cls_name).from_pretrained(repo, torch_dtype=torch.bfloat16, token=token)
    for name in ("text_encoder", "text_encoder_2", "vae"):
        getattr(pipe, name).to(DEVICE)
    pipe.transformer.enable_group_offload(onload_device=DEVICE, offload_device=torch.device("cpu"),
                                          offload_type="block_level", num_blocks_per_group=2)
    type(pipe)._execution_device = property(lambda self: DEVICE)   # the transformer reports "cpu" between blocks
    mem("kontext loaded")
    return pipe


def run_batch_qwen(repo: str, cls_name: str, token, items: list, kw: dict, res: int) -> None:
    """Text-to-image for MANY prompts with one model load (Qwen-Image-2.1 on a 24 GB GPU): encode every prompt first
    with the text encoder on the GPU, drop it, load the transformer once, then denoise each prompt in turn
    (the transformer is parked on the CPU during each VAE decode, as in ``load_pipeline_staged``)."""
    pipeline_cls = getattr(diffusers, cls_name)
    text_encoder = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        repo, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda", token=token)
    pipe = pipeline_cls.from_pretrained(repo, text_encoder=text_encoder, transformer=None,
                                        torch_dtype=torch.bfloat16, token=token)
    pipe.vae.to(DEVICE)
    pipeline_cls._execution_device = property(lambda self: DEVICE)
    embeds = []
    for it in items:
        with torch.no_grad():       # without this the saved embeddings keep the autograd graph, and with it the 16 GB encoder
            emb, mask, _ = pipe.encode_prompt(prompt=it["prompt"], device=DEVICE)
        embeds.append((emb.detach().cpu(), None if mask is None else mask.detach().cpu()))
    pipe.text_encoder = None
    del text_encoder
    gc.collect(); torch.cuda.empty_cache()
    pipe.register_modules(transformer=diffusers.QwenImage21Transformer2DModel.from_pretrained(
        repo, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda", token=token))
    vae_decode = pipe.vae.decode

    def decode_after_freeing_transformer(*args, **kwargs):
        pipe.transformer.to("cpu"); gc.collect(); torch.cuda.empty_cache()
        return vae_decode(*args, **kwargs)

    pipe.vae.decode = decode_after_freeing_transformer
    for it, (emb, mask) in zip(items, embeds):
        pipe.transformer.to(DEVICE)
        out = pipe(prompt_embeds=emb.to(DEVICE), prompt_embeds_mask=None if mask is None else mask.to(DEVICE),
                   generator=torch.Generator("cpu").manual_seed(it.get("seed", 0)), output_resolution=res, **kw).images[0]
        out.convert("RGB").save(it["out"], format="PNG")
        print(f"saved {it['out']} {out.size}", flush=True)


def run_batch(model: str, items: list, res: int) -> None:
    """Text-only generation for a list of {prompt, out, seed, size?} items with one model load. Existing outputs are
    skipped, so an interrupted batch can simply be re-run."""
    repo, cls_name, kw = MODELS[model]
    token = os.environ.get("HF_TOKEN")
    todo = [it for it in items if not os.path.exists(it["out"])]
    print(f"{len(items) - len(todo)} already done, {len(todo)} to generate with {model}", flush=True)
    if not todo:
        return
    if model == "qwen_image_2_1":
        run_batch_qwen(repo, cls_name, token, todo, kw, res)
        return
    pipe = load_pipeline_cpu_offload(repo, cls_name, token)
    for it in todo:
        w, h = (int(v) for v in (it.get("size") or "1024x1024").split("x"))
        out = pipe(prompt=it["prompt"], generator=torch.Generator("cpu").manual_seed(it.get("seed", 0)),
                   height=h, width=w, **kw).images[0]
        out.convert("RGB").save(it["out"], format="PNG")
        print(f"saved {it['out']} {out.size}", flush=True)


LOADERS = {"qwen_image_2_1": load_pipeline_staged, "flux1_kontext_dev": load_pipeline_group_offload}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--image", help="source photo (not needed with --text-only)")
    ap.add_argument("--prompt")
    ap.add_argument("--out")
    ap.add_argument("--batch", help="JSON list of {prompt, out, seed, size}: text-to-image for all of them with one model load")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-only", action="store_true", help="create the image from the prompt alone (no source photo)")
    ap.add_argument("--size", help="WxH of the output in --text-only mode (default 1024x1024)")
    ap.add_argument("--res", type=int, default=1024, help="output resolution (Qwen-Image-2.1 output_resolution)")
    args = ap.parse_args()
    if args.batch:
        run_batch(args.model, json.load(open(args.batch)), args.res)
        return 0
    if not args.prompt or not args.out:
        ap.error("--prompt and --out are required (or use --batch)")
    if not args.text_only and not args.image:
        ap.error("--image is required unless --text-only is given")

    repo, cls_name, kw = MODELS[args.model]
    if not hasattr(diffusers, cls_name):
        print(f"diffusers {diffusers.__version__} has no {cls_name}; point $SYNTHSCOPE_GEN_PYTHONPATH at a newer diffusers",
              file=sys.stderr)
        return 3
    pipe = LOADERS.get(args.model, load_pipeline_cpu_offload)(repo, cls_name, os.environ.get("HF_TOKEN"))

    src = None if args.text_only else fit(Image.open(args.image).convert("RGB"))
    call = dict(kw)
    if args.model.startswith("flux"):
        if src is not None:
            call.update(height=src.height, width=src.width)
        else:
            w, h = (int(v) for v in (args.size or "1024x1024").split("x"))
            call.update(height=h, width=w)
    if args.model.startswith("qwen"):
        call["output_resolution"] = args.res
    cond = {} if src is None else {"image": src}
    out = pipe(prompt=args.prompt, generator=torch.Generator("cpu").manual_seed(args.seed), **cond, **call).images[0]
    out.convert("RGB").save(args.out, format="PNG")
    print(f"saved {args.out} {out.size} via {cls_name} ({repo})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
