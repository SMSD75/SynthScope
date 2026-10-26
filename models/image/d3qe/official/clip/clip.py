# Vendored from github.com/Zhangyr2022/D3QE (MIT License, Copyright (c) 2025 Yanran Zhang), file networks/clip/clip.py,
# itself OpenAI's CLIP loader. Reduced to what D3QE uses, ``load(name, device="cpu")`` on the non-JIT path; ``_download``
# and ``_transform`` are unchanged. Changes: the text tokenizer is dropped (D3QE only uses the image tower);
# ``from pkg_resources import packaging`` (gone from current setuptools) became ``import packaging.version``;
# ``import urllib`` became ``import urllib.request``, which the download needs.
import hashlib
import os
import urllib.request
import warnings
from typing import List, Union

import packaging.version
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from .model import build_model

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load"]

_MODELS = {
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cpu", download_root: str = None):
    """The official ``clip.load`` on its non-JIT path (the one D3QE takes): build the model from the OpenAI
    checkpoint and keep it in float32 on the CPU. ``$SYNTHSCOPE_CLIP_DIR`` overrides the download folder."""
    if name in _MODELS:
        root = download_root or os.environ.get("SYNTHSCOPE_CLIP_DIR") or os.path.expanduser("~/.cache/clip")
        model_path = _download(_MODELS[name], root)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            state_dict = torch.load(opened_file, map_location="cpu")

    model = build_model(state_dict or model.state_dict()).to(device)
    if str(device) == "cpu":
        model.float()
    return model, _transform(model.visual.input_resolution)
