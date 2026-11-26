"""Image loading of the official VLM inference stacks: what the server does to an image file before the model's processor
sees it. Three stacks so far: vLLM (``vllm_rgb``, Veritas++), ms-swift's Qwen-VL template (``swift_rgb`` +
``swift_qwen_vl_resize``, Align, Don't Blend) and transformers 4.45.2's CLIP image processor (``transformers_rgb`` +
``clip_resize_center_crop``, FakeVLM).

vLLM. Ported from vLLM 0.31.0 (Apache-2.0, Copyright contributors to the vLLM project): ``vllm/multimodal/media/image.py``
``ImageMediaIO.load_bytes`` with the default ``image_mode="RGB"`` and ``rgba_background_color=(255, 255, 255)``, and
the helpers it calls in ``vllm/multimodal/image.py`` (``normalize_image``, ``rgba_to_rgb``, ``_has_transparency``,
``convert_image_mode``), unchanged in behaviour:

    1. ``ImageOps.exif_transpose`` (any failure ignored): pixels follow the EXIF orientation, as displayed;
    2. RGB images pass through; RGBA images are composited on white; other modes that carry transparency (LA, PA, a
       ``transparency`` entry, e.g. a palette PNG with tRNS) are converted to RGBA and composited on white; every other
       mode is ``convert("RGB")``.

SynthScope's other detectors use a plain ``convert("RGB")`` (no EXIF rotation, alpha dropped). A VLM detector served
through vLLM (Veritas++) sees the image this way, so its ``VLMConfig.image_loader`` is ``vllm_rgb``; pass it the
decoded file before any RGB conversion (``media.decode_image_raw``) so that transparency is still there.

ms-swift (``swift infer``, Qwen2-VL / Qwen2.5-VL templates). Ported from ms-swift 4.4.2 (Apache-2.0, Copyright the
ModelScope team) and qwen-vl-utils 0.0.14 (Apache-2.0, Copyright the Qwen team, Alibaba), unchanged in behaviour:

    1. ``swift_rgb``: ``swift/template/vision_utils.py`` ``load_image``: the file is opened and ``convert("RGB")``-ed
       unless already RGB; no EXIF rotation, and transparency is DROPPED (PIL's RGBA -> RGB keeps the colour channels),
       unlike vLLM;
    2. ``swift_qwen_vl_resize(image, max_pixels, min_pixels)``: the template's ``_preprocess_inputs`` first shrinks an
       image above ``--max_pixels`` with ``rescale_image`` (torchvision ``Resize`` to int(sqrt(max_pixels / ratio)) rows,
       bilinear), then the Qwen-VL template's ``replace_tag`` calls qwen-vl-utils ``fetch_image``: ``to_rgb`` and a PIL
       ``resize`` (Pillow's default filter, bicubic) to ``smart_resize``'s size (both sides multiples of 28 = patch 14 x
       merge 2, pixels within [min_pixels, fetch cap]; swift's ``compat_qwen_vl_utils`` sets qwen-vl-utils'
       IMAGE_MAX_TOKEN_NUM to ``int(MAX_PIXELS) // 28^2`` tokens, so the fetch cap is ``max_pixels // 784 * 784``:
       ``max_pixels`` itself when it is a multiple of 784 (1,003,520), 1,209,712 for 1,210,000). The processor is then
       called with ``do_resize=False`` (``Qwen2VLTemplate._encode``).

transformers 4.45.2 CLIP image processor (FakeVLM: its official ``eval.py`` pins transformers 4.45.2 and calls the
llava-hf LLaVA-1.5 processor on the file as ``Image.open`` returns it). Ported from transformers 4.45.2 (Apache-2.0,
Copyright The HuggingFace Team): ``image_transforms.convert_to_rgb`` and ``CLIPImageProcessor.preprocess``'s resize
and centre crop (``get_resize_output_image_size`` with ``default_to_square=False``, ``image_transforms.resize``
through PIL, ``image_transforms.center_crop``), unchanged in behaviour:

    1. ``transformers_rgb``: ``convert("RGB")`` unless already RGB (no EXIF rotation; alpha dropped), what the processor
       does with ``do_convert_rgb=True`` (4.45.2 and 5.x alike);
    2. ``clip_resize_center_crop(image, shortest_edge, crop_size)``: a PIL bicubic resize of the shorter side to
       ``shortest_edge`` (the longer side ``int(shortest_edge * long / short)``), then the central ``crop_size`` square
       (top / left offsets floored).
transformers 5 resizes with torchvision instead of PIL (up to 0.03 apart after normalisation on noise images); with
this hook as ``VLMConfig.preprocess`` and the processor called with ``do_resize=False, do_center_crop=False``, the
pixel values equal 4.45.2's to float rounding (tests/test_fakevlm_smoke.py).
"""
import contextlib
import math
from typing import Optional, Tuple

import torchvision.transforms as T
from PIL import Image, ImageOps

WHITE = (255, 255, 255)


def normalize_image(image: Image.Image) -> Image.Image:
    """Normalize EXIF orientation so the pixel data matches visual display (vLLM ``normalize_image``)."""
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    return image


def rgba_to_rgb(image: Image.Image, background_color=WHITE) -> Image.Image:
    """An RGBA image composited on ``background_color`` (vLLM ``rgba_to_rgb``)."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])
    return converted


def has_transparency(image: Image.Image) -> bool:
    """RGBA, LA, PA, or a tRNS ``transparency`` entry (vLLM ``_has_transparency``)."""
    if image.mode in ("RGBA", "LA", "PA"):
        return True
    return "transparency" in getattr(image, "info", {})


def convert_image_mode(image: Image.Image, to_mode: str, background_color=WHITE) -> Image.Image:
    """vLLM ``convert_image_mode``: transparency is composited on ``background_color`` when converting to RGB."""
    if image.mode == to_mode:
        return image
    if to_mode == "RGB" and has_transparency(image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return rgba_to_rgb(image, background_color)
    return image.convert(to_mode)


def vllm_rgb(image: Image.Image) -> Image.Image:
    """The RGB image vLLM's ``ImageMediaIO.load_bytes`` (``image_mode="RGB"``) makes of a decoded file."""
    image = normalize_image(image)
    image.load()
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        return rgba_to_rgb(image, WHITE)
    return convert_image_mode(image, "RGB", WHITE)


# --- ms-swift 4.4.2 / qwen-vl-utils 0.0.14 (Qwen2-VL / Qwen2.5-VL) ----------------------------------------------------------
QWEN_VL_FACTOR = 28                                  # qwen-vl-utils: image_patch_size 14 x SPATIAL_MERGE_SIZE 2
QWEN_VL_MIN_PIXELS = 4 * QWEN_VL_FACTOR ** 2         # IMAGE_MIN_TOKEN_NUM 4 -> 3136
QWEN_VL_MAX_RATIO = 200                              # qwen-vl-utils MAX_RATIO


def swift_rgb(image: Image.Image) -> Image.Image:
    """ms-swift ``load_image``: ``convert("RGB")`` unless the image already is RGB (no EXIF transpose; alpha dropped)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def swift_rescale_image(image: Image.Image, max_pixels: int) -> Image.Image:
    """ms-swift ``rescale_image``: an image above ``max_pixels`` resized (torchvision ``Resize``, bilinear) to
    ``int(sqrt(max_pixels / ratio))`` x ``int(that * ratio)``; others unchanged."""
    width, height = image.width, image.height
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return image
    ratio = width / height
    height_scaled = math.sqrt(max_pixels / ratio)
    width_scaled = height_scaled * ratio
    return T.Resize((int(height_scaled), int(width_scaled)))(image)


def qwen_vl_smart_resize(height: int, width: int, factor: int = QWEN_VL_FACTOR, min_pixels: Optional[int] = None,
                         max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """qwen-vl-utils ``smart_resize``: (height, width) as multiples of ``factor``, closest to the aspect ratio, with the
    pixel count within [min_pixels, max_pixels] (round, then floor / ceil after scaling by the square-root ratio)."""
    max_pixels = max_pixels if max_pixels is not None else 16384 * factor ** 2
    min_pixels = min_pixels if min_pixels is not None else 4 * factor ** 2
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > QWEN_VL_MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {QWEN_VL_MAX_RATIO}, "
                         f"got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_vl_to_rgb(image: Image.Image) -> Image.Image:
    """qwen-vl-utils ``to_rgb``: RGBA composited on white, anything else ``convert("RGB")``."""
    if image.mode == "RGBA":
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.split()[3])
        return white
    return image.convert("RGB")


def qwen_vl_fetch_image(image: Image.Image, min_pixels: int = QWEN_VL_MIN_PIXELS,
                        max_pixels: Optional[int] = None) -> Image.Image:
    """qwen-vl-utils ``fetch_image`` for a PIL image: ``to_rgb``, then a PIL resize (default filter) to ``smart_resize``'s
    size."""
    image = qwen_vl_to_rgb(image)
    width, height = image.size
    resized_height, resized_width = qwen_vl_smart_resize(height, width, QWEN_VL_FACTOR, min_pixels, max_pixels)
    return image.resize((resized_width, resized_height))


def swift_fetch_max_pixels(max_pixels: int) -> int:
    """qwen-vl-utils' ``fetch_image`` cap under ``MAX_PIXELS=max_pixels``: swift's ``compat_qwen_vl_utils`` sets
    IMAGE_MAX_TOKEN_NUM = ``int(max_pixels) // 28**2`` and ``fetch_image`` multiplies it back by 28**2."""
    return int(max_pixels) // QWEN_VL_FACTOR ** 2 * QWEN_VL_FACTOR ** 2


def swift_qwen_vl_resize(image: Image.Image, max_pixels: int, min_pixels: int = QWEN_VL_MIN_PIXELS) -> Image.Image:
    """What ``swift infer --max_pixels max_pixels`` (with ``MAX_PIXELS=max_pixels`` in the environment, as the official
    ``predict.py`` sets it) does to an (already ``swift_rgb``) image of a Qwen2-VL / Qwen2.5-VL model before the
    processor (called with ``do_resize=False``): ``rescale_image`` at ``max_pixels``, then ``fetch_image`` at the
    token-rounded cap ``swift_fetch_max_pixels(max_pixels)``."""
    return qwen_vl_fetch_image(swift_rescale_image(image, max_pixels), min_pixels=min_pixels,
                               max_pixels=swift_fetch_max_pixels(max_pixels))


# --- transformers 4.45.2 CLIPImageProcessor (LLaVA-1.5 / FakeVLM) ------------------------------------------------------
def transformers_rgb(image: Image.Image) -> Image.Image:
    """transformers ``convert_to_rgb``: ``convert("RGB")`` unless the image already is RGB (no EXIF transpose; alpha
    dropped)."""
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def clip_resize_center_crop(image: Image.Image, shortest_edge: int = 336, crop_size: int = 336) -> Image.Image:
    """transformers 4.45.2 ``CLIPImageProcessor`` resize (``size={"shortest_edge": shortest_edge}``, PIL bicubic) and
    centre crop (``crop_size`` square) of an RGB image."""
    width, height = image.size
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = shortest_edge, int(shortest_edge * long / short)
    new_height, new_width = (new_long, new_short) if width <= height else (new_short, new_long)
    image = image.resize((new_width, new_height), resample=Image.BICUBIC, reducing_gap=None)
    top, left = (new_height - crop_size) // 2, (new_width - crop_size) // 2
    if top < 0 or left < 0:
        raise ValueError(f"crop {crop_size} larger than the resized image {new_width}x{new_height}")
    return image.crop((left, top, left + crop_size, top + crop_size))
