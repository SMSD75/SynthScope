"""Image loading of the official VLM inference stacks: what the server does to an image file before the model's processor
sees it. Two stacks so far: vLLM (``vllm_rgb``, Veritas++) and ms-swift's Qwen-VL template (``swift_rgb`` +
``swift_qwen_vl_resize``, Align, Don't Blend).

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
       merge 2, pixels within [min_pixels, max_pixels]; swift sets qwen-vl-utils' IMAGE_MAX_TOKEN_NUM from the
       MAX_PIXELS environment variable, i.e. max_pixels / 28^2 tokens). The processor is then called with
       ``do_resize=False`` (``Qwen2VLTemplate._encode``).
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


def swift_qwen_vl_resize(image: Image.Image, max_pixels: int, min_pixels: int = QWEN_VL_MIN_PIXELS) -> Image.Image:
    """What ``swift infer --max_pixels max_pixels`` does to an (already ``swift_rgb``) image of a Qwen2-VL / Qwen2.5-VL
    model before the processor (called with ``do_resize=False``): ``rescale_image``, then ``fetch_image``."""
    return qwen_vl_fetch_image(swift_rescale_image(image, max_pixels), min_pixels=min_pixels, max_pixels=max_pixels)
