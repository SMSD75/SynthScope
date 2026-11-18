"""Image loading of the official VLM servers: what vLLM does to an image file before the model's processor sees it.

Ported from vLLM 0.31.0 (Apache-2.0, Copyright contributors to the vLLM project): ``vllm/multimodal/media/image.py``
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
"""
import contextlib

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
