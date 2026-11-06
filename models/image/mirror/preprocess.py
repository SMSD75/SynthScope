"""MIRROR's test-time preprocessing; ``compress_image`` and the resize rule are vendored from github.com/handsome-rich/MIRROR
(commit 7f108fd, ``data/datasets.py``; licence unclear: README badge MIT, no LICENSE file - see LICENSES.md).

The eval path of the official ``data/datasets.py`` (``Get_Transforms(mode='preprocess')`` eval branch and
``TestDataset.__getitem__``), PIL image in, the 224x224 RGB PIL crop out; the eval list then
applies ``to_dtype`` (= ToTensor). ImageNet normalization happens inside the model, as in the official
``MIRROR_Detector.forward``.

Official per image (``RandomScaleCropOrDirect224(train=False, infer_policy="short256_center", eval_resize_short=512)``,
then ``CenterCrop(224)``, then the "format-aligned input" step, then ``ToTensor``):
  1. if the shorter side exceeds 512, resize it to 512 (bicubic, antialias; long side ``int(round(w * 512 / short))``),
  2. center-crop 224 (torchvision pads smaller images with zeros), a second CenterCrop(224) is a no-op,
  3. if the SOURCE FILE is .png / .bmp / .tiff, JPEG round-trip at quality 96 (``compress_image``: PIL, optimize=True) -
     the fake training images were JPEG q96 round-tripped, so lossless inputs are aligned to that format,
  4. ToTensor (the ``to_dtype`` step after this one).

Step 3 depends on the file name, which a decoded PIL image has lost. SynthScope's image loaders record the decoded format
in ``img.info["synthscope_src_format"]`` (``SRC_FORMAT_KEY``): ``bench_eval._open_rgb`` and ``interface/backend/media``
(``to_rgb`` / ``decode_image`` / ``open_rgb``, used by the web UI, the synthscope API for paths, bytes and PIL images,
``benchmark/evaluate.py`` and ``benchmark/generator_bench.py``). PNG / BMP there take the JPEG step; anything else, or
an image without the hint (e.g. the training loaders' tensors), does not. For correctly named files this is the official
extension test.
"""
import io

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models.image.cure.preprocess import SRC_FORMAT_KEY

CROP_SIZE = 224
EVAL_RESIZE_SHORT = 512
ALIGN_JPEG_QUALITY = 96
# Official: path.lower().endswith(('.png', '.bmp', '.tiff', '.PNG')). TIFF is kept for completeness, but the official
# TestDataset never loads .tif/.tiff files (it lists .png/.jpg/.jpeg/.bmp/.webp only), so no official number involves it.
ALIGNED_FORMATS = ("PNG", "BMP", "TIFF")


def compress_image(img_pil, quality=75):
    """Official ``data/datasets.py`` compress_image, unchanged."""
    img = img_pil.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=int(quality), optimize=True)
    buffer.seek(0)

    return Image.open(buffer).convert('RGB')


def resize_short(img: Image.Image, target_short: int) -> Image.Image:
    """Official ``RandomScaleCropOrDirect224._resize_short`` (bicubic, antialias)."""
    w, h = img.size
    short = min(w, h)
    if short == target_short:
        return img
    scale = target_short / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)


class MirrorEvalTransform:
    """PIL image -> the official MIRROR eval crop, a 224x224 RGB PIL image (stateless, picklable)."""

    def __call__(self, img: Image.Image):
        source_format = img.info.get(SRC_FORMAT_KEY) or img.format
        image = img.convert("RGB")
        w, h = image.size
        if min(w, h) > EVAL_RESIZE_SHORT:
            image = resize_short(image, EVAL_RESIZE_SHORT)
        image = TF.center_crop(image, [CROP_SIZE, CROP_SIZE])
        image = TF.center_crop(image, [CROP_SIZE, CROP_SIZE])   # the official second transforms.CenterCrop(224)
        if source_format in ALIGNED_FORMATS:
            image = compress_image(image, quality=ALIGN_JPEG_QUALITY)
        return image

    def __repr__(self):
        return "MirrorEvalTransform()"
