import random
import re
from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import tv_tensors

from models.image.aide.official_arch import AIDEFiveView
from models.image.cure.preprocess import CuReTransform
from models.image.effort.preprocess import Cv2SquareResize
from models.image.mirror.preprocess import MirrorEvalTransform
from models.image.moejepa.preprocess import VJEPA2ImageTransform

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
DINOV3_MEAN, DINOV3_STD = [0.430, 0.411, 0.296], [0.213, 0.156, 0.143]
SQUARE_RESIZES = (224, 256, 336, 512)        # resize_<n>: both sides to n (aspect not kept)
SHORT_RESIZES = (256, 384, 448)              # resize_short_<n>: shorter side to n
CROPS = (224, 256, 336, 384, 448)            # center_crop_<n> (zero-padded when the image is smaller)
# The sized resize / crop names and their allowed sizes.
_SIZED = re.compile(r"(resize_short|resize|center_crop)_([1-9]\d*)")
_SIZES = {"resize_short": SHORT_RESIZES, "resize": SQUARE_RESIZES, "center_crop": CROPS}
# Name prefixes of steps that resize inside (whole official pipelines, SPAI's cap, aide_five_view*): an eval list holding
# one has no native-resolution form.
_RESIZING_STEPS = ("cap_long_side_1024", "effort_cv2_resize_224", "cure_eval", "mirror_eval", "vjepa2_processor",
                   "aide_five_view")


class CNNDetAugment:
    """The blur + JPEG augmentation of the CNNDetection code base (``data_augment``), which CNNDetection, D3QE and other
    detectors built on it train with: with probability ``blur_prob`` a Gaussian blur (scipy, sigma drawn from
    ``blur_sig``, in place per channel), then with probability ``jpg_prob`` a JPEG round trip (cv2 or PIL, quality drawn
    from ``jpg_qual``). Ops copied from D3QE's data/process.py; takes and returns a uint8 [3, H, W] tensor."""

    def __init__(self, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual):
        self.blur_prob, self.blur_sig = blur_prob, list(blur_sig)
        self.jpg_prob, self.jpg_method, self.jpg_qual = jpg_prob, list(jpg_method), list(jpg_qual)

    @staticmethod
    def _sample_continuous(s):
        return s[0] if len(s) == 1 else random.random() * (s[1] - s[0]) + s[0]

    @staticmethod
    def _cv2_jpg(img, quality):
        _, encimg = cv2.imencode(".jpg", img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(encimg, 1)[:, :, ::-1]

    @staticmethod
    def _pil_jpg(img, quality):
        out = BytesIO()
        Image.fromarray(img).save(out, format="jpeg", quality=quality)
        img = np.array(Image.open(out))
        out.close()
        return img

    def __call__(self, image):
        img = np.ascontiguousarray(image.permute(1, 2, 0).numpy() if isinstance(image, torch.Tensor) else np.array(image))
        if random.random() < self.blur_prob:
            sig = self._sample_continuous(self.blur_sig)
            for c in range(3):
                gaussian_filter(img[:, :, c], output=img[:, :, c], sigma=sig)
        if random.random() < self.jpg_prob:
            method = random.choice(self.jpg_method)
            quality = random.choice(self.jpg_qual)
            img = self._cv2_jpg(img, quality) if method == "cv2" else self._pil_jpg(img, quality)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()


class RandomMask:
    def __init__(self, ratio=(0.0, 0.75), patch_size=16, p=0.5):
        self.ratio = ratio
        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor

        _, height, width = tensor.shape
        ratio = random.uniform(*self.ratio)
        num_masks = int((height * width * ratio) / (self.patch_size ** 2))
        mask = torch.ones((height, width), dtype=tensor.dtype, device=tensor.device)
        max_top = max((height // self.patch_size) - 1, 0)
        max_left = max((width // self.patch_size) - 1, 0)

        positions = set()
        while len(positions) < num_masks:
            top = random.randint(0, max_top) * self.patch_size
            left = random.randint(0, max_left) * self.patch_size
            positions.add((top, left))

        for top, left in positions:
            mask[top:top + self.patch_size, left:left + self.patch_size] = 0
        return tensor * mask.expand_as(tensor)


class ToFloat:
    """``to_dtype``: an image as a float tensor in [0, 1]. A PIL image goes through ``ToTensor`` (what the inference
    transforms always used, so their pixels are unchanged); a uint8 tensor (the training loaders) through
    ``ToDtype(float32, scale=True)``."""

    def __init__(self):
        self._tensor = v2.ToDtype(torch.float32, scale=True)

    def __call__(self, image):
        return TF.to_tensor(image) if isinstance(image, Image.Image) else self._tensor(image)

    def __repr__(self):
        return "ToFloat()"


class CapLongSide:
    """Downscale a PIL image (keeping aspect, min side 224) only if its long side exceeds ``max_side``; otherwise
    leave it at native resolution (SPAI's any-resolution path)."""

    def __init__(self, max_side: int):
        self.max_side = max_side

    def __call__(self, im):
        w, h = im.size
        if max(w, h) > self.max_side:
            s = self.max_side / max(w, h)
            im = im.resize((max(int(w * s), 224), max(int(h * s), 224)))
        return im

    def __repr__(self):
        return f"CapLongSide({self.max_side})"


class AIGITransformations:
    """A named preprocessing pipeline. Each detector's lists live in its model config (``conf/model/.../<model>.yaml``:
    ``eval_transform_list``, ``train_transform_list``, optional ``val_transform_list``); the same names build the
    inference transform (PIL in) and the training transforms (uint8 tensors in).

    With ``pil_input=True`` a uint8 ``[3, H, W]`` image tensor (what the training loaders decode) is converted to a PIL
    image first, so the pipeline runs exactly the inference code; ``[T, C, H, W]`` video clips are left as they are."""

    def __init__(self, transform_list: list[str], pil_input: bool = False):
        self.transform_list = transform_list
        self.pil_input = pil_input
        self.transforms = [self._create_transform(transform) for transform in self.transform_list]

    def _create_transform(self, transform_name: str):
        sized = _SIZED.fullmatch(transform_name)
        if sized:
            kind, n = sized.group(1), int(sized.group(2))
            if n not in _SIZES[kind]:
                raise ValueError(f"Invalid transform name: {transform_name}")
            if kind == "resize_short":
                return v2.Resize(n)
            if kind == "resize":                           # square, e.g. NPR/FatFormer (Resize((256, 256))), OmniAID
                return v2.Resize((n, n))
            return v2.CenterCrop(n)
        elif transform_name == "cap_long_side_1024":       # SPAI: native resolution, long side capped at 1024 (PIL)
            return CapLongSide(1024)
        elif transform_name == "to_tensor":
            return v2.ToTensor()
        elif transform_name == "to_dtype":
            return ToFloat()
        elif transform_name == "gaussian_blur":
            return v2.RandomApply(
                [v2.GaussianBlur(kernel_size=11, sigma=(1e-3, 3.0))], p=0.5
            )
        elif transform_name == "random_horizontal_flip":
            return v2.RandomHorizontalFlip(p=0.5)
        elif transform_name == "random_vertical_flip":
            return v2.RandomVerticalFlip(p=0.5)
        elif transform_name == "blur_jpeg_cnndet":         # CNNDetection/D3QE training: blur 0.1 (sigma 0-3), JPEG 0.1 (cv2/pil, q30-100)
            return CNNDetAugment(0.1, (0.0, 3.0), 0.1, ("cv2", "pil"), range(30, 101))
        elif transform_name == "blur_jpeg_cnndet_val":     # the same on the validation split, fixed: sigma 1.5, PIL q65
            return CNNDetAugment(0.1, (1.5,), 0.1, ("pil",), (65,))
        elif transform_name == "jpeg_compression":
            return v2.RandomApply([v2.JPEG(quality=(30, 100))], p=0.5)
        elif transform_name == "random_crop_224":
            return v2.RandomCrop(size=(224, 224))
        elif transform_name == "random_crop_256":
            return v2.RandomCrop(size=(256, 256))
        elif transform_name == "random_crop_256_pad":
            return v2.RandomCrop(size=(256, 256), pad_if_needed=True)
        elif transform_name == "random_crop_384":
            return v2.RandomCrop(size=(384, 384))
        elif transform_name == "random_crop_448":
            return v2.RandomCrop(size=(448, 448))
        elif transform_name == "random_crop_384_pad":
            return v2.RandomCrop(size=(384, 384), pad_if_needed=True)
        elif transform_name == "pad_random_crop_224":      # PGC: pad-then-crop 224 (official PadRandomCrop)
            return v2.RandomCrop(size=(224, 224), pad_if_needed=True)
        elif transform_name == "random_rotation_180":
            return v2.RandomRotation(180)
        elif transform_name == "color_jitter_safe":
            return v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        elif transform_name == "color_jitter_pgc":         # PGC: official RandomApply(ColorJitter, p=0.2)
            return v2.RandomApply([v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)], p=0.2)
        elif transform_name == "random_mask_safe":
            return RandomMask(ratio=(0.0, 0.75), patch_size=16, p=0.5)
        elif transform_name == "aide_five_view":
            # AIDE native-resolution 5-view preprocessing: PIL/tensor -> [5,3,256,256] (DCT patch
            # selection at native res, then resize the selected patches). Must be the LAST transform.
            return AIDEFiveView()
        elif transform_name == "aide_five_view_train":
            # Same, but caps the long side at 1024 before the DCT to bound dataloader-worker memory
            # on large reals (training only; inference uses the uncapped aide_five_view).
            return AIDEFiveView(max_side=1024)
        # normalize_dinov2: the earlier name of normalize_imagenet (DINOv2 uses ImageNet statistics), kept for old configs.
        elif transform_name in ("normalize_imagenet", "normalize_dinov2"):
            return v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        elif transform_name == "normalize_clip":
            return v2.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        elif transform_name == "normalize_dinov3":
            return v2.Normalize(mean=DINOV3_MEAN, std=DINOV3_STD)
        # Whole official eval pipelines that are not a chain of the steps above (PIL in, normalized tensor out).
        elif transform_name == "effort_cv2_resize_224":    # Effort (DeepfakeBench load_rgb): cv2 bicubic squash, PIL out
            return Cv2SquareResize(224)
        elif transform_name == "cure_eval":                # CuRe: PNG->JPEG96, 390 crop, 336 resize, JPEG70, mean=std=0.5
            return CuReTransform()
        elif transform_name == "mirror_eval":              # MIRROR: short side >512 -> 512, center 224, PNG/BMP->JPEG96 (PIL out)
            return MirrorEvalTransform()
        elif transform_name == "vjepa2_processor":         # MoE-JEPA: the V-JEPA 2 video processor on one frame
            return VJEPA2ImageTransform()
        else:
            raise ValueError(f"Invalid transform name: {transform_name}")

    def geometric_steps(self) -> list:
        """The ``resize_*`` / ``center_crop_*`` steps, in order (they take PIL images too: the web overlay's geometry)."""
        return [t for name, t in zip(self.transform_list, self.transforms) if _SIZED.fullmatch(name)]

    def normalize_step(self) -> v2.Normalize | None:
        """The ``Normalize`` step, or None for pipelines that feed [0, 1] pixels (or normalize inside a custom step)."""
        return next((t for t in self.transforms if isinstance(t, v2.Normalize)), None)

    def without_resize(self) -> "AIGITransformations":
        """The same pipeline without its ``resize_*`` / ``center_crop_*`` steps, to score images at native resolution.
        Raises ValueError when a remaining step resizes by itself, as the result would then not be native."""
        kept = [name for name in self.transform_list if not _SIZED.fullmatch(name)]
        resizing = [name for name in kept if name.startswith(_RESIZING_STEPS)]
        if resizing:
            raise ValueError(f"{self}: {resizing} resize the image themselves, so it has no native-resolution form")
        return AIGITransformations(kept, pil_input=self.pil_input)

    def __repr__(self):
        return f"AIGITransformations({self.transform_list})"

    def __call__(self, image):
        # A 4D [T, C, H, W] tensor is a video clip. Wrap it as tv_tensors.Video so the v2
        # transforms (crops, flips, normalize, ...) apply with the SAME sampled parameters to
        # every frame (temporal consistency) instead of treating the frames as an independent
        # batch. Single images ([C, H, W] tensors or PIL) pass through unchanged.
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            image = tv_tensors.Video(image)
        elif self.pil_input and isinstance(image, torch.Tensor) and image.ndim == 3 and image.dtype == torch.uint8:
            image = TF.to_pil_image(image)
        for transform in self.transforms:
            image = transform(image)
        return image
