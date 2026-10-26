from io import BytesIO
from typing import Callable
import cv2
import numpy as np
import torchvision.transforms.v2 as v2
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import tv_tensors
import torch
import random


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

class AIGITransformations:
    def __init__(self, transform_list: list[Callable]):
        self.transform_list = transform_list
        self.transforms = [self._create_transform(transform) for transform in self.transform_list]
    
    def _create_transform(self, transform_name: str):
        if transform_name == "resize_224":
            return v2.Resize((224, 224))
        elif transform_name == "resize_short_256":
            return v2.Resize(256)
        elif transform_name == "resize_short_384":
            return v2.Resize(384)
        elif transform_name == "resize_short_448":
            return v2.Resize(448)
        elif transform_name == "center_crop_224":
            return v2.CenterCrop(224)
        elif transform_name == "center_crop_256":
            return v2.CenterCrop(256)
        elif transform_name == "center_crop_384":
            return v2.CenterCrop(384)
        elif transform_name == "center_crop_448":
            return v2.CenterCrop(448)
        elif transform_name == "to_tensor":
            return v2.ToTensor()
        elif transform_name == "to_dtype":
            return v2.ToDtype(torch.float32, scale=True)
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
            from models.image.aide.official_arch import AIDEFiveView
            return AIDEFiveView()
        elif transform_name == "aide_five_view_train":
            # Same, but caps the long side at 1024 before the DCT to bound dataloader-worker memory
            # on large reals (training only; inference uses the uncapped aide_five_view).
            from models.image.aide.official_arch import AIDEFiveView
            return AIDEFiveView(max_side=1024)
        elif transform_name == "normalize_dinov2":
            return v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif transform_name == "normalize_clip":
            return v2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        elif transform_name == "normalize_dinov3":
            return v2.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143),
            )
        else:
            raise ValueError(f"Invalid transform name: {transform_name}")
    
    def __call__(self, image):
        # A 4D [T, C, H, W] tensor is a video clip. Wrap it as tv_tensors.Video so the v2
        # transforms (crops, flips, normalize, ...) apply with the SAME sampled parameters to
        # every frame (temporal consistency) instead of treating the frames as an independent
        # batch. Single images ([C, H, W] tensors or PIL) pass through unchanged.
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            image = tv_tensors.Video(image)
        for transform in self.transforms:
            image = transform(image)
        return image
