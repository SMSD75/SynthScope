from typing import Callable
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
import torch
import random


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
