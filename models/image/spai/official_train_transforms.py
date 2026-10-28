# Vendored from github.com/mever-team/spai (Apache-2.0, Copyright (c) 2025 Centre for Research and Technology Hellas and
# University of Amsterdam), file spai/data/data_finetune.py: ``build_transform`` unchanged. ``SPAI_CONFIG`` holds the
# values it reads, from spai/config.py defaults overridden by configs/spai.yaml (the released training config); the
# released code pins albumentations==1.4.14 / albucore==0.0.16, whose random draws and defaults this relies on.
from collections.abc import Callable
from types import SimpleNamespace as _NS

import albumentations as A
import numpy as np
from albumentations.augmentations.transforms import ImageCompressionType
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

SPAI_CONFIG = _NS(
    DATA=_NS(IMG_SIZE=224, AUGMENTED_VIEWS=4),
    MODEL=_NS(REQUIRED_NORMALIZATION="positive_0_1"),
    AUG=_NS(
        MIN_CROP_AREA=0.2, MAX_CROP_AREA=1.0,
        HORIZONTAL_FLIP_PROB=0.5, VERTICAL_FLIP_PROB=0.5,
        ROTATION_PROB=0.5, ROTATION_DEGREES=90,
        GAUSSIAN_BLUR_PROB=0.5, GAUSSIAN_NOISE_PROB=0.5,
        JPEG_COMPRESSION_PROB=0.5, JPEG_MIN_QUALITY=50, JPEG_MAX_QUALITY=100,
        WEBP_COMPRESSION_PROB=0.0, WEBP_MIN_QUALITY=50, WEBP_MAX_QUALITY=100,
        COLOR_JITTER=0.0, COLOR_JITTER_BRIGHTNESS_RANGE=(0.8, 1.2), COLOR_JITTER_CONTRAST_RANGE=(0.8, 1.2),
        COLOR_JITTER_SATURATION_RANGE=(0.8, 1.2), COLOR_JITTER_HUE_RANGE=(-0.1, 0.1),
        SHARPEN_PROB=0.0, SHARPEN_ALPHA_RANGE=(0.01, 0.4), SHARPEN_LIGHTNESS_RANGE=(0.95, 1),
    ),
    TEST=_NS(
        JPEG_COMPRESSION=False, JPEG_QUALITY=100, WEBP_COMPRESSION=False, WEBP_QUALITY=100,
        GAUSSIAN_BLUR=False, GAUSSIAN_BLUR_KERNEL_SIZE=3, GAUSSIAN_NOISE=False, GAUSSIAN_NOISE_SIGMA=1.0,
        SCALE=False, SCALE_FACTOR=1.0, MAX_SIZE=None, ORIGINAL_RESOLUTION=True, CROP=True,
    ),
)


def build_transform(is_train, config) -> Callable[[np.ndarray], np.ndarray]:
    # resize_im: bool = config.DATA.IMG_SIZE > 32
    # # this should always dispatch to transforms_imagenet_train
    # transform = create_transform(
    #     input_size=config.DATA.IMG_SIZE,
    #     is_training=True,
    #     color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
    #     auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
    #     re_prob=config.AUG.REPROB,
    #     re_mode=config.AUG.REMODE,
    #     re_count=config.AUG.RECOUNT,
    #     interpolation=config.DATA.INTERPOLATION,
    # )
    # if not resize_im:
    #     # replace RandomResizedCropAndInterpolation with
    #     # RandomCrop
    #     transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
    # transform.transforms.insert(0, torchvision.transforms.v2.JPEG((50, 100)))
    # transform.transforms.insert(4, torchvision.transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.01, 0.5)))

    if is_train:  # Training augmentations
        transforms_list = []

        if config.AUG.MIN_CROP_AREA == config.AUG.MAX_CROP_AREA:
            transforms_list.append(
                A.PadIfNeeded(min_height=config.DATA.IMG_SIZE, min_width=config.DATA.IMG_SIZE)
            )
            transforms_list.append(
                A.RandomCrop(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE)
            )
        else:
            transforms_list.append(
                A.RandomResizedCrop(size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                    scale=(config.AUG.MIN_CROP_AREA, config.AUG.MAX_CROP_AREA))
            )
        transforms_list.extend([
            A.HorizontalFlip(p=config.AUG.HORIZONTAL_FLIP_PROB),
            A.VerticalFlip(p=config.AUG.VERTICAL_FLIP_PROB),
            A.Rotate(limit=config.AUG.ROTATION_DEGREES,
                     crop_border=True,
                     p=config.AUG.ROTATION_PROB)
        ])
        if config.AUG.ROTATION_PROB > .0:
            # Rotation with crop_border set to True leads to images smaller than the target
            # size. So, restore the target size.
            transforms_list.append(
                A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE)
            )
        transforms_list.extend([
            A.GaussianBlur(blur_limit=(3, 9),
                           sigma_limit=(0.01, 0.5),
                           p=config.AUG.GAUSSIAN_BLUR_PROB),
            A.GaussNoise(p=config.AUG.GAUSSIAN_NOISE_PROB),
            A.ColorJitter(
                p=config.AUG.COLOR_JITTER,
                brightness=config.AUG.COLOR_JITTER_BRIGHTNESS_RANGE,
                contrast=config.AUG.COLOR_JITTER_CONTRAST_RANGE,
                saturation=config.AUG.COLOR_JITTER_SATURATION_RANGE,
                hue=config.AUG.COLOR_JITTER_HUE_RANGE,
            ),
            A.Sharpen(p=config.AUG.SHARPEN_PROB,
                      alpha=config.AUG.SHARPEN_ALPHA_RANGE,
                      lightness=config.AUG.SHARPEN_LIGHTNESS_RANGE),
            A.ImageCompression(quality_lower=config.AUG.JPEG_MIN_QUALITY,
                               quality_upper=config.AUG.JPEG_MAX_QUALITY,
                               compression_type=ImageCompressionType.JPEG,
                               p=config.AUG.JPEG_COMPRESSION_PROB),
            A.ImageCompression(quality_lower=config.AUG.WEBP_MIN_QUALITY,
                               quality_upper=config.AUG.WEBP_MAX_QUALITY,
                               compression_type=ImageCompressionType.WEBP,
                               p=config.AUG.WEBP_COMPRESSION_PROB),
        ])
        if config.MODEL.REQUIRED_NORMALIZATION == "imagenet":
            transforms_list.append(
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            )
        elif config.MODEL.REQUIRED_NORMALIZATION == "positive_0_1":
            transforms_list.append(
                A.Normalize(mean=0., std=1.)
            )
        else:
            raise RuntimeError(f"Unsupported Normalization: {config.MODEL.REQUIRED_NORMALIZATION}")
        transforms_list.append(ToTensorV2())
        transform = A.Compose(transforms_list)

    else:  # Inference augmentations
        transforms_list = [
            A.ImageCompression(quality_lower=config.TEST.JPEG_QUALITY,
                               quality_upper=config.TEST.JPEG_QUALITY,
                               compression_type=ImageCompressionType.JPEG,
                               p=1.0 if config.TEST.JPEG_COMPRESSION else .0),
            A.ImageCompression(quality_lower=config.TEST.WEBP_QUALITY,
                               quality_upper=config.TEST.WEBP_QUALITY,
                               compression_type=ImageCompressionType.WEBP,
                               p=1.0 if config.TEST.WEBP_COMPRESSION else .0),
            A.GaussianBlur(blur_limit=(config.TEST.GAUSSIAN_BLUR_KERNEL_SIZE,
                                       config.TEST.GAUSSIAN_BLUR_KERNEL_SIZE),
                           sigma_limit=0,
                           p=1.0 if config.TEST.GAUSSIAN_BLUR else .0),
            A.GaussNoise(var_limit=(config.TEST.GAUSSIAN_NOISE_SIGMA**2,
                                    config.TEST.GAUSSIAN_NOISE_SIGMA**2),
                         p=1.0 if config.TEST.GAUSSIAN_NOISE else .0),
            A.RandomScale(scale_limit=(config.TEST.SCALE_FACTOR-1, config.TEST.SCALE_FACTOR-1),
                          p=1.0 if config.TEST.SCALE else .0)
        ]
        if config.TEST.MAX_SIZE is not None:
            transforms_list.append(A.SmallestMaxSize(max_size=config.TEST.MAX_SIZE))

        if config.TEST.ORIGINAL_RESOLUTION:
            transforms_list.append(A.PadIfNeeded(min_height=config.DATA.IMG_SIZE,
                                                 min_width=config.DATA.IMG_SIZE))
        elif config.TEST.CROP:
            transforms_list.append(A.PadIfNeeded(min_height=config.DATA.IMG_SIZE,
                                                 min_width=config.DATA.IMG_SIZE))
            transforms_list.append(A.CenterCrop(height=config.DATA.IMG_SIZE,
                                                width=config.DATA.IMG_SIZE))
        else:
            transforms_list.append(A.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
        if config.MODEL.REQUIRED_NORMALIZATION == "imagenet":
            transforms_list.append(A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
        elif config.MODEL.REQUIRED_NORMALIZATION == "positive_0_1":
            transforms_list.append(A.Normalize(mean=0., std=1.))
        else:
            raise RuntimeError(f"Unsupported Normalization: {config.MODEL.REQUIRED_NORMALIZATION}")
        transforms_list.append(ToTensorV2())
        transform = A.Compose(transforms_list)

    return transform


