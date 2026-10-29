# Vendored from github.com/roy-ch/Dual-Data-Alignment (Apache-2.0, Copyright the Dual Data Alignment authors), file
# Training/data/datasets.py: ``create_train_transforms`` unchanged, and the body of ``RealFakeDataset.__getitem__`` split
# into ``build_item`` (the pair construction) so SynthScope's dataset can call it; the logic and the order of the random
# draws are unchanged. MEAN/STD are the module constants of the same file.
import random

import torchvision.transforms as transforms
from PIL import Image

from .official_train_transforms import (
    JPEG_Compression,
    MedianBlur,
    MotionBlur,
    PadRandomCrop,
    RandomGaussianNoise,
    RandomJPEGCompression,
    RandomPepperNoise,
    RandomSharpen,
    apply_resize,
    freq_blend_mix,
    pixel_blend_mix,
)

MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}
STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def create_train_transforms(
    size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    is_crop=True,
):
    if is_crop:
        resize_func = PadRandomCrop(size)
    else:
        print(f"Using Resize to {size} x {size}")
        resize_func = transforms.Resize((size, size))

    transform_list = [
        RandomJPEGCompression(quality_lower=55, quality_upper=100, p=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomGaussianNoise(p=0.1),
        RandomPepperNoise(p=0.1),
        transforms.RandomApply(
            [
                transforms.RandomChoice(
                    [
                        transforms.GaussianBlur(kernel_size=3),
                        transforms.GaussianBlur(kernel_size=5),
                        MedianBlur(kernel_size=3),
                        MotionBlur(kernel_size=5),
                    ]
                )
            ],
            p=0.2,
        ),
        RandomSharpen(p=0.1),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.175
                )
            ],
            p=0.2,
        ),
        transforms.RandomGrayscale(p=0.1),
        resize_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transform_list)


def build_item(real_img, fake_img, jpeg_quality, p_pixelmix=0.2, r_pixelmix=0.8, p_freqmix=0.2, r_freqmix=0.8):
    """``RealFakeDataset.__getitem__`` up to (not including) the shared transform: the matched JPEG of the fake, the
    pixel/frequency blending and the two resized copies. Returns the official ``img_dict``."""
    resampling_methods = [
        Image.NEAREST,
        Image.BOX,
        Image.BILINEAR,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
    ]

    img_dict = {
        "real": None,
        "fake": None,
        "real_resized": [],
        "fake_resized": [],
    }
    img_dict["real"] = real_img

    # DDA frequency-domain alignment: JPEG the VAE reconstruction at its real image's quality (p 0.5).
    if random.random() < 0.5:
        jpeg_quality_factor = int(jpeg_quality)
        fake_img = JPEG_Compression(fake_img, jpeg_quality_factor)

    blending_operations = []

    if p_pixelmix > 0 and random.random() < p_pixelmix:
        blending_operations.append("pixel")

    if p_freqmix > 0 and random.random() < p_freqmix:
        blending_operations.append("frequency")

    if blending_operations:
        random.shuffle(blending_operations)

        for operation in blending_operations:
            if operation == "pixel":

                resize_method = random.choice(resampling_methods)
                fake_img_resized = fake_img.resize(real_img.size, resize_method)
                fake_img = pixel_blend_mix(
                    real_img=real_img,
                    fake_img=fake_img_resized,
                    ratios=[0, r_pixelmix],
                )

            elif operation == "frequency":
                fake_img = freq_blend_mix(
                    real_img=real_img,
                    fake_img=fake_img,
                    ratios=[0, r_freqmix],
                    patch=-1,
                )

    img_dict["fake"] = fake_img

    down_resize_factor = random.uniform(0.2, 1.0)
    upper_resize_factor = random.uniform(1.0, 3.5)

    resize_methods = random.sample(resampling_methods, 2)

    for resize_factor, resize_method in zip(
        [down_resize_factor, upper_resize_factor], resize_methods
    ):

        real_resized = apply_resize(real_img, resize_factor, resize_method)
        fake_resized = apply_resize(fake_img, resize_factor, resize_method)

        img_dict["real_resized"].append(real_resized)
        img_dict["fake_resized"].append(fake_resized)

    return img_dict
