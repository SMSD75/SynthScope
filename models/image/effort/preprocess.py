import cv2
import numpy as np
from PIL import Image


class Cv2SquareResize:
    """DeepfakeBench's test loader (``abstract_dataset.load_rgb``): the whole image squashed to ``size`` x ``size`` with
    cv2's bicubic interpolation (not PIL's, whose bicubic differs), then returned as a PIL image for ToTensor."""

    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = cv2.resize(np.asarray(img.convert("RGB")), (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(arr)

    def __repr__(self):
        return f"Cv2SquareResize({self.size}, INTER_CUBIC)"
