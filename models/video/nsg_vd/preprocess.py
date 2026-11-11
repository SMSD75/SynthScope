"""NSG-VD's per-frame preprocessing (official data/feature_dataset/score_feature_dataset.py
``_preprocess_frame_from_path``): the extracted JPEG frame resized to 224x224 with PIL's LANCZOS (aspect ratio not
kept), then ``torch.from_numpy(np.array(img)).float() / 255`` as [3, H, W]. No normalization: the diffusion score
function maps [0, 1] to [-1, 1] itself."""
import numpy as np
import torch
from PIL import Image


class NSGVDFrame:
    """``nsgvd_frame_224``: PIL frame -> float [3, 224, 224] in [0, 1], exactly the official frame loader."""

    def __init__(self, size: int = 224):
        self.size = (size, size)

    def __call__(self, frame_img: Image.Image) -> torch.Tensor:
        frame_img = frame_img.resize(self.size, Image.Resampling.LANCZOS)   # imagnet's shape 224x224
        frame_tensor = torch.from_numpy(np.array(frame_img)).float()       # (h,w,3)
        frame_tensor = frame_tensor / 255.0                                  # normalize to (0,1)
        return frame_tensor.permute(2, 0, 1)                                 # (3,h,w)

    def __repr__(self):
        return f"NSGVDFrame({self.size[0]}, LANCZOS)"
