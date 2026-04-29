from typing import Callable
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms.v2 as v2
import torch

class AIGITransformations:
    def __init__(self, transform_list: list[Callable]):
        self.transform_list = transform_list
        self.transforms = [self._create_transform(transform) for transform in self.transform_list]
    
    def _create_transform(self, transform_name: str):
        if transform_name == "resize":
            return v2.Resize((224, 224))
        elif transform_name == "resize_short":
            return v2.Resize(256)
        elif transform_name == "center_crop":
            return v2.CenterCrop(224)
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
        elif transform_name == "random_crop":
            return v2.RandomCrop(size=(224, 224))
        elif transform_name == "normalize":
            return v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif transform_name == "normalize_clip":
            return v2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        else:
            raise ValueError(f"Invalid transform name: {transform_name}")
    
    def __call__(self, image: Image.Image):
        for transform in self.transforms:
            image = transform(image)
        return image
    

if __name__ == "__main__":
    from config_utils import compose_app_config

    config = compose_app_config()
    transform = AIGITransformations(list(config.DatasetConfig.eval_transform_list))
    image = Image.open("//nvmestore/ssalehi/AGID/AIGIBench/train/car/0_real/00000.png")
    plt.imshow(image)
    plt.show()
    plt.savefig(fname='original_image.png')
    image = transform(image)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    plt.savefig(fname='transformations_test.png')
    print(image.shape)
