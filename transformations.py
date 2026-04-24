from typing import Callable
from PIL import Image
from omegaconf import OmegaConf
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
        elif transform_name == "to_tensor":
            return v2.ToTensor()
        elif transform_name == "to_dtype":
            return v2.ToDtype(torch.float32, scale=True)
        elif transform_name == "normalize":
            return v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            raise ValueError(f"Invalid transform name: {transform_name}")
    
    def __call__(self, image: Image.Image):
        for transform in self.transforms:
            image = transform(image)
        return image
    

if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    transform = AIGITransformations(config.transform_list)
    image = Image.open("//nvmestore/ssalehi/AGID/AIGIBench/train/car/0_real/00000.png")
    plt.imshow(image)
    plt.show()
    plt.savefig(fname='original_image.png')
    image = transform(image)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    plt.savefig(fname='transformations_test.png')
    print(image.shape)