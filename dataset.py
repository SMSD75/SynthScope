from torch.utils.data import Dataset
from typing import Literal, Callable
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformations import AIGITransformations
from omegaconf import OmegaConf
from torchvision.io import decode_image, ImageReadMode

class AIGIBenchDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        setting: Literal["Setting-I", "Setting-II"],
        transform: AIGITransformations | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.setting = setting
        self.transform = transform
        self.unique_folder_names = ["train"]
        if self.split != "train":
            self.unique_folder_names += [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        self.images, self.labels, self.folder_names = self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = decode_image(str(self.images[idx]), mode=ImageReadMode.RGB)
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path, folder_names: list[str] | None = None) -> int:
        name = filepath.parent.name
        if "real" in name:
            return 0
        elif "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")
    
    def _iter_images(self, directory: Path):
        return (p for p in directory.rglob("*") if p.suffix.lower() in (".jpg", ".png"))

    def _load_data(self):
        images, labels, folder_names = [], [], []
        split_dir = self.root_dir

        for filepath in self._iter_images(split_dir):
            if self.split == "train" and self.setting == "Setting-I":
                if "sdv1.4" in str(filepath):
                    continue

            images.append(filepath)
            labels.append(self._get_label(filepath, self.unique_folder_names))
            folder_names.append(
                filepath.parent.parent.name if self.split != "train" else "train"
            )

        return images, labels, folder_names
class AIGIDatasetModule:
    def __init__(self, config):
        self.config = config
        if config.DatasetConfig.transform_list != []:
            self.transform = AIGITransformations(config.DatasetConfig.transform_list)
        else:
            self.transform = None
        self.train_dataset = AIGIBenchDataset(config.DatasetConfig.train_dir, "train", config.DatasetConfig.setting, self.transform)
        self.val_dataset = AIGIBenchDataset(config.DatasetConfig.val_dir, "val", config.DatasetConfig.setting, self.transform)
        self.test_dataset = AIGIBenchDataset(config.DatasetConfig.test_dir, "test", config.DatasetConfig.setting, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')
    dataset_module = AIGIDatasetModule(config)
    train_dataloader = dataset_module.train_dataloader()
    val_dataloader = dataset_module.val_dataloader()
    test_dataloader = dataset_module.test_dataloader()
    for batch in train_dataloader:
        print(batch)
        break
    for batch in val_dataloader:
        print(batch)
        break
    for batch in test_dataloader:
        print(batch)
        plt.imshow(batch[0].permute(1, 2, 0))
        plt.savefig(fname='test_image.png')
        break