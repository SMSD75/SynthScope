from torch.utils.data import Dataset
from typing import Literal
from pathlib import Path
import zipfile
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformations import AIGITransformations
from omegaconf import OmegaConf
from torchvision.io import decode_image, ImageReadMode
import os
os.environ["HF_TOKEN"] = "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"

class AIGIBenchDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        setting: Literal["Setting-I", "Setting-II"],
        transform: AIGITransformations | None = None,
        prefetch_hf: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.setting = setting
        self.transform = transform
        self.unique_folder_names = ["train"]
        self.destination_dir = self.root_dir / self.split
        self.huggingface_path = "https://huggingface.co/datasets/HorizonTEL/AIGIBench/tree/main"
        self.huggingface_repo_id = "HorizonTEL/AIGIBench"
        if prefetch_hf:
            if os.is_dir(self.destination_dir):
                print(f"on prefetch_hf mode, {self.destination_dir} already exists")
        else:
            print(f"Downloading {self.split} split from Hugging Face...")
            self._download_data()
        if self.split != "train":
            self.unique_folder_names += [d.name for d in self.destination_dir.iterdir() if d.is_dir()]
        self.images, self.labels, self.folder_names = self._load_data()

    @staticmethod
    def _split_has_images(directory: Path) -> bool:
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".png", ".jpeg", ".webp"):
                return True
        return False

    @classmethod
    def prepare_split(
        cls,
        destination_dir: str | Path,
        split: Literal["train", "val", "test"],
        repo_id: str = "HorizonTEL/AIGIBench",
        force: bool = False,
    ) -> None:
        """Download all zip archives for one split from the Hub and extract under root_dir."""
        from huggingface_hub import hf_hub_download, list_repo_files

        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{split}/"
        files = list_repo_files(repo_id, repo_type="dataset")
        zips = sorted(f for f in files if f.startswith(prefix) and f.lower().endswith(".zip"))
        if not zips:
            raise RuntimeError(f"No .zip files under {prefix!r} in dataset {repo_id!r}")

        for rel in zips:
            stem = Path(rel).stem
            target_dir = destination_dir / stem
            if not force and target_dir.is_dir() and cls._split_has_images(target_dir):
                continue

            local_zip = hf_hub_download(
                repo_id=repo_id,
                filename=rel,
                repo_type="dataset",
            )
            with zipfile.ZipFile(local_zip, "r") as zf:
                zf.extractall(destination_dir)

    def _download_data(self, force: bool = False) -> None:
        """Download and extract this dataset split from Hugging Face (skips subsets that already contain images)."""
        self.prepare_split(
            destination_dir=self.destination_dir,
            split=self.split,
            repo_id=self.huggingface_repo_id,
            force=force,
        )

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
        split_dir = self.destination_dir

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


class AIGIBenchDatasetModule:
    def __init__(self, config):
        self.config = config
        train_list = list(config.DatasetConfig.train_transform_list)
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        self.train_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "train", config.DatasetConfig.setting, self.train_transform)
        self.val_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "val", config.DatasetConfig.setting, self.eval_transform)
        self.test_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "test", config.DatasetConfig.setting, self.eval_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')
    dataset_module = AIGIBenchDatasetModule(config)
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