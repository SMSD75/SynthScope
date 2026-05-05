from torch.utils.data import Dataset
from typing import Literal
from pathlib import Path
import hashlib
import re
import zipfile
import tarfile
import shutil
import subprocess
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformations import AIGITransformations
from torchvision.io import decode_image, ImageReadMode
from config_utils import compose_config


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

def _iter_images(directory: Path):
    return (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _has_images(directory: Path) -> bool:
    return directory.is_dir() and any(_iter_images(directory))


def _safe_extract_zip(zip_path: str | Path, destination_dir: str | Path) -> None:
    """Extract a zip file without allowing path traversal outside destination_dir."""
    zip_path = Path(zip_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_root = destination_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (destination_dir / member.filename).resolve()
            if not member_path.is_relative_to(destination_root):
                raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
        zf.extractall(destination_dir)


def _safe_extract_tar(tar_path: str | Path, destination_dir: str | Path) -> None:
    """Extract a tar file without allowing path traversal outside destination_dir."""
    tar_path = Path(tar_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_root = destination_dir.resolve()

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            member_path = (destination_dir / member.name).resolve()
            if not member_path.is_relative_to(destination_root):
                raise RuntimeError(f"Refusing to extract unsafe tar member: {member.name}")
        tf.extractall(destination_dir)


def _download_hf_file(
    repo_id: str,
    filename: str,
    download_dir: str | Path,
    repo_type: str = "dataset",
) -> Path:
    """Download one file from the Hugging Face Hub into download_dir and return its local path."""
    from huggingface_hub import hf_hub_download

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=download_dir,
        )
    )

class GenImageDataset(Dataset):
    google_drive_folder_url = "https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
    split_aliases = {
        "train": "train",
        "val": "test",
        "test": "test",
    }
    source_split_markers = {
        "train": ("train",),
        "test": ("val", "validation", "test"),
    }
    class_dir_to_label = {
        "nature": "real",
        "real": "real",
        "ai": "fake",
        "fake": "fake",
        "false": "fake",
    }

    def __init__(self, root_dir: str | Path, split: Literal["train", "val", "test"], transform: AIGITransformations | None = None, prefetch: bool = False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.requested_split = split
        self.split = self.split_aliases[split]
        self.transform = transform
        self.prefetch = prefetch
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print(f"Downloading {self.split} split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @staticmethod
    def _tokens(path: Path) -> list[str]:
        tokens: list[str] = []
        for part in path.parts:
            tokens.extend(token for token in re.split(r"[^a-z0-9]+", part.lower()) if token)
        return tokens

    @classmethod
    def download_files(
        cls,
        download_dir: str | Path,
        url: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download the shared GenImage Google Drive folder into download_dir."""
        download_dir = Path(download_dir)
        if download_dir.exists() and any(download_dir.iterdir()) and not force:
            return download_dir

        if force:
            shutil.rmtree(download_dir, ignore_errors=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "Downloading the GenImage Google Drive folder requires `gdown`. "
                "Install it with `pip install gdown`."
            ) from exc

        gdown.download_folder(
            url=url or cls.google_drive_folder_url,
            output=str(download_dir),
            quiet=False,
            use_cookies=False,
        )
        return download_dir

    @classmethod
    def _extract_downloaded_archives(cls, download_dir: Path) -> None:
        for zip_path in sorted(download_dir.rglob("*.zip")):
            extract_dir = zip_path.with_suffix("")
            if _has_images(extract_dir):
                continue
            _safe_extract_zip(zip_path, extract_dir)

    @classmethod
    def _source_split_for_path(cls, filepath: Path, destination_split: Literal["train", "test"]) -> bool:
        tokens = cls._tokens(filepath.parent)
        return any(marker in tokens for marker in cls.source_split_markers[destination_split])

    @classmethod
    def _label_dir_for_path(cls, filepath: Path) -> str | None:
        tokens = cls._tokens(filepath.parent)
        for source_name, label_dir in cls.class_dir_to_label.items():
            if source_name in tokens:
                return label_dir
        return None

    @classmethod
    def _relative_image_path(cls, filepath: Path) -> Path:
        parts = list(filepath.parts)
        lower_parts = [part.lower() for part in parts]
        marker_index = -1
        for idx, part in enumerate(lower_parts[:-1]):
            tokens = cls._tokens(Path(part))
            if any(token in cls.source_split_markers["train"] + cls.source_split_markers["test"] for token in tokens):
                marker_index = idx
            if any(token in cls.class_dir_to_label for token in tokens):
                marker_index = idx
        relative_parts = parts[marker_index + 1 :] if marker_index >= 0 else [filepath.name]
        return Path(*relative_parts)

    @staticmethod
    def _deduplicated_destination(destination_path: Path, source_path: Path) -> Path:
        if not destination_path.exists():
            return destination_path

        digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
        return destination_path.with_name(f"{destination_path.stem}_{digest}{destination_path.suffix}")

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        url: str | None = None,
        force: bool = False,
        keep_downloads: bool = True,
    ) -> None:
        """Download GenImage and normalize it to root/train and root/test."""
        root_dir = Path(root_dir)
        train_dir = root_dir / "train"
        test_dir = root_dir / "test"
        if not force and _has_images(train_dir) and _has_images(test_dir):
            return

        if force:
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(test_dir, ignore_errors=True)

        download_dir = root_dir / "_downloads" / "GenImage"
        cls.download_files(download_dir, url=url, force=force)
        cls._extract_downloaded_archives(download_dir)

        copied = {"train": 0, "test": 0}
        for destination_split in ("train", "test"):
            destination_split_dir = root_dir / destination_split
            destination_split_dir.mkdir(parents=True, exist_ok=True)

            for source_path in _iter_images(download_dir):
                source_relative_path = source_path.relative_to(download_dir)
                if not cls._source_split_for_path(source_relative_path, destination_split):
                    continue

                label_dir = cls._label_dir_for_path(source_relative_path)
                if label_dir is None:
                    continue

                relative_path = cls._relative_image_path(source_relative_path)
                destination_path = cls._deduplicated_destination(
                    destination_split_dir / label_dir / relative_path,
                    source_path,
                )
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                copied[destination_split] += 1

        missing = [split for split, count in copied.items() if count == 0]
        if missing:
            raise RuntimeError(
                "GenImage download did not contain recognizable images for "
                f"{', '.join(missing)}. Expected train/val split folders and ai/nature class folders."
            )

        if not keep_downloads:
            shutil.rmtree(root_dir / "_downloads", ignore_errors=True)

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        url: str | None = None,
        force: bool = False,
        keep_downloads: bool = True,
    ) -> None:
        """Download and normalize GenImage; val is stored as the test split."""
        destination_split = cls.split_aliases[split]
        destination_dir = Path(root_dir) / destination_split
        if not force and _has_images(destination_dir):
            return
        cls.prepare_dataset(root_dir, url=url, force=force, keep_downloads=keep_downloads)

    def _download_data(self, force: bool = False) -> None:
        """Download and normalize this dataset split from Google Drive."""
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.requested_split,
            url=self.google_drive_folder_url,
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
    
    def _iter_images(self, directory: Path):
        return _iter_images(directory)
    
    def _load_data(self):
        images, labels, folder_names = [], [], []
        split_dir = self.destination_dir
        for filepath in self._iter_images(split_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(self.split)
        return images, labels, folder_names

    @staticmethod
    def _get_label(filepath: Path) -> int:
        names = [part.lower() for part in filepath.parent.parts]
        if "real" in names:
            return 0
        if "fake" in names or "false" in names:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")


class CNNDetectionDataset(Dataset):
    google_drive_folder_url = "https://drive.google.com/drive/u/0/folders/1RwCSaraEUctIwFgoQXWMKFvW07gM80_3"
    archive_name_markers = {
        "train": ("progan_train", "train"),
        "val": ("progan_val", "val"),
        "test": ("cnn_synth_testset", "test"),
    }

    def __init__(self, root_dir: str | Path, split: Literal["train", "val", "test"], transform: AIGITransformations | None = None, prefetch: bool = False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.prefetch = prefetch
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print(f"Downloading {self.split} split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @classmethod
    def download_archives(
        cls,
        archive_dir: str | Path,
        url: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download the shared CNNDetection Google Drive folder into archive_dir."""
        archive_dir = Path(archive_dir)
        if archive_dir.exists() and any(archive_dir.iterdir()) and not force:
            return archive_dir

        archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "Downloading a Google Drive folder requires `gdown`. "
                "Install it with `pip install gdown`."
            ) from exc

        gdown.download_folder(
            url=url or cls.google_drive_folder_url,
            output=str(archive_dir),
            quiet=False,
            use_cookies=False,
        )
        return archive_dir

    @classmethod
    def _find_split_archives(
        cls,
        archive_dir: Path,
        split: Literal["train", "val", "test"],
    ) -> list[Path]:
        markers = cls.archive_name_markers[split]
        archive_suffixes = (".zip", ".7z", ".001")
        archives = [
            p
            for p in archive_dir.rglob("*")
            if p.is_file()
            and p.name.lower().endswith(archive_suffixes)
            and any(marker in p.name.lower() for marker in markers)
        ]

        if split == "train":
            zip_archives = [p for p in archives if p.suffix.lower() == ".zip"]
            if zip_archives:
                return sorted(zip_archives)
            first_7z_parts = [p for p in archives if p.name.lower().endswith(".7z.001")]
            if first_7z_parts:
                return sorted(first_7z_parts)

        return sorted(archives)

    @staticmethod
    def _extract_7z(archive_path: Path, destination_dir: Path) -> None:
        seven_zip = shutil.which("7z") or shutil.which("7za")
        if seven_zip is None:
            raise RuntimeError("Extracting .7z archives requires the `7z` command. Install p7zip-full.")

        destination_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [seven_zip, "x", str(archive_path), f"-o{destination_dir}", "-y"],
            cwd=archive_path.parent,
            check=True,
        )

    @classmethod
    def _extract_archive(cls, archive_path: Path, destination_dir: Path) -> None:
        if archive_path.suffix.lower() == ".zip":
            _safe_extract_zip(archive_path, destination_dir)
            return

        if archive_path.name.lower().endswith(".7z") or archive_path.name.lower().endswith(".7z.001"):
            cls._extract_7z(archive_path, destination_dir)
            for zip_path in destination_dir.rglob("*.zip"):
                _safe_extract_zip(zip_path, destination_dir)
            return

        raise RuntimeError(f"Unsupported archive type: {archive_path}")

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download and extract one CNNDetection split under root_dir/split."""
        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_images(destination_dir):
            return

        archive_dir = root_dir / "_archives" / "CNNDetection"
        cls.download_archives(archive_dir)
        archive_paths = cls._find_split_archives(archive_dir, split)
        if not archive_paths:
            raise RuntimeError(f"No archive found for CNNDetection {split!r} split in {archive_dir}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        for archive_path in archive_paths:
            cls._extract_archive(archive_path, destination_dir)

        if not keep_archives:
            shutil.rmtree(archive_dir, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        splits: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test"),
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download and extract the requested CNNDetection splits."""
        for split in splits:
            cls.prepare_split(root_dir, split, force=force, keep_archives=keep_archives)

    def _download_data(self, force: bool = False) -> None:
        """Download and extract this dataset split under root_dir/split."""
        self.prepare_split(self.root_dir, self.split, force=force)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = decode_image(str(self.images[idx]), mode=ImageReadMode.RGB)
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")
    
    def _iter_images(self, directory: Path):
        return _iter_images(directory)

    def _load_data(self):
        images, labels, folder_names = [], [], []
        split_dir = self.destination_dir
        for filepath in self._iter_images(split_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name if filepath.parent.parent != split_dir.parent else self.split)
        return images, labels, folder_names


class CNNDetectionDatasetModule:
    def __init__(self, config):
        self.config = config
        train_list = list(config.DatasetConfig.train_transform_list)
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        self.train_dataset = CNNDetectionDataset(config.DatasetConfig.root_dir, "train", self.train_transform, prefetch=prefetch)
        self.val_dataset = CNNDetectionDataset(config.DatasetConfig.root_dir, "val", self.eval_transform, prefetch=prefetch)
        self.test_dataset = CNNDetectionDataset(config.DatasetConfig.root_dir, "test", self.eval_transform, prefetch=prefetch)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class GenImageDatasetModule:
    def __init__(self, config):
        self.config = config
        train_list = list(config.DatasetConfig.train_transform_list)
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        source_url = config.DatasetConfig.get("source_url", GenImageDataset.google_drive_folder_url)
        GenImageDataset.google_drive_folder_url = source_url
        self.train_dataset = GenImageDataset(config.DatasetConfig.root_dir, "train", self.train_transform, prefetch=prefetch)
        self.val_dataset = GenImageDataset(config.DatasetConfig.root_dir, "val", self.eval_transform, prefetch=prefetch)
        self.test_dataset = GenImageDataset(config.DatasetConfig.root_dir, "test", self.eval_transform, prefetch=prefetch)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class AIGIBenchDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        setting: Literal["Setting-I", "Setting-II"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.setting = setting
        self.transform = transform
        self.unique_folder_names = ["train"]
        self.destination_dir = self.root_dir / self.split
        self.huggingface_path = "https://huggingface.co/datasets/HorizonTEL/AIGIBench/tree/main"
        self.huggingface_repo_id = "HorizonTEL/AIGIBench"
        if prefetch:
            if _has_images(self.destination_dir):
                print(f"on prefetch_hf mode, {self.destination_dir} already exists")
            else:
                print(f"Downloading {self.split} split from Hugging Face...")
                self._download_data()
        if self.split != "train":
            self.unique_folder_names += [d.name for d in self.destination_dir.iterdir() if d.is_dir()] if self.destination_dir.exists() else []
        self.images, self.labels, self.folder_names = self._load_data()

    @staticmethod
    def _split_has_images(directory: Path) -> bool:
        return _has_images(directory)

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
            _safe_extract_zip(local_zip, destination_dir)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        splits: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test"),
        repo_id: str = "HorizonTEL/AIGIBench",
        force: bool = False,
    ) -> None:
        """Download and extract AIGIBench split archives under root_dir/train, root_dir/val, root_dir/test."""
        root_dir = Path(root_dir)
        for split in splits:
            cls.prepare_split(
                destination_dir=root_dir / split,
                split=split,
                repo_id=repo_id,
                force=force,
            )

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
        return _iter_images(directory)

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


class DiTFakeDataset(Dataset):
    repo_id = "lioooox/DiTFake"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["test"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        repo_id: str | None = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.repo_id = repo_id or self.repo_id
        self.destination_dir = self.root_dir / self.split
        self._normalize_existing_layout()

        if prefetch and not _has_images(self.destination_dir):
            print("Downloading DiTFake test split from Hugging Face...")
            self._download_data()
            self.destination_dir = self.root_dir / self.split
            self._normalize_existing_layout()

        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    def _normalize_existing_layout(self) -> None:
        nested_split_dir = self.root_dir / "DiTFake" / self.split
        if _has_images(nested_split_dir) and not _has_images(self.destination_dir):
            if self.destination_dir.exists():
                shutil.rmtree(self.destination_dir)
            self.destination_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(nested_split_dir), str(self.destination_dir))
        nested_root = self.root_dir / "DiTFake"
        if nested_root.exists() and not any(nested_root.rglob("*")):
            shutil.rmtree(nested_root, ignore_errors=True)

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        repo_id: str = "lioooox/DiTFake",
        force: bool = False,
    ) -> None:
        """Download one DiTFake split subtree into root_dir/split."""
        from huggingface_hub import snapshot_download

        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if not force and _has_images(destination_dir):
            return

        root_dir.mkdir(parents=True, exist_ok=True)
        download_dir = root_dir / "_downloads" / "DiTFake"
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=download_dir,
            allow_patterns=[f"DiTFake/{split}/**"],
        )
        downloaded_split_dir = download_dir / "DiTFake" / split
        if downloaded_split_dir.exists() and downloaded_split_dir != destination_dir:
            if destination_dir.exists() and (force or not _has_images(destination_dir)):
                shutil.rmtree(destination_dir)
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded_split_dir), str(destination_dir))
        shutil.rmtree(root_dir / "_downloads", ignore_errors=True)
        nested_root = root_dir / "DiTFake"
        if nested_root.exists():
            shutil.rmtree(nested_root, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        repo_id: str = "lioooox/DiTFake",
        force: bool = False,
    ) -> None:
        """Download DiTFake test split under root_dir."""
        cls.prepare_split(root_dir=root_dir, split="test", repo_id=repo_id, force=force)

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            repo_id=self.repo_id,
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
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")

    def _load_data(self):
        images, labels, folder_names = [], [], []
        for filepath in _iter_images(self.destination_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name)
        return images, labels, folder_names



class FreqNetDetectionDataset(Dataset):
    google_drive_folder_url = "https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["test"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        source_url: str | None = None,
    ):
        super().__init__()
        if split != "test":
            raise ValueError("FreqNetDetectionDataset only supports the test split.")
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.prefetch = prefetch
        self.source_url = source_url or self.google_drive_folder_url
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print("Downloading FreqNetDetection test split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    @classmethod
    def download_archives(
        cls,
        archive_dir: str | Path,
        url: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download the shared FreqNetDetection Google Drive folder into archive_dir."""
        archive_dir = Path(archive_dir)
        if archive_dir.exists() and any(archive_dir.iterdir()) and not force:
            return archive_dir

        if force:
            shutil.rmtree(archive_dir, ignore_errors=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "Downloading the FreqNetDetection Google Drive folder requires `gdown`. "
                "Install it with `pip install gdown`."
            ) from exc

        gdown.download_folder(
            url=url or cls.google_drive_folder_url,
            output=str(archive_dir),
            quiet=False,
            use_cookies=False,
        )
        return archive_dir

    @staticmethod
    def _archive_paths(archive_dir: Path) -> list[Path]:
        archive_suffixes = (".tar.gz", ".tgz", ".tar")
        return sorted(
            p
            for p in archive_dir.rglob("*")
            if p.is_file() and p.name.lower().endswith(archive_suffixes)
        )

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        source_url: str | None = None,
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download FreqNetDetection generator archives and extract them under root_dir/test."""
        if split != "test":
            raise ValueError("FreqNetDetectionDataset only supports the test split.")

        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if _has_images(destination_dir) and not force:
            return
        if force:
            shutil.rmtree(destination_dir, ignore_errors=True)

        archive_dir = root_dir / "_archives" / "FreqNetDetection"
        cls.download_archives(archive_dir, url=source_url, force=force)
        archive_paths = cls._archive_paths(archive_dir)
        if not archive_paths:
            raise RuntimeError(f"No .tar archives found for FreqNetDetection in {archive_dir}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        for archive_path in archive_paths:
            _safe_extract_tar(archive_path, destination_dir)

        if not _has_images(destination_dir):
            raise RuntimeError("FreqNetDetection archives did not contain recognizable images.")
        if not keep_archives:
            shutil.rmtree(archive_dir, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        source_url: str | None = None,
        force: bool = False,
        keep_archives: bool = True,
    ) -> None:
        """Download FreqNetDetection as a test-only dataset under root_dir/test."""
        cls.prepare_split(
            root_dir=root_dir,
            split="test",
            source_url=source_url,
            force=force,
            keep_archives=keep_archives,
        )

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            source_url=self.source_url,
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
    def _get_label(filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")

    def _load_data(self):
        images, labels, folder_names = [], [], []
        for filepath in _iter_images(self.destination_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name)
        return images, labels, folder_names


class UniversialFakeDetectionDataset(Dataset):
    source_url = "https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["test"],
        transform: AIGITransformations | None = None,
        prefetch: bool = False,
        source_url: str | None = None,
    ):
        super().__init__()
        if split != "test":
            raise ValueError("UniversialFakeDetectionDataset only supports the test split.")
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.prefetch = prefetch
        self.source_url = source_url or self.source_url
        self.destination_dir = self.root_dir / self.split
        if prefetch and not _has_images(self.destination_dir):
            print(f"Downloading {self.split} split from Google Drive...")
            self._download_data()
        self.images, self.labels, self.folder_names = self._load_data()
        self.unique_folder_names = sorted(set(self.folder_names)) or [self.split]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = decode_image(str(self.images[idx]), mode=ImageReadMode.RGB)
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label, self.unique_folder_names.index(self.folder_names[idx])

    @staticmethod
    def _flatten_extracted_tree(extract_dir: Path, destination_dir: Path) -> None:
        source_dir = extract_dir / "diffusion_models"
        if not source_dir.exists():
            nested = [p for p in extract_dir.rglob("diffusion_models") if p.is_dir()]
            source_dir = nested[0] if nested else extract_dir
        container_names = {"diffusion_datasets", "diffusion_dataset", "datasets", "data"}
        child_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
        if len(child_dirs) == 1 and child_dirs[0].name.lower() in container_names:
            source_dir = child_dirs[0]

        destination_dir.mkdir(parents=True, exist_ok=True)
        for child in source_dir.iterdir():
            target = destination_dir / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))

    @classmethod
    def prepare_split(
        cls,
        root_dir: str | Path,
        split: Literal["test"] = "test",
        source_url: str | None = None,
        force: bool = False,
        keep_downloads: bool = False,
    ) -> None:
        """Download the Universal Fake Detection zip and flatten diffusion_models under root_dir/test."""
        if split != "test":
            raise ValueError("UniversialFakeDetectionDataset only supports the test split.")

        root_dir = Path(root_dir)
        destination_dir = root_dir / split
        if _has_images(destination_dir) and not force:
            return
        if force:
            shutil.rmtree(destination_dir, ignore_errors=True)

        import gdown

        download_dir = root_dir / "_downloads" / "UniversialFakeDetection"
        extract_dir = download_dir / "extracted"
        zip_path = download_dir / "universial_fake_detection.zip"
        shutil.rmtree(download_dir, ignore_errors=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        download_source = source_url or cls.source_url
        file_id_match = re.search(r"/d/([^/]+)", download_source) or re.search(r"[?&]id=([^&]+)", download_source)
        download_kwargs = {"id": file_id_match.group(1)} if file_id_match else {"url": download_source}
        gdown.download(
            output=str(zip_path),
            quiet=False,
            **download_kwargs,
        )
        _safe_extract_zip(zip_path, extract_dir)
        cls._flatten_extracted_tree(extract_dir, destination_dir)

        if not _has_images(destination_dir):
            raise RuntimeError(
                "Universal Fake Detection download did not contain images after "
                "flattening diffusion_models into the test split."
            )
        if not keep_downloads:
            shutil.rmtree(download_dir, ignore_errors=True)

    @classmethod
    def prepare_dataset(
        cls,
        root_dir: str | Path,
        source_url: str | None = None,
        force: bool = False,
        keep_downloads: bool = False,
    ) -> None:
        """Download Universal Fake Detection as a test-only dataset under root_dir/test."""
        cls.prepare_split(
            root_dir=root_dir,
            split="test",
            source_url=source_url,
            force=force,
            keep_downloads=keep_downloads,
        )

    def _load_data(self):
        images, labels, folder_names = [], [], []
        for filepath in _iter_images(self.destination_dir):
            images.append(filepath)
            labels.append(self._get_label(filepath))
            folder_names.append(filepath.parent.parent.name)
        return images, labels, folder_names

    def _get_label(self, filepath: Path) -> int:
        name = filepath.parent.name.lower()
        if "real" in name:
            return 0
        if "fake" in name or "false" in name:
            return 1
        raise ValueError(f"Cannot determine label from filename: {filepath}")

    def _download_data(self, force: bool = False) -> None:
        self.prepare_split(
            root_dir=self.root_dir,
            split=self.split,
            source_url=self.source_url,
            force=force,
        )

class AIGIBenchDatasetModule:
    def __init__(self, config):
        self.config = config
        train_list = list(config.DatasetConfig.train_transform_list)
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.train_transform = AIGITransformations(train_list) if len(train_list) > 0 else None
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        self.train_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "train", config.DatasetConfig.setting, self.train_transform, prefetch=prefetch)
        self.val_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "val", config.DatasetConfig.setting, self.eval_transform, prefetch=prefetch)
        self.test_dataset = AIGIBenchDataset(config.DatasetConfig.root_dir, "test", config.DatasetConfig.setting, self.eval_transform, prefetch=prefetch)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.DatasetConfig.train_batch_size, shuffle=True, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.DatasetConfig.val_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class DiTFakeDatasetModule:
    def __init__(self, config):
        self.config = config
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        repo_id = config.DatasetConfig.get("repo_id", DiTFakeDataset.repo_id)
        self.test_dataset = DiTFakeDataset(config.DatasetConfig.root_dir, "test", self.eval_transform, prefetch=prefetch, repo_id=repo_id)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class FreqNetDetectionDatasetModule:
    def __init__(self, config):
        self.config = config
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        source_url = config.DatasetConfig.get("source_url", FreqNetDetectionDataset.google_drive_folder_url)
        self.test_dataset = FreqNetDetectionDataset(
            config.DatasetConfig.root_dir,
            "test",
            self.eval_transform,
            prefetch=prefetch,
            source_url=source_url,
        )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)


class UniversialFakeDetectionDatasetModule:
    def __init__(self, config):
        self.config = config
        eval_list = list(config.DatasetConfig.eval_transform_list)
        self.eval_transform = AIGITransformations(eval_list) if len(eval_list) > 0 else None
        prefetch = getattr(config.DatasetConfig, "prefetch", False)
        source_url = config.DatasetConfig.get("source_url", UniversialFakeDetectionDataset.source_url)
        self.test_dataset = UniversialFakeDetectionDataset(
            config.DatasetConfig.root_dir,
            "test",
            self.eval_transform,
            prefetch=prefetch,
            source_url=source_url,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.DatasetConfig.test_batch_size, shuffle=False, num_workers=self.config.DatasetConfig.num_workers, persistent_workers=self.config.DatasetConfig.persistent_workers, pin_memory=self.config.DatasetConfig.pin_memory)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None


class AIGIDatasetModule:
    def __init__(self, config):
        self.delegate = AGIDModuleDict[config.DatasetConfig.name](config)

    def train_dataloader(self):
        return self.delegate.train_dataloader()

    def val_dataloader(self):
        return self.delegate.val_dataloader()

    def test_dataloader(self):
        return self.delegate.test_dataloader()

AGIDModuleDict = {
    "aigibench": AIGIBenchDatasetModule,
    "cnndetection": CNNDetectionDatasetModule,
    "ditfake": DiTFakeDatasetModule,
    "freqnet": FreqNetDetectionDatasetModule,
    "freqnetdetection": FreqNetDetectionDatasetModule,
    "genimage": GenImageDatasetModule,
    "universialfakedetection": UniversialFakeDetectionDatasetModule,
}


if __name__ == "__main__":
    config = compose_config()
    dataset_module = AGIDModuleDict[config.DatasetConfig.name](config)
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
