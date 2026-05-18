"""Smoke test for CoSpyDataset / CoSpyDatasetModule integration.

Builds a fake staging tree that mimics the post-download CO-SPY-Bench
layout, then exercises:

* Hydra composition of ``conf/dataset/cospy.yaml``.
* ``CoSpyDataset._restructure_to_cnndet_layout`` -- verifies the final
  on-disk tree is identical in shape to CNNDetection (``<class>/
  {0_real,1_fake}/*.png``), with hardlinked reals shared across all
  models of a source dataset.
* The CNNDetection-style ``_get_label`` / ``_load_data`` parses the
  resulting tree correctly: 110 unique ``<source>__<model>`` folder
  names, real images labelled 0, fake images labelled 1.
* ``CoSpyDatasetModule`` returns a working ``test_dataloader`` and
  ``None`` for train / val (test-only convention).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import compose_config
from dataset import CoSpyDataset, CoSpyDatasetModule


def _make_dummy_image(path: Path, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(123, 45, 67)).save(path, format="PNG")


def _populate_staging_tree(
    root: Path,
    sources: tuple[str, ...],
    models: tuple[str, ...],
    n_fake_per_model: int = 2,
    n_real_per_source: int = 3,
) -> tuple[Path, Path, int, int]:
    """Lay out a fake post-download staging tree under ``root``.

    Returns (synthetic_root, real_root, n_fake_total, n_real_total)
    where the tree shape matches what ``prepare_split`` produces just
    before ``_restructure_to_cnndet_layout`` runs.
    """

    synthetic_root = root / "_downloads" / "Co-Spy-Bench"
    real_root = root / "_downloads" / "Co-Spy-Misc" / "extracted" / "real_image_examples"

    n_fake_total = 0
    for source in sources:
        for model in models:
            for i in range(n_fake_per_model):
                _make_dummy_image(synthetic_root / source / model / f"fake_{i}.png")
                n_fake_total += 1
            (synthetic_root / source / model / "metadata.json").write_text("{}", encoding="utf-8")

    n_real_total = 0
    for source in sources:
        for i in range(n_real_per_source):
            _make_dummy_image(real_root / source / f"real_{i}.png")
            n_real_total += 1

    return synthetic_root, real_root, n_fake_total, n_real_total


def test_hydra_compose() -> None:
    print("[1] Composing conf/dataset/cospy.yaml ...")
    cfg = compose_config(
        overrides=[
            "dataset@DatasetConfig=cospy",
            "DatasetConfig.root_dir=/tmp/__cospy_dummy__",
            "DatasetConfig.prefetch=false",
        ]
    )
    assert cfg.DatasetConfig.name == "cospy"
    assert cfg.DatasetConfig.repo_id == CoSpyDataset.repo_id
    assert cfg.DatasetConfig.real_repo_id == CoSpyDataset.real_repo_id
    assert len(cfg.DatasetConfig.source_datasets) == len(CoSpyDataset.default_source_datasets)
    assert len(cfg.DatasetConfig.generator_models) == len(CoSpyDataset.default_generator_models)
    print("    OK")


def test_restructure_and_load() -> None:
    print("[2] Restructuring a fake staging tree into CNNDetection layout ...")
    sources = ("cc3m", "mscoco")
    models = (
        "CompVis@stable-diffusion-v1-4",
        "black-forest-labs@FLUX.1-dev",
    )
    n_fake_per_model = 2
    n_real_per_source = 3

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        synthetic_root, real_root, expected_fake, expected_real = _populate_staging_tree(
            root, sources=sources, models=models,
            n_fake_per_model=n_fake_per_model, n_real_per_source=n_real_per_source,
        )
        destination_dir = root / "test"

        CoSpyDataset._restructure_to_cnndet_layout(
            synthetic_root=synthetic_root,
            real_root=real_root,
            destination_dir=destination_dir,
            sources=sources,
            models=models,
        )

        sep = CoSpyDataset.CLASS_SEPARATOR
        for source in sources:
            for model in models:
                class_dir = destination_dir / f"{source}{sep}{model}"
                assert (class_dir / CoSpyDataset.FAKE_LABEL_DIR).is_dir(), class_dir
                assert (class_dir / CoSpyDataset.REAL_LABEL_DIR).is_dir(), class_dir
                fakes = sorted((class_dir / CoSpyDataset.FAKE_LABEL_DIR).iterdir())
                reals = sorted((class_dir / CoSpyDataset.REAL_LABEL_DIR).iterdir())
                assert len(fakes) == n_fake_per_model, (class_dir, fakes)
                assert len(reals) == n_real_per_source, (class_dir, reals)

        print("    layout matches CNNDetection (110-style)")

        print("[3] Verifying real images are hardlinked, not duplicated ...")
        first = destination_dir / f"{sources[0]}{sep}{models[0]}" / CoSpyDataset.REAL_LABEL_DIR / "real_0.png"
        second = destination_dir / f"{sources[0]}{sep}{models[1]}" / CoSpyDataset.REAL_LABEL_DIR / "real_0.png"
        assert first.exists() and second.exists()
        assert first.stat().st_ino == second.stat().st_ino, (
            "real images should share an inode via os.link"
        )
        print(f"    inode shared across models: {first.stat().st_ino}")

        print("[4] Iterating CoSpyDataset over the restructured tree ...")
        ds = CoSpyDataset(
            root_dir=root,
            split="test",
            transform=None,
            prefetch=False,
            source_datasets=list(sources),
            generator_models=list(models),
        )
        expected_total = expected_fake + (n_real_per_source * len(sources) * len(models))
        assert len(ds) == expected_total, f"len mismatch: {len(ds)} vs {expected_total}"
        n_real = sum(1 for label in ds.labels if label == 0)
        n_fake = sum(1 for label in ds.labels if label == 1)
        assert n_fake == expected_fake, f"fake mismatch: {n_fake} vs {expected_fake}"
        assert n_real == n_real_per_source * len(sources) * len(models)
        expected_folders = sorted(
            f"{source}{sep}{model}" for source in sources for model in models
        )
        assert ds.unique_folder_names == expected_folders, (
            f"folders wrong: {ds.unique_folder_names} vs {expected_folders}"
        )
        print(f"    folders ({len(expected_folders)}): {expected_folders}")

        print("[5] Pulling one item via __getitem__ ...")
        image, label, folder_id = ds[0]
        assert isinstance(image, torch.Tensor) and image.dim() == 3
        assert label.item() in (0, 1)
        assert 0 <= folder_id < len(expected_folders)
        print(
            f"    image.shape = {tuple(image.shape)}, label = {int(label)}, "
            f"folder = {expected_folders[folder_id]}"
        )

        print("[6] Building CoSpyDatasetModule and iterating one batch ...")
        cfg = compose_config(
            overrides=[
                "dataset@DatasetConfig=cospy",
                f"DatasetConfig.root_dir={root}",
                "DatasetConfig.prefetch=false",
                "DatasetConfig.test_batch_size=4",
                "DatasetConfig.num_workers=0",
                "DatasetConfig.persistent_workers=false",
                f"DatasetConfig.source_datasets=[{','.join(sources)}]",
                f"DatasetConfig.generator_models=[{','.join(models)}]",
            ]
        )
        module = CoSpyDatasetModule(cfg)
        assert module.train_dataloader() is None
        assert module.val_dataloader() is None
        loader = module.test_dataloader()
        batch = next(iter(loader))
        images, labels, folder_ids = batch
        assert images.dim() == 4 and images.shape[0] <= 4
        assert images.dtype == torch.float32
        assert labels.shape == (images.shape[0],)
        assert folder_ids.shape == (images.shape[0],)
        print(
            f"    batch -> images {tuple(images.shape)}, labels {labels.tolist()}, "
            f"folder_ids {folder_ids.tolist()}"
        )

    print("\nAll CO-SPY dataset smoke checks passed.")


def main() -> None:
    test_hydra_compose()
    test_restructure_and_load()


if __name__ == "__main__":
    main()
