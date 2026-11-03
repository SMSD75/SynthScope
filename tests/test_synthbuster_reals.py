"""Checks where the Synthbuster loader gets its real images from (no network: the Zenodo zip is built locally).

Synthbuster ships fakes only, so the loader pairs every generator with a pool of real photos:

* an explicit ``real_pool`` is used,
* otherwise the sibling GenImage folder ``<root>/../GenImage/test/real``,
* otherwise it raises a clear error before downloading, instead of laying out fakes only,
* a legacy fakes-only layout (generator folders with ``1_fake`` but no ``0_real``) is completed, not accepted,
* a complete layout is left alone,
* ``ensure_dataset`` (bench_eval ``--dataset``) repairs a fakes-only layout instead of returning it,
* ``DatasetConfig.real_pool`` reaches ``prepare_split`` through ``SynthbusterDatasetModule``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

from config_utils import compose_config
from datasets_util import ensure_dataset
from dataset import SynthbusterDataset, SynthbusterDatasetModule

GENERATORS = ("dalle2", "sd14")


def _image(path: Path, shade: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(shade, 20, 30)).save(path, format="PNG")


def _real_pool(directory: Path, n: int) -> Path:
    for i in range(n):
        _image(directory / f"real_{i}.png", shade=200)
    return directory


def _synthbuster_root(parent: Path, n_fake: int = 3) -> Path:
    """``<parent>/Synthbuster`` holding the Zenodo zip (synthbuster/<gen>/*.png), so nothing is downloaded."""
    root = parent / "Synthbuster"
    src = parent / "zip_src" / "synthbuster"
    for gen in GENERATORS:
        for i in range(n_fake):
            _image(src / gen / f"{i:03d}.png")
    (root / "_downloads").mkdir(parents=True)
    with zipfile.ZipFile(root / "_downloads" / SynthbusterDataset.zenodo_file, "w") as z:
        for p in src.rglob("*.png"):
            z.write(p, p.relative_to(src.parent))
    return root


def _count(directory: Path) -> int:
    return sum(1 for p in directory.glob("*") if p.is_file())


def _assert_paired(root: Path, n_fake: int, n_real: int) -> None:
    for gen in GENERATORS:
        assert _count(root / "test" / gen / "1_fake") == n_fake, gen
        assert _count(root / "test" / gen / "0_real") == n_real, gen


def test_explicit_pool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = _synthbuster_root(tmp)
        pool = _real_pool(tmp / "elsewhere" / "reals", 5)
        SynthbusterDataset.prepare_split(root, real_pool=str(pool), cap=4)
        _assert_paired(root, n_fake=3, n_real=4)      # the first 4 sorted reals (cap), 3 fakes
        names = sorted(p.name for p in (root / "test" / "sd14" / "0_real").iterdir())
        assert names == [f"{i:05d}_real_{i}.png" for i in range(4)], names


def test_sibling_genimage_pool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = _synthbuster_root(tmp)
        _real_pool(tmp / "GenImage" / "test" / "real", 3)
        SynthbusterDataset.prepare_split(root)
        _assert_paired(root, n_fake=3, n_real=3)


def test_no_pool_raises_before_downloading() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "Synthbuster"                     # no zip and no pool: a download would hit the network
        try:
            SynthbusterDataset.prepare_split(root)
        except FileNotFoundError as e:
            msg = str(e)
            assert "no reals" in msg and "real_pool" in msg and "--reals" in msg, msg
        else:
            raise AssertionError("expected a FileNotFoundError without a real pool")
        assert not (root / "test").exists(), "a fakes-only layout must not be written"
        try:                                           # an explicit pool without images is not a pool either
            SynthbusterDataset.prepare_split(root, real_pool=str(tmp / "missing"))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected a FileNotFoundError for an empty real_pool")


def test_legacy_fakes_only_layout_is_completed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "Synthbuster"
        for gen in GENERATORS:
            _image(root / "test" / gen / "1_fake" / "00000_a.png")
        assert SynthbusterDataset._is_prepared(root / "test") is False
        try:
            SynthbusterDataset.prepare_split(root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a fakes-only layout without a pool must raise, not be accepted")
        pool = _real_pool(tmp / "reals", 2)
        SynthbusterDataset.prepare_split(root, real_pool=str(pool))
        _assert_paired(root, n_fake=1, n_real=2)
        assert SynthbusterDataset._is_prepared(root / "test") is True


def test_complete_layout_is_left_alone() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Synthbuster"
        for gen in GENERATORS:
            _image(root / "test" / gen / "1_fake" / "00000_a.png")
            _image(root / "test" / gen / "0_real" / "00000_b.png")
        SynthbusterDataset.prepare_split(root)         # no pool and no zip needed: returns early
        _assert_paired(root, n_fake=1, n_real=1)


def test_ensure_dataset_repairs_incomplete_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "Synthbuster"
        for gen in GENERATORS:
            _image(root / "test" / gen / "1_fake" / "00000_a.png")
        try:                                           # no pool: the clear error, not a silent return
            ensure_dataset("synthbuster", root=str(root))
        except FileNotFoundError as e:
            assert "real_pool" in str(e), e
        else:
            raise AssertionError("ensure_dataset returned a fakes-only layout")
        _real_pool(tmp / "GenImage" / "test" / "real", 2)
        assert ensure_dataset("synthbuster", root=str(root)) == root
        _assert_paired(root, n_fake=1, n_real=2)
        assert ensure_dataset("synthbuster", root=str(root)) == root      # complete now: nothing to do


def test_config_reaches_the_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = _synthbuster_root(tmp)
        pool = _real_pool(tmp / "elsewhere" / "reals", 2)
        cfg = compose_config(["dataset@DatasetConfig=image/synthbuster", f"DatasetConfig.root_dir={root}"])
        assert cfg.DatasetConfig.real_pool is None
        cfg = compose_config(["dataset@DatasetConfig=image/synthbuster", f"DatasetConfig.root_dir={root}",
                              f"DatasetConfig.real_pool={pool}"])
        module = SynthbusterDatasetModule(cfg)
        counts = {(f, l): 0 for f in module.test_dataset.unique_folder_names for l in (0, 1)}
        for f, l in zip(module.test_dataset.folder_names, module.test_dataset.labels):
            counts[(f, l)] += 1
        assert all(c > 0 for c in counts.values()), counts
        assert sorted(module.test_dataset.unique_folder_names) == sorted(GENERATORS)
        assert sum(module.test_dataset.labels) == 3 * len(GENERATORS)
        assert len(module.test_dataset) == (3 + 2) * len(GENERATORS)


if __name__ == "__main__":
    os.chdir(ROOT)
    test_explicit_pool()
    print("explicit real_pool: ok")
    test_sibling_genimage_pool()
    print("sibling GenImage pool: ok")
    test_no_pool_raises_before_downloading()
    print("no pool -> clear error, nothing written: ok")
    test_legacy_fakes_only_layout_is_completed()
    print("legacy fakes-only layout completed: ok")
    test_complete_layout_is_left_alone()
    print("complete layout untouched: ok")
    test_ensure_dataset_repairs_incomplete_layout()
    print("ensure_dataset repairs a fakes-only layout: ok")
    test_config_reaches_the_loader()
    print("DatasetConfig.real_pool wired through the module: ok")
    print("Synthbuster real-pool checks passed")
