"""Real-network smoke test for ``CoSpyDataset`` downloading.

This script intentionally downloads only the smallest single
(source_dataset, generator_model) pair from the CO-SPY-Bench HuggingFace
dataset plus the shared real-image zip, runs the restructuring logic,
and verifies the resulting CNNDetection-style layout.

Run manually via:

    HF_HOME=.cache/hf python tests/test_cospy_download_real.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import CoSpyDataset


def _human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:7.2f} {unit}"
        num /= 1024.0
    return f"{num:7.2f} PB"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            continue
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / ".cache" / "cospy_dl_test"),
        help="Root directory for the download/restructure test.",
    )
    parser.add_argument(
        "--source",
        default="textcaps",
        choices=list(CoSpyDataset.default_source_datasets),
    )
    parser.add_argument(
        "--model",
        default="CompVis@ldm-text2im-large-256",
        help="Generator model archive to fetch from the CO-SPY-Bench repo.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the downloaded data in place after the test.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    sources = (args.source,)
    models = (args.model,)

    print(f"[plan] root              = {root}")
    print(f"[plan] sources           = {sources}")
    print(f"[plan] generator_models  = {models}")
    print(f"[plan] keep after test   = {args.keep}")
    print()

    t0 = time.time()
    print("[1/3] running CoSpyDataset.prepare_split (download + extract + restructure)...")
    CoSpyDataset.prepare_split(
        root_dir=root,
        split="test",
        source_datasets=sources,
        generator_models=models,
        include_real=True,
        keep_downloads=False,
    )
    print(f"      done in {time.time() - t0:.1f}s")
    print()

    print("[2/3] inspecting resulting tree...")
    test_dir = root / "test"
    assert test_dir.is_dir(), f"missing test dir: {test_dir}"

    class_name = f"{args.source}{CoSpyDataset.CLASS_SEPARATOR}{args.model}"
    class_dir = test_dir / class_name
    assert class_dir.is_dir(), f"missing class dir: {class_dir}"

    real_dir = class_dir / CoSpyDataset.REAL_LABEL_DIR
    fake_dir = class_dir / CoSpyDataset.FAKE_LABEL_DIR
    assert real_dir.is_dir(), f"missing real dir: {real_dir}"
    assert fake_dir.is_dir(), f"missing fake dir: {fake_dir}"

    real_files = sorted(real_dir.iterdir())
    fake_files = sorted(fake_dir.iterdir())
    assert len(real_files) > 0, "no real images found"
    assert len(fake_files) > 0, "no fake images found"

    pool_dir = test_dir / CoSpyDataset.REAL_POOL_DIR / args.source
    pool_files = sorted(pool_dir.iterdir()) if pool_dir.is_dir() else []
    print(f"      class folder      : {class_dir.relative_to(test_dir)}")
    print(f"      real images       : {len(real_files):6d} (sample: {real_files[0].name})")
    print(f"      fake images       : {len(fake_files):6d} (sample: {fake_files[0].name})")
    print(f"      real pool images  : {len(pool_files):6d}")

    if pool_files:
        a, b = real_files[0], pool_dir / real_files[0].name
        if b.exists():
            same_inode = a.stat().st_ino == b.stat().st_ino
            print(f"      hardlink check    : {'OK (shared inode)' if same_inode else 'FAILED (different inodes)'}")
            assert same_inode, "real image in class dir is NOT a hardlink to the pool image"

    total_size = _dir_size(test_dir)
    print(f"      total disk used   : {_human_bytes(total_size)}")
    print()

    print("[3/3] instantiating CoSpyDataset to verify __getitem__...")
    ds = CoSpyDataset(
        root_dir=root,
        split="test",
        prefetch=False,
        source_datasets=sources,
        generator_models=models,
        include_real=True,
    )
    print(f"      total samples     : {len(ds)}")
    print(f"      unique folders    : {len(ds.unique_folder_names)} -> {ds.unique_folder_names[:3]}{'...' if len(ds.unique_folder_names) > 3 else ''}")

    img0, lbl0, folder_idx0 = ds[0]
    real_idx = next(i for i, l in enumerate(ds.labels) if l == 0)
    fake_idx = next(i for i, l in enumerate(ds.labels) if l == 1)
    img_real, lbl_real, _ = ds[real_idx]
    img_fake, lbl_fake, _ = ds[fake_idx]
    print(f"      sample[0]         : tensor shape={tuple(img0.shape)} label={int(lbl0)} folder='{ds.unique_folder_names[folder_idx0]}'")
    print(f"      sample[real]      : tensor shape={tuple(img_real.shape)} label={int(lbl_real)}")
    print(f"      sample[fake]      : tensor shape={tuple(img_fake.shape)} label={int(lbl_fake)}")

    assert int(lbl_real) == 0
    assert int(lbl_fake) == 1
    assert img_real.ndim == 3 and img_real.shape[0] == 3
    assert img_fake.ndim == 3 and img_fake.shape[0] == 3

    print()
    print("[ok] CoSpyDataset real-download smoke test passed.")

    if not args.keep:
        print(f"[cleanup] removing {root}")
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"[keep] data left at {root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
