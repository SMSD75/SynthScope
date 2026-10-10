"""Shared helpers for the benchmark scripts: download a dataset with its built-in loader and find where to evaluate it.

``ensure_dataset`` composes the dataset's Hydra config (``conf/dataset/<modality>/<name>.yaml``), and, if the data is not
on disk yet, constructs its dataset module, whose loader (``prefetch: true``) downloads the data and restructures it into
``<generator>/{0_real,1_fake}``. ``find_eval_root`` then locates the folder to evaluate and picks the ``bench_eval.py``
profile (forensynths / ufd / pergen / pool) from the layout.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGE_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_NAMES = {"0_real", "real", "reals"}
FAKE_NAMES = {"1_fake", "fake", "fakes", "ai"}
# datasets whose generator grouping / real pairing is special-cased by a bench_eval profile
PROFILE_OVERRIDE = {"cnndetection": "forensynths", "cospy_subset": "forensynths", "spai_ldm": "forensynths",
                    "universialfakedetection": "ufd"}


def dataset_config(name: str, modality: str = "image", root: str | None = None):
    """The composed Hydra config for ``conf/dataset/<modality>/<name>.yaml`` (``root`` overrides ``root_dir``)."""
    from config_utils import compose_config
    conf = ROOT / "conf" / "dataset"
    if not (conf / modality / f"{name}.yaml").is_file():
        other = [m for m in ("image", "video") if (conf / m / f"{name}.yaml").is_file()]
        if other:
            raise SystemExit(f"{name} is a {other[0]} dataset; use the {other[0]} evaluator "
                             f"({'benchmark/bench_eval_video.py' if other[0] == 'video' else 'benchmark/bench_eval.py'})")
        known = sorted(p.stem for p in (conf / modality).glob("*.yaml"))
        raise SystemExit(f"unknown {modality} dataset '{name}'. Available: {', '.join(known)}")
    overrides = [f"dataset@DatasetConfig={modality}/{name}"]
    if root:
        overrides.append(f"DatasetConfig.root_dir={root}")
    return compose_config(overrides)


def has_images(root: Path) -> bool:
    """True as soon as one image file is found under ``root`` (cheap: stops at the first hit)."""
    if not root.is_dir():
        return False
    for _, _, files in os.walk(root):
        if any(os.path.splitext(f)[1].lower() in IMAGE_EXT for f in files):
            return True
    return False


def download_dataset(name: str, modality: str = "image", root: str | None = None) -> Path:
    """Run the dataset's loader (download + restructure). Returns the root directory."""
    from dataset import AIGIDatasetModule
    cfg = dataset_config(name, modality, root)
    AIGIDatasetModule(cfg)        # constructing the datasets runs the loader (prefetch: true downloads + restructures)
    return Path(cfg.DatasetConfig.root_dir)


def ensure_dataset(name: str, modality: str = "image", root: str | None = None, download: bool = True) -> Path:
    """The dataset's root directory, downloading it first when no images are there yet."""
    cfg = dataset_config(name, modality, root)
    path = Path(cfg.DatasetConfig.root_dir).expanduser()
    if has_images(path):
        return path
    if not download:
        raise SystemExit(f"{name}: no data under {path} (drop --no-download to fetch it, or pass --root)")
    print(f"[dataset] {name}: nothing under {path}, downloading with the built-in loader ...", flush=True)
    return download_dataset(name, modality, root)


def _subdirs(d: Path) -> list[Path]:
    try:
        return sorted(c for c in d.iterdir() if c.is_dir() and not c.name.startswith((".", "_")))
    except OSError:
        return []


def _has_real_fake(d: Path, depth: int = 3) -> bool:
    names = {c.name.lower() for c in d.iterdir() if c.is_dir()} if d.is_dir() else set()
    if names & REAL_NAMES and names & FAKE_NAMES:
        return True
    return depth > 0 and any(_has_real_fake(c, depth - 1) for c in _subdirs(d)[:60])


def find_eval_root(name: str, root: Path) -> tuple[Path, str]:
    """(folder to evaluate, bench_eval profile). Looks in ``root/test``, ``root/test_full`` and ``root`` itself."""
    if name in PROFILE_OVERRIDE:       # fixed layouts handled by their own bench_eval profile (e.g. UFD: fakes and reals in separate folders)
        for cand in (root / "test", root):
            if cand.is_dir() and _subdirs(cand):
                return cand, PROFILE_OVERRIDE[name]
    for cand in (root / "test", root / "test_full", root):
        if not cand.is_dir():
            continue
        names = {c.name.lower() for c in cand.iterdir() if c.is_dir()}
        if names & REAL_NAMES and names & FAKE_NAMES:            # one pooled real/fake set
            return cand, PROFILE_OVERRIDE.get(name, "pool")
        if any(_has_real_fake(g) for g in _subdirs(cand)):       # one folder per generator
            return cand, PROFILE_OVERRIDE.get(name, "pergen")
    raise SystemExit(f"{name}: could not find real/fake folders under {root}; expected "
                     "<root>/[test/]<generator>/{0_real,1_fake} (see docs/BENCHMARKS.md)")
