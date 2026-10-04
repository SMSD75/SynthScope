"""Download and prepare one of SynthScope's benchmark datasets with its built-in loader.

    python benchmark/download_dataset.py --list                       # datasets and where each comes from
    python benchmark/download_dataset.py genimage --root /data/GenImage
    python benchmark/download_dataset.py synthbuster --root /data/Synthbuster --dry-run   # only show what would happen

The loaders download from the dataset's official source (Hugging Face, Zenodo, ModelScope or Google Drive; see
``conf/dataset/image/<name>.yaml``) and restructure it into ``<root>/<generator>/{0_real,1_fake}``, which
``benchmark/bench_eval.py`` reads. Some hosts rate-limit (Google Drive) or ask you to accept terms first; if a download
fails, fetch the data by hand from the ``source_url`` and unpack it into ``--root`` in the same layout.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config_utils import compose_config

SOURCE_KEYS = ("source_url", "hf_repo_id", "zenodo_record", "repo_id", "modelscope_id", "setting")


def available() -> list[Path]:
    return sorted((ROOT / "conf" / "dataset").glob("*/*.yaml"))


def describe(path: Path) -> str:
    """One line: modality/name and where the data comes from."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    src = ", ".join(f"{k}={cfg[k]}" for k in SOURCE_KEYS if k in cfg)
    return f"{path.parent.name}/{path.stem:24s} {src or '(see the yaml)'}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="dataset config name, e.g. genimage, cnndetection, aigibench, synthbuster")
    ap.add_argument("--root", help="where to put the data (default: root_dir from the dataset's yaml)")
    ap.add_argument("--modality", default="image", choices=["image", "video"])
    ap.add_argument("--list", action="store_true", help="list the available datasets and their sources")
    ap.add_argument("--dry-run", action="store_true", help="only print the resolved config; download nothing")
    args = ap.parse_args()

    if args.list or not args.name:
        print("\n".join(describe(p) for p in available()))
        return
    overrides = [f"dataset@DatasetConfig={args.modality}/{args.name}"]
    if args.root:
        overrides.append(f"DatasetConfig.root_dir={args.root}")
    cfg = compose_config(overrides)
    dc = cfg.DatasetConfig
    print(f"dataset {dc.name}: root_dir={dc.root_dir}  prefetch={dc.get('prefetch')}")
    for k in SOURCE_KEYS:
        if k in dc:
            print(f"  {k}: {dc[k]}")
    if args.dry_run:
        return
    from dataset import AIGIDatasetModule
    AIGIDatasetModule(cfg)           # constructing the datasets runs the loader (prefetch: true downloads + restructures)
    print(f"done: {dc.root_dir}")


if __name__ == "__main__":
    main()
