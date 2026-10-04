"""Publish a trained checkpoint into the central directory under its canonical name.

The trainer writes ``model_<epoch>.pth`` to the staging dir on each val-loss improvement
(with save_policy=min the last/highest-epoch file is the best). This copies that file to
``<key>.pth`` so the interface registry picks it up, then refreshes the manifest.

  conda run -n SynthScope python interface/scripts/promote_checkpoint.py <key> [--epoch N] [--copy]
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import config              # noqa: E402
from backend.registry import get_spec   # noqa: E402


def best_epoch_file(stage_dir: Path, epoch: int | None) -> Path:
    files = list(stage_dir.glob("model_*.pth"))
    if not files:
        raise FileNotFoundError(f"No model_*.pth in {stage_dir}")
    if epoch is not None:
        f = stage_dir / f"model_{epoch}.pth"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found")
        return f
    # Most recently written = the current run's best (save_policy=min writes the best last).
    # Use mtime, NOT the epoch number, so stale higher-epoch checkpoints left in the staging dir
    # by a *previous* run (possibly a different architecture) are never picked.
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    spec = get_spec(args.key)
    if spec.checkpoint is None:
        print(f"{args.key} is training-free; nothing to promote.")
        return
    stage = config.CHECKPOINTS_DIR / "_train" / args.key
    src = best_epoch_file(stage, args.epoch)
    dst = config.CHECKPOINTS_DIR / spec.checkpoint
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if args.copy:
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")
    else:
        dst.symlink_to(src)
        print(f"linked {dst} -> {src}")


if __name__ == "__main__":
    main()
