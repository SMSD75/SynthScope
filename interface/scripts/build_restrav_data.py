"""Stage the local d3_repro GenVideo data into the VideoFolder layout ReStraV training needs:

    <root>/<split>/genvideo/{0_real,1_fake}/*.mp4         (symlinks)

Sources (already on cn023):
  * train / val : d3_repro/extracted/*.mp4  (flat; filename `<idx>_<gen>_<id>.mp4`,
                  gen == 'real' => real/label 0, else fake/label 1)
  * test        : d3_repro/genvideo_val/GenVideo-Val/{Real, Fake/<gen>}  (GenVideo-Val)

The in-loop ``test`` split is a balanced SUBSAMPLE (videos are re-decoded every epoch, so the
full 18k-clip benchmark would dominate runtime). The full GenVideo-Val also gets staged under
``<root>_fulltest/test`` for a faithful final evaluation (see eval_restrav_full.py).

Single 'genvideo' folder per split: with one folder_id the per-folder AP/AUROC AIGIMetrics
reports equal the global real-vs-fake metric (per-generator folders would be single-class).

  conda run -n AGID python interface/scripts/build_restrav_data.py
Override OUT with AGID_RESTRAV_DIR.
"""
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

EXTRACTED = Path("/hddstore/ssalehi/AGID/d3_repro/extracted")
GENVIDEO_VAL = Path("/hddstore/ssalehi/AGID/d3_repro/genvideo_val/GenVideo-Val")
OUT = Path(os.environ.get("AGID_RESTRAV_DIR", "/hddstore/ssalehi/AGID/restrav_data"))
FULLTEST = Path(str(OUT) + "_fulltest")

N_TRAIN_PER_CLASS = int(os.environ.get("RESTRAV_N_TRAIN", "8000"))
N_VAL_PER_CLASS = int(os.environ.get("RESTRAV_N_VAL", "500"))
N_TEST_PER_CLASS = int(os.environ.get("RESTRAV_N_TEST", "400"))  # balanced in-loop test subsample
SEED = 42
NAME_RE = re.compile(r"^\d+_(.+?)_[^_]+\.mp4$")


def gen_of(name: str) -> str:
    m = NAME_RE.match(name)
    if m:
        return m.group(1)
    stem = name[:-4]
    parts = stem.split("_", 1)
    return parts[1].split("-")[0] if len(parts) > 1 else "unknown"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def stage_train_val():
    reals, fakes = [], []
    for p in sorted(EXTRACTED.glob("*.mp4")):
        (reals if gen_of(p.name) == "real" else fakes).append(p)
    random.Random(SEED).shuffle(reals)
    random.Random(SEED + 1).shuffle(fakes)
    splits = {
        "train": (reals[:N_TRAIN_PER_CLASS], fakes[:N_TRAIN_PER_CLASS]),
        "val": (reals[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS],
                fakes[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS]),
    }
    for split, (rl, fk) in splits.items():
        for p in rl:
            link(p, OUT / split / "genvideo" / "0_real" / p.name)
        for p in fk:
            link(p, OUT / split / "genvideo" / "1_fake" / p.name)
        print(f"{split}: real={len(rl)} fake={len(fk)}")


def _genvideo_real_fake():
    reals = sorted(GENVIDEO_VAL.glob("Real/**/*.mp4"))
    fakes = []
    for gen_dir in sorted((GENVIDEO_VAL / "Fake").iterdir()):
        if gen_dir.is_dir():
            fakes += sorted(gen_dir.glob("**/*.mp4"))
    return reals, fakes


def stage_test():
    reals, fakes = _genvideo_real_fake()
    # full test (faithful eval)
    for p in reals:
        link(p, FULLTEST / "test" / "genvideo" / "0_real" / p.name)
    for p in fakes:
        link(p, FULLTEST / "test" / "genvideo" / "1_fake" / p.name)
    print(f"fulltest: real={len(reals)} fake={len(fakes)}")
    # balanced subsample for the in-loop test
    random.Random(SEED + 2).shuffle(reals)
    random.Random(SEED + 3).shuffle(fakes)
    for p in reals[:N_TEST_PER_CLASS]:
        link(p, OUT / "test" / "genvideo" / "0_real" / p.name)
    for p in fakes[:N_TEST_PER_CLASS]:
        link(p, OUT / "test" / "genvideo" / "1_fake" / p.name)
    print(f"test(subsample): real={min(len(reals),N_TEST_PER_CLASS)} fake={min(len(fakes),N_TEST_PER_CLASS)}")


def main():
    for d in (OUT, FULLTEST):
        if d.exists():
            shutil.rmtree(d)
    stage_train_val()
    stage_test()
    print("staged into", OUT, "and", FULLTEST)


if __name__ == "__main__":
    main()
