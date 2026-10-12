"""Build SynthScope's online-demo data pack: a few files per generator from several openly licensed benchmarks, in the
layout the web app reads (<Folder>/test/<generator>/{0_real,1_fake} or EvalCrafter's test/1_fake/<generator>), plus a
README with sources and licenses. Files are picked evenly across each folder (deterministic)."""
import shutil
import sys
from pathlib import Path

S = Path("/ssdstore/ssalehi/SynthScope")
OUT = S / "_demo_pack"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".gif", ".mov"}


def pick(folder: Path, k: int) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXT and not p.name.startswith("._"))
    if len(files) <= k:
        return files
    return [files[round(i * (len(files) - 1) / (k - 1))] for i in range(k)] if k > 1 else files[:1]


def copy(files, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copy2(p, dest / p.name)


PLAN = [  # (folder, per-class count, classes)
    ("CommGen15", 6, ("0_real", "1_fake")),
    ("RRDataset", 12, ("0_real", "1_fake")),
    ("EvalGEN", 8, ("1_fake",)),
    ("VideoPhy", 3, ("1_fake",)),
]
shutil.rmtree(OUT, ignore_errors=True)
for folder, k, classes in PLAN:
    for gen in sorted(p for p in (S / folder / "test").iterdir() if p.is_dir()):
        for c in classes:
            copy(pick(gen / c, k), OUT / folder / "test" / gen.name / c)
for gen in sorted(p for p in (S / "EvalCrafter" / "test" / "1_fake").iterdir() if p.is_dir()):
    copy(pick(gen, 3), OUT / "EvalCrafter" / "test" / "1_fake" / gen.name)

(OUT / "README.md").write_text("""---
license: other
pretty_name: SynthScope online-demo data
---
# SynthScope online-demo data

A small sample of public AI-generated-image and -video benchmarks for the dataset browser of the SynthScope online demo.
A local SynthScope install downloads the full datasets with one command each.

| Folder | Source | License | Sample |
|---|---|---|---|
| CommGen15 | PGC (Zhou et al.), Hugging Face `xiaoyuzhou68/CommGen15` | Apache-2.0 | 6 real + 6 generated per generator |
| RRDataset | "Bridging the Gap Between Ideal and Real-world Evaluation" (ICCV 2025), Zenodo record 14963880 | CC BY 4.0 | 12 real + 12 generated per scenario |
| EvalGEN | DDA (Chen et al., NeurIPS 2025), Hugging Face `Junwei-Xi/EvalGEN` | Apache-2.0 | 8 generated images per generator |
| EvalCrafter | EvalCrafter (Liu et al., CVPR 2024), Hugging Face `RaphaelLiu/EvalCrafter_T2V_Dataset` | Apache-2.0 | 3 generated clips per generator |
| VideoPhy | VideoPhy (Bansal et al., 2024), `videophysics/videophy_test_public` | MIT | 3 generated clips per generator |

All files are unchanged copies from those sources; credit belongs to their authors.
""")
n = sum(1 for p in OUT.rglob("*") if p.is_file())
size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"demo pack: {n} files, {size / 1e6:.0f} MB at {OUT}")
for d in sorted(OUT.iterdir()):
    if d.is_dir():
        print(f"  {d.name}: {sum(1 for p in d.rglob('*') if p.is_file())} files, "
              f"{sum(p.stat().st_size for p in d.rglob('*') if p.is_file()) / 1e6:.0f} MB")
