"""Build docs/WILD_BENCHMARKS.md: every image detector on the recent / in-the-wild benchmarks.

Reads repro/bench_results/<model>_<dataset>.json (written by benchmark/bench_eval.py, one per run, each traceable to
its job log) and writes one table of balanced accuracy at 0.5 and AUROC per detector and dataset. Rerun after new
results land:

    python repro/build_wild_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "repro" / "bench_results"
OUT = ROOT / "docs" / "WILD_BENCHMARKS.md"

DATASETS = [  # (config name, column title, what the set is, caveat)
    ("realchain", "RealChain", "7 recent generators (FLUX.1, SD 3.5, Qwen-Image, Hunyuan 3, NanoBanana, Seedream 4, "
     "image-to-image) vs 7000 reals, everything JPEG q60", "the 7000 unpaired reals are split into 7 slices of 1000"),
    ("evalgen", "EvalGEN", "Flux, GoT, Infinity, OmniGen, NOVA on GenEval prompts, about 11k fakes each",
     "fakes only; paired with MSCOCO val2017 reals (our choice)"),
    ("synthwildx", "SynthWildX", "DALL-E 3, Firefly, Midjourney v5 images posted on X, vs real X images",
     "1713 of 2000 images are still online"),
    ("wildrf", "WildRF", "real and fake images from Facebook, Reddit and Twitter", "full test split, 1251 / 1252"),
    ("bfree_online", "BFree-Online", "real and fake images that went viral, several web copies each",
     "807 of about 1400 images are still online"),
    ("rrdataset", "RRDataset", "real vs AI images as created, after sharing online and after re-digitization",
     "test archive only; three scenarios, about 8500 / 8500 each"),
]

# Display names, in the order of the README's detector table.
DETECTORS = [("cure", "CuRe"), ("spai", "SPAI"), ("dinov3", "DINOv3 probe"), ("co_spy", "Co-Spy"), ("aide", "AIDE"),
             ("vib_net", "VIB-Net"), ("pgc", "PGC"), ("npr", "NPR"), ("safe", "SAFE"), ("fat_former", "FatFormer"),
             ("clip", "UnivFD / CLIP"), ("cnn", "CNNDetection"), ("dinov2", "DINOv2 probe"),
             ("adapted_dino", "Adapted DINOv2")]


def load(model, dataset):
    p = RESULTS / f"{model}_{dataset}.json"
    return json.loads(p.read_text())["means"] if p.is_file() else None


def cell(m):
    if m is None:
        return "…"
    text = f"{m['acc']:.1f} / {m['mauroc']:.1f}"
    return f"**{text}**" if m["acc"] >= 75 else text


def main():
    rows = []
    for key, name in DETECTORS:
        ms = [load(key, d) for d, *_ in DATASETS]
        done = [m for m in ms if m]
        mean = f"{sum(m['acc'] for m in done) / len(done):.1f}" if len(done) == len(DATASETS) else "…"
        rows.append((name, ms, mean))
    rows.sort(key=lambda r: -float(r[2]) if r[2] != "…" else 0)

    lines = [
        "# Detectors on recent and in-the-wild benchmarks",
        "",
        "Every SynthScope image detector, with its default (published) weights, on six benchmarks built from recent",
        "commercial generators or from images collected online. Each cell is **balanced accuracy at a 0.5 threshold /",
        "AUROC**, both the unweighted mean over the dataset's subsets (generators or platforms), in percent; bold marks",
        "a balanced accuracy of 75 or more. `…` means the run has not finished yet.",
        "",
        "All six datasets download and lay themselves out on first use, for example",
        "`python benchmark/bench_eval.py --model cure --dataset realchain --root /data/RealChain`.",
        "Each number comes from `repro/bench_results/<model>_<dataset>.json`; regenerate this page with",
        "`python repro/build_wild_table.py`.",
        "",
        "| Detector | " + " | ".join(t for _, t, *_ in DATASETS) + " | Mean acc |",
        "| --- | " + " | ".join("---:" for _ in DATASETS) + " | ---: |",
    ]
    for name, ms, mean in rows:
        lines.append(f"| {name} | " + " | ".join(cell(m) for m in ms) + f" | {mean} |")
    lines += [
        "",
        "## The datasets",
        "",
        "| Dataset (`config name`) | What it is | Caveat |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {t} (`{d}`) | {what} | {caveat} |" for d, t, what, caveat in DATASETS]
    lines += [
        "",
        "## Reading the table",
        "",
        "- Only CuRe's paper reports these benchmarks, and CuRe reproduces it: RealChain 79.69 vs 79.7, EvalGEN 98.14 vs",
        "  98.1, SynthWildX 92.69 vs 92.7, WildRF 95.89 vs 95.8, RRDataset 82.04 vs 81.5, BFree-Online 91.64 vs 90.9 (fewer images; see",
        "  [the CuRe README](../models/image/cure/README.md)). The other detectors were never evaluated on these sets by",
        "  their authors, so their rows are new measurements, not reproductions.",
        "- Balanced accuracy uses a fixed 0.5 threshold. A detector with a high AUROC but a balanced accuracy near 50",
        "  separates real from fake but is miscalibrated for this data (it calls nearly everything one class).",
        "- An AUROC well below 50 means the detector ranks real images as more fake than the generated ones. We traced",
        "  the same effect for NPR on CommGen15 to how the real and fake images were stored (JPEG reals vs PNG fakes,",
        "  different sizes), not to a bug ([write-up](../repro/investigate/npr_commgen15.md)).",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}: {sum(m is not None for _, ms, _ in rows for m in ms)} of "
          f"{len(rows) * len(DATASETS)} cells filled")


if __name__ == "__main__":
    main()
