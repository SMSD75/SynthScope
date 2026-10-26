"""Dataset explorer backend: discover on-disk datasets, sample random batches, serve
thumbnails, and report which dataset each detector was trained on.

The dataset *configs* (conf/dataset/*.yaml) point at paths that are partly stale on the GPU
node (the /nvmestore roots are not mounted on cn023, and the video sets live in ad-hoc repro
dirs). So this module keeps an explicit registry of candidate roots and auto-detects which
ones actually exist, rather than trusting the configs. Labels are inferred from path
components (``0_real``/``real``/``nature`` -> real, ``1_fake``/``fake``/``ai`` -> fake), which
covers every on-disk layout (ForenSynths ``<gen>/{0_real,1_fake}``, GenImage ``{ai,nature}``,
GenVideo ``Real`` + ``Fake/<gen>``, ReStraV ``genvideo/{0_real,1_fake}``).
"""
from __future__ import annotations

import os
import _synthscope_env  # noqa: F401  (maps legacy AGID_* variables to SYNTHSCOPE_*)
import random
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Directory-name tokens that identify a sample's ground-truth label.
REAL_TOKENS = {"0_real", "real", "reals", "nature", "natural", "coco", "imagenet"}
FAKE_TOKENS = {"1_fake", "fake", "fakes", "ai", "generated", "gen", "synthesis"}

# How many paths to keep in the in-memory pool per label (reservoir-sampled over the full tree)
# and the hard cap on tree entries scanned, so sampling stays fast on 100k+ -image datasets.
POOL_PER_LABEL = 4000
SCAN_CAP = 400_000


@dataclass
class DatasetDef:
    key: str
    name: str
    modality: str                 # "image" | "video"
    roots: list[str]              # candidate roots; first existing one wins
    splits: list[str] = field(default_factory=lambda: ["train", "val", "test"])
    note: str = ""


# Explicit registry (auto-detected against disk at request time).
DATASET_DEFS: list[DatasetDef] = [
    DatasetDef("cnndetection", "CNNDetection / ForenSynths (ProGAN)", "image",
               ["/hddstore/ssalehi/SynthScope/CNNDetection"], ["train", "val", "test"],
               "ProGAN-family GAN images; the ForenSynths benchmark."),
    DatasetDef("genimage", "GenImage (SD-v1.4)", "image",
               ["/hddstore/ssalehi/SynthScope/GenImage"], ["train", "test"],
               "ImageNet-class diffusion (SD-v1.4) vs real ImageNet."),
    DatasetDef("cospy", "CoSpyBench", "image",
               ["/hddstore/ssalehi/SynthScope/CoSpyBench"], ["test"],
               "Co-Spy cross-generator evaluation benchmark."),
    DatasetDef("cospy_subset", "Co-Spy ProGAN subset", "image",
               ["/hddstore/ssalehi/SynthScope/spai_train/cospy_subset"], ["train", "val", "test"]),
    DatasetDef("spai_ldm", "SPAI LDM trainset (vs COCO)", "image",
               ["/hddstore/ssalehi/SynthScope/spai_train/staged"], ["train", "val", "test"],
               "Latent-diffusion fakes vs COCO reals (SPAI from-scratch)."),
    DatasetDef("genvideo_val", "GenVideo-Val", "video",
               ["/hddstore/ssalehi/SynthScope/d3_repro/genvideo_val/GenVideo-Val"], ["."],
               "GenVideo validation pool (msrvtt reals + 10 generators)."),
    DatasetDef("restrav_data", "ReStraV data (VidProM → GenVideo split)", "video",
               ["/hddstore/ssalehi/SynthScope/restrav_data"], ["train", "val", "test"],
               "Train = VidProM/DeMamba (VSC reals); test = GenVideo (msrvtt reals)."),
    DatasetDef("ditfake", "DiTFake (DiT/FLUX/SD3 images)", "image",
               ["/hddstore/ssalehi/SynthScope/DiTFake"], ["test"],
               "DiT/FLUX/SD3-generated images vs COCO reals."),
    DatasetDef("aigibench", "AIGIBench (multi-generator)", "image",
               ["/hddstore/ssalehi/SynthScope/AIGIBench"], ["test"],
               "25-generator AI-image benchmark (GAN/diffusion/face-swap) vs real."),
    DatasetDef("evalcrafter", "EvalCrafter (T2V, generated)", "video",
               ["/hddstore/ssalehi/SynthScope/EvalCrafter"], ["test"],
               "14 text-to-video generators — all AI-generated (no reals)."),
    DatasetDef("genvidbench", "GenVidBench", "video",
               ["/hddstore/ssalehi/SynthScope/GenVidBench"], ["test"],
               "Real (Vript, HD-VG-130M) vs 8 video generators."),
    DatasetDef("freqnetdetection", "FreqNet test set", "image",
               ["/hddstore/ssalehi/SynthScope/freqnet"], ["test"],
               "FreqNet evaluation set (ForenSynths-style real/fake)."),
    DatasetDef("universialfakedetection", "UniversalFakeDetect (Ojha)", "image",
               ["/hddstore/ssalehi/SynthScope/universialfakedetection"], ["test"],
               "Ojha et al. CVPR'23 cross-generator test set."),
    # Cross-generator benchmarks added for the full reproduction suite.
    DatasetDef("genimage_full", "GenImage (8 generators)", "image",
               ["/hddstore/ssalehi/SynthScope/GenImage_full"], ["test"],
               "Full 8-generator GenImage test (Midjourney, SD-v1.4/1.5, ADM, GLIDE, Wukong, VQDM, BigGAN)."),
    DatasetDef("synthbuster", "Synthbuster (9 diffusion generators)", "image",
               ["/hddstore/ssalehi/SynthScope/Synthbuster"], ["test"],
               "Synthbuster (Zenodo 10066460) — DALL-E 2/3, Firefly, GLIDE, Midjourney-v5, SD 1.3/1.4/2/XL; "
               "used by SPAI. Proxy reals (GenImage nature) since Synthbuster ships fakes only."),
    DatasetDef("diffusionforensics", "DiffusionForensics / DIRE", "image",
               ["/hddstore/ssalehi/SynthScope/DiffusionForensics"], ["test"],
               "8 diffusion subsets (ADM, DDPM, IDDPM, LDM, PNDM, SDv1, SDv2, VQ-Diffusion); NPR Table 3."),
    DatasetDef("diffusion1kstep", "Diffusion-1kStep", "image",
               ["/hddstore/ssalehi/SynthScope/Diffusion1kStep"], ["test"],
               "DALLE, DDPM, guided-diffusion, improved-diffusion, midjourney; NPR Table 5."),
    DatasetDef("aigcdetect", "AIGCDetectBenchmark (17 gen)", "image",
               ["/hddstore/ssalehi/SynthScope/AIGCDetectBenchmark"], ["test"],
               "17-generator GAN+diffusion benchmark; Co-Spy / AIDE / CuRe cross-generator table."),
    DatasetDef("commgen15", "CommGen15 (15 commercial gen)", "image",
               ["/hddstore/ssalehi/SynthScope/CommGen15"], ["test"],
               "15 commercial generators (Midjourney, Sora, Flux, Kling, ...); PGC Table 3."),
    DatasetDef("realchain", "RealChain (7 recent generators)", "image",
               ["/hddstore/ssalehi/SynthScope/RealChain"], ["test"],
               "Flux1, SDv3.5, QwenImage, Hunyuan3, NanoBanana, Seedream4, i2i vs 7000 reals, all JPEG q60; "
               "CuRe Table 1. CC BY-NC 4.0 (non-commercial)."),
    DatasetDef("evalgen", "EvalGEN (5 diffusion + AR gen)", "image",
               ["/hddstore/ssalehi/SynthScope/EvalGEN"], ["test"],
               "Flux, GoT, Infinity, OmniGen, NOVA on GenEval prompts (DDA); paired with MSCOCO val2017 reals."),
    DatasetDef("synthwildx", "SynthWildX (images from X)", "image",
               ["/hddstore/ssalehi/SynthScope/SynthWildX"], ["test"],
               "DALL-E 3, Firefly, Midjourney v5 vs real, all posted on X; fetched from X, so deleted posts are missing."),
    DatasetDef("wildrf", "WildRF (Facebook, Reddit, Twitter)", "image",
               ["/hddstore/ssalehi/SynthScope/WildRF"], ["test"],
               "In-the-wild real and fake images from three social platforms (LaDeDa paper); one folder per platform."),
    DatasetDef("bfree_online", "BFree-Online (viral images)", "image",
               ["/hddstore/ssalehi/SynthScope/BFreeOnline"], ["test"],
               "Real and fake images that went viral, several web copies each (B-Free); fetched from the URL list, "
               "dead links skipped. Nonprofit use only."),
    DatasetDef("rrdataset", "RRDataset (original / transfer / redigital)", "image",
               ["/hddstore/ssalehi/SynthScope/RRDataset"], ["test"],
               "Real vs AI images as created, after sharing online, and after re-digitization (ICCV 2025)."),
    DatasetDef("arforensics", "ARForensics (7 autoregressive generators)", "image",
               ["/hddstore/ssalehi/SynthScope/ARForensics"], ["test"],
               "LlamaGen, VAR, Infinity, Janus-Pro, RAR, Switti and Open-MAGVIT2 vs ImageNet reals, 6000 each (D3QE, ICCV 2025)."),
    DatasetDef("videofeedback", "VideoFeedback (TIGER-Lab)", "video",
               ["/hddstore/ssalehi/SynthScope/VideoFeedback"], ["test"],
               "11 video generators vs DiDeMo+Panda70M real; STALL benchmark."),
    DatasetDef("comgenvid", "ComGenVid (Sora/VEO3)", "video",
               ["/hddstore/ssalehi/SynthScope/ComGenVid"], ["test"],
               "Sora + VEO3 fakes vs MSVD real; STALL's new benchmark."),
    DatasetDef("genvideo", "GenVideo-Val (10 generators)", "video",
               ["/hddstore/ssalehi/SynthScope/GenVideo"], ["test"],
               "DeMamba's GenVideo-Val per generator vs 10,000 MSR-VTT reals; D3 Table 1, STALL."),
    DatasetDef("videophy", "VideoPhy (text-to-video)", "video",
               ["/hddstore/ssalehi/SynthScope/VideoPhy"], ["test"],
               "VideoPhy test clips from up to 10 generators vs MSR-VTT reals; D3 Table 3."),
    DatasetDef("vidprom", "VidProM (4 generators, subsample)", "video",
               ["/hddstore/ssalehi/SynthScope/VidProM"], ["test"],
               "First 1,000 clips per generator (ModelScope, Pika, Text2Video-Zero, VideoCrafter2) vs MSR-VTT reals; "
               "CC BY-NC 4.0."),
]

# Which dataset each detector was trained on (best-effort, from interface/scripts/train_model.sh
# and experiments/*). `official` flags checkpoints whose deployed weights are the authors' (the
# SynthScope from-scratch dataset is noted, but the served weights are the official ProGAN-trained ones).
MODEL_TRAIN_DATASET: dict[str, dict] = {
    "clip": {"dataset": "cnndetection"},
    "cnn": {"dataset": "cnndetection"},
    "dinov2": {"dataset": "cnndetection"},
    "dinov3": {"dataset": "cnndetection"},
    "adapted_dino": {"dataset": "cnndetection"},
    "safe": {"dataset": "cnndetection"},
    "pgc": {"dataset": "cospy_subset"},
    "co_spy": {"dataset": "cospy_subset"},
    "vib_net": {"dataset": "genimage"},
    "aide": {"dataset": "genimage", "official": True},
    "npr": {"dataset": "genimage", "official": True},
    "spai": {"dataset": "spai_ldm"},
    "fat_former": {"dataset": "cnndetection", "official": True},
    "restrav": {"dataset": "restrav_data"},
    "d3": {"dataset": None, "training_free": True},
    "stall": {"dataset": None, "training_free": True},
}


# Node-local scratch differs per node (cn023 -> /hddstore, cees -> /ssdstore, cn030/31 -> /nvmestore).
# Try each configured root, then the same tail remapped onto the other scratch mounts, so the explorer
# works wherever the data was staged. Override the prefix set with SYNTHSCOPE_DATA_PREFIXES (colon-separated).
_SCRATCH_PREFIXES = tuple(p for p in (
    os.environ.get("SYNTHSCOPE_DATA_PREFIXES") or "/hddstore:/ssdstore:/nvmestore").split(":") if p)


def _root_candidates(r: str) -> list[str]:
    cands = [r]
    # SYNTHSCOPE_DATA_DIR: one folder holding datasets by their usual folder name (e.g. <dir>/CommGen15), as on a Space
    if os.environ.get("SYNTHSCOPE_DATA_DIR"):
        cands.insert(0, str(Path(os.environ["SYNTHSCOPE_DATA_DIR"]).expanduser() / Path(r).name))
    for src in _SCRATCH_PREFIXES:
        if r.startswith(src + "/"):
            tail = r[len(src):]
            cands += [pre + tail for pre in _SCRATCH_PREFIXES if pre != src]
            break
    return cands


def _resolve_root(d: DatasetDef) -> Path | None:
    for r in d.roots:
        for c in _root_candidates(r):
            p = Path(c)
            if p.is_dir():
                return p
    return None


def _exts_for(modality: str) -> set[str]:
    return VIDEO_EXTS if modality == "video" else IMAGE_EXTS


def _split_dir(root: Path, split: str) -> Path:
    return root if split in (".", "", None) else root / split


def label_of(rel: Path) -> str | None:
    """Infer 'real'/'fake' from a relative path's directory tokens (None if unknown)."""
    parts = [p.lower() for p in rel.parts]
    for p in parts:
        if p in FAKE_TOKENS:
            return "fake"
    for p in parts:
        if p in REAL_TOKENS:
            return "real"
    return None


def available_splits(d: DatasetDef, root: Path) -> list[str]:
    out = []
    for s in d.splits:
        if _split_dir(root, s).is_dir():
            out.append(s)
    return out


def list_datasets() -> list[dict]:
    """All registry datasets whose data is present on disk, with resolved path + splits."""
    out = []
    for d in DATASET_DEFS:
        root = _resolve_root(d)
        if root is None:
            continue
        out.append({
            "key": d.key,
            "name": d.name,
            "modality": d.modality,
            "note": d.note,
            "path": str(root),
            "splits": available_splits(d, root),
        })
    return out


def get_def(key: str) -> DatasetDef:
    for d in DATASET_DEFS:
        if d.key == key:
            return d
    raise KeyError(f"Unknown dataset '{key}'")


def _iter_files(base: Path, exts: set[str]):
    scanned = 0
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            if Path(fn).suffix.lower() in exts:
                yield Path(dirpath) / fn
                scanned += 1
                if scanned >= SCAN_CAP:
                    return


def _pools(base: Path, exts: set[str]) -> dict[str, list[Path]]:
    """Reservoir-sample up to POOL_PER_LABEL paths per label across the whole tree."""
    pools: dict[str, list[Path]] = {"real": [], "fake": []}
    seen: dict[str, int] = {"real": 0, "fake": 0}
    rng = random.Random(1234)
    for fp in _iter_files(base, exts):
        lab = label_of(fp.relative_to(base))
        if lab is None:
            continue
        seen[lab] += 1
        pool = pools[lab]
        if len(pool) < POOL_PER_LABEL:
            pool.append(fp)
        else:                                   # reservoir replacement
            j = rng.randint(0, seen[lab] - 1)
            if j < POOL_PER_LABEL:
                pool[j] = fp
    return pools


def sample_batch(key: str, split: str, n: int, seed: int) -> list[dict]:
    """Return ``n`` random items (balanced real/fake when possible) from ``key``/``split``."""
    d = get_def(key)
    root = _resolve_root(d)
    if root is None:
        raise FileNotFoundError(f"Dataset '{key}' not present on disk")
    base = _split_dir(root, split)
    if not base.is_dir():
        raise FileNotFoundError(f"Split '{split}' not found for '{key}'")
    pools = _pools(base, _exts_for(d.modality))
    rng = random.Random(seed)
    half = max(1, n // 2)
    items: list[Path] = []
    for lab in ("real", "fake"):
        p = pools[lab]
        rng.shuffle(p)
        items += [(fp, lab) for fp in p[:half]]
    # top up if one class is short
    if len(items) < n:
        extra = [(fp, lab) for lab in ("real", "fake") for fp in pools[lab][half:]]
        rng.shuffle(extra)
        items += extra[: n - len(items)]
    rng.shuffle(items)
    items = items[:n]
    return [
        {"rel": str(fp.relative_to(root)), "label": lab,
         "name": fp.name, "modality": d.modality}
        for fp, lab in items
    ]


def resolve_media(key: str, rel: str) -> Path:
    """Resolve ``rel`` under the dataset root, guarding against ``..`` traversal.

    Some datasets (e.g. ``restrav_data``) populate their split dirs with *symlinks* into other
    SynthScope data folders, so we must NOT ``.resolve()`` and then reject by the symlink's target
    (that would 403 every restrav_data thumbnail). Instead we forbid absolute paths and ``..``
    components in the user-supplied ``rel`` — which keeps the addressed path lexically under the
    dataset root — then follow the symlink (``is_file``) to serve the actual file.
    """
    d = get_def(key)
    root = _resolve_root(d)
    if root is None:
        raise FileNotFoundError(f"Dataset '{key}' not present on disk")
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise PermissionError("Path escapes dataset root")
    target = root / rel_path
    if not target.is_file():            # follows the symlink to the real video/image
        raise FileNotFoundError(str(target))
    return target


# Hydra dataset configs that are training sets, not benchmarks (excluded from the benchmark count shown on the page).
_TRAINING_ONLY = {"cospy_subset", "spai_ldm"}


def supported_counts() -> dict:
    """How many benchmark datasets SynthScope can download and lay out (one Hydra config each)."""
    conf = Path(__file__).resolve().parents[2] / "conf" / "dataset"
    return {m: sum(1 for p in (conf / m).glob("*.yaml") if p.stem not in _TRAINING_ONLY) for m in ("image", "video")}
