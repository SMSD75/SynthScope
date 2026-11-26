"""Unified benchmark eval for SynthScope detectors — reproduce a paper's per-generator table.

Scores a detector over a benchmark's per-generator real/fake folders through the SAME verified
synthscope inference path the web app uses (engine + registry + per-model preprocessing + fake-prob),
computes per-generator + mean acc@0.5 / AP / AUROC, writes a JSON, and APPENDS a verified line to
``repro/MASTER_RESULTS.jsonl`` so every reported number traces to this run + its job log.

    python benchmark/bench_eval.py --model clip --benchmark forensynths \
        --data /hddstore/ssalehi/SynthScope/CNNDetection/test --tag clip_synthscope_forensynths

    # one step: download the dataset if needed (its built-in loader arranges it), pick the profile, evaluate
    python benchmark/bench_eval.py --model npr --dataset commgen15 --root /data/CommGen15

Benchmark profiles encode each dataset's correct generator grouping + real-pairing:
  forensynths  CNNDetection/test    per top-level generator; reals+fakes collected recursively
                                    (handles progan/<class>/{0_real,1_fake} nesting)
  ufd          UniversalFakeDetect  fakes per generator; reals paired Ojha-style
                                    (guided->imagenet, ldm/glide/dalle->laion)
  pergen       freqnet/aigibench/   each <gen>/{0_real,1_fake}; optional shared --reals
               cospybench/generic
  pool         GenImage/single-pool one group from <root>/{real,fake} or {0_real,1_fake}

Use --ckpt to override the checkpoint (official vs SynthScope-trained, for side-by-side runs).
Use --model-config NAME with --ckpt for a checkpoint trained by a recipe whose model config (conf/model/NAME.yaml) has its own
test preprocessing, e.g. Effort's UFD/ProGAN recipe (image/effort_ufd: center crop 224, not the served model's squash).
Use --jpeg-fakes Q for protocols that JPEG-compress only the synthetic images (DDA: Q=96 on GenImage,
ForenSynths and AIGCDetect, to remove the PNG-fake/JPEG-real format bias).
Use --degrade SPEC for a robustness slice: one deterministic degradation applied to EVERY image (reals and fakes) after it is
loaded and before --jpeg-fakes (fakes only) and the detector's own transform, in that order. SPEC is one token:
jpeg<Q> | down<F> | blur<S> | social | social2 (see ``parse_degrade``). Use --results-log PATH to append the run's summary line
to PATH instead of repro/MASTER_RESULTS.jsonl. Use --per-image to record every image's P(fake) (with its key, relative to
--data, and the file's sha1) in the output JSON (``per_image``) for a non-VLM detector, e.g. to fuse it with a VLM's records
(Align, Don't Blend's full method, repro/adb_full/fuse.py).

VLM detectors (registry ``predict_kind="vlm"``, e.g. veritaspp) are scored one image at a time through ``Engine.vlm_score`` on the
file decoded without an RGB conversion (``media.decode_image_raw``; the model's own ``image_loader`` does the official EXIF / alpha
handling). With --degrade / --jpeg-fakes that loader runs first and the degradation is applied to its RGB output. Metrics:
accuracy from the parsed verdict (an unparsed answer counts as wrong; ``*_drop`` fields drop it, as the authors' script does),
AUROC / AP from the fake-minus-real logit, plus the unparsed count, the unscored count (a parsed verdict but no verdict token, so
P(fake) = 0.5 / logit 0 enters AUROC / AP; ``n_unscored``) and the mean generated tokens; the per-image records go into the
output JSON (``per_image``). --vlm-decoding official samples with the authors' settings, seeded per image with --vlm-seed.
--vlm-cache DIR keeps every scored image as a JSON line (one file per writer, keyed by its path relative to --data and checked
against the file's sha1), so a preempted or repeated run only scores what is missing. The cache folder is also keyed by a hash of
the weights revision (or --ckpt) and a LoRA base's revision, max_new_tokens and sampling settings, the prompts, the detector's
other VLM settings (max_pixels, model class, verdict pairs, markers, parser and its verdict-offset hook, image loader, pre-resize
hook and its parameters) and the VLM code version (``vlm_identity``, written to ``identity.json`` in the folder), so a cache
reused after any of them changed starts empty (repro/vlm_bench/migrate_vlm_cache.py carries records over a re-key that does
not change them).
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import dataclasses
import datetime
import hashlib
import io
import json
import os
import re
import socket
import time
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score

import synthscope
from interface.backend import media
from interface.backend.registry import get_spec, model_transform
from synthscope.api import Detector
from models.image.cure.preprocess import SRC_FORMAT_KEY
from datasets_util import dataset_config, ensure_dataset, find_eval_root

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None   # benchmark data is trusted; some real photos (e.g. RealChain) exceed PIL's 179 MP bomb guard


def _limit_heap_growth() -> None:
    """Serve large allocations from mmap and give freed heap back to the OS (glibc ``mallopt``; the same as
    ``MALLOC_MMAP_THRESHOLD_`` / ``MALLOC_TRIM_THRESHOLD_`` = 128 KiB). Scoring one native-resolution image after another
    (B-Free, SPAI: a differently sized decode and tensor each time) otherwise fragments the heap and the process grows by
    ~1 GB a minute until it is killed (B-Free on EvalGEN); with it the run stays flat and the scores are identical."""
    libc = ctypes.util.find_library("c")
    if libc:
        mallopt = ctypes.CDLL(libc).mallopt
        mallopt(-3, 128 * 1024)    # M_MMAP_THRESHOLD
        mallopt(-1, 128 * 1024)    # M_TRIM_THRESHOLD


_limit_heap_growth()

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG"}
REAL_DIRNAMES = ("0_real", "real", "reals")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "ai")
SHARED_REAL_DIRS = {"imagenet", "laion", "_real_pool", "real", "0_real", "reals"}
RESULTS_LOG = Path(__file__).resolve().parents[1] / "repro" / "MASTER_RESULTS.jsonl"
# Ojha UniversalFakeDetect real pairing (verified against the official dataset_paths.py).
UFD_REAL = {"guided": "imagenet"}  # everything else -> laion
UFD_DEFAULT_REAL = "laion"


def _images(d: Path) -> list[Path]:
    if not d or not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in {e.lower() for e in IMAGE_EXTS})


def _collect(gen_dir: Path, dirnames: tuple[str, ...]) -> list[Path]:
    """All images under any subdir of gen_dir whose name matches one of dirnames (any depth)."""
    out: list[Path] = []
    for name in dirnames:
        for sub in gen_dir.rglob(name):
            if sub.is_dir():
                out += [p for p in sub.rglob("*") if p.suffix.lower() in {e.lower() for e in IMAGE_EXTS}]
    # de-dup while keeping order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return sorted(uniq)


def discover(benchmark: str, root: Path, shared_reals: Path | None, select: set[str] | None):
    """Return [(generator, real_paths, fake_paths)] for the benchmark."""
    shared = _images(shared_reals) if shared_reals else []

    if benchmark == "pool":
        reals = _collect(root, REAL_DIRNAMES) or shared
        fakes = _collect(root, FAKE_DIRNAMES)
        return [("all", reals, fakes)] if reals and fakes else []

    if benchmark == "ufd":
        out = []
        gens = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in SHARED_REAL_DIRS)
        for gen in gens:
            if select and gen not in select:
                continue
            real_src = root / UFD_REAL.get(gen, UFD_DEFAULT_REAL)
            reals = _collect(real_src, REAL_DIRNAMES)
            fakes = _collect(root / gen, FAKE_DIRNAMES)
            if reals and fakes:
                out.append((gen, reals, fakes))
            else:
                print(f"  skip {gen}: reals={len(reals)} fakes={len(fakes)}")
        return out

    if benchmark == "forensynths":
        out = []
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if select and gen_dir.name not in select:
                continue
            reals = _collect(gen_dir, REAL_DIRNAMES) or shared
            fakes = _collect(gen_dir, FAKE_DIRNAMES)
            if reals and fakes:
                out.append((gen_dir.name, reals, fakes))
            else:
                print(f"  skip {gen_dir.name}: reals={len(reals)} fakes={len(fakes)}")
        return out

    # pergen / generic: each top-level dir (excluding shared-real dirs) is a generator
    out = []
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if gen_dir.name in SHARED_REAL_DIRS and gen_dir.name != root.name:
            continue
        if select and gen_dir.name not in select:
            continue
        reals = _collect(gen_dir, REAL_DIRNAMES) or shared
        fakes = _collect(gen_dir, FAKE_DIRNAMES)
        if reals and fakes:
            out.append((gen_dir.name, reals, fakes))
        else:
            print(f"  skip {gen_dir.name}: reals={len(reals)} fakes={len(fakes)}")
    return out


def native_res_transform(tf):
    """Drop the up-scaling resize/center-crop steps from a model's eval transform, keeping
    to_dtype + normalize, so images are scored at NATIVE resolution (ValueError for eval lists
    whose own steps resize, see ``AIGITransformations.without_resize``).

    Some methods (e.g. NPR on GANGen-Detection/freqnet) detect up-sampling artifacts and
    MUST NOT be fed a bilinearly up-scaled image — resizing a 128x128 GAN crop to 256x256
    injects the exact artifact they key on, collapsing the real class. The official NPR
    GANGen recipe runs with ``no_resize=True``; this reproduces it. Safe only when every
    image in a per-generator batch shares one resolution (true for freqnet's per-gen sets),
    since bench_eval stacks a batch into a single tensor."""
    return tf.without_resize()


@torch.no_grad()
def _open_rgb(path: Path) -> Image.Image:
    """Open ``path`` as RGB, recording the source file format in ``img.info`` so format-aware eval
    transforms (CuRe's PNG->JPEG96 step) see what the official loader sees."""
    source = Image.open(path)
    source_format = source.format
    image = source.convert("RGB")
    image.info[SRC_FORMAT_KEY] = source_format
    return image


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    """In-memory JPEG round trip, as DDA's ``JPEG_Compression`` (Training/data/custom_transforms.py)."""
    buf = io.BytesIO()
    image.save(buf, format="jpeg", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.info[SRC_FORMAT_KEY] = "JPEG"
    return out


SOCIAL_LONG_SIDE = 1080     # "social": the long side an image is resized to (when larger) before the JPEG q75 re-encode
SOCIAL_QUALITY = 75


def _resized(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Bicubic resize that keeps the source-format hint."""
    out = image.resize(size, Image.BICUBIC)
    out.info.update(image.info)
    return out


def _down(image: Image.Image, factor: float) -> Image.Image:
    """Downscale by ``factor`` (bicubic), then upsample back to the original size (bicubic)."""
    w, h = image.size
    small = _resized(image, (max(1, round(w * factor)), max(1, round(h * factor))))
    return _resized(small, (w, h))


def _blur(image: Image.Image, sigma: float) -> Image.Image:
    out = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    out.info.update(image.info)
    return out


def _social(image: Image.Image) -> Image.Image:
    """A social-network upload: long side to 1080 if larger (bicubic), then JPEG q75 (4:2:0)."""
    w, h = image.size
    long_side = max(w, h)
    if long_side > SOCIAL_LONG_SIDE:
        scale = SOCIAL_LONG_SIDE / long_side
        size = (SOCIAL_LONG_SIDE if w >= h else max(1, round(w * scale)),
                SOCIAL_LONG_SIDE if h > w else max(1, round(h * scale)))
        image = _resized(image, size)
    return _jpeg(image, SOCIAL_QUALITY)


def parse_degrade(spec: str):
    """The deterministic degradation (``PIL image -> RGB PIL image``) a ``--degrade`` token names; ValueError if malformed.

        jpeg<Q>   in-memory PIL JPEG re-encode at quality Q (1-100), default 4:2:0 chroma subsampling
        down<F>   downscale by factor F (0 < F < 1, e.g. down0.5) with bicubic, then back to the original size with bicubic
        blur<S>   Gaussian blur, sigma S > 0 (PIL ``ImageFilter.GaussianBlur(radius=S)``)
        social    long side resized to 1080 if larger (bicubic), then JPEG q75 (4:2:0)
        social2   ``social`` applied twice

    No RNG. Source-format hint (``SRC_FORMAT_KEY``, read by CuRe / MIRROR for their official PNG->JPEG step): a degradation
    that JPEG-encodes (jpeg, social, social2) leaves a JPEG, so the hint becomes "JPEG"; down and blur keep the file's own."""
    number = r"(\d+(?:\.\d+)?)"
    if spec == "social":
        return _social
    if spec == "social2":
        return lambda image: _social(_social(image))
    if m := re.fullmatch(r"jpeg(\d+)", spec):
        quality = int(m.group(1))
        if not 1 <= quality <= 100:
            raise ValueError(f"--degrade {spec}: JPEG quality must be in 1..100")
        return lambda image: _jpeg(image, quality)
    if m := re.fullmatch(rf"down{number}", spec):
        factor = float(m.group(1))
        if not 0 < factor < 1:
            raise ValueError(f"--degrade {spec}: downscale factor must be in (0, 1)")
        return lambda image: _down(image, factor)
    if m := re.fullmatch(rf"blur{number}", spec):
        sigma = float(m.group(1))
        if sigma <= 0:
            raise ValueError(f"--degrade {spec}: blur sigma must be > 0")
        return lambda image: _blur(image, sigma)
    raise ValueError(f"--degrade {spec!r}: expected jpeg<Q>, down<F>, blur<S>, social or social2")


def score(det, model, spec, paths: list[Path], bs: int, transform=None,
          jpeg_q: int | None = None, jpeg_from: int = 0, degrade=None) -> tuple[torch.Tensor, list[int]]:
    """Fake probabilities for the images that could be read, and the indices (into ``paths``) they belong to.
    ``degrade`` (a ``parse_degrade`` callable) is applied to every image right after loading; then, with ``jpeg_q``,
    images from index ``jpeg_from`` on (the fakes) are JPEG-compressed at that quality; then the detector's transform."""
    device = det._engine.device
    tf = transform if transform is not None else spec.transform
    # Native-resolution specs (SPAI any-res) produce variable-size tensors that cannot be stacked,
    # so they MUST be scored one image at a time.
    bs = spec.eval_batch_size(bs)
    probs, kept = [], []
    for i in range(0, len(paths), bs):
        batch = []
        for j, p in enumerate(paths[i:i + bs], start=i):
            try:
                image = _open_rgb(p)
                if degrade is not None:
                    image = degrade(image)
                if jpeg_q is not None and j >= jpeg_from and jpeg_q < 100:
                    image = _jpeg(image, jpeg_q)
                batch.append(tf(image))
                kept.append(j)
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        probs.append(det._engine._fake_prob(model, spec, x).detach().float().cpu())
    return (torch.cat(probs) if probs else torch.empty(0)), kept


def metrics(fake_probs: torch.Tensor, labels: torch.Tensor) -> dict | None:
    from torchmetrics.functional import auroc, average_precision
    if labels.unique().numel() < 2:
        return None
    ap = float(average_precision(fake_probs, labels, task="binary")) * 100
    au = float(auroc(fake_probs, labels, task="binary")) * 100
    preds = (fake_probs >= 0.5).int()
    real_acc = float((preds[labels == 0] == 0).float().mean()) * 100
    fake_acc = float((preds[labels == 1] == 1).float().mean()) * 100
    return {"ap": round(ap, 2), "auroc": round(au, 2),
            "real_acc": round(real_acc, 2), "fake_acc": round(fake_acc, 2),
            "acc": round((real_acc + fake_acc) / 2, 2)}


# --- VLM detectors -------------------------------------------------------------------------------------------------------------
REALS_KEY_PREFIX = "@reals/"     # cache key of an image of the shared --reals pool (outside --data)


def image_key(path: Path, data_root: Path, reals_root: Path | None) -> str:
    """An image's cache key: its path relative to the benchmark root (``@reals/<rel>`` for the shared real pool), so a
    copy of the same files under another root (a staged subset, another node) finds the same records."""
    for root, prefix in ((data_root, ""), (reals_root, REALS_KEY_PREFIX)):
        if root is not None:
            try:
                return prefix + path.relative_to(root).as_posix()
            except ValueError:
                pass
    return path.as_posix()


VLM_CODE_DIR = Path(__file__).resolve().parents[1] / "models" / "vlm"


def vlm_code_version() -> str:
    """Short hash of the VLM scoring code (``models/vlm/*.py``: family adapters, parsers, image loading): an edit to any
    of it changes the cache identity, so records written by other code are never reused. A comment-only edit also
    invalidates a cache; that is the price of not trusting a hand-kept version number."""
    h = hashlib.sha1()
    for f in sorted(VLM_CODE_DIR.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:10]


VLM_CONFIG_FIELDS = ("model_class", "family", "max_pixels", "verdict_pairs", "verdict_sequences", "verdict_mode",
                     "teacher_forced_suffix", "logits", "answer_marker", "answer_end", "parser", "verdict_offset",
                     "image_loader", "preprocess", "preprocess_kwargs", "processor", "processor_overrides",
                     "processor_kwargs", "base_model", "dtype", "attn_implementation")
# VLMConfig fields added later: part of the fingerprint only when set, so the identities of detectors that leave them unset
# (and the caches keyed by them) do not change
VLM_CONFIG_FIELDS_IF_SET = ("answer_parsed",)
NO_SCORE_SOURCES = ("unparsed", "unscored")      # score_source of a record without a verdict token (P 0.5, logit 0)


def vlm_config_fingerprint(cfg) -> str:
    """Short hash of the per-detector VLM settings that change a score (``VLMConfig`` fields: model class, pixel cap,
    verdict words / sequences / mode, logits kind, answer marker and end tag, parser and its verdict-offset hook, image
    loader, pre-resize hook and its parameters, processor and base folders, dtype, attention): callables by qualified
    name, a family by its class."""
    def name(v):
        if callable(v) or isinstance(v, type):
            return f"{getattr(v, '__module__', '')}.{getattr(v, '__qualname__', repr(v))}"
        if hasattr(v, "__class__") and type(v).__module__ != "builtins" and not isinstance(v, (dict, list, tuple)):
            return f"{type(v).__module__}.{type(v).__qualname__}"
        return v
    fields = {f: name(getattr(cfg, f, None)) for f in VLM_CONFIG_FIELDS}
    fields.update({f: name(getattr(cfg, f)) for f in VLM_CONFIG_FIELDS_IF_SET if getattr(cfg, f, None) is not None})
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def vlm_identity(spec, decoding: str, ckpt: str | None = None) -> dict:
    """Everything besides the image that determines a cached record: the weights (pinned HF revision, or the explicit
    ``--ckpt`` path; a LoRA's base revision), the generation limit and sampling settings of the active decoding, a hash of the prompts and chat
    template arguments, a fingerprint of the rest of the detector's VLM config (``vlm_config_fingerprint``: max_pixels,
    model class, verdict pairs, markers, parser, image loader, ...), and the code version. A cache folder is keyed by a
    hash of this dict."""
    cfg = spec.vlm
    dec = dict(cfg.official_decoding if decoding == "official" else cfg.decoding)
    prompts = json.dumps([cfg.system_prompt, cfg.user_prompt, cfg.chat_template_kwargs, cfg.chat_template],
                         sort_keys=True, default=str)
    return {"weights_revision": cfg.hf_revision or None, "base_weights_revision": cfg.base_hf_revision or None,
            "ckpt": ckpt, "max_new_tokens": dec.get("max_new_tokens"),
            "generation": dec, "prompt_sha1": hashlib.sha1(prompts.encode()).hexdigest()[:10],
            "config_sha1": vlm_config_fingerprint(cfg), "code_version": vlm_code_version()}


def vlm_variant(model: str, decoding: str, seed: int | None, degrade: str | None, jpeg_q: int | None,
                identity: dict | None = None) -> str:
    """The cache folder name of one scoring setup: records of different setups never mix. ``identity`` (``vlm_identity``)
    adds a hash of the weights revision, max_new_tokens, prompt, the rest of the VLM config and the code version, so a cache reused after any of them
    changed starts empty instead of returning stale records."""
    return (f"{model}__{decoding}" + (f"_seed{seed}" if decoding == "official" else "")
            + (f"__{degrade}" if degrade else "") + (f"__jpegfakes{jpeg_q}" if jpeg_q is not None else "")
            + ("" if identity is None else
               "__" + hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]))


class VLMCache:
    """Per-image VLM results as JSON lines under ``<directory>/<variant>/``: this process appends to its own file
    (``<host>.<pid>.jsonl``, flushed per line) and reads every file there, so several writers never share a file and a
    preempted run resumes from what any earlier run stored. A record is used only when the file's sha1 matches."""

    def __init__(self, directory: Path | None, variant: str, identity: dict | None = None):
        self.records: dict[str, dict] = {}
        self.folder = None if directory is None else Path(directory) / variant
        self._fh = None
        if self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        if identity is not None:                   # human-readable record of what the folder's hash stands for
            (self.folder / "identity.json").write_text(json.dumps(identity, indent=1) + "\n")
        for f in sorted(self.folder.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:       # a line cut by a kill
                    continue
                self.records[rec["key"]] = rec
        print(f"[vlm-cache] {self.folder}: {len(self.records)} cached images", flush=True)

    def get(self, key: str, sha1: str) -> dict | None:
        rec = self.records.get(key)
        return rec if rec is not None and rec["sha1"] == sha1 else None

    def put(self, rec: dict) -> None:
        self.records[rec["key"]] = rec
        if self.folder is None:
            return
        if self._fh is None:
            self._fh = (self.folder / f"{socket.gethostname().split('.')[0]}.{os.getpid()}.jsonl").open("a")
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()


def score_vlm(det, spec, paths: list[Path], keys: list[str], labels: list[int], cache: VLMCache, *, decoding: str,
              seed: int | None, degrade=None, jpeg_q: int | None = None, jpeg_from: int = 0) -> tuple[list[dict], list[int]]:
    """One ``Engine.vlm_score`` call per image (cached ones are read back), and the indices (into ``paths``) scored.
    The file is decoded without an RGB conversion; with ``degrade`` / ``jpeg_q`` the model's ``image_loader`` runs first
    and the degradation is applied to its RGB output (the scorer's loader then leaves an RGB image unchanged)."""
    recs, kept = [], []
    n_new = secs = 0
    for j, (p, key) in enumerate(zip(paths, keys)):
        try:
            data = p.read_bytes()
            sha1 = hashlib.sha1(data).hexdigest()
            rec = cache.get(key, sha1)
            if rec is None:
                image = media.decode_image_raw(data)
                fake_jpeg = jpeg_q is not None and j >= jpeg_from and jpeg_q < 100
                if degrade is not None or fake_jpeg:
                    image = spec.vlm.image_loader(image)
                    if degrade is not None:
                        image = degrade(image)
                    if fake_jpeg:
                        image = _jpeg(image, jpeg_q)
        except Exception as e:  # noqa: BLE001
            print(f"    bad image {p}: {e}", flush=True)
            continue
        cached = rec is not None
        if not cached:
            t0 = time.time()
            out = det._engine.vlm_score(spec, image, decoding=decoding, seed=seed)
            dt = time.time() - t0
            rec = {"key": key, "sha1": sha1, "logit": out["logit"], "verdict": out["verdict"],
                   "answer_parsed": out.get("answer_parsed", out["verdict"] is not None),
                   **({"logit_unclamped": out["logit_unclamped"]} if "logit_unclamped" in out else {}),
                   "score_source": out["score_source"], "new_tokens": out["new_tokens"], "finished": out["finished"],
                   "verdict_token": out["verdict_token"], "input_size": out["input_size"], "seconds": round(dt, 3),
                   "decoding": decoding, "seed": seed, "response": out["response"]}
            cache.put(rec)
            n_new += 1
            secs += dt
        recs.append(rec)
        kept.append(j)
        print(f"    vlm {key} label={labels[j]} verdict={rec['verdict']} logit={rec['logit']:.4f} "
              f"src={rec['score_source']} tok={rec['new_tokens']} {rec['seconds']:.1f}s" + (" [cached]" if cached else ""),
              flush=True)
    if n_new:
        print(f"    scored {n_new} new image(s), {secs / n_new:.2f} s/img; {len(recs) - n_new} from the cache", flush=True)
    return recs, kept


def answer_parsed(rec: dict, cfg=None) -> bool:
    """Whether the official parser read the record's answer: the record's ``answer_parsed`` (written since the field
    exists), else a verdict and, for a config with an ``answer_parsed`` hook (ADB: a rejected answer defaults to Real),
    the hook on the stored response."""
    if "answer_parsed" in rec:
        return bool(rec["answer_parsed"])
    if rec["verdict"] is None:
        return False
    hook = getattr(cfg, "answer_parsed", None) if cfg is not None else None
    return True if hook is None else bool(hook(rec.get("response", "")))


def vlm_metrics(recs: list[dict], labels: list[int], cfg=None) -> dict | None:
    """Accuracy from the parsed verdicts (unparsed = wrong; ``*_drop``: unparsed answers left out, as the authors'
    ``cal_metrics.py``), AUROC / AP from the fake-minus-real logit (float64, rank-based), plus counts (``n_unscored``: a
    parsed verdict without a verdict token, i.e. logit 0 / P 0.5 in AUROC / AP)."""
    if len(set(labels)) < 2:
        return None
    logits = [r["logit"] for r in recs]
    verdicts = [r["verdict"] for r in recs]

    def pct(hits: list[bool]) -> float | None:
        return round(100.0 * sum(hits) / len(hits), 2) if hits else None

    correct = [v == ("fake" if y else "real") for v, y in zip(verdicts, labels)]
    parsed = [answer_parsed(r, cfg) for r in recs]
    by_class = {y: [c for c, l in zip(correct, labels) if l == y] for y in (0, 1)}
    by_class_drop = {y: [c for c, l, ok in zip(correct, labels, parsed) if l == y and ok] for y in (0, 1)}
    real_acc, fake_acc = pct(by_class[0]), pct(by_class[1])
    real_drop, fake_drop = pct(by_class_drop[0]), pct(by_class_drop[1])
    return {"ap": round(100 * average_precision_score(labels, logits), 2),
            "auroc": round(100 * roc_auc_score(labels, logits), 2),
            "real_acc": real_acc, "fake_acc": fake_acc, "acc": round((real_acc + fake_acc) / 2, 2),
            "acc_pooled": pct(correct),
            "real_acc_drop": real_drop, "fake_acc_drop": fake_drop,
            "acc_drop": None if real_drop is None or fake_drop is None else round((real_drop + fake_drop) / 2, 2),
            "acc_pooled_drop": pct([c for c, ok in zip(correct, parsed) if ok]),
            "n_real": len(by_class[0]), "n_fake": len(by_class[1]),
            "n_unparsed": parsed.count(False),
            "n_unscored": sum(ok and r["score_source"] in NO_SCORE_SOURCES for ok, r in zip(parsed, recs)),
            "n_unfinished": sum(not r["finished"] for r in recs),
            "n_logit_verdict_disagree": sum(v is not None and (v == "fake") != (r["logit"] > 0)
                                            for v, r in zip(verdicts, recs)),
            "mean_tokens": round(sum(r["new_tokens"] for r in recs) / len(recs), 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="SynthScope detector key (see `synthscope list`)")
    ap.add_argument("--benchmark", choices=["forensynths", "ufd", "pergen", "pool"],
                    help="how the folders are grouped (chosen automatically with --dataset)")
    ap.add_argument("--data", help="benchmark root (not needed with --dataset)")
    ap.add_argument("--dataset", help="a supported dataset (see `python benchmark/download_dataset.py --list`): downloads it "
                    "if missing, finds the folder to evaluate and the right --benchmark profile")
    ap.add_argument("--root", help="with --dataset: where the dataset lives / should be downloaded (default: root_dir in its yaml)")
    ap.add_argument("--no-download", action="store_true", help="with --dataset: fail instead of downloading missing data")
    ap.add_argument("--dry-run", action="store_true", help="with --dataset: only show the resolved folder and profile")
    ap.add_argument("--reals", default=None, help="shared real dir (pergen/forensynths fallback; with --dataset also the real pool of loaders that ship "
                    "fakes only, e.g. Synthbuster)")
    ap.add_argument("--ckpt", default=None, help="checkpoint override (official vs SynthScope weights)")
    ap.add_argument("--model-config", default=None, metavar="NAME",
                    help="score --ckpt with another model config of the same detector (conf/model/NAME.yaml, e.g. image/effort_ufd: "
                         "a training recipe whose own test preprocessing differs from the served model's); needs --ckpt")
    ap.add_argument("--select", default=None, help="comma list: only these generators")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class per generator")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--native-res", action="store_true",
                    help="score at native resolution (strip up-scaling Resize/CenterCrop); "
                         "for artifact-based methods like NPR on low-res GAN sets (freqnet)")
    ap.add_argument("--jpeg-fakes", type=int, default=None, metavar="Q",
                    help="JPEG-compress only the fake images at quality Q before scoring (DDA protocol: 96)")
    ap.add_argument("--degrade", default=None, metavar="SPEC",
                    help="robustness slice: degrade EVERY image (reals and fakes) after loading, before --jpeg-fakes and the "
                         "detector's transform. SPEC: jpeg<Q> | down<F> | blur<S> | social | social2, e.g. jpeg75, down0.5, "
                         "blur2 (deterministic; recorded as `degrade` in the output)")
    ap.add_argument("--results-log", default=None, metavar="PATH",
                    help="append the run's summary line to PATH instead of repro/MASTER_RESULTS.jsonl")
    ap.add_argument("--vlm-decoding", choices=["greedy", "official"], default="greedy",
                    help="VLM detectors: greedy decoding (default) or the authors' sampling (seeded per image with --vlm-seed)")
    ap.add_argument("--vlm-seed", type=int, default=0, help="VLM detectors, --vlm-decoding official: the per-image seed")
    ap.add_argument("--vlm-cache", default=None, metavar="DIR",
                    help="VLM detectors: per-image result cache (resume); one subfolder per model / decoding / degradation")
    ap.add_argument("--vlm-dtype", choices=["bfloat16", "float16", "float32"], default=None,
                    help="VLM detectors: load the model in this dtype instead of the registry's (a precision check; part of "
                         "the cache identity)")
    ap.add_argument("--vlm-attn", choices=["sdpa", "eager", "flash_attention_2"], default=None,
                    help="VLM detectors: this attention implementation instead of the registry's (part of the cache identity)")
    ap.add_argument("--per-image", action="store_true",
                    help="non-VLM detectors: record each image's key, sha1, label and P(fake) in the output JSON (per_image)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", help="run tag (output filename + log); default <model>_<dataset> (+ _<degrade> with --degrade and no --out)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.model_config and not args.ckpt:
        ap.error("--model-config needs --ckpt")
    degrade = None
    if args.degrade is not None:     # "" is rejected by parse_degrade, not silently ignored
        try:
            degrade = parse_degrade(args.degrade)
        except ValueError as e:
            ap.error(str(e))

    if args.dataset:
        overrides = []
        if args.reals and "real_pool" in dataset_config(args.dataset, "image", args.root).DatasetConfig:
            # loaders that ship fakes only (Synthbuster) pair them with this pool when building the layout
            overrides.append(f"DatasetConfig.real_pool={Path(args.reals).expanduser().resolve()}")
        root = ensure_dataset(args.dataset, "image", args.root, download=not args.no_download and not args.dry_run,
                              overrides=overrides)
        eval_root, profile = find_eval_root(args.dataset, root)
        args.data = args.data or str(eval_root)
        args.benchmark = args.benchmark or profile
        # a degraded run must never overwrite the clean default repro/bench_results/<model>_<dataset>.json
        degrade_suffix = f"_{args.degrade}" if args.degrade is not None and not args.out else ""
        args.tag = args.tag or f"{args.model}_{args.dataset}{degrade_suffix}"
        print(f"[dataset] {args.dataset}: evaluating {args.data} with the '{args.benchmark}' profile", flush=True)
        if args.dry_run:
            return 0
    if not (args.benchmark and args.data):
        ap.error("give --dataset NAME, or both --benchmark and --data")
    if not args.tag:
        ap.error("--tag is required without --dataset")

    load_kw = {}
    if args.ckpt:
        load_kw["checkpoint"] = args.ckpt
    if args.model_config:
        spec = get_spec(args.model)
        # only the model config changes; the spec's other overrides (e.g. backbone options) stay
        overrides = [o for o in spec.overrides if not o.startswith("model@ModelConfig=")]
        spec = dataclasses.replace(spec, overrides=overrides + [f"model@ModelConfig={args.model_config}"],
                                   transform=model_transform(args.model_config))
        det = Detector(spec, device=args.device, **load_kw)
    else:
        det = synthscope.load(args.model, device=args.device, **load_kw)
    if getattr(det.spec, "predict_kind", None) == "fusion":
        ap.error(f"{args.model} combines two detectors: score both per image (the VLM with --vlm-cache, the expert with "
                 "--per-image) and fuse them with repro/adb_full/fuse.py (fuse_components, models/image/adb_full/adb_full.py)")
    if (args.vlm_dtype or args.vlm_attn) and getattr(det.spec, "vlm", None) is None:
        ap.error("--vlm-dtype / --vlm-attn apply to VLM detectors only")
    if args.vlm_dtype or args.vlm_attn:
        vlm_cfg = dataclasses.replace(det.spec.vlm, dtype=args.vlm_dtype or det.spec.vlm.dtype,
                                      attn_implementation=args.vlm_attn or det.spec.vlm.attn_implementation)
        det.spec = dataclasses.replace(det.spec, vlm=vlm_cfg)
        print(f"[vlm] dtype={vlm_cfg.dtype} attn={vlm_cfg.attn_implementation} (override)", flush=True)
    det._ensure_weights()
    spec = det.spec
    model = det._engine._get_model(spec)
    is_vlm = getattr(spec, "predict_kind", None) == "vlm"
    if args.per_image and is_vlm:
        ap.error("--per-image is for non-VLM detectors (a VLM's per-image records are always written)")
    if is_vlm and (args.native_res or args.model_config):
        ap.error("--native-res / --model-config do not apply to VLM detectors")
    eval_transform = native_res_transform(spec.transform) if args.native_res else spec.transform
    if args.native_res:
        print(f"[native-res] scoring at native resolution; transform={eval_transform}", flush=True)

    root = Path(args.data).expanduser()
    reals_root = Path(args.reals).expanduser() if args.reals else None
    select = set(args.select.split(",")) if args.select else None
    gens = discover(args.benchmark, root, reals_root, select)
    if not gens:
        print(f"No generators with real+fake under {root} (benchmark={args.benchmark})")
        return 1

    print(f"=== bench_eval {args.tag} | model={args.model} ckpt={args.ckpt or 'registry-default'} "
          f"| {args.benchmark} | {len(gens)} generators | device={det._engine.device}"
          + (f" | fakes JPEG q{args.jpeg_fakes}" if args.jpeg_fakes else "")
          + (f" | degrade {args.degrade}" if args.degrade is not None else "") + " ===", flush=True)
    if is_vlm:
        vlm_seed = args.vlm_seed if args.vlm_decoding == "official" else None
        identity = vlm_identity(spec, args.vlm_decoding, args.ckpt)
        cache = VLMCache(args.vlm_cache, vlm_variant(args.model, args.vlm_decoding, vlm_seed, args.degrade, args.jpeg_fakes,
                                                     identity), identity)
        per_image = []

    per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
    if args.per_image:
        per_image = []
    for name, reals, fakes in gens:
        if args.limit:
            reals, fakes = reals[:args.limit], fakes[:args.limit]
        labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
        print(f"  {name}: {len(reals)} real + {len(fakes)} fake ...", flush=True)
        if is_vlm:
            paths = reals + fakes
            recs, kept = score_vlm(det, spec, paths, [image_key(p, root, reals_root) for p in paths], labels.tolist(), cache,
                                   decoding=args.vlm_decoding, seed=vlm_seed, degrade=degrade, jpeg_q=args.jpeg_fakes,
                                   jpeg_from=len(reals))
        else:
            probs, kept = score(det, model, spec, reals + fakes, args.bs, transform=eval_transform,
                                jpeg_q=args.jpeg_fakes, jpeg_from=len(reals), degrade=degrade)
        if len(kept) != labels.numel():     # unreadable images are skipped: keep each score paired with its own label
            print(f"    {labels.numel() - len(kept)} unreadable image(s) left out of {name}", flush=True)
            labels = labels[kept]
        if is_vlm:
            m = vlm_metrics(recs, labels.tolist(), spec.vlm)
            per_image += [{"generator": name, "key": r["key"], "label": int(y), "logit": round(r["logit"], 4),
                           "verdict": r["verdict"], "answer_parsed": answer_parsed(r, spec.vlm),
                           "score_source": r["score_source"], "new_tokens": r["new_tokens"],
                           "finished": r["finished"], "seconds": r["seconds"], "sha1": r["sha1"]}
                          for r, y in zip(recs, labels.tolist())]
        else:
            m = metrics(probs, labels)
            if args.per_image:
                paths = reals + fakes
                per_image += [{"generator": name, "key": image_key(paths[j], root, reals_root),
                               "sha1": hashlib.sha1(paths[j].read_bytes()).hexdigest(), "label": int(y), "prob": float(p)}
                              for j, y, p in zip(kept, labels.tolist(), probs.tolist())]
        if m is None:
            print(f"    skip {name}: one class only")
            continue
        per_gen[name] = m
        for k in agg:
            agg[k].append(m[k])
        print(f"    {name}: acc={m['acc']} AP={m['ap']} AUROC={m['auroc']} "
              f"(real_acc={m['real_acc']} fake_acc={m['fake_acc']})"
              + (f" unparsed={m['n_unparsed']} unscored={m['n_unscored']} mean_tokens={m['mean_tokens']}" if is_vlm else ""),
              flush=True)

    if not per_gen:
        print("No generator produced valid metrics.")
        return 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2)

    means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "mauroc": mean(agg["auroc"])}
    result = {
        "tag": args.tag,
        "model": args.model,
        "ckpt": args.ckpt or "registry-default",
        "model_config": args.model_config,
        "benchmark": args.benchmark,
        "data_root": str(root),
        "jpeg_fakes": args.jpeg_fakes,
        "n_generators": len(per_gen),
        "means": means,
        "per_generator": per_gen,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "synthscope_version": getattr(synthscope, "__version__", "?"),
    }
    if args.degrade is not None:
        result["degrade"] = args.degrade
    if args.limit:
        result["limit"] = args.limit
    if is_vlm:
        n = len(per_image)
        unparsed = sum(not r["answer_parsed"] for r in per_image)
        result["vlm"] = {
            "decoding": args.vlm_decoding, "seed": vlm_seed, "n_images": n, "n_unparsed": unparsed,
            "n_unscored": sum(r["answer_parsed"] and r["score_source"] in NO_SCORE_SOURCES for r in per_image),
            "n_unfinished": sum(not r["finished"] for r in per_image),
            "mean_tokens": round(sum(r["new_tokens"] for r in per_image) / n, 1),
            "mean_seconds_scored": round(sum(r["seconds"] for r in per_image) / n, 2),
            "acc_pooled": round(100 * sum(r["verdict"] == ("fake" if r["label"] else "real") for r in per_image) / n, 2),
            "acc_pooled_drop": round(100 * sum(r["verdict"] == ("fake" if r["label"] else "real") for r in per_image
                                               if r["answer_parsed"]) / (n - unparsed), 2) if n > unparsed else None,
            "cache": str(cache.folder) if cache.folder else None, "identity": identity,
            "metric_note": "acc / real_acc / fake_acc from the parsed verdict: an unparsed answer is wrong, or, for a "
                           "detector whose official decision gives it a default verdict (adb_vlm: Real), that verdict; "
                           "n_unparsed counts both kinds; *_drop leave unparsed "
                           "answers out (the authors' cal_metrics.py); AUROC / AP from the fake-minus-real logit at the "
                           "verdict token (n_unscored: a parsed verdict without a verdict token enters AUROC / AP with "
                           "logit 0, P(fake) 0.5). "
                           "acc_pooled counts every scored (generator, image) pair once: a real pool shared by several "
                           "generators (ufd, synthbuster layouts) is counted once per generator; the headline acc is the "
                           "balanced mean over generators and is unaffected"}
        result["per_image"] = per_image
        if args.vlm_dtype or args.vlm_attn:
            result["vlm"]["dtype"], result["vlm"]["attn_implementation"] = spec.vlm.dtype, spec.vlm.attn_implementation
    elif args.per_image:
        result["per_image"] = per_image
    if torch.cuda.is_available() and str(det._engine.device).startswith("cuda"):
        result["gpu"] = torch.cuda.get_device_name(0)
    out = Path(args.out).expanduser() if args.out else (
        RESULTS_LOG.parent / "bench_results" / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    results_log = Path(args.results_log).expanduser() if args.results_log else RESULTS_LOG
    results_log.parent.mkdir(parents=True, exist_ok=True)
    with results_log.open("a") as f:
        f.write(json.dumps({k: result[k] for k in
                            ("date", "tag", "model", "ckpt", "benchmark", "data_root",
                             "n_generators", "means") + (("model_config",) if args.model_config else ())
                            + (("degrade",) if args.degrade is not None else ())}) + "\n")

    print(f"\n=== RESULT {args.tag}: mAP={means['map']} mAUROC={means['mauroc']} acc={means['acc']} "
          f"over {len(per_gen)} generators -> {out} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
