# SynthScope capabilities (orientation for an agent building a new detector)

SynthScope is a **reproduction + benchmarking library for AI-generated *image and video* detection**. It
hosts many published detectors behind one interface, reproduces each on its own benchmark, and
serves a web UI + a Python/CLI API. Read this before adding a method, then follow
[`docs/ADD_DETECTOR.md`](ADD_DETECTOR.md) for the exact checklist.

> Environment: everything runs in the **`SynthScope` conda env** —
> `/home/ssalehi/miniconda3/bin/conda run -n SynthScope <cmd>`. Config is **Hydra** (`conf/`).

---

## 1. What's already integrated (17 detectors)

`synthscope.list_models()` / `from synthscope.api import list_specs`. Each is an `SynthScope<Name>Model` with a
matching `conf/model/<modality>/<key>.yaml`, a `trainer.py._build_model` branch, an
`interface/backend/registry.py` `ModelSpec`, a per-model `README.md`, and a `tests/test_<k>_smoke.py`.

| key | modality | predict_kind | weights | interp map | method |
|---|---|---|---|---|---|
| `clip` | image | ce | synthscope | – | UnivFD / CLIP linear probe |
| `cnn` | image | ce | synthscope | – | CNNDetection (ResNet-50) |
| `dinov2`,`dinov3`,`adapted_dino` | image | ce | synthscope | – | DINO probes / adapter-tuned |
| `npr` | image | bce_npr | official | npr_residual | Neighboring-Pixel Relationships |
| `safe` | image | ce | synthscope | safe_hf | High-freq DWT artifacts |
| `aide` | image | ce | official | aide_dct | ConvNeXt-XXL + DCT/SRM (native-res 5-view) |
| `fat_former` | image | ce | official | – | Forgery-aware adaptive transformer (CLIP) |
| `spai` | image | bce | synthscope* | spai_patch | Spectral, any-resolution (MFM ViT-B/16) |
| `co_spy` | image | bce | synthscope* | – | Semantic + artifact fusion (SigLIP + SD-VAE) |
| `pgc` | image | bce | official | pgc_peak | LoRA-DINOv2 + YCbCr residual, peak-guided |
| `cure` | image | ce | official | – | Frozen PE-Core L/14 + Q/V LoRA + source-response subspace (mixing-ratio regression) |
| `vib_net` | image | vib | synthscope | – | Variational information bottleneck (FLAVA) |
| `d3` | video | video | training-free | – | Disentangled detail dynamics (X-CLIP) |
| `restrav` | video | video | synthscope | – | Representation-trajectory geometry (DINOv2) |
| `stall` | video | video | training-free | – | Spatio-temporal whitened log-likelihoods (DINOv3) |

`*` served via official weights remapped into the SynthScope class. **`predict_kind`** sets the score
convention the engine applies to `forward()` logits: `ce`→`softmax[:,1]`, `bce`→`sigmoid`,
`bce_npr` (NPR residual transform then sigmoid), `vib` (tuple out), `video` (clip in).

---

## 2. The detector contract (what a new method must implement)

`models/<modality>/<key>/<key>.py`:
```python
class SynthScope<Name>Model(nn.Module):
    def __init__(self, config: DictConfig): ...        # build arch from config.ModelConfig
    def forward(self, images) -> logits: ...           # RAW logits, not probs; model-specific
                                                       # preprocessing (residual/DWT/DCT) goes INSIDE forward
    def compute_loss(self, batch) -> {"loss", "outputs"}  # outputs = [N,2] real/fake logits, used by training
```
Repo rule: **all imports + helper classes at module top level** (no nested defs/imports). Vendor the
official arch pieces verbatim where possible and document deviations in the per-model README.

To wire it in: `conf/model/<modality>/<key>.yaml` (`model_type: <key>`), a branch in
`trainer.py._build_model` (dispatches on `model_type`), and a `ModelSpec` in the interface registry
(see `ADD_DETECTOR.md`). The web engine (`interface/backend/engine.py`) lazy-imports by `model_type`.

---

## 3. Training infrastructure

- **`trainer.py`** (`SynthScopeModelTrainer`) + **`main.py`** entrypoint; Hydra config in `conf/config.yaml`
  with groups `model/`, `dataset/`, `trainer/`. Multi-GPU via 🤗 `accelerate` (`accelerate_config.yaml`).
- **`dataset.py`** — image/video dataset loaders; `_decode_rgb` feeds uint8 `[3,H,W]` tensors;
  `sample_video_frames(..., edge_crop=)` for video.
- **`transformations.py`** — composable named transforms / augmentations: `resize_224`,
  `resize_short_{256,384,448}`, `center_crop_{224,256,384,448}`, `random_crop_*`, `pad_random_crop_224`
  (PGC), `gaussian_blur`, `jpeg_compression`, `random_horizontal/vertical_flip`, `random_rotation_180`,
  `color_jitter_safe`, `to_tensor`/`to_dtype`. **Eval-transform fidelity is critical** — artifact/
  low-level detectors (SAFE/AIDE/SPAI/PGC/NPR) are wrecked by a wrong resize; match the official eval
  exactly (see `repro/investigate/pgc_gap_investigation.md` and the SAFE crop-not-resize lesson).
- **`metrics.py`** — torchmetrics binary `Accuracy` (real/fake), `AveragePrecision` (AP),
  `AUROC`. Threshold-free AP/AUROC are the headline metrics (esp. video, where per-gen real:fake
  ratios make mAP degenerate — prefer AUROC there).

---

## 4. Evaluation / benchmarking

- **Image**: `benchmark/bench_eval.py --model <k> --benchmark {forensynths|ufd|pergen|pool} --data <root> [--ckpt <pth>] [--native-res] --tag <t>`. Auto-discovers `<gen>/{0_real,1_fake}` (or `{ai,nature}`, shared-reals) layouts; prints per-generator acc/AP/AUROC + mean; writes `repro/bench_results/<tag>.json`.
- **Video**: `benchmark/bench_eval_video.py --model {d3|restrav|stall} --benchmark {genvideo|fakeonly|pergen}`.
- `--ckpt` overrides the registry checkpoint (for official-vs-SynthScope side-by-side).
- `benchmark/aggregate_results.py`, `leaderboard.py`, `evaluate.py`, `schema.json` aggregate runs.
- Results live in `repro/`: `RESULTS_SUMMARY.md` (read first), `AGGREGATED_BENCHMARK.md`,
  `BENCHMARK_COVERAGE_MATRIX.md`, `REPRODUCED_MODELS_BENCHMARKS.md`, `repro/bench_results/*.json`,
  and per-method deep-dives in `repro/investigate/`.

---

## 5. Datasets (explorer registry: `interface/backend/datasets.py`, `DATASET_DEFS`)

Known image sets: `cnndetection` (ForenSynths/ProGAN), `genimage`, `genimage_full` (8-gen),
`universialfakedetection` (Ojha UFD), `aigibench`, `synthbuster`, `ditfake`, `diffusionforensics`,
`diffusion1kstep`, `aigcdetect`, `commgen15`, `realchain`, `evalgen`, `cospy`/`cospy_subset`, `spai_ldm`, `freqnetdetection`.
Video: `genvideo_val`, `restrav_data`, `evalcrafter`, `genvidbench`, `videofeedback`, `comgenvid`.

Each `DatasetDef` lists candidate roots; **`_resolve_root` is node-portable** — it remaps the configured
`/hddstore` path onto whatever scratch mount exists (`/hddstore`↔`/ssdstore`↔`/nvmestore`), overridable
with `SYNTHSCOPE_DATA_PREFIXES`. Labels are inferred from path tokens (`0_real`/`nature`→real,
`1_fake`/`ai`→fake). **Not every dataset is physically present on every node** — data is node-local
scratch; check what's staged (e.g. CNNDetection, GenImage, GenImage_full are on cn029 `/ssdstore`).

---

## 6. The `synthscope` package (Python + CLI)

```python
import synthscope
det = synthscope.load("pgc")                  # or load(key, checkpoint=PATH, device=...)
det.predict("img.png")                  # -> {fake_probability, verdict, confidence, interpret_map, ...}
synthscope.list_models(); synthscope.available_models()   # registered / weights-present
```
CLI: `synthscope list`, `synthscope detect <file> --model <k>`, `synthscope pull <k>` (fetch weights from the HF model
zoo), `synthscope push` (maintainer). Weights resolve under **`$SYNTHSCOPE_CKPT_DIR`** (default
`/hddstore/ssalehi/SynthScope/checkpoints`; override per node) or are auto-downloaded.

---

## 7. Web interface (`interface/`, FastAPI + uvicorn)

`bash interface/run_backend.sh` (`SYNTHSCOPE_PORT`, `SYNTHSCOPE_HOST`). Serves `frontend/dist` if built, else the
no-build `static/index.html`. Endpoints (`interface/backend/app.py`): `/api/models`, `/api/predict`
(image/video, lazy model load + LRU), `/api/datasets[.../sample]`, `/api/predict_dataset`,
`/api/score_batch`, `/api/leaderboard`, `/api/generator_benchmarks`, and a **SOTA-vs-detector** demo
(`/api/sota/{generators,real_image,compare}`) that generates an image from a prompt (`generators.py`,
API-keyed SOTA generators incl. local Ideogram) and scores real vs generated across detectors.
On-cluster launch scripts: `repro/webui_cn029_serve.sbatch` (GPU on free cees node, checkpoints staged
to `/ssdstore`), `repro/webui_cn023_cpu.sbatch`. Reach it via SSH tunnel through the `ivi-h1` login host.

### Interpretability maps
The engine returns a per-detector artifact map for the 5 native-signal detectors (`spec.interpret`):
`safe_hf` (DWT high-freq band), `npr_residual` (pixel residual), `pgc_peak` (peak-guided residual
score), `aide_dct` (native-res DCT patch energy), `spai_patch` (per-patch spectral fake-evidence). Each
yields **`interpret_map`** (bare jet heatmap) **+ `interpret_overlay`** (heatmap blended on the input,
transform-aligned). A new method exposes one by setting `interpret="<kind>"` on its `ModelSpec` and
adding a branch in `engine._interpret_map` that produces a 2-D saliency tensor.

---

## 8. Trending pipeline & repo hygiene

`repro/fetch_arxiv_trending.py` → `repro/generate_trending_readme.py` keep a curated
`interface/backend/trending.json` (image+video) and `TRENDING.md` (full list) + a short teaser in the README `<!-- TRENDING -->` block + a web tab.
Daily rule: new detectors are integrated **on a `trending/<key>` branch and pushed for review — never
auto-merged to `main`**; only `trending.json`/`TRENDING.md`/README docs go to `main` from the pipeline. Never commit
weights (`*.pth`), `.env`, or caches.

---

## 9. Cluster cheat-sheet (IvI SLURM)

- **cn023** (`quva`/`quvausers`, A6000) holds all detector checkpoints + most datasets on node-local
  **`/hddstore`**; GPUs are often booked for days. `SYNTHSCOPE_CKPT_DIR=/hddstore/ssalehi/SynthScope/checkpoints`.
- **cees** (`ceesusers`: cn029 etc., a5000, free) and **cees6000** (cn030/31, rtx6000) use node-local
  `/ssdstore` / `/nvmestore`. To run there, stage checkpoints/data from cn023 (`rsync -aL`, dereference
  the symlinks) — note `pam_slurm_adopt` only allows ssh to a node where you hold an active job.
- `/ivi/zfs/...` is the shared home (24 GiB user quota — don't park big checkpoints there);
  HF cache is shared at `/ivi/.../SynthScope/.cache_shared/hf` (`HF_HOME`). DINOv3 weights gated, under
  `dinov3_checkpoints/` (`DINO_V3_WEIGHTS`); SPAI MFM backbone via `SYNTHSCOPE_MFM_CKPT`.

---

## 10. Starting a new method — TL;DR

1. Confirm it passes both trending rules (public repo + strong results). Read `ADD_DETECTOR.md`.
2. `git checkout -b trending/<key>`; add the 5 files (model, conf, trainer dispatch, registry ModelSpec, README) + smoke test.
3. **Match the official eval transform exactly** in the `ModelSpec` (resize vs crop, norm) — this is the #1 silent-failure source.
4. Reproduce on the paper's benchmark with official weights via `bench_eval.py --ckpt`; fix `remap_<k>` for key mismatches; record per-gen AP/AUROC in the Paper/SynthScope_official/SynthScope_train schema.
5. (Optional) add an interpretability map. Push the branch; do **not** merge to `main`.
