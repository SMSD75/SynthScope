# SynthScope — AI-Generated Image & Video Detection

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/SMSD75/AGID/actions/workflows/ci.yml/badge.svg)](https://github.com/SMSD75/AGID/actions/workflows/ci.yml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

SynthScope is a unified library for **detecting AI-generated (synthetic) images and videos**. It re-implements a range of state-of-the-art real-vs-fake detectors behind one [Hydra](https://hydra.cc/) config system and an [Accelerate](https://huggingface.co/docs/accelerate) training loop, so any method can be run, trained, and benchmarked on the same datasets, transforms, and metrics — through a one-line Python/CLI API, a web demo, or a standardized leaderboard.

> **Renamed:** SynthScope was previously called *AGID*. `import agid` and the `agid` command still work as deprecated aliases,
> and old `AGID_*` environment variables and `AGID` conda-env launch scripts keep working. The GitHub URL below still uses the old name until the repo is renamed.

Every detector is a binary classifier (`real` vs `fake`), and results are reported **per generator**, the standard way to measure generalization to unseen generative models. **17 detectors** are included (14 image, 3 video), each verified against its paper.

---

## Install

```bash
git clone https://github.com/SMSD75/AGID.git
cd SynthScope
pip install -e .                 # editable install
# extras:  ".[web]" web demo · ".[train]" wandb + dataset download · ".[dev]" contributor tooling
```

Install the [PyTorch](https://pytorch.org) build that matches your CUDA. A couple of detectors
(`clip`, `fat_former`) also need OpenAI CLIP: `pip install git+https://github.com/openai/CLIP.git`.
See [LICENSES.md](LICENSES.md) for per-method/weight licenses.

## Quick start

> **Try it without installing:** open the [Colab quickstart](notebooks/synthscope_quickstart.ipynb) or launch
> the [Gradio demo](demo/) (`pip install -e ".[demo]" && python demo/app.py`).

Run any detector on an image or video and get the probability it is AI-generated:

```python
import synthscope

synthscope.list_models()                         # detectors + which weights are present
det = synthscope.load("fat_former")              # builds lazily; downloads backbones on first use
det.predict("photo.png")
# -> {'model': 'fat_former', 'fake_probability': 0.98, 'verdict': 'fake', 'confidence': 0.98, ...}

det.predict("clip.mp4")                    # image detectors sample frames; video detectors score the clip
synthscope.load("d3").predict("clip.mp4")        # D3 is training-free — no checkpoint needed
```

From the shell:

```bash
synthscope list                                  # list detectors and weight availability
synthscope detect photo.png --model fat_former   # human-readable verdict
synthscope detect clip.mp4  --model d3 --json    # machine-readable
```

Checkpoints resolve from `$SYNTHSCOPE_CKPT_DIR` (default `~/.cache/synthscope/checkpoints`); pass
`synthscope.load(..., checkpoint="/path/to.pth")` to use your own weights. Training-free detectors
(D3, STALL) need none.

### Model zoo (Hugging Face Hub)

When a detector's weights aren't present locally, `synthscope.load(...)` **auto-downloads** them from a Hub
model-zoo repo (set `SYNTHSCOPE_HUB_REPO`, or pass `synthscope.load(..., repo="user/synthscope-checkpoints")`):

```bash
synthscope pull --model clip                                 # fetch one checkpoint
synthscope push --repo user/synthscope-checkpoints --public        # publish SynthScope-trained checkpoints (license-safe)
```

`push` uploads only SynthScope-trained checkpoints by default; weights derived from authors' official
releases are held back unless you opt in and confirm the upstream license permits redistribution.

---

## Implemented detectors

Select a detector with `model@ModelConfig=image/<name>` (image) or `model@ModelConfig=video/<name>`. Most
methods freeze a large pretrained backbone and train a lightweight head/adapter; see each model's
`README.md` under [models/](models/) for details.

| `model_type` | Method | Idea |
| --- | --- | --- |
| `clip` | **UnivFD / CLIP** (Ojha, CVPR'23) | Linear probe on frozen CLIP ViT-L/14 features. |
| `cnn` | **CNNDetection** (Wang, CVPR'20) | ImageNet ResNet-50 fine-tuned as real/fake. |
| `npr` | **NPR** (Tan, CVPR'24) | Small ResNet on neighboring-pixel residuals (up-sampling artifacts). |
| `fat_former` | **FatFormer** (Liu, CVPR'24) | Frozen CLIP + forgery-aware adapters + learnable text prompts. |
| `aide` | **AIDE** (Yan, ICLR'25) | ConvNeXt semantic features fused with low-level DCT/SRM cues (native-res 5-view). |
| `safe` | **SAFE** (Li, KDD'25) | Lightweight DWT high-frequency artifact detector + targeted augmentation. |
| `co_spy` | **Co-Spy** (Cheng, CVPR'25) | Adaptive fusion of a SigLIP semantic stream and an SD-VAE artifact stream. |
| `spai` | **SPAI** (Karageorgiou, CVPR'25) | Spectral reconstruction-similarity + context attention, any-resolution. |
| `vib_net` | **VIB-Net** (Zhang, CVPR'25) | Variational information bottleneck on frozen CLIP features. |
| `pgc` | **PGC** (Zhou, 2026) | LoRA-tuned DINOv2 + YCbCr-residual CNN, peak-guided calibration. |
| `cure` | **CuRe** (manic-cui, 2026) | Frozen Perception-Encoder Core L/14 + Q/V LoRA + source-response subspace, trained by regressing real↔generated mixing ratios on SD‑v1.4 only (official adapter). |
| `dinov2` / `dinov3` / `adapted_dino` | DINO probes | DINOv2/v3 backbone + linear head (or per-block adapters). |

**Video:** `d3` — **D3** (Zheng, ICCV'25), training-free X-CLIP trajectory dynamics · `restrav` —
**ReStraV** (Internò, 2025), DINOv2 trajectory geometry + MLP · `stall` — **STALL** (Ben Hayun,
CVPR'26), training-free spatio-temporal whitened log-likelihoods.

---

## Benchmarks

SynthScope ships a **versioned, standardized benchmark** so detectors are comparable under one protocol —
fixed test sets, per-generator metrics, one evaluation pipeline. See the ranked
**[LEADERBOARD.md](LEADERBOARD.md)** and the protocol in **[benchmark/](benchmark/)**.

```bash
python benchmark/evaluate.py --model fat_former --track forensynths --data <root> \
    --out benchmark/submissions/fat_former.json
python benchmark/leaderboard.py        # refresh the table
```

Submit your own result via PR — one JSON under [benchmark/submissions/](benchmark/submissions/); CI
validates it against the [schema](benchmark/schema.json).

**Every detector is verified against its paper** — headline numbers below (mAP unless noted; "official"
= authors' weights run through SynthScope). Full per-benchmark tables (paper / official / SynthScope-trained, all
generators) live in [`repro/AGGREGATED_BENCHMARK.md`](repro/AGGREGATED_BENCHMARK.md) and each model's
`README.md`.

| Detector | Headline benchmark | Paper | SynthScope |
| --- | --- | ---: | ---: |
| CLIP / UnivFD | UniversalFakeDetect | 95.0 | 96.44 |
| CNNDetection | ForenSynths | 90.8 | 94.42 |
| FatFormer | ForenSynths-8 | 99.7 | 99.70 (official) |
| NPR | GANGen-Detection | 96.6 | 96.49 (official) |
| AIDE | GenImage | 97.1 | 96.42 |
| SAFE | GenImage-8 | 99.3 | 99.51 |
| Co-Spy | Co-Spy-Bench | 96.0 | 96.60 |
| SPAI | GenImage | ~91 AUROC | 94.22 |
| PGC | GenImage | 100 | 100 (official) |
| CuRe | AIGIBench (20 subsets, balanced acc) | 92.8 acc | 92.85 acc (official) |
| VIB-Net | GenImage SD-v1.4 | 95.5 | 83.94 (no official weights released) |
| D3 (video) | GenVideo-Val | ~98 | 95.95 AUROC |
| ReStraV (video) | GenVidBench | 93.0 | 92.7 |
| STALL (video) | GenVideo-Val | ~80 AUC | 83.5 AUROC |

---

## Web demo

A self-contained web app to run any detector on an uploaded image/video lives in
[`interface/`](interface/) — a FastAPI backend (model registry + lazy inference engine) and a React UI:

```bash
pip install -r interface/requirements.txt
python interface/scripts/collect_checkpoints.py   # assemble the checkpoint folder
bash interface/run_backend.sh                      # serves http://<host>:8000
```

It exposes every detector with its exact preprocessing, runs image detectors frame-by-frame on videos
and video detectors through their native clip pipeline, shows which weights back each model
(official / SynthScope-trained / training-free), and includes a benchmark view, the **What's Trending** feed ([TRENDING.md](TRENDING.md)), and a *SOTA vs
detectors* demo that generates images with hosted models (Nano Banana, GPT Image, Grok, FLUX.1) or **local open-weight
ones (FLUX.2 [klein] 4B, Qwen-Image-2.1, Ideogram 4.0)** and scores them with every detector. See
[interface/README.md](interface/README.md).

---

## Train your own

Compose three Hydra config groups and override any field with dot-paths:

```bash
accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
  trainer@TrainerConfig=image/aide model@ModelConfig=image/aide dataset@DatasetConfig=image/genimage \
  DatasetConfig.train_batch_size=16 logger.name=aide
```

Or use a method's ready-made recipe in [experiments/](experiments/) (e.g. `bash experiments/image/aide.sh`).
Datasets (`dataset@DatasetConfig=<name>`) auto-download to their `root_dir` and restructure into a
common `real/`/`fake/` layout; transforms are named pipelines in [transformations.py](transformations.py).
Training interleaves validation/test each epoch and logs per-generator metrics to Weights & Biases.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/ADD_DETECTOR.md](docs/ADD_DETECTOR.md) to add a detector.

---

<!-- TRENDING:START -->
## 📈 What's Trending

A curated, daily-refreshed list of recent image & video AIGC-detection papers with public code: **15 image · 6 video** tracked (13 already integrated in SynthScope), last updated **2026-10-04**. → **[TRENDING.md](TRENDING.md)** (also in the web UI's *What's Trending* tab).
<!-- TRENDING:END -->
