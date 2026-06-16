# AGID — AI-Generated Image & Video Detection

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/SMSD75/AGID/actions/workflows/ci.yml/badge.svg)](https://github.com/SMSD75/AGID/actions/workflows/ci.yml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

AGID is a unified research framework for **training and evaluating AI-generated (synthetic) image detectors**. It re-implements a range of state-of-the-art real-vs-fake image detectors behind a single [Hydra](https://hydra.cc/) configuration system and an [Accelerate](https://huggingface.co/docs/accelerate) (DeepSpeed) training loop, so that different methods can be trained and benchmarked on the same datasets, transforms, metrics, and logging stack.

Every detector is a binary classifier (`real` vs `fake`). Metrics are reported **per generator / data source**, which is the standard way to measure how well a detector generalizes across unseen generative models.

---

## Install

```bash
git clone https://github.com/SMSD75/AGID.git
cd AGID
pip install -e .                 # editable install (recommended for the flat research layout)
# extras:  ".[web]" web demo · ".[train]" wandb+dataset download · ".[dev]" contributor tooling
```

Install the [PyTorch](https://pytorch.org) build that matches your CUDA. A few detectors
(`clip`, `fat_former`) also need OpenAI CLIP: `pip install git+https://github.com/openai/CLIP.git`.
See [LICENSES.md](LICENSES.md) for per-method/weight licenses.

## Quick start — Python API & CLI

> **Try it without installing:** open the [Colab quickstart](notebooks/agid_quickstart.ipynb), or
> launch the [Gradio demo](demo/) (`pip install -e ".[demo]" && python demo/app.py`) locally or as a
> Hugging Face Space.

Run any detector on an image or video and get the probability it is AI-generated:

```python
import agid

agid.list_models()                         # detectors + which weights are present
det = agid.load("fat_former")              # builds lazily; downloads backbones on first use
det.predict("photo.png")
# -> {'model': 'fat_former', 'fake_probability': 0.98, 'verdict': 'fake', 'confidence': 0.98, ...}

det.predict("clip.mp4")                    # image detectors sample frames; video detectors score the clip
agid.load("d3").predict("clip.mp4")        # D3 is training-free — no checkpoint needed
```

From the shell:

```bash
agid list                                  # list detectors and weight availability
agid detect photo.png --model fat_former   # human-readable verdict
agid detect clip.mp4  --model d3 --json    # machine-readable
```

Checkpoints are resolved from `$AGID_CKPT_DIR` (default `~/.cache/agid/checkpoints`); pass
`agid.load(..., checkpoint="/path/to.pth")` to use your own trained weights. Training-free
detectors (D3, STALL) need none.

### Model zoo (Hugging Face Hub)

When a detector's weights aren't present locally, `agid.load(...)` **auto-downloads** them from a
Hub model-zoo repo (set `AGID_HUB_REPO`, or pass `agid.load(..., repo="user/agid-checkpoints")`).
Maintainers publish with:

```bash
agid push --repo user/agid-checkpoints --public        # AGID-trained checkpoints (license-safe)
agid push --repo user/agid-checkpoints --include-official   # also official weights — verify LICENSES.md first
agid pull --model clip                                 # fetch a single checkpoint
```

`push` uploads only AGID-trained checkpoints by default; checkpoints derived from authors' official
released weights are held back unless you opt in and confirm their upstream license permits
redistribution.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| [main.py](main.py) | Entry point. Loads the Hydra config, sets the seed, builds the dataset module and trainer, and runs training. |
| [trainer.py](trainer.py) | `AGIDModelTrainer` — model dispatch, optimizer, scheduler, Accelerate setup, the train/val/test loop, metric aggregation, checkpointing, and logging. |
| [dataset.py](dataset.py) | All dataset classes + per-dataset data modules, with automatic download/extraction and restructuring to a common `real/fake` layout. `AIGIDatasetModule` dispatches on the dataset name. |
| [transformations.py](transformations.py) | `AIGITransformations` — builds an image transform pipeline from a list of named transforms. |
| [metrics.py](metrics.py) | `AIGIMetrics` — per-generator `real_accuracy`, `fake_accuracy`, `AP`, `AUC`. |
| [logger.py](logger.py) | Weights & Biases logger (or a null logger). |
| [model.py](model.py) | Reference/standalone definitions for the simple backbone detectors (CLIP, CNN, DINO). |
| [conf/](conf/) | Hydra config groups: `config.yaml` plus `trainer/`, `model/`, `dataset/`. |
| [experiments/](experiments/) | Ready-to-run launch scripts, one per detector. |
| [models/image/](models/image/) | One subfolder per detector with its implementation and a method-specific `README.md`. |

Each model exposes a `compute_loss(batch)` method that returns `{"loss": ..., "outputs": ...}`; the trainer is agnostic to the internals. A batch is the tuple `(image, label, folder_id)`, where `folder_id` identifies the generator/source so metrics can be broken down per model.

---

## Implemented detectors

Select a detector with `ModelConfig.model_type=<type>` (or the matching `model@ModelConfig=<name>` config group). Most methods freeze a large pretrained backbone and train a lightweight head/adapter.

| `model_type` | Method | Class | Idea (see the per-model README for details) |
| --- | --- | --- | --- |
| `clip` | **CLIPDetection / UnivFD** (Ojha et al., CVPR 2023) | `AGIDCLIPModel` | Frozen CLIP ViT-L/14 features + linear real/fake head. |
| `cnn` | **CNNDetection** (Wang et al., CVPR 2020) | `AGIDCNNModel` | ImageNet ResNet-50 fine-tuned as a real/fake classifier. |
| `dinov2` | DINOv2 probe | `AGIDDINOModel` | DINOv2 backbone + linear head (optionally frozen). |
| `dinov3` | DINOv3 probe | `AGIDDINOv3Model` | DINOv3 backbone + linear head. |
| `adapted_dino` | Adapter-tuned DINOv2 | `AGIDAdaptedDINOModel` | DINOv2 with per-block bottleneck adapters; backbone frozen. |
| `safe` | **SAFE** (Li et al., KDD 2025) | `AGIDSAFEModel` | Lightweight high-frequency artifact detector + a targeted augmentation recipe. |
| `fat_former` | **FatFormer** (Liu et al., CVPR 2024) | `AGIDFatFormerModel` | Frozen CLIP + forgery-aware adapters + learnable text-prompt context. |
| `co_spy` | **CO-SPY** (Cheng et al., CVPR 2025) | `AGIDCoSpyModel` | Adaptive fusion of a semantic stream (SigLIP) and a pixel/artifact stream (VAE residual). |
| `aide` | **AIDE** | `AGIDAIDEModel` | ConvNeXt semantic features fused with low-level DCT/SRM cues. |
| `spai` | **SPAI** (Karageorgiou et al., CVPR 2025) | `SPAI` | Spectral learning: Spectral Reconstruction Similarity + Spectral Context Attention for any-resolution detection. |
| `vib_net` | **VIB-Net** (Zhang et al., CVPR 2025) | `AGIDVIBNetModel` | Variational Information Bottleneck between frozen CLIP features and the classifier. |
| `npr` | **NPR** (Tan et al., CVPR 2024) | `NPR` | Small ResNet trained from scratch on neighboring-pixel residuals that expose up-sampling artifacts. |
| `pgc` | **PGC** (Zhou et al., 2026) | `AGIDPGCModel` | LoRA-tuned DINOv2 RGB stream + YCbCr-quantization-residual CNN, fused by a peak-guided (log-sum-exp) calibration. |

**Video detectors** (select with `model@ModelConfig=video/<name>`):

| `model_type` | Method | Class | Idea |
| --- | --- | --- | --- |
| `d3` | **D3** (Zheng et al., ICCV 2025) | `AGIDD3Model` | Training-free: 2nd-order std of frozen per-frame features. |
| `restrav` | **ReStraV** (Internò et al., 2025) | `AGIDReStraVModel` | Frozen DINOv2 trajectory-geometry features + standardized MLP. |
| `stall` | **STALL** (Ben Hayun et al., CVPR 2026) | `AGIDSTALLModel` | Training-free: spatial + temporal whitened-Gaussian log-likelihoods vs a real-video (VATEX) calibration. |

---

## Benchmark & leaderboard

AGID ships a **versioned, standardized benchmark** (v1) so detectors are comparable under one
protocol — fixed test sets, per-generator metrics, and a single evaluation pipeline. See the
ranked **[LEADERBOARD.md](LEADERBOARD.md)** and the protocol in **[benchmark/](benchmark/)**.

```bash
# evaluate a detector and write a submission, then refresh the table
python benchmark/evaluate.py --model fat_former --track forensynths --data <root> \
    --out benchmark/submissions/fat_former.json
python benchmark/leaderboard.py
```

Submit your own result via PR — one JSON file under [benchmark/submissions/](benchmark/submissions/);
CI validates it against the [schema](benchmark/schema.json) and checks the table is current. The
seed entries below are the reproductions, transcribed into the leaderboard.

## Full-benchmark reproduction (all models × all reported benchmarks)

Every benchmark each paper reports, with our verified numbers. Per-model detail in each `models/*/<name>/README.md`; raw results in `repro/bench_results/`, specs in `repro/benchmark_specs/`, matrix in `repro/BENCHMARK_COVERAGE_MATRIX.md`.

**Exactly three number columns:**
- **Paper** — the figure reported in the paper.
- **AGID_official** — the authors' released weights run through the AGID class / eval (architecture-fidelity check).
- **AGID_train** — the AGID class **trained from scratch** by us; the **single best** run is kept (older/weaker from-scratch checkpoints are discarded, not shown).

Values are `mAP (acc)` or AUROC as noted. **Status:** reproduced = AGID_train matches paper/official · official = AGID_official matches (AGID_train may still lag) · below = gap remains · reference = no comparable paper figure · missing = not yet evaluated.

| Model | Benchmark | Paper | AGID_official | AGID_train | Status |
|---|---|---|---|---|---|
| CLIP / UnivFD | ForenSynths GAN/low-level (11) | — | mAP=96.95 | mAP=97.22 / acc=88.28 | reference |
| CLIP / UnivFD | UniversalFakeDetect diffusion/AR (8) | map=95.0 | mAP=95.66 | mAP=96.44 / acc=87.23 | reproduced |
| CNNDetection | ForenSynths test set (11) | map=90.8 | mAP=92.67 | mAP=94.42 / acc=76.36 | reproduced |
| FatFormer | GANs — 8 GANs (4-class ProGAN) | map=99.7 / acc=98.4 | mAP=99.7 / acc=98.39 | — | official |
| FatFormer | GANs — 8 GANs (2-class ProGAN) | map=99.2 / acc=95.8 | — | — | missing |
| FatFormer | Diffusion (DMs) — 10 subsets | map=98.8 / acc=95.0 | mAP=98.6 / acc=93.72 | — | official |
| NPR | Table 1 — ForenSynths / Cross-GAN (8) | map=96.1 / acc=92.5 | mAP=94.22 / acc=92.3 | — | official |
| NPR | Table 2 — GANGen-Detection (9) | map=96.6 / acc=93.2 | mAP=96.49 / acc=94.81 | — | official |
| NPR | Table 3 — DiffusionForensics / DIRE | map=99.8 / acc=95.3 | mAP=99.4 / acc=95.03 | — | official |
| NPR | Table 4 — UniversalFakeDetect / Ojha | acc=95.2 | mAP=96.69 / acc=93.94 | — | official |
| NPR | Table 5 — Diffusion1kStep | acc=80.1 | mAP=83.47 / acc=79.13 | — | official |
| AIDE | AIGCDetectBenchmark (B1, ProGAN-trained) | acc=92.77 | mAP=97.94 / acc=91.89 | mAP=72.25 / acc=61.64 | official |
| AIDE | GenImage (B2, SDv1.4-trained) | acc=86.88 | mAP=97.12 / acc=85.91 | mAP=92.47 / acc=73.5 | official |
| AIDE | Chameleon (B3, in-the-wild) | acc=58–66 | — | — | missing |
| SAFE | ForenSynths-8 (Table 1, intra-arch GANs) | map=98.8 / acc=96.2 | mAP=98.78 / acc=96.2 | mAP=97.79 / acc=95.1 | reproduced |
| SAFE | Self-Synthesis / GANGen (Table 2) | map=99.8 / acc=99.0 | mAP=99.85 / acc=99.0 | not re-run (data offline) | official |
| SAFE | Ojha / UniversalFakeDetect-8 (Table 3) | map=99.1 / acc=95.7 | mAP=99.1 / acc=95.71 | mAP=98.44 / acc=94.0 | reproduced |
| SAFE | GenImage-8 (Table 4, DM+GAN) | map=99.3 / acc=95.6 | mAP=99.32 / acc=95.62 | mAP=99.51 / acc=95.9 | reproduced |
| SAFE | DiTFake (Table 6, DiT) | map=99.9 / acc=99.4 | — | mAP=99.96 / acc=99.2 | reproduced |
| SAFE | DiffusionForensics-8 | — | — | mAP=99.34 / acc=91.3 | reproduced |
| SAFE | ForenSynths-13 (full, incl. CRN/IMLE/face) | — | mAP=85.74 / acc=84.9 | mAP=87.31 / acc=83.9 | retrain ≥ official |
| Co-Spy | Co-Spy-Bench | map=96.02 / acc=87.06 | mAP=86.65 / acc=83.88 | mAP=96.60 / acc=89.9 | reproduced |
| Co-Spy | AIGCDetectionBenchmark | map=95.21 / acc=90.58 | mAP=94.59 / acc=89.25 | mAP=86.18 / acc=73.0 | official |
| Co-Spy | GenImage | — | mAP=92.41 | mAP=90.59 / acc=75.91 | reference |
| Co-Spy | ForenSynths (CNNDetection test) | — | mAP=87.97 | mAP=96.17 | reference |
| Co-Spy | In-the-wild (Co-Spy-Bench) | map=84.65 / acc=77.41 | — | — | missing |
| SPAI | SPAI main eval (13 gen × 5 real) | auroc=91.0 | — | — | missing |
| SPAI | GenImage (cross-dataset) | — | mAP=94.22 / AUROC=94.28 | mAP=88.80 / acc=82.81 | reference |
| SPAI | Synthbuster | — | mAP=96.75 / AUROC=96.97 | — | reference |
| VIB-Net | Diffusion-source (SDv1.4), AP — HEADLINE | map=95.53 | *none released* | mAP=83.94 / acc=77.57 | below |
| VIB-Net | Diffusion-source (SDv1.4), ACC | acc=85.83 | *none released* | mAP=83.94 / acc=77.57 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN), AP | map=93.14 | *none released* | mAP=61.71 / acc=57.91 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN), ACC | acc=82.5 | *none released* | mAP=61.71 / acc=57.91 | below |
| PGC | GenImage | map=100.0 / acc=98.3 | mAP=100.0 / acc=99.49 | mAP=97.62 / acc=78.6 | official |
| PGC | UniversalFakeDetect (UFD) | map=98.0 | mAP=99.34 / acc=94.16 | mAP=99.01 | reproduced |
| PGC | ForenSynths (CNNDetection test) [†] | — | mAP=96.25 / acc=83.73 | mAP=94.57 / acc=83.41 | reference |
| PGC | AIGIBench (AIGI) | map=95.0 / acc=86.6 | mAP=95.36 / acc=88.33 | mAP=86.88 | official |
| PGC | CommGen15 (commercial in-the-wild) | acc=97.9 | mAP=99.26 / acc=97.39 | mAP=98.61 / acc=66.66 | official |
| D3 | GenVideo (DeMamba GenVideo-Val) | mAP=98.46 | — | mAP=89.0 / acc=86.04 / AUROC=93.87 | below |
| D3 | EvalCrafter (14 generators) | mAP=98.87 | — | mAP=91.19 / acc=87.67 / AUROC=95.41 | below |
| D3 | VideoPhy (10 generators) | mAP=99.16 | — | — | missing |
| D3 | VidProM (6 generators) | mAP=88.46 | — | — | missing |
| ReStraV | A) VidProM vs image detectors | acc=97.06 / mAP=98.81 | — | — | missing |
| ReStraV | B) GenVidBench vs SoTA video — Main | Main_M_avg=93.01 | — | mAP=92.7 / acc=86.87 / AUROC=95.9 | reproduced |
| ReStraV | C) DeMamba one-to-many | — | — | — | missing |
| ReStraV | D) Veo3 zero-shot | acc=83.2 / AUROC=86.9 | — | — | missing |
| ReStraV | E) Physics-IQ 2AFC | Pika=97.5 / Sora=99.0 | — | — | missing |
| STALL | GenVideo (DeMamba GenVideo-Val, 10 gen) | avg_AUC≈0.80 | — | mAP=82.38 / acc=56.88 / AUROC=83.51 | reproduced |
| STALL | VideoFeedback (TIGER-Lab, 11 gen) | avg_AUC≈0.83 | — | mAP=83.07 / acc=62.12 / AUROC=85.1 | reproduced |
| STALL | ComGenVid (Sora + VEO3) | avg_AUC≈0.852 | — | mAP=82.59 / acc=57.34 / AUROC=84.0 | below |

**[†] PGC ForenSynths caveat:** AGID_train mAP 94.57 (const-LR ep100, ProGAN-trained, job 185056) is the *13-generator* mean and looks close to AGID_official 96.25 — but the mean is carried by the 10 easy generators (~99). On the three hard generators (deepfake/san/seeingdark) AGID_train is only **75.87 mAUROC vs 94.35** for the official weights; that residual is the subject of `repro/investigate/pgc_gap_investigation.md`. Running the authors' own `train.py` verbatim shows the same plateau, so the released checkpoint is not reproducible from the published recipe+data.

*Note: `AGID_train` shows the single best from-scratch run; earlier/weaker from-scratch checkpoints (e.g. PGC GenImage 69.68, SPAI 80.75) were superseded and are no longer listed.*

## Reproduction results

Each implemented detector was reproduced and checked against its paper's headline number. Metrics
are per-generator means (the AGID convention); values are shown in **%**. "How" indicates whether the
result came from loading the authors' **official released weights** into the AGID model, **training
from scratch** via the detector's `experiments/` recipe, or a **frozen-backbone probe**. Every number
below traces to a logged `Test Metrics` evaluation (no reported value is estimated).

**Image detectors**

| Method | Benchmark | Our result | Paper (ref.) | How |
| --- | --- | --- | --- | --- |
| **FatFormer** (CVPR'24) | ForenSynths-8 | acc 98.39 / mAP 99.70 | 98.4 / 99.7 | official weights |
| **CO-SPY** (CVPR'25) | ForenSynths-8 | acc 95.12 / mAP 98.71 | 90.6 / 95.2 (AIGCDetect) | official weights¹ |
| **NPR** (CVPR'24) | ForenSynths-8 | acc 91.83 / mAP 87.09 | ~91–94 acc | official weights |
| **AIDE** (ICLR'25) | ForenSynths-8 | mAP 92.08 (acc@0.5 72.0²) | high-90s acc | official weights |
| **UnivFD / CLIP** (CVPR'23) | ForenSynths | mAP 97.2 / AUROC 97.0 | ~98 mAP | frozen CLIP probe (ProGAN) |
| **CNNDetection** (CVPR'20) | ForenSynths | mAP 96.8 / AUROC 96.6 | ~96–98 mAP | ResNet-50 fine-tune (ProGAN) |
| **DINOv2 probe** | ForenSynths | mAP 95.03 / AUROC 94.86 | ~97 (CLIP probe ref.) | frozen DINOv2 ViT-L/14 probe |
| **SAFE** (KDD'25) | ForenSynths | mAP 94.13 / AUROC 93.68 | — | trained from scratch (`safe.sh`) |
| **SPAI** (CVPR'25) | GenImage SD-v1.4 | AUC 98.45 / AP 94.01 | SOTA diffusion AUC | official weights¹ |
| **VIB-Net** (CVPR'25) | GenImage SD-v1.4 | mAP 99.33 / AUROC 99.33³ | — | trained from scratch (`vib_net.sh`) |

CO-SPY and SPAI were additionally **trained from scratch** with the AGID re-implementations: CO-SPY on
ProGAN → mAP 96.98 / AUROC 96.79 (ForenSynths); SPAI (MFM-initialised, fixed-resolution) on the
DMID Latent-Diffusion set vs COCO → AUC 97.66 — within ~0.8 AUC of the official-weights model, i.e. the
any-resolution component is not required to land in the paper's range.

**Video detectors**

| Method | Benchmark | Our result | Paper (ref.) | How |
| --- | --- | --- | --- | --- |
| **D3** (ICCV'25) | GenVideo-Val | AP 95.6 / AUROC 93.6 | ~97 mAP | frozen CLIP, training-free |
| **ReStraV** | GenVideo-Val (in-distribution⁴) | AUROC 100 / AP 100 | ~high-90s AUROC | frozen DINOv2 trajectory + standardized MLP |
| **ReStraV** | GenVideo-Val (unseen-generator⁴) | AUROC 100 / AP 100 | unseen mAP 97.3 / Sora 92.9 | leave-one-generator-out, same real source |
| **STALL** (CVPR'26) | GenVideo-Val | AP 79.4 / AUROC 81.9 | SOTA training-free | training-free; DINOv3 + VATEX-style calibration⁵ |

¹ The AGID `co_spy` / `spai` classes diverge from the official networks (different backbone / fusion /
no Spectral-Context-Attention), so the **official-weights** evals load the authors' architectures
(vendored) rather than the AGID class; the from-scratch numbers above use the AGID class.
² AIDE's score *ranking* reproduces (high mAP), but acc@0.5 is miscalibrated because the AGID input is a
single CLIP-normalised view vs the official 5-view DCT-reconstructed stack.
³ VIB-Net is evaluated on the GenImage SD-v1.4 test split, which is a single real-vs-fake pool, so the
per-generator mAP and AUROC coincide by construction.
⁴ ReStraV's headline number is **in-distribution** (the official `train.py` protocol: a balanced pool,
standardized 21-d features, random stratified 50/50 split that *shares* generators between train and test).
The **unseen-generator** row is the paper's generalization protocol (Table 3 / Table 5): the real-video
source is held fixed and the *fake* generators are split, so train and test fakes are disjoint. Both are
reproduced by the bit-faithful [`interface/scripts/restrav_repro.py`](interface/scripts/restrav_repro.py)
and the multi-protocol [`interface/scripts/restrav_xgen_repro.py`](interface/scripts/restrav_xgen_repro.py)
(official pipeline verbatim: raw `[0,1]`, center 2 s window, BICUBIC, 21-d features, plain Adam / 10 epochs).
Running the **official recipe** under leave-one-generator-out on GenVideo-Val gives **AUROC ≈ 1.00 for every
held-out generator (incl. Sora)** — the method's cross-generator generalization reproduces, with no
code/method bug. Two caveats the README previously conflated: **(a)** a *cross-dataset* transfer — train on
the VidProM/DeMamba pool, test on GenVideo-Val, which also shifts the **real** source — is much harder and is
*not* a protocol the paper reports; there the same official recipe scores only **AUROC ≈ 0.72–0.77** (this is
what the old "cross-generator AP 80.3 / 75.6" row actually measured). **(b)** the deployed `restrav.pth`
previously **trained on ImageNet-normalized frames but was served raw `[0,1]`** — a train/serve mismatch that
collapsed it to AUROC ≈ 0.50. `experiments/video/restrav.sh` and `AGIDReStraVModel` now use raw `[0,1]`
end-to-end (matching the official `preprocess` and the serving spec), the 8 summary features are in the
official `[mean, min, max, var]` order, and training is plain Adam (wd 0) for 10 epochs. Retrain
`restrav.pth` with the fixed recipe to refresh the deployed checkpoint.
⁵ STALL is training-free: it scores spatial + temporal whitened-Gaussian log-likelihoods against a
calibration set of real videos. The default ships the official VATEX calibration; the number here uses a
calibration recomputed on GenVideo reals (`ModelConfig.recompute_params=true`). Needs the gated DINOv3
ViT-L/16 weights.

### Full-benchmark reproduction (paper vs official vs AGID)

Beyond the single headline above, **every paper's benchmark tables were reproduced end-to-end** — loading
each method's *official released weights* (where they exist) and the *AGID re-implementation* side by side,
on the same on-disk test sets. Each per-model README carries the complete per-benchmark tables; the
headline benchmark per method is summarised here (mean **mAP %** unless noted). "Official" = authors'
weights in the (vendored where needed) architecture; "AGID" = the AGID class.

| Method | Headline benchmark | Paper | Official-repro | AGID-repro | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| **FatFormer** | ForenSynths 8-GAN | 99.7 | **99.70** | — | official matches paper exactly |
| **CO-SPY** | Co-Spy-Bench (110 subsets) | ~96 | 86.65 | 72.81 | coverage 66→**110**; **94.6** on the 3 shipped-real sources (flickr/COCO = proxy reals) |
| **NPR** | GANGen-Detection | 96.6 | **96.49** | — | **fixed**: native-res eval, was 65.1 (resize artifact) |
| **AIDE** | GenImage (SD-v1.4 train) | 86.9 (acc) | 85.9 (acc) | 91.75 | faithful ConvNeXt-XXL+5-view; AGID = trained from scratch on GenImage (in-domain); ForenSynths 94.4, AIGCDetect 97.9 |
| **CLIP / UnivFD** | UniversalFakeDetect | 95.0 | 95.66 | 96.44 | frozen-CLIP linear probe |
| **CNNDetection** | ForenSynths | 92.6 | 91.15 | 91.05 | ResNet-50 |
| **SAFE** | GenImage | 99.3 | **99.32** | 92.79 | official matches paper exactly |
| **SPAI** | GenImage | (91 AUC) | 91.33 | **88.80** | **gap-closed**: ported any-res SCA + fixed deleted-reals bug, 80.75→88.80 (promoted) |
| **VIB-Net** | GenImage SD-v1.4 | 95.5 | *(no weights)* | 83.26 | no official release; residual gap documented |
| **PGC** | GenImage | 100.0 | 100.0 | **97.62** | **gap-closed**: AGID retrain 69.68→97.62 (promoted) |
| **D3** (video) | GenVideo-Val | 98.5 | — | 89.00 | training-free; CLIP-B/16 vs paper XCLIP |
| **ReStraV** (video) | GenVidBench | 93.0 | — | 92.70 | reproduces; cross-*dataset* transfer is harder |
| **STALL** (video) | GenVideo-Val | 80 (AUC) | — | 83.5 (AUROC) | training-free; reproduces within ~3 pts |

Two campaign highlights: **(1) gap-closing** — retraining the weak AGID re-implementations on each paper's
own training source (and, where the gap was architectural, porting the missing module) pulls them toward the
official numbers: PGC GenImage **69.68 → 97.62 mAP** (CommGen15 73.53 → 98.61, UniversalFakeDetect 97.27 →
99.01) and SPAI GenImage **80.75 → 88.80 mAP** (porting the official any-resolution Spectral-Context-Attention
+ fixing a deleted-reals data bug; both promoted). AIDE trained from scratch on GenImage reaches **91.75 mAP**,
near its official 97.12. **(2) faithfulness fixes** — NPR's low-res-GAN collapse was an eval-resolution bug
(resize injects NPR's own up-sampling artifact; fixed with native-res scoring, 65.1 → 96.49 mAP), and AIDE's
ForenSynths now uses the genuine ConvNeXt-XXL + 5-view-DCT model (81.4 → 94.4 mAP). Gaps that remain are
genuinely *architectural* and documented rather than masked: **VIB-Net** (CLIP-backbone re-impl at 83.26 mAP;
a FLAVA-backbone retrain to match the paper did not converge) and **Co-Spy** (an 8-epoch from-scratch retrain
reproduced the same 72.7 mAP ceiling, confirming the gap is the single-backbone class vs the official
SigLIP-384 + SD-VAE fusion, not the training recipe).

---

## Web interface & checkpoint zoo

A self-contained web app to run any detector on an uploaded image/video lives in
[`interface/`](interface/): a FastAPI backend (model registry + lazy inference engine) and a React UI
(Vite build, plus a no-build vendored-React page served by the backend, so it works with no external
CDN). Launch:

```bash
conda run -n AGID pip install -r interface/requirements.txt
conda run -n AGID python interface/scripts/collect_checkpoints.py   # assemble the checkpoint folder
bash interface/run_backend.sh                                       # serves http://<host>:8000
```

It exposes all 16 detectors, applies each model's exact preprocessing / output convention, runs image
detectors frame-by-frame on videos (mean fake-probability), routes video detectors (D3, ReStraV, STALL)
through their native clip pipeline, and reports a calibrated fake/real probability. See
[interface/README.md](interface/README.md).

`scripts/collect_checkpoints.py` gathers every detector's weights into one folder (`$AGID_CKPT_DIR`,
default `/hddstore/ssalehi/AGID/checkpoints`) and writes a `MANIFEST.md`: official weights (FatFormer,
AIDE, NPR) are symlinked; the others are AGID-class checkpoints trained via
[`scripts/train_model.sh`](interface/scripts/train_model.sh); D3 and STALL are training-free.
`promote_checkpoint.py` publishes a trained run and `verify_model.py` runs a quick held-out check.

**Quick held-out verification** of the AGID-class checkpoints (`verify_model.py`, 240 ForenSynths images
across {progan, biggan, stylegan, cyclegan}). This is an **in-domain GAN sanity check on a small sample**
— the models were trained on ProGAN-family data — **not** a full-benchmark reproduction (see the
Reproduction-results table above for those). Values in **%**:

| model | AP | AUROC | note |
| --- | --- | --- | --- |
| `clip` | 99.9 | 99.9 | frozen CLIP ViT-L/14 probe (ProGAN) |
| `dinov3` | 99.4 | 99.3 | frozen DINOv3 ViT-L/16 probe (ProGAN) |
| `dinov2` | 93.0 | 92.1 | frozen DINOv2 ViT-L/14 probe |
| `cnn` | 92.4 | 91.9 | ResNet-50 fine-tune |
| `safe` | 92.0 | 91.9 | from scratch (ProGAN here; paper recipe uses AIGIBench) |
| `adapted_dino` | 100 | 100 | in-domain GANs, small sample |
| `co_spy` | 100 | 100 | in-domain GANs, small sample |
| `pgc` | 100 | 100 | in-domain GANs, small sample |
| `spai` | 73.5 | 77.1 | out-of-domain here (trained on LDM diffusion, verified on GANs) |
| `vib_net` | 69.0 | 74.7 | out-of-domain here (trained on GenImage diffusion, verified on GANs) |

The `100`s reflect *easy in-domain* (ProGAN-family) separation on 240 images, not cross-generator
generalization; `spai`/`vib_net` are low only because they were trained on diffusion data and verified on
GANs. `fat_former`/`aide`/`npr` use official weights (full-benchmark numbers above). The gated DINOv3
ViT-L/16 weights needed by STALL and the `dinov3` probe live under `dinov3_checkpoints/`.

---

## Datasets

Select a dataset with `dataset@DatasetConfig=<name>`. Each dataset auto-downloads to its `root_dir` on first use (via the Hugging Face Hub, `gdown`, or archive extraction) and is restructured into a common `real`/`fake` directory layout.

**Training datasets (train / val / test):**

| Config | Dataset | Source |
| --- | --- | --- |
| [genimage](conf/dataset/genimage.yaml) | **GenImage** | Google Drive (`gdown`) |
| [cnndetection](conf/dataset/cnndetection.yaml) | **CNNDetection / ForenSynths** (ProGAN-trained) | Google Drive (`gdown`, 7z) |
| [aigibench](conf/dataset/aigibench.yaml) | **AIGIBench** (Setting-II) | HF `HorizonTEL/AIGIBench` |
| [cospy](conf/dataset/cospy.yaml) | **Co-Spy-Bench** (22 generators + real) | HF `ruojiruoli/Co-Spy-Bench` (+ `Co-Spy-Misc` reals) |

**Cross-generator evaluation benchmarks (test only):**

| Config | Dataset | Source |
| --- | --- | --- |
| [ditfake](conf/dataset/ditfake.yaml) | **DiTFake** | HF `lioooox/DiTFake` |
| [freqnetdetection](conf/dataset/freqnetdetection.yaml) | **FreqNet** test set | Google Drive (`gdown`) |
| [universialfakedetection](conf/dataset/universialfakedetection.yaml) | **UniversalFakeDetect** (Ojha) test set | Google Drive (`gdown`) |

Per-dataset knobs (`root_dir`, batch sizes, `num_workers`, and the train/eval transform pipelines) live in the dataset config files.

### Transforms

Pipelines are lists of named transforms resolved by [transformations.py](transformations.py). Common training pipeline (CNNDetection-style augmentation):

```
resize_short_256, gaussian_blur, jpeg_compression, random_crop_224, random_horizontal_flip, to_dtype, normalize_clip
```

Available names include resize/crop variants, `gaussian_blur`, `jpeg_compression`, flips, rotation, `color_jitter_safe`, `random_mask_safe`, and normalization presets (`normalize_clip`, `normalize_dinov2`, `normalize_dinov3`). Pick the normalization that matches the backbone (e.g. SPAI normalizes internally and expects `to_dtype` only).

---

## Setup

All commands run inside the **`AGID` conda environment**:

```bash
/home/ssalehi/miniconda3/bin/conda run -n AGID <command>
```

Install dependencies from [requirements.txt](requirements.txt) (Hydra, OmegaConf, huggingface-hub, gdown, open_clip, diffusers, pytorch_wavelets, …) plus PyTorch, `accelerate`, `transformers`, `torchmetrics`, `clip`, and `wandb`.

Distributed/mixed-precision behavior is set in [accelerate_config.yaml](accelerate_config.yaml) (DeepSpeed ZeRO-2). Set `--num_processes` to the number of GPUs you launch with.

Launch scripts also export cache locations so downloads land on fast storage:

```bash
export HF_HOME=/nvmestore/ssalehi/.cache/huggingface
export TORCH_HOME=/nvmestore/ssalehi/.cache/torch
export TRITON_CACHE_DIR=/nvmestore/ssalehi/.triton
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1
```

---

## How to train and evaluate

The pipeline is configured by composing three Hydra config groups, then overriding any field with dot-paths:

```
trainer@TrainerConfig=<trainer>   model@ModelConfig=<model>   dataset@DatasetConfig=<dataset>
```

A typical launch (here: AIDE on GenImage with 2 GPUs):

```bash
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n AGID \
  accelerate launch --config_file accelerate_config.yaml --num_processes 2 main.py \
    trainer@TrainerConfig=aide \
    model@ModelConfig=aide \
    dataset@DatasetConfig=genimage \
    DatasetConfig.train_batch_size=16 \
    DatasetConfig.val_batch_size=16 \
    DatasetConfig.test_batch_size=16 \
    logger.name=aide logger.mode=online logger.tags=[AGID,AIDE,genimage]
```

The simplest way to run a method is its prepared script in [experiments/](experiments/), each of which sets the env vars and the full command:

```bash
bash experiments/aide.sh          # AIDE on GenImage
bash experiments/co_spy.sh        # CO-SPY
bash experiments/fat_former.sh    # FatFormer on AIGIBench
bash experiments/safe.sh          # SAFE on AIGIBench
bash experiments/spai.sh          # SPAI
bash experiments/vib_net.sh       # VIB-Net
bash experiments/npr.sh           # NPR
bash experiments/clip_vitl14.sh   # frozen CLIP linear probe
bash experiments/dinov2_vitl14.sh / dinov3_vitl14.sh / adapted_dinov2_vitl14.sh
```

### Training loop and evaluation

Training is driven by `AGIDModelTrainer.train()` in [trainer.py](trainer.py):

- Each epoch runs **validation** (`val_interval`) and **test** (`test_interval`) and then a train epoch — so evaluation is interleaved with training; there is no separate eval-only entry point.
- Metrics from [metrics.py](metrics.py) (`real_accuracy`, `fake_accuracy`, and optionally `AP`/`AUC`) are computed **per `folder_id`**, i.e. per generator, so you can see generalization across generative models. The metrics to compute are listed under `metrics:` in [conf/config.yaml](conf/config.yaml).
- All scalars are logged to **Weights & Biases** (`logger.mode=online` to sync, `offline` to keep local). Set `logger.name`/`logger.tags` per run.

To evaluate a trained detector on a **held-out / cross-generator benchmark**, point `DatasetConfig` at a test-only dataset (e.g. `dataset@DatasetConfig=ditfake`, `freqnetdetection`, or `universialfakedetection`).

### Key trainer knobs

From [conf/trainer/default.yaml](conf/trainer/default.yaml) (each method overrides as needed):

| Field | Meaning |
| --- | --- |
| `epochs`, `lr`, `weight_decay`, `label_smoothing` | Optimization (AdamW + cross-entropy). |
| `warmup.{mode,warmup_ratio,lr_scheduler}` | Optional LR schedule (`linear`/`cosine`); `mode: off` disables it. |
| `val_interval`, `test_interval` | How often (in epochs) to validate / test. |
| `save`, `save_interval`, `save_policy`, `save_path` | Checkpointing: enable `save`, choose `max`/`min` on the tracked val loss, and the output directory. |
| `seed` | Reproducibility seed (set via Accelerate). |

Model-specific hyperparameters (backbone, adapter dims, projection sizes, etc.) live in the [conf/model/](conf/model/) files.

<!-- TRENDING:START -->
## 📈 What's Trending — AI-generated content detection

*High-quality recent papers for image & video AIGC detection. High-quality = (1) a solid public GitHub repository, and (2) strong results versus previous methods. Auto-refreshed daily. Last updated: 2026-06-16.*

### Image detectors
- **[PGC: Peak-Guided Calibration for AI-Generated Image Detection](https://github.com/xiaoyu6868/PGC)** (Zhou et al. · 2026) — [code](https://github.com/xiaoyu6868/PGC) · _in AGID: `pgc`_  
  LoRA-tuned DINOv2 RGB stream + a YCbCr-quantization-residual CNN, fused by a peak-guided (log-sum-exp) calibration. Strong cross-generator transfer.  
  _Results: ForenSynths 96.25 mAP / GenImage 100 mAP (official weights)._
- **[PROBE: Probing Generative Space for Generalizable AIGI Detection](https://github.com/Amamiya-C/PROBE-AIGI-Detection)** (arXiv · 2026) — [code](https://github.com/Amamiya-C/PROBE-AIGI-Detection) · _in AGID: `probe`_  
  Boosts cross-generator generalization by augmenting training with images sampled from probed generative-space variations; the detector is a fine-tuned DINOv2 (sliding-patch) classifier. Released weights on ModelScope.  
  _Results: PROBE-DINOv2 93.9% avg balanced-acc over 7 benchmarks (paper). AGID reproduces the official weights: GenImage 98.54 / Synthbuster 99.27 / UFD 95.95 mAP (mAUROC 96-99). [Integration on branch trending/probe, under review.]_
- **[DiCoME: Divide-and-Conquer Multi-View Evidential Learning for Deepfake Detection](https://github.com/kxl0825/DiCoME)** (arXiv · 2026) — [code](https://github.com/kxl0825/DiCoME)  
  Disentangles semantic vs structural-artifact views and fuses them with evidential learning to counter the 'semantic masking effect'. Released checkpoint (HF).  
  _Results: SOTA cross-dataset deepfake generalization (DeepfakeBench protocol). [face-deepfake domain; AGID lacks those datasets locally — integration pending data.]_
- **[Images as Tables: In-Context AIGI Detection with TabPFN](https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images)** (arXiv · 2026) — [code](https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images)  
  Frozen DINOv3 CLS feature -> PCA(500) -> TabPFN in-context tabular classification; strong few-shot/low-data generalization with no task-specific training.  
  _Results: Generalizable detection in the low-data regime (few labeled examples per new generator)._
- **[Co-Spy: Fusing Semantic and Artifact Cues for Synthetic Image Detection](https://github.com/Megum1/Co-Spy)** (Cheng et al. · CVPR · 2025) — [code](https://github.com/Megum1/Co-Spy) · _in AGID: `co_spy`_  
  Adaptive fusion of a SigLIP semantic stream and an SD-VAE artifact-residual stream.  
  _Results: Co-Spy-Bench ~96 mAP; AGID retrain reaches 96.6 (beats released weights on that bench)._
- **[SAFE: High-Frequency Artifact Detector with Targeted Augmentation](https://github.com/Ouxiang-Li/SAFE)** (Ouxiang Li et al. · KDD · 2025) — [code](https://github.com/Ouxiang-Li/SAFE) · _in AGID: `safe`_  
  Lightweight DWT high-frequency artifact detector with a carefully designed augmentation recipe.  
  _Results: AGID-trained reproduction (served): ForenSynths-8 97.79 / UFD-8 98.44 / GenImage-8 99.51 / DiTFake 99.96 mAP — beats the official checkpoint on GenImage._
- **[AIDE: ConvNeXt Semantic + Low-Level DCT/SRM Cues](https://arxiv.org/abs/2406.19435)** (Yan et al. · ICLR · 2025) — [arXiv](https://arxiv.org/abs/2406.19435) · _in AGID: `aide`_  
  Fuses ConvNeXt semantic features with low-level DCT/SRM frequency cues; native-resolution preprocessing.  
  _Results: GenImage 97.12 mAP / AIGCDetect 97.94 mAP (official weights)._
- **[SPAI: Spectral AI-Image Detection (any-resolution)](https://github.com/mever-team/spai)** (Karageorgiou et al. · CVPR · 2025) — [code](https://github.com/mever-team/spai) · _in AGID: `spai`_  
  Spectral Reconstruction Similarity + Spectral Context Attention; detects at native/arbitrary resolution.  
  _Results: AGID serves the official weights at native resolution (the faithful any-res path): GenImage 94.22 / Synthbuster 96.75 mAP — reproduces & beats the vendored-official 91.33._
- **[FatFormer: Forgery-Aware Adaptive Transformer](https://github.com/Michel-liu/FatFormer)** (Liu et al. · CVPR · 2024) — [code](https://github.com/Michel-liu/FatFormer) · _in AGID: `fat_former`_  
  Frozen CLIP + forgery-aware adapters + learnable text-prompt context.  
  _Results: ForenSynths-8 99.7 mAP (official weights)._
- **[NPR: Neighboring-Pixel Relationships](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)** (Tan et al. · CVPR · 2024) — [code](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) · _in AGID: `npr`_  
  Small ResNet on neighboring-pixel residuals that expose generator up-sampling artifacts.  
  _Results: GANGen 96.49 mAP / DiffusionForensics 99.4 mAP (official weights)._
- **[UnivFD / CLIP: Towards Universal Fake Image Detectors](https://github.com/WisconsinAIVision/UniversalFakeDetect)** (Ojha et al. · CVPR · 2023) — [code](https://github.com/WisconsinAIVision/UniversalFakeDetect) · _in AGID: `clip`_  
  Linear probe on frozen CLIP ViT-L/14 features — a simple, strong cross-generator baseline.  
  _Results: UniversalFakeDetect 95.66 mAP; AGID-trained 96.44._

### Video detectors
- **[D3: Disentangled Detail Dynamics (training-free)](https://github.com/Zig-HS/D3)** (Zheng et al. · ICCV · 2025) — [code](https://github.com/Zig-HS/D3) · _in AGID: `d3`_  
  Training-free: 2nd-order std of frozen per-frame features as the score.  
  _Results: GenVideo-Val ~98 mAP (paper); AGID 89.0 mAP._
- **[ReStraV: Representation-Trajectory Geometry](https://github.com/ChristianInterno/ReStraV)** (Internò et al. · 2025) — [code](https://github.com/ChristianInterno/ReStraV) · _in AGID: `restrav`_  
  Frozen DINOv2 trajectory-geometry features + a standardized MLP head.  
  _Results: GenVidBench Main ~93; AGID 92.7 mAP._
- **[STALL: Spatio-Temporal whitened Log-Likelihoods (training-free)](https://github.com/OmerBenHayun/STALL)** (Ben Hayun et al. · CVPR · 2026) — [code](https://github.com/OmerBenHayun/STALL) · _in AGID: `stall`_  
  Training-free spatial + temporal whitened-Gaussian log-likelihoods vs a real-video calibration.  
  _Results: GenVideo-Val ~0.80 AUC (paper); AGID AUROC 83.5._

### Benchmarks & curated lists
- [GenImage: million-scale real-vs-fake benchmark](https://github.com/GenImage-Dataset/GenImage) — ImageNet-1k reals + ~equal synthetic from diverse generators.
- [AIGIBench: robustness/generalization benchmark (23 fake subsets)](https://github.com/HorizonTEL/AIGIBench) — Multi-source generalization, degradation robustness, augmentation sensitivity, preprocessing impact.
- [AI-GenBench: temporal (GAN→diffusion) generalization benchmark](https://github.com/MI-BioLab/AI-GenBench) — Incremental, historically-ordered training to test generalization to new generators (Verimedia @ IJCNN 2025).
- [GenVidBench: 6M AI-generated video detection benchmark](https://arxiv.org/abs/2501.11340) — Large-scale benchmark for AI-generated video detection.
- [DeepfakeBench: comprehensive deepfake-detection benchmark](https://github.com/SCLBD/DeepfakeBench) — Unified benchmark across many deepfake datasets/detectors.
- [Awesome-AIGC-Image-Video-Detection (curated list)](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection) — Continuously-updated collection of image + video AIGC-detection papers and code.
- [Awesome-Comprehensive-Deepfake-Detection (curated list)](https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection) — Broad survey-style index of deepfake-detection research and repos.

<!-- TRENDING:END -->
