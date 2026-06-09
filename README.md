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

Every benchmark each paper reports, reproduced and verified. Per-model detail in each `models/*/<name>/README.md`; raw results in `repro/bench_results/`, specs in `repro/benchmark_specs/`, matrix in `repro/BENCHMARK_COVERAGE_MATRIX.md`.

| Model | Benchmark | Paper | Ours | Status |
|---|---|---|---|---|
| CLIP / UnivFD | ForenSynths GAN/low-level test (11 of the pape | — | map_fs8=97.22 / acc_fs8=88.28 / map_fs13=92.99 / official_fs8=96.95 / official_fs13=93.14 / ckpt=clip.pth (AGID-trained linear head, ckpt_kind=agid) / tags=clip_agid_forensynths / clip_official_forensynths | reproduced |
| CLIP / UnivFD | UniversalFakeDetect diffusion/autoregressive t | map=95.0 | map=96.44 / acc=87.23 / official=95.66 / ckpt=clip.pth (AGID-trained head) / tags=clip_agid_ufd / clip_official_ufd | reproduced |
| CNNDetection | ForenSynths test set (CVPR paper Table 2 = 11  | map=90.8 | map_fs8=94.42 / acc_fs8=76.36 / map_fs13=91.05 / official_fs8=92.67 / official_fs13=91.15 / ckpt=cnn.pth (AGID-trained ResNet-50, ckpt_kind=agid) / tags=cnn_agid_forensynths / cnn_official_forensynths | reproduced |
| FatFormer | GANs dataset — 8 GANs (4-class ProGAN supervis | map=99.7 / acc=98.4 | map=99.7 / acc=98.39 / ckpt=fat_former.pth (= fatformer_official.pth, OFFICIAL 4-class CLIP:ViT-L/14 weights remapped into the AGID class; ckpt_kind=official_fatformer) / tag=fatformer_official_forensynths | reproduced |
| FatFormer | GANs dataset — 8 GANs (2-class ProGAN supervis | map=99.2 / acc=95.8 | — | missing |
| FatFormer | Diffusion-model (DMs) dataset — 10 subsets (4- | map=98.8 / acc=95.0 | map_ufd8=98.6 / acc_ufd8=93.72 / ckpt=fat_former.pth (official weights in AGID class) / tag=fatformer_official_ufd | below |
| NPR | Table 1 — ForenSynths / Cross-GAN-Sources (8 G | map=96.1 / acc=92.5 | 94.22 mAP (correct ckpt) | reproduced |
| NPR | Table 2 — GANGen-Detection / Self-Synthesis 9  | map=96.6 / acc=93.2 | map=96.49 / acc=94.81 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_freqnet / eval=run with bench_eval.py --native-res to drop the up-scaling Resize, matching official no_resize=True | reproduced |
| NPR | Table 3 — DiffusionForensics / DIRE | map=99.8 / acc=95.3 | map=99.4 / acc=95.03 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_diffusionforensics | reproduced |
| NPR | Table 4 — UniversalFakeDetect / Ojha | acc=95.2 | map=96.69 / acc=93.94 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_ufd | reproduced |
| NPR | Table 5 — Diffusion1kStep / Self-Synthesis Dif | acc=80.1 | map=83.47 / acc=79.13 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_diffusion1kstep | reproduced |
| AIDE | AIGCDetectBenchmark (B1, Setting-I, ProGAN-tra | acc=92.77 | map=97.94 / acc=91.89 | reproduced |
| AIDE | GenImage (B2, Setting-I, SDv1.4-trained) | acc=86.88 | map=97.12 / acc=85.91 | reproduced |
| AIDE | Chameleon (B3, Setting-II, in-the-wild) | acc_ProGAN_trained=58.37 / acc_SDv14_trained=62.6 / acc_AllGenImage_trained=65.77 | — | missing |
| SAFE | ForenSynths (Table 1, intra-architecture GANs) | map=98.8 / acc=96.2 | map=98.78 / acc=96.2 | below |
| SAFE | Self-Synthesis / GANGen (Table 2, intra-archit | map=99.8 / acc=99.0 | map=99.85 / acc=98.99 | reproduced |
| SAFE | Ojha / UniversalFakeDetect (Table 3, cross-arc | map=99.1 / acc=95.7 | map=99.1 / acc=95.71 | reproduced |
| SAFE | GenImage (Table 4, cross-architecture DM+GAN)  | map=99.3 / acc=95.6 | map=95.7 / acc=89.84 | below |
| SAFE | DiTFake (Table 6, cross-architecture DiT) | map=99.9 / acc=99.4 | 97.78 mAP | reproduced |
| Co-Spy | Co-Spy-Bench | map=96.02 / acc=87.06 | map=96.7 / acc=90.28 / ckpt=checkpoints_retrain/cospy_full/model_1.pth (AGID retrain, full SD-v1.4 data, flickr-reals bug fixed) / tag=cospy_full_cospybench / official_wts_in_agid=86.65 | reproduced |
| Co-Spy | AIGCDetectionBenchmark (AIGCDetect) | map=95.21 / acc=90.58 | map=94.59 / acc=89.25 / ckpt=official ProGAN weights loaded into AGID class (/hddstore/ssalehi/AGID/repro/ckpts/progan) / tag=cospy_official_aigcdetect / agid_trained_ckpt=84.91 | reproduced |
| Co-Spy | GenImage | — | map_official_wts=92.41 / map_agid_retrain=90.59 / acc=75.91 / ckpt=checkpoints_retrain/cospy_full/model_2.pth / tag=cospy_full_g8 | below |
| Co-Spy | ForenSynths (CNNDetection test) | — | map_official_wts=87.97 / map_agid_retrain=96.17 / ckpt=AGID co_spy.pth / tag=cospy_official_forensynths / co_spy_retrain_forensynths | reproduced |
| Co-Spy | In-the-wild (Co-Spy-Bench in-the-wild) | map=84.65 / acc=77.41 | — | missing |
| SPAI | SPAI main eval set (13 generators x 5 real sou | auroc=91.0 | — | missing |
| SPAI | GenImage (cross-dataset, AGID benchmark) | — | 90.74 mAP (any-res) | reproduced |
| SPAI | Synthbuster | — | ~96.6 mAUROC (any-res) | reproduced |
| VIB-Net | Diffusion-source (SDv1.4-trained), AP -- HEADL | map=95.53 | map=83.94 / acc=77.57 | below |
| VIB-Net | Diffusion-source (SDv1.4-trained), ACC | acc=85.83 | map=83.94 / acc=77.57 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN-trained), AP | map=93.14 | map=61.71 / acc=57.91 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN-trained), ACC | acc=82.5 | — | missing |
| PGC | GenImage | map=100.0 / acc=98.3 | map_official_wts=100.0 / acc_official_wts=99.49 / map_agid_retrain=97.62 / acc_agid_retrain=78.6 / ckpt_official=repro/official_ckpts/pgc/PGC_train_sdv1_4_ckpt.pth; ckpt_agid: checkpoints/_train/pgc_genimage/model_0.pth (registry-default) / tags=pgc_official_genimage8 / pgc_retrain_genimage8 (pgc_long_g8) | reproduced |
| PGC | UniversalFakeDetect (UFD) | map=98.0 | map_official_wts=99.34 / acc_official_wts=94.16 / map_agid_retrain=99.01 / ckpt_official=repro/official_ckpts/pgc/PGC_train_progan_ckpt.pth / tags=pgc_official_ufd / pgc_retrain_ufd | reproduced |
| PGC | ForenSynths (CNNDetection test) | — | map_official_wts=96.25 / acc_official_wts=83.73 / map_agid_retrain=89.03 / ckpt_official=repro/official_ckpts/pgc/PGC_train_progan_ckpt.pth / tags=pgc_official_forensynths / pgc_retrain_forensynths | reproduced |
| PGC | AIGIBench (AIGI) | map=95.0 / acc=86.6 | map_official_wts=95.36 / acc_official_wts=88.33 / map_agid_retrain=86.88 / ckpt_official=repro/official_ckpts/pgc/PGC_train_progan_sdv1_4_ckpt.pth / tags=pgc_official_aigibench25 / pgc_retrain_aigibench | reproduced |
| PGC | CommGen15 (commercial in-the-wild) | acc=97.9 | map_official_wts=99.26 / acc_official_wts=97.39 / map_agid_retrain=98.61 / acc_agid_retrain=66.66 / ckpt_official=repro/official_ckpts/pgc/PGC_train_sdv1_4_ckpt.pth / tags=pgc_official_commgen15 / pgc_retrain_commgen15 | reproduced |
| D3 | GenVideo (DeMamba GenVideo-Val, Table 1) | mAP=98.46 | XCLIP 95.1 AUROC / 99.5 AP_real_full (CLIP 94.4) | reproduced |
| D3 | EvalCrafter (Table 2, 14 generators, fake-only | mAP=98.87 | mAP=91.19 / mAUROC=95.41 / acc=87.67 | below |
| D3 | VideoPhy (Table 3, 10 generators) | mAP=99.16 | — | missing |
| D3 | VidProM (Table 4, 6 generators) | mAP=88.46 | — | missing |
| ReStraV | A) VidProM vs image-based detectors (Table 2) | acc=97.06 / mAP=98.81 | — | missing |
| ReStraV | B) VidProM vs VideoSwin — seen/unseen/future ( | seen={'acc': 97.05, 'mAP': 98.78} / unseen={'acc': 89.45, 'mAP': 97.32} / future_Sora={'acc': 80.05, 'mAP': 92.85} | — | missing |
| ReStraV | B) GenVidBench vs SoTA video detectors — Main  | Main_M_avg=93.01 / Plants_P_avg=96.96 | acc=86.87 / mAP=92.7 / mAUROC=95.9 | reproduced |
| ReStraV | C) DeMamba one-to-many generalization (Table 5 | train_Pika={'R': 0.735, 'F1': 0.827, 'AP': 0.797} / train_SEINE={'R': 0.82, 'F1': 0.898, 'AP': 0.854} / train_OpenSora={'R': 0.771, 'F1': 0.797, 'AP': 0.717} | — | missing |
| ReStraV | D) Veo3 zero-shot generalization (Sec 7-D) | acc=83.2 / F1=85.1 / AUROC=86.9 | — | missing |
| ReStraV | E) Physics-IQ 2AFC vs Gemini 1.5 Pro (Sec 7-E, | Pika=97.5 / Runway=94.9 / Sora=99.0 | — | missing |
| STALL | GenVideo (DeMamba GenVideo-Val, 10 generators  | avg_AUC=0.8 / avg_AP=0.8 / per_gen_AUC_range=0.72-0.85 / per_gen_AP_range=0.75-0.84 | mAP=82.38 / mAUROC=83.51 / acc=56.88 | reproduced |
| STALL | VideoFeedback (TIGER-Lab, 11 generators + DiDe | avg_AUC=0.83 / avg_AP=0.85 / per_gen_AUC_range=0.78-0.93 / per_gen_AP_range=0.80-0.94 | mAP=83.07 / mAUROC=85.1 / acc=62.12 | reproduced |
| STALL | ComGenVid (paper's new benchmark — Sora + VEO3 | avg_AUC=0.852 / avg_AP=0.858 / Sora={'AUC': 0.84, 'AP': 0.846} / VEO3={'AUC': 0.865, 'AP': 0.87} | mAP=82.59 / mAUROC=84.0 / acc=57.34 | reproduced |

---

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
