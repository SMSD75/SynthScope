# AGID — AI-Generated Image & Video Detection

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/SMSD75/AGID/actions/workflows/ci.yml/badge.svg)](https://github.com/SMSD75/AGID/actions/workflows/ci.yml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

AGID is a unified library for **detecting AI-generated (synthetic) images and videos**. It re-implements a range of state-of-the-art real-vs-fake detectors behind one [Hydra](https://hydra.cc/) config system and an [Accelerate](https://huggingface.co/docs/accelerate) training loop, so any method can be run, trained, and benchmarked on the same datasets, transforms, and metrics — through a one-line Python/CLI API, a web demo, or a standardized leaderboard.

Every detector is a binary classifier (`real` vs `fake`), and results are reported **per generator**, the standard way to measure generalization to unseen generative models. **17 detectors** are included (14 image, 3 video), each verified against its paper.

---

## Install

```bash
git clone https://github.com/SMSD75/AGID.git
cd AGID
pip install -e .                 # editable install
# extras:  ".[web]" web demo · ".[train]" wandb + dataset download · ".[dev]" contributor tooling
```

Install the [PyTorch](https://pytorch.org) build that matches your CUDA. A couple of detectors
(`clip`, `fat_former`) also need OpenAI CLIP: `pip install git+https://github.com/openai/CLIP.git`.
See [LICENSES.md](LICENSES.md) for per-method/weight licenses.

## Quick start

> **Try it without installing:** open the [Colab quickstart](notebooks/agid_quickstart.ipynb) or launch
> the [Gradio demo](demo/) (`pip install -e ".[demo]" && python demo/app.py`).

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

Checkpoints resolve from `$AGID_CKPT_DIR` (default `~/.cache/agid/checkpoints`); pass
`agid.load(..., checkpoint="/path/to.pth")` to use your own weights. Training-free detectors
(D3, STALL) need none.

### Model zoo (Hugging Face Hub)

When a detector's weights aren't present locally, `agid.load(...)` **auto-downloads** them from a Hub
model-zoo repo (set `AGID_HUB_REPO`, or pass `agid.load(..., repo="user/agid-checkpoints")`):

```bash
agid pull --model clip                                 # fetch one checkpoint
agid push --repo user/agid-checkpoints --public        # publish AGID-trained checkpoints (license-safe)
```

`push` uploads only AGID-trained checkpoints by default; weights derived from authors' official
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

AGID ships a **versioned, standardized benchmark** so detectors are comparable under one protocol —
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
= authors' weights run through AGID). Full per-benchmark tables (paper / official / AGID-trained, all
generators) live in [`repro/AGGREGATED_BENCHMARK.md`](repro/AGGREGATED_BENCHMARK.md) and each model's
`README.md`.

| Detector | Headline benchmark | Paper | AGID |
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
(official / AGID-trained / training-free), and includes a benchmark view and the **What's Trending**
feed below. See [interface/README.md](interface/README.md).

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
## 📈 What's Trending — AI-generated content detection

*High-quality recent papers for image & video AIGC detection. High-quality = (1) a solid public GitHub repository, and (2) strong results versus previous methods. Auto-refreshed daily. Last updated: 2026-10-03.*

### Image detectors
- **[CuRe: Learning Continuous Source Responses for Generalizable AI-Generated Image Detection](https://github.com/manic-cui/CuRe)** (manic-cui · arXiv · 2026) — [code](https://github.com/manic-cui/CuRe) · _in AGID: `cure`_  
  Reframes backbone adaptation as regressing real-generated MIXING RATIOS instead of binary classification (finer supervision), with a compact source-response subspace that suppresses nuisance variation/shortcut cues. Frozen Perception-Encoder (PE-Core L/14-336) backbone + LoRA + subspace + linear head; the 25 MB adapter ships in the repo and infer.py scores a single image.  
  _Results: 89.7% average balanced accuracy over ten public benchmarks, +5.2 pts over the second-best method, robust to common degradations (paper). [Integrated in AGID as `cure` (merged to main 2026-10-03): AGID's class is identical to the official model (395/395 state-dict tensors, logit diff 4.8e-6, 0 verdict flips). AGID-protocol results with the official adapter: UFD-8 93.9 / ForenSynths-13 82.0 / GenImage-8 85.2 balanced acc (SD-v1.4, its training set, 99.7). AIGIBench REPRODUCED: on the paper's own 20 full-image-synthesis subsets (full test data) the port scores 92.85 balanced acc / 97.48 AUROC vs the paper's 92.8 / 97.5, every subset within 0.10 pt (per-subset mean abs diff 0.03; the paper excludes the 5 face-swap subsets, which is why all 25 gives 85.95). Still unconfirmed: UnivFD (aggregation unspecified), AIGCDetect (data only on cn023, not run), and 7 of the paper's 10 benchmarks (not in AGID).]_
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
  _Results: Generalizable detection in the low-data regime (few labeled examples per new generator). [integration pending: in-context TabPFN paradigm — a clean AGID forward->logit needs the released fitted TabPFN+PCA+labeled-support-set checkpoint baked in (GDrive), plus an external version-sensitive tabpfn dependency; a different paradigm from the trained-detector template, and the headline is a low-data/few-shot niche (multi-multi acc 0.8765 / AUROC 0.9501 @625 samples) rather than full-data SOTA. Verified repo 2026-06-20: weights on GDrive, frozen DINOv3->PCA(500)->TabPFN.]_
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
- **[MoE-JEPA: Unifying Semantic Priors and High-Frequency Traces (V-JEPA for synthetic image forensics)](https://github.com/ALCOR-Lab-DIAG/MoE-JEPA)** (ALCOR-Lab-DIAG · ECCV'26 Workshop (AI for Multimedia Forensics) · 2026) — [code](https://github.com/ALCOR-Lab-DIAG/MoE-JEPA)  
  Frozen V-JEPA 2 ViT-L backbone with a Residual Mixture-of-Experts injected into the last blocks, a Bayar-conv noise branch and Gated-Attention MIL pooling; uses JEPA 'world understanding' as a semantic prior.  
  _Results: 95.54% acc / 94.21 F1 on SID-Set (300K images), outperforming much larger models (paper). [integration pending (assessed 2026-10-02): the authors' trained weights are NOT linked (only the frozen V-JEPA 2 backbone is downloaded separately) and there is no single-image inference entrypoint; can only be retrained.]_
- **[RAID: Towards Robust AI-Generated Image Detection with Bit-Reversed Images](https://github.com/renxi-seu/RAID)** (renxi-seu · arXiv · 2026) — [code](https://github.com/renxi-seu/RAID)  
  Detects via bit-planes: a bit-reversed image, gradient-based patch selection and a small convolutional classifier, with a theoretical analysis; ~100x faster than counterparts.  
  _Results: Claims to outperform existing methods on 40+ benchmarks across cross-generator, cross-dataset and zero-shot settings (paper; README gives no numbers). [integration pending (assessed 2026-10-02): the SD-v1.4-trained weights are only shared via Baidu Cloud (not fetchable programmatically), and the README leaves the architecture and patch-score aggregation unspecified; revisit if a direct weight link appears.]_
- **[Fleet: Few Shots Lead Effective AI-generated Image Detection](https://github.com/ICTMCG/Fleet)** (ICTMCG · arXiv · 2026) — [code](https://github.com/ICTMCG/Fleet)  
  Dynamic few-shot adaptation: 'constrained routing correction' redirects novel AI samples away from non-AI-dominated routes in decoupled subspaces (DINOv3 backbone), so the detector adapts to rapidly-evolving/commercial generators from a handful of labeled examples. Introduces the Treasure benchmark (64 generators, 360k images, 20 closed-source commercial engines).  
  _Results: On a modern generator (Doubao Seedream 4.0), 20.4% -> 73.1% with only 10-shot adaptation, where static SOTA collapses (paper). Repo now live with code (re-listed 2026-10-03; it had been pruned 2026-07-04 while an empty placeholder). [integration pending (assessed 2026-10-03): not a single-checkpoint detector — the Hugging Face repo wangjiaan/Fleet holds NINE per-target-generator few-shot adapted models (ADM, FLUX.2, GPT4O, HunyuanImage-3.0, NextStep, Qwen-Image, StarGAN, wan2.5, ...; 1.3 GB each), the generic stage-1 base is only on Baidu Pan, inference requires few-shot support data per target generator, and the Treasure benchmark is gated. No clean forward->logit that represents the method; revisit if a generic checkpoint appears on HF.]_

### Video detectors
- **[D3: Disentangled Detail Dynamics (training-free)](https://github.com/Zig-HS/D3)** (Zheng et al. · ICCV · 2025) — [code](https://github.com/Zig-HS/D3) · _in AGID: `d3`_  
  Training-free: 2nd-order std of frozen per-frame features as the score.  
  _Results: GenVideo-Val ~98 mAP (paper). AGID faithful protocol (X-CLIP-B/16 + 10% edge-crop): AUROC 95.95 / acc 91.7 (the paper's mAP isn't comparable on AGID's 10000:56 real:fake pool — AUROC is the robust metric)._
- **[ReStraV: Representation-Trajectory Geometry](https://github.com/ChristianInterno/ReStraV)** (Internò et al. · 2025) — [code](https://github.com/ChristianInterno/ReStraV) · _in AGID: `restrav`_  
  Frozen DINOv2 trajectory-geometry features + a standardized MLP head.  
  _Results: GenVidBench Main ~93; AGID 92.7 mAP._
- **[STALL: Spatio-Temporal whitened Log-Likelihoods (training-free)](https://github.com/OmerBenHayun/STALL)** (Ben Hayun et al. · CVPR · 2026) — [code](https://github.com/OmerBenHayun/STALL) · _in AGID: `stall`_  
  Training-free spatial + temporal whitened-Gaussian log-likelihoods vs a real-video calibration.  
  _Results: GenVideo-Val ~0.80 AUC (paper); AGID AUROC 83.5._
- **[ATSS: Anomalous Temporal Self-Similarity for AI-Generated Video Detection](https://github.com/hwang-cs-ime/ATSS)** (Wang et al. (hwang-cs-ime) · arXiv · 2026) — [code](https://github.com/hwang-cs-ime/ATSS)  
  Multimodal detector that exploits anomalous temporal self-similarity fingerprints across frames of AI-generated video. Inference code + a pretrained checkpoint (best.pt) are released; the data sampling/extraction pipeline is still being cleaned for release.  
  _Results: Claims SOTA in AP/AUC/ACC over GenVideo, EvalCrafter, VideoPhy, and VidProM vs prior methods (e.g. DeMamba). Repo live 2026-06-21 with code+weights; per-generator numbers not yet in the README. [integration pending: the released checkpoint's forward() consumes PRECOMPUTED multimodal temporal-self-similarity tensors (sim_img/sim_txt/sim_cross, BLIP2-based), not raw video — and the feature-extraction/data-sampling front-end that produces them is 'still being cleaned for release (coming soon)'. AGID feeds raw video clips, so there is no faithful way to run ATSS until that pipeline is published; will revisit when released.]_
- **[G2VD: Generalizable AI-Generated Video Detection via Counterfactual Intervention and Causal Disentanglement](https://github.com/DMOSCAR-98/G2VD)** (Du et al. (dumeng98) · arXiv · 2026) — [code](https://github.com/DMOSCAR-98/G2VD)  
  Counterfactual-intervention + causal-disentanglement detector: a training-time CFIPipeline synthesizes controlled counterfactual samples (VAEs + frequency/pixel alignment), and a causal-disentanglement classifier with two domain-anchored branches and an HSIC independence constraint isolates the generalizable artifact signal from domain-specific content.  
  _Results: GenVidBench cross-domain: >90% accuracy / AUC ~0.95 using only 10% of the training data (paper). Repo live 2026-07-07 (CLIP/XCLIP/DeMamba backbones — the same family as AGID's D3). [integration pending (re-checked 2026-10-02): repo moved from dumeng98/G2VD (now 404) to DMOSCAR-98/G2VD; checkpoints are still NOT released ('not stored in this repository', MODEL_ZOO.md lists no links) and the CFIPipeline VAE-pool is part of the staged pipeline. Revisit when checkpoints appear.]_
- **[RIFT: Mind the Rift - Cross-Scale Coupling Mismatch for AI-Generated Video Detection](https://github.com/Litsay/RIFT)** (Litsay · arXiv (submitted ACM MM'26) · 2026) — [code](https://github.com/Litsay/RIFT)  
  Treats a cross-scale coupling mismatch between macro temporal dynamics (differential geometry / persistent homology on frozen DINOv2 features) and micro residuals (steganalytic filtering + RAFT motion) as the forensic signal, scored by a coupling-divergence module.  
  _Results: Own benchmark: 99.22% acc / 99.33 F1 / 99.96 AUC, 97.87% leave-one-generator-out (paper; no head-to-head against named prior detectors in the abstract, so not directly comparable). [integration pending (assessed 2026-10-02): trained weights are 'released upon acceptance' (not available), and it needs DINOv2 + RAFT features over ~120K training videos.]_

### Benchmarks & curated lists
- [GenImage: million-scale real-vs-fake benchmark](https://github.com/GenImage-Dataset/GenImage) — ImageNet-1k reals + ~equal synthetic from diverse generators.
- [AIGIBench: robustness/generalization benchmark (23 fake subsets)](https://github.com/HorizonTEL/AIGIBench) — Multi-source generalization, degradation robustness, augmentation sensitivity, preprocessing impact.
- [AI-GenBench: temporal (GAN→diffusion) generalization benchmark](https://github.com/MI-BioLab/AI-GenBench) — Incremental, historically-ordered training to test generalization to new generators (Verimedia @ IJCNN 2025).
- [GenVidBench: 6M AI-generated video detection benchmark](https://arxiv.org/abs/2501.11340) — Large-scale benchmark for AI-generated video detection.
- [DeepfakeBench: comprehensive deepfake-detection benchmark](https://github.com/SCLBD/DeepfakeBench) — Unified benchmark across many deepfake datasets/detectors.
- [Awesome-AIGC-Image-Video-Detection (curated list)](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection) — Continuously-updated collection of image + video AIGC-detection papers and code.
- [Awesome-Comprehensive-Deepfake-Detection (curated list)](https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection) — Broad survey-style index of deepfake-detection research and repos.

<!-- TRENDING:END -->
