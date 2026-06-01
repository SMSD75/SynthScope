# AGID — AI-Generated Image Detection

AGID is a unified research framework for **training and evaluating AI-generated (synthetic) image detectors**. It re-implements a range of state-of-the-art real-vs-fake image detectors behind a single [Hydra](https://hydra.cc/) configuration system and an [Accelerate](https://huggingface.co/docs/accelerate) (DeepSpeed) training loop, so that different methods can be trained and benchmarked on the same datasets, transforms, metrics, and logging stack.

Every detector is a binary classifier (`real` vs `fake`). Metrics are reported **per generator / data source**, which is the standard way to measure how well a detector generalizes across unseen generative models.

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
| **ReStraV** | GenVideo-Val | AP 96.67 / AUROC 97.06 | ~high-90s AUROC | frozen DINOv2 trajectory + MLP |

¹ The AGID `co_spy` / `spai` classes diverge from the official networks (different backbone / fusion /
no Spectral-Context-Attention), so the **official-weights** evals load the authors' architectures
(vendored) rather than the AGID class; the from-scratch numbers above use the AGID class.
² AIDE's score *ranking* reproduces (high mAP), but acc@0.5 is miscalibrated because the AGID input is a
single CLIP-normalised view vs the official 5-view DCT-reconstructed stack.
³ VIB-Net is evaluated on the GenImage SD-v1.4 test split, which is a single real-vs-fake pool, so the
per-generator mAP and AUROC coincide by construction.

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
