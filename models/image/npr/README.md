# NPR: Neighboring Pixel Relationships for Generalizable Deepfake Detection

## Short method explanation

**NPR** is a generalizable deepfake and synthetic-image detector from the paper **"Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection"** by Tan et al., accepted to CVPR 2024.

The paper studies artifacts caused by up-sampling operations in CNN-based generative networks. Instead of focusing only on global frequency artifacts, NPR extracts local spatial relationships between neighboring pixels. The core idea is that many GAN and diffusion pipelines use up-sampling layers to expand low-resolution latent features into high-resolution images, and those operations can leave local pixel-dependency traces that transfer across generator families.

Official repo: https://github.com/chuangchuangtan/NPR-DeepfakeDetection  
Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.pdf

## Pipeline

Given an input image:

1. Down-sample the image by a factor of `0.5` using nearest-neighbor interpolation.
2. Up-sample it back to the original resolution using nearest-neighbor interpolation.
3. Subtract the reconstructed image from the original image.
4. Scale the residual by `2.0 / 3.0`.
5. Feed the NPR residual image into a lightweight ResNet-style binary classifier.
6. Output a real/fake prediction score.

In this implementation, the NPR transform is:

```python
x_down = F.interpolate(x, scale_factor=0.5, mode="nearest")
x_up = F.interpolate(x_down, scale_factor=2.0, mode="nearest")
npr = (x - x_up) * (2.0 / 3.0)
```

The detector uses label `0` for real images and label `1` for generated images.

## Why NPR works

The paper argues that up-sampling leaves local spatial traces in generated images. For example, nearest-neighbor up-sampling makes nearby pixels in a local grid strongly related before later convolution layers refine the image. Even after those convolutions, some local dependency patterns remain in the final image.

NPR represents these traces through local residual relationships. This makes the detector less dependent on one generator's global frequency fingerprint and more focused on a structural artifact shared by many CNN-based generator pipelines.

## Training dataset

NPR follows the CNNDetection / ForenSynths training setup for its main experiments.

The official paper trains on the ForenSynths training set, which contains LSUN real images and ProGAN-generated fake images from 20 object categories. For consistency with prior baselines, the NPR paper uses a 4-class training setup:

| Split | Source | Description |
|---|---|---|
| Real | LSUN | Real images from selected object categories |
| Fake | ProGAN | ProGAN-generated images from matching categories |
| Training classes | car, cat, chair, horse | Official 4-class NPR setup |
| Task | Binary classification | Real vs fake |

The key design choice is that the detector is trained on a limited ProGAN source and evaluated on unseen generator families.

## Test datasets

The paper evaluates NPR across 28 generation models from GAN and diffusion families.

| Test group | Generators |
|---|---|
| ForenSynths | ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake |
| GANGen-Detection | AttGAN, BEGAN, CramerGAN, InfoMaxGAN, MMDGAN, RelGAN, S3GAN, SNGAN, STGAN |
| DIRE / DiffusionForensics | ADM, DDPM, IDDPM, LDM, PNDM, VQDiffusion, Stable Diffusion v1, Stable Diffusion v2 |
| UniversalFakeDetect-style diffusion tests | ADM, GLIDE, DALL-E mini, LDM |
| Diffusion1kStep | DDPM, IDDPM, ADM, Midjourney, DALLE |

The paper reports both accuracy and average precision, with the main result showing strong cross-generator generalization from the NPR representation.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

Values are `mAP / acc`. **SynthScope_official** = the authors' released weights run through the SynthScope class / eval. **SynthScope_train** = the SynthScope class trained from scratch by us with the official recipe (`experiments/image/npr.sh`; see *Training in SynthScope* below), evaluated so far on Tables 1 and 4.

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| Table 1 — ForenSynths / Cross-GAN (8) | 96.1 / 92.5 | 94.22 / 92.3 | 93.83 / 84.09 | official; trained: AP close, acc below (as the official training code here) |
| Table 2 — GANGen-Detection (9) | 96.6 / 93.2 | 96.49 / 94.81 | — | official |
| Table 3 — DiffusionForensics / DIRE | 99.8 / 95.3 | 99.4 / 95.03 | — | official |
| Table 4 — UniversalFakeDetect / Ojha | acc 95.2 | 96.69 / 93.94 | 97.49 / 88.91 | official; trained: AP matches, acc below |
| Table 5 — Diffusion1kStep | acc 80.1 | 83.47 / 79.13 | — | official |

**reproduced** = SynthScope-trained checkpoint matches the paper · **official** = official weights loaded into the SynthScope class match (the SynthScope-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Training in SynthScope

`experiments/image/npr.sh` runs the official recipe (the README's `train.py` command and `networks/trainer.py`): ProGAN
4-class (car, cat, chair, horse; 144,024 images), `Resize((256, 256))` → RandomCrop 224 → flip → ImageNet normalization
(no blur/JPEG: the official `data_augment` call is commented out), Adam lr 2e-4 without weight decay, batch 32 on one
GPU, 50 epochs, learning rate ×0.9 after epochs 10, 20, 30 and 40, last-epoch checkpoint, seed 100. The SynthScope NPR
uses the official convolution initialization for this.

To tell our pipeline apart from run-to-run variance, we also ran the **official `train.py`, unmodified**, on the same
data and machine, and scored every checkpoint with the same `bench_eval` command (`mAP / acc`):

| Checkpoint | Table 1 — ForenSynths (8 GANs) | Table 4 — UniversalFakeDetect |
|---|---|---|
| official `NPR.pth` (served as `npr.pth`) | 94.22 / 92.30 | 97.16 / 94.57 |
| official `model_epoch_last_3090.pth` | 92.81 / 91.83 | 96.69 / 93.94 |
| official `train.py` rerun here | 92.72 / 81.47 | 96.56 / 87.61 |
| **SynthScope recipe (ours)** | **93.83 / 84.09** | **97.49 / 88.91** |

Our recipe matches the official code's own rerun on every number (it is slightly better), so the SynthScope pipeline is
faithful; neither run reaches the released checkpoints' accuracy. AP — how well real and fake are ranked — reproduces;
the accuracy gap is a threshold shift (our models call more unseen-generator fakes real, e.g. CycleGAN fake recall 48 vs
96 for `NPR.pth`), which the released checkpoints did not show and which this environment (PyTorch 2.11 instead of 1.8,
another GPU) did not reproduce. Runs: training job 265720, official-code rerun 265815, evaluation 265808/265814/266028
(`repro/bench_results/npr_{official_nprpth,official3090,officialcode_rerun,trained_last,trained_best}_*.json`).

## Model details

The SynthScope implementation lives in:

```bash
models/image/NPR/NPR.py
```

The current model uses:

| Component | SynthScope NPR setting |
|---|---|
| NPR transform | nearest down-sample, nearest up-sample, subtract, scale by `2.0 / 3.0` |
| Stem | `3x3` convolution, stride `2`, padding `1` |
| Backbone | lightweight ResNet-style Bottleneck trunk |
| Residual layers | `layer1`: 3 Bottleneck blocks, `layer2`: 4 Bottleneck blocks |
| Pooling | adaptive average pooling |
| Head | one binary logit |
| Loss | `BCEWithLogitsLoss` |
| Metric output | two-class logits formed as `[-logit, logit]` |

This keeps the public SynthScope model interface:

```python
loss_dict = model.compute_loss(batch)
loss = loss_dict["loss"]
outputs = loss_dict["outputs"]
```

## Similarity to the official NPR implementation

The SynthScope implementation now matches the method-defining parts of the official NPR code closely:

| Area | Similarity |
|---|---|
| Artifact representation | Same nearest-neighbor down/up reconstruction and residual subtraction |
| NPR scaling | Same `2.0 / 3.0` scaling before classification |
| Stem | Same `3x3`, stride-2 convolution style |
| Classifier depth | Same lightweight early-ResNet layout using `layer1` and `layer2` only |
| Binary objective | Same one-logit real/fake detector concept |

## Integration differences

The implementation is adapted to SynthScope rather than copied line-for-line from the official repository.

| Difference | Why it exists |
|---|---|
| Uses `torchvision.models.resnet.Bottleneck` | Reuses the local PyTorch dependency instead of maintaining a separate ResNet copy |
| Uses SynthScope's `compute_loss(batch)` interface | Keeps training, validation, metrics, and Accelerate integration consistent with the project |
| Returns two-class metric logits | SynthScope metrics expect class logits, so the one binary logit is converted to `[-logit, logit]` |
| Does not include the official standalone train/test scripts | SynthScope handles training through its shared trainer and Hydra-style configuration |

One practical caveat from the official repo is image size: NPR reconstruction is simplest when spatial dimensions are even. The official testing notes crop odd image dimensions before applying NPR in some settings. SynthScope transforms should likewise keep inputs at even spatial sizes.

## Core novelties

### 1. Local up-sampling artifacts

NPR focuses on local pixel dependencies caused by up-sampling, rather than only global frequency-domain artifacts.

### 2. Source-invariant artifact representation

Because up-sampling is common in GAN and diffusion generator pipelines, the representation can generalize beyond the generator family used for training.

### 3. Simple residual preprocessing

The method does not require a complex feature extractor to define the artifact. It uses a simple nearest-neighbor reconstruction residual before classification.

### 4. Lightweight classifier

The official detector uses a compact ResNet-style classifier, making the main contribution the NPR representation rather than a large architecture.

## Summary

NPR is a compact detector built around local neighboring-pixel residuals caused by generator up-sampling operations. In SynthScope, `NPR` implements the nearest-neighbor NPR transform, the official scaling, and a lightweight ResNet-style binary classifier adapted to the shared model API. It is best understood as a method-aligned SynthScope implementation of Tan et al.'s CVPR 2024 NPR detector, not a byte-for-byte copy of the official repository.

## Known limitation: resizing and image format

NPR reads local resampling traces, so it is sensitive to how images were resized and compressed. On CommGen15, where every
real image is a 500-640 px JPEG and every fake a 320 px PNG, the paper's square resize to 256 inverts its scores (mean AUROC
about 23); scoring at native resolution (`bench_eval.py --native-res --bs 1`) recovers most of the signal (AUROC 37-96 per
generator). We verified that SynthScope's NPR matches the official network, preprocessing and weights exactly, and still scores
AUROC 100 on ProGAN; details in [`repro/investigate/npr_commgen15.md`](../../../repro/investigate/npr_commgen15.md).

## License, credit and permission

The official NPR repository has no license. Its first author, Chuangchuang Tan, gave SynthScope written permission
(2026-10-11) to include this implementation under Apache-2.0 and to use the released `NPR.pth` weights for benchmarking
and in a free, non-commercial public web demo, with the original work and repository credited. Please cite the paper:

```bibtex
@InProceedings{Tan_2024_CVPR,
    author    = {Tan, Chuangchuang and Zhao, Yao and Wei, Shikui and Gu, Guanghua and Liu, Ping and Wei, Yunchao},
    title     = {Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {28130-28139}
}
```
