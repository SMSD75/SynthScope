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

## Model details

The AGID implementation lives in:

```bash
models/image/NPR/NPR.py
```

The current model uses:

| Component | AGID NPR setting |
|---|---|
| NPR transform | nearest down-sample, nearest up-sample, subtract, scale by `2.0 / 3.0` |
| Stem | `3x3` convolution, stride `2`, padding `1` |
| Backbone | lightweight ResNet-style Bottleneck trunk |
| Residual layers | `layer1`: 3 Bottleneck blocks, `layer2`: 4 Bottleneck blocks |
| Pooling | adaptive average pooling |
| Head | one binary logit |
| Loss | `BCEWithLogitsLoss` |
| Metric output | two-class logits formed as `[-logit, logit]` |

This keeps the public AGID model interface:

```python
loss_dict = model.compute_loss(batch)
loss = loss_dict["loss"]
outputs = loss_dict["outputs"]
```

## Similarity to the official NPR implementation

The AGID implementation now matches the method-defining parts of the official NPR code closely:

| Area | Similarity |
|---|---|
| Artifact representation | Same nearest-neighbor down/up reconstruction and residual subtraction |
| NPR scaling | Same `2.0 / 3.0` scaling before classification |
| Stem | Same `3x3`, stride-2 convolution style |
| Classifier depth | Same lightweight early-ResNet layout using `layer1` and `layer2` only |
| Binary objective | Same one-logit real/fake detector concept |

## Integration differences

The implementation is adapted to AGID rather than copied line-for-line from the official repository.

| Difference | Why it exists |
|---|---|
| Uses `torchvision.models.resnet.Bottleneck` | Reuses the local PyTorch dependency instead of maintaining a separate ResNet copy |
| Uses AGID's `compute_loss(batch)` interface | Keeps training, validation, metrics, and Accelerate integration consistent with the project |
| Returns two-class metric logits | AGID metrics expect class logits, so the one binary logit is converted to `[-logit, logit]` |
| Does not include the official standalone train/test scripts | AGID handles training through its shared trainer and Hydra-style configuration |

One practical caveat from the official repo is image size: NPR reconstruction is simplest when spatial dimensions are even. The official testing notes crop odd image dimensions before applying NPR in some settings. AGID transforms should likewise keep inputs at even spatial sizes.

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

NPR is a compact detector built around local neighboring-pixel residuals caused by generator up-sampling operations. In AGID, `NPR` implements the nearest-neighbor NPR transform, the official scaling, and a lightweight ResNet-style binary classifier adapted to the shared model API. It is best understood as a method-aligned AGID implementation of Tan et al.'s CVPR 2024 NPR detector, not a byte-for-byte copy of the official repository.
