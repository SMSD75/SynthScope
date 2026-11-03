# VIB-Net: Variational Information Bottleneck for Universal AI-Generated Image Detection

## Short method explanation

**VIB-Net** is an AI-generated image detector based on the paper **"Towards Universal AI-Generated Image Detection by Variational Information Bottleneck Network"** by Zhang et al., CVPR 2025.

The paper argues that large pretrained visual representations, such as CLIP features, contain both authentication-related cues and many authentication-irrelevant cues. A detector trained directly on the full feature vector may overfit to generator-specific shortcuts and generalize poorly to unseen generation families. VIB-Net addresses this by inserting a **Variational Information Bottleneck (VIB)** between the frozen image representation and the binary real/fake classifier.

Instead of classifying CLIP features directly, the model predicts a Gaussian latent distribution, samples a compact latent code, and classifies from that code. The training objective encourages the latent variable to preserve label-relevant evidence while suppressing unnecessary information.

Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Towards_Universal_AI-Generated_Image_Detection_by_Variational_Information_Bottleneck_Network_CVPR_2025_paper.pdf

## Pipeline

Given an input image:

1. Encode the image with a CLIP visual backbone.
2. Flatten the visual features if the backbone returns spatial features.
3. Apply dropout to the CLIP feature vector.
4. Pass the feature through a two-layer MLP.
5. Predict Gaussian statistics `(mu, std)` for a latent distribution.
6. Sample a latent code with the reparameterization trick: `z = mu + eps * std`.
7. Decode the sampled latent code into a single binary logit.
8. Train with binary classification loss plus a beta-weighted KL regularizer.

The detector uses label `0` for real images and label `1` for generated images.

## VIB objective

The paper's VIB objective combines prediction loss with a KL penalty on the latent distribution:

```text
L = prediction_loss + beta * KL(q(z | x) || r(z))
```

In this implementation:

| Term | SynthScope implementation |
|---|---|
| Prediction loss | `BCEWithLogitsLoss` on the binary real/fake logit |
| Latent posterior | diagonal Gaussian parameterized by `mu` and `std` |
| Prior `r(z)` | standard normal `N(0, I)` |
| KL term | analytic Gaussian KL, summed over latent dimensions and averaged over the batch |
| Beta | `ModelConfig.vib_beta` when provided, otherwise `1e-3` |

The analytic KL used here is:

```text
0.5 * sum(mu^2 + std^2 - log(std^2) - 1)
```

This is the standard closed-form KL divergence from `N(mu, diag(std^2))` to `N(0, I)`, matching the Gaussian VIB formulation described in the paper.

## Model details

The current SynthScope model is implemented in `models/image/vib_net/vib_net.py` as `SynthScopeVIBNetModel`.

| Component | Setting |
|---|---|
| Backbone | CLIP visual encoder |
| Default backbone | `CLIP:ViT-L/14` |
| Backbone training | frozen by default |
| Hidden dimension | `1024` |
| Latent dimension | `256` |
| Dropout | `0.5` |
| Standard deviation parameterization | `softplus(raw_std - std_shift)` |
| Default `std_shift` | `5.0` |
| Decoder | `Linear(latent_dim, 1)` |
| Output | one binary fake/real logit |

The model follows the architecture described by the paper: pretrained image features are projected into a stochastic bottleneck, and the classifier operates on the sampled bottleneck representation rather than on the full feature vector.

## Why this is considered an implementation of the paper

This SynthScope implementation reproduces the method-defining pieces of VIB-Net:

| Paper concept | SynthScope implementation |
|---|---|
| Use a pretrained CLIP image representation | `clip.load(backbone_name)` and `backbone.encode_image(inputs)` |
| Avoid directly classifying the full representation | features are passed through a stochastic latent bottleneck |
| Predict latent Gaussian statistics | `fc_3` outputs `2 * latent_dim`, split into `mu` and `std` |
| Use a compact latent code | default `vib_latent_dim: 256` |
| Use reparameterized sampling | `_reparameterize(mu, std)` |
| Decode the latent variable for binary detection | `decode = nn.Linear(latent_dim, 1)` |
| Optimize the VIB objective | binary prediction loss plus beta-weighted Gaussian KL |
| Use real/fake binary labels | `BCEWithLogitsLoss` with labels `0` and `1` |

The implementation is therefore best understood as a **paper-faithful SynthScope reproduction** of VIB-Net's architecture and objective, adapted to this repository's Hydra configuration, Accelerate trainer, dataset module, and metrics interface.

## Official model (`vib_net_official`)

`conf/model/image/vib_net_official.yaml` (registry key `vib_net_official`) is the official model exactly as the released
test code builds it (`models/clip_models.py`): CLIP ViT-L/14 loaded in fp32 (the official loader runs on the CPU and
moves the model; `clip.load` on a GPU would give fp16), no feature normalization, `softplus(x - 5)` without a floor, one
latent sample per image also at inference, and the official test transform (a 224 center crop of the image at its own
size, no resize). The released checkpoints map onto it with `remap_vibnet` (`model.*` -> `backbone.*`). The older
`vib_net` key keeps the earlier SynthScope variant (feature BatchNorm, std floor, resize 256), which the web demo serves.

## Training in SynthScope

No training code is released (the official repository has the test code and the checkpoints only), so the recipe is
reconstructed from three sources and documented in `experiments/image/vib_net.sh`:

| Setting | Value | Source |
|---|---|---|
| Data | GenImage SD v1.4 train split, a seeded 16k real + 16k fake subset; validation 1k + 1k held out from the same split | paper ("32,000 real and generated images"); the GenImage val alias is its test split, so it is not used to select |
| Augmentation | blur p 0.1 (sigma 0-3) + JPEG p 0.1 (cv2/PIL, q 30-100), random 224 crop, flip, CLIP normalization | released `options/base_options.py` values on UniversalFakeDetect's `data_augment` (`blur_jpeg_cnndet`) |
| Scale | the image at its own size (no resize) | the official test is a 224 center crop without resize; UniversalFakeDetect's pipeline resizes to 256 first, which on GenImage's 512 px images trains at half scale |
| Model | frozen fp32 CLIP ViT-L/14 + VIB head, K = 256, beta 1e-3 | paper Fig. 4 / released code; beta is not reported (1e-3 is the standard VIB value) |
| Optimizer | Adam(W) lr 2e-4, no weight decay, batch 256 | paper implementation details; released options |
| Schedule | validation accuracy every epoch; after 5 epochs without a 0.001 gain the learning rate is divided by 10, stopping below 1e-6 (at most 100 epochs) | UniversalFakeDetect `train.py` (trainer `plateau`) |

Two runs, evaluated on the paper's protocol (7 GenImage generators + 8 ForenSynths GANs, official test transform):

| Run | best val acc | 15-generator AP / acc | GenImage-7 AP / acc | ForenSynths-8 AP / acc |
|---|---|---|---|---|
| resize 256 then crop (UniversalFakeDetect pipeline) | 95.5 | 82.98 / 70.20 | 89.64 / 74.76 | 77.15 / 66.25 |
| **native-resolution crop** (default) | 98.3 | **94.08 / 85.22** | 97.18 / 87.16 | 91.37 / 83.52 |

Per generator (native resolution, AP / acc; paper in brackets): SD v1.4 99.98 / 99.29 (100.0 / 99.55) · SD v1.5 99.98 /
98.90 (99.97 / 99.20) · ADM 89.96 / 66.80 (95.49 / 73.85) · GLIDE 95.48 / 75.23 (97.13 / 74.25) · Midjourney 98.35 / 88.33
(97.81 / 88.05) · Wukong 99.96 / 99.17 (99.93 / 98.25) · VQDM 96.56 / 82.37 (97.00 / 89.35) · ProGAN 97.09 / 90.83 (96.59 /
89.70) · CycleGAN 98.90 / 91.90 (98.44 / 88.60) · BigGAN 96.96 / 91.17 (97.17 / 91.20) · StyleGAN 85.66 / 75.53 (84.31 /
74.10) · StarGAN 96.81 / 84.74 (97.60 / 80.70) · GauGAN 98.05 / 91.62 (96.94 / 87.15) · Deepfake 68.46 / 60.61 (81.32 / 72.00)
· SAN 89.00 / 81.74 (93.27 / 81.50). Runs: jobs 267625 / 267846 (training), 267842 / 267928 (evaluation); results in
`repro/bench_results/vib_net_trained*.json`.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| Diffusion-source (SDv1.4-trained), 15 generators, AP -- headline | map=95.53 | *not available* | **mAP=94.08** / acc=85.22 | **reproduced** (within 1.5 AP) |
| Diffusion-source (SDv1.4-trained), 15 generators, ACC | acc=85.83 | *not available* | mAP=94.08 / **acc=85.22** | **reproduced** (within 0.7) |
| GAN-source (ForenSynths/ProGAN-trained), AP | map=93.14 | *not available* | — | missing (not trained on ProGAN yet) |
| GAN-source (ForenSynths/ProGAN-trained), ACC | acc=82.5 | *not available* | — | missing (not trained on ProGAN yet) |

**SynthScope_official** = authors' released weights run through the SynthScope class (the checkpoints are on a
restricted Google Drive; access has been requested) · **SynthScope_train** = trained in SynthScope with the recipe above
· **Status:** reproduced = SynthScope_train matches the paper · missing = not yet evaluated. The earlier entry for the
GAN-source rows (61.71 / 57.91) was the SD v1.4-trained model tested on ForenSynths, not a ProGAN-trained model, and has
been removed.

## Integration notes

The VIB-Net model is designed to be called through the shared `compute_loss(batch)` interface used by the SynthScope trainer. It returns:

| Key | Meaning |
|---|---|
| `loss` | scalar training objective |
| `outputs` | two-class logits for SynthScope metrics |

For the trainer to instantiate VIB-Net, `SynthScopeVIBNetModel` should be registered in `trainer.py` before the generic CLIP branch. Since `vib_net` uses a `CLIP:` backbone name, a generic CLIP condition placed first can accidentally instantiate `SynthScopeCLIPModel` instead of `SynthScopeVIBNetModel`.

For metric compatibility, the binary VIB logit should be converted to two-class logits in a way that preserves the same decision boundary as `sigmoid(logit) > 0.5`. A common mapping is:

```python
outputs = torch.stack([torch.zeros_like(logit), logit], dim=1)
```

This makes `argmax(outputs, dim=1)` equivalent to predicting fake when the VIB logit is positive.

## Differences from the paper text

The paper specifies the method and objective, but not every engineering detail needed for integration into SynthScope. The current implementation makes the following practical choices:

| Difference | Reason |
|---|---|
| Uses SynthScope's `compute_loss(batch)` model interface | Keeps VIB-Net compatible with the shared trainer |
| Uses `BCEWithLogitsLoss` for the one-logit binary classifier | Matches the model's single-logit output and avoids an unnecessary two-logit head |
| Uses an analytic KL to `N(0, I)` | Direct implementation of the Gaussian VIB objective |
| Uses Hydra configs | Matches the rest of SynthScope |
| Uses Accelerate trainer and SynthScope metrics | Keeps training, validation, testing, logging, and distributed support consistent |

These are integration differences rather than conceptual changes: the detector still uses CLIP features, a stochastic Gaussian bottleneck, reparameterized latent sampling, and the VIB loss.

## Core novelties

### 1. Stochastic bottleneck between representation and classifier

VIB-Net does not classify directly from the full CLIP feature vector. It learns a compact latent distribution and samples a latent code for classification.

### 2. Information suppression through KL regularization

The KL term discourages the latent representation from carrying unnecessary information. This is intended to reduce shortcut learning and improve cross-generator generalization.

### 3. CLIP features with task-focused compression

The method keeps the benefit of large pretrained CLIP representations while adding a trainable bottleneck that emphasizes authentication-relevant information.

### 4. Binary detection from a sampled latent code

The classifier predicts real/fake from the sampled latent variable, so the final decision depends on the information that passes through the bottleneck.

## Summary

VIB-Net in SynthScope is a method-level reproduction of Zhang et al.'s CVPR 2025 VIB-Net paper. It uses CLIP image features, predicts Gaussian bottleneck statistics, samples a latent code with reparameterization, and trains with binary prediction loss plus a beta-weighted Gaussian KL term. The implementation is adapted to SynthScope's model, config, trainer, and metrics structure, so it is not a line-by-line official-code reproduction, but it implements the central architecture and objective that define the paper.
