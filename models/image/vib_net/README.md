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

| Term | AGID implementation |
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

The current AGID model is implemented in `models/image/vib_net/vib_net.py` as `AGIDVIBNetModel`.

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

This AGID implementation reproduces the method-defining pieces of VIB-Net:

| Paper concept | AGID implementation |
|---|---|
| Use a pretrained CLIP image representation | `clip.load(backbone_name)` and `backbone.encode_image(inputs)` |
| Avoid directly classifying the full representation | features are passed through a stochastic latent bottleneck |
| Predict latent Gaussian statistics | `fc_3` outputs `2 * latent_dim`, split into `mu` and `std` |
| Use a compact latent code | default `vib_latent_dim: 256` |
| Use reparameterized sampling | `_reparameterize(mu, std)` |
| Decode the latent variable for binary detection | `decode = nn.Linear(latent_dim, 1)` |
| Optimize the VIB objective | binary prediction loss plus beta-weighted Gaussian KL |
| Use real/fake binary labels | `BCEWithLogitsLoss` with labels `0` and `1` |

The implementation is therefore best understood as a **paper-faithful AGID reproduction** of VIB-Net's architecture and objective, adapted to this repository's Hydra configuration, Accelerate trainer, dataset module, and metrics interface.

## Training setup

The VIB-Net config lives in:

```bash
conf/model/vib_net.yaml
conf/trainer/vib_net.yaml
```

The current model config sets:

| Setting | Value |
|---|---:|
| `model_type` | `vib_net` |
| `backbone_name` | `CLIP:ViT-L/14` |
| `vib_hidden_dim` | `1024` |
| `vib_latent_dim` | `256` |
| `dropout` | `0.5` |
| `std_shift` | `5.0` |
| `freeze_backbone` | `true` |

The current trainer config uses AdamW through the AGID trainer:

| Setting | Value |
|---|---:|
| Epochs | `10` |
| Learning rate | `0.0001` |
| Weight decay | `0.05` |
| Label smoothing | `0.1` |
| Validation interval | `1` |
| Test interval | `1` |

`label_smoothing` is part of the shared trainer configuration and is used by the cross-entropy models. VIB-Net uses `BCEWithLogitsLoss`, so label smoothing is not part of its current loss unless it is explicitly added to `AGIDVIBNetModel.compute_loss`.

## Reproduced benchmark results

All benchmarks the paper reports, with our reproduced numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Metric | Paper | Ours | Status |
|---|---|---|---|---|
| Diffusion-source (SDv1.4-trained), AP -- HEADL | AP | map=95.53 | map=83.94 / acc=77.57 | below |
| Diffusion-source (SDv1.4-trained), ACC | acc | acc=85.83 | map=83.94 / acc=77.57 | below |
| GAN-source (ForenSynths/ProGAN-trained), AP | AP | map=93.14 | map=61.71 / acc=57.91 | below |
| GAN-source (ForenSynths/ProGAN-trained), ACC | acc | acc=82.5 | — | missing |

## Integration notes

The VIB-Net model is designed to be called through the shared `compute_loss(batch)` interface used by the AGID trainer. It returns:

| Key | Meaning |
|---|---|
| `loss` | scalar training objective |
| `outputs` | two-class logits for AGID metrics |

For the trainer to instantiate VIB-Net, `AGIDVIBNetModel` should be registered in `trainer.py` before the generic CLIP branch. Since `vib_net` uses a `CLIP:` backbone name, a generic CLIP condition placed first can accidentally instantiate `AGIDCLIPModel` instead of `AGIDVIBNetModel`.

For metric compatibility, the binary VIB logit should be converted to two-class logits in a way that preserves the same decision boundary as `sigmoid(logit) > 0.5`. A common mapping is:

```python
outputs = torch.stack([torch.zeros_like(logit), logit], dim=1)
```

This makes `argmax(outputs, dim=1)` equivalent to predicting fake when the VIB logit is positive.

## Differences from the paper text

The paper specifies the method and objective, but not every engineering detail needed for integration into AGID. The current implementation makes the following practical choices:

| Difference | Reason |
|---|---|
| Uses AGID's `compute_loss(batch)` model interface | Keeps VIB-Net compatible with the shared trainer |
| Uses `BCEWithLogitsLoss` for the one-logit binary classifier | Matches the model's single-logit output and avoids an unnecessary two-logit head |
| Uses an analytic KL to `N(0, I)` | Direct implementation of the Gaussian VIB objective |
| Uses Hydra configs | Matches the rest of AGID |
| Uses Accelerate trainer and AGID metrics | Keeps training, validation, testing, logging, and distributed support consistent |

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

VIB-Net in AGID is a method-level reproduction of Zhang et al.'s CVPR 2025 VIB-Net paper. It uses CLIP image features, predicts Gaussian bottleneck statistics, samples a latent code with reparameterization, and trains with binary prediction loss plus a beta-weighted Gaussian KL term. The implementation is adapted to AGID's model, config, trainer, and metrics structure, so it is not a line-by-line official-code reproduction, but it implements the central architecture and objective that define the paper.
