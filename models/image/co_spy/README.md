# CO-SPY: Combining Semantic and Pixel Features

## Short method explanation

**CO-SPY** is an AI-generated image detector from the paper **"CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI"** by Cheng et al., CVPR 2025.

The paper observes that the two dominant families of synthetic-image detectors are complementary but not synergistic:

1. **Artifact / pixel detectors** (e.g. NPR, LNP) generalize well across new generators and unseen objects but degrade sharply under lossy transformations such as JPEG compression, because the upsampling-style artifacts they latch onto are low-level and easily destroyed.
2. **Semantic detectors** (e.g. UnivFD, Fusing, DRCT) are robust to lossy transformations but generalize poorly to new generative models, because they rely on content-style cues specific to the training data.

CO-SPY first enhances each side independently and then fuses them adaptively, so the two streams reinforce one another instead of being a simple additive ensemble:

- **Enhanced semantic detector** -- a strong frozen vision-language encoder (OpenCLIP `ViT-SO400M-14-SigLIP-384`) used purely as a feature extractor, plus a feature-space interpolation augmentation that creates soft-label samples between real and fake to mitigate overfitting.
- **Enhanced artifact detector** -- the upsampling-residual cue is replaced by a *VAE-reconstruction residual* `|x - VAE(x)|`. Because the VAE shares the same decoding stage that GANs and LDMs use, the resulting artifacts are higher-level and survive lossy transformations far better than raw upsampling residuals. A small ResNet-50-style stem is trained on these residuals.
- **Adaptive fusion** -- two regulator heads emit per-sample scalar coefficients `alpha` and `beta` that scale the semantic and artifact features before concatenation. During training, with some probability one of the two coefficients is randomly zeroed (never both), so the classifier cannot overfit to either branch.

Official repo: https://github.com/Megum1/Co-Spy
Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Cheng_CO-SPY_Combining_Semantic_and_Pixel_Features_to_Detect_Synthetic_Images_CVPR_2025_paper.pdf

## Pipeline

Given an input image `x` in `[0, 1]^{3 x H x W}` (no normalization at the dataset side):

1. **Semantic encoding**. Resize to `semantic_image_size x semantic_image_size`, apply SigLIP normalization (`mean = std = 0.5`), and feed through the frozen OpenCLIP visual encoder to get `v_sem`.
2. **Artifact encoding**. Resize to `artifact_image_size x artifact_image_size`. Encode with the frozen SD-v1.4 VAE: `mu = E(x)`, `x' = D(mu)`. Compute the scaled residual `r = (x - x') * artifact_residual_scale` (default `100 / 7`, matching upstream). Feed `r` through a small ResNet-50-style stem (`conv1 + bn1 + maxpool + layer1(3 x Bottleneck 64) + layer2(4 x Bottleneck 128) + avgpool`) to obtain `v_art` (512-d).
3. **Adaptive fusion**. Two linear regulators produce per-sample scalar coefficients `alpha = sem_fc(v_sem)` and `beta = art_fc(v_art)`. During training, with probability `coeff_dropout`, randomly zero one of them. The fused feature is `[alpha * v_sem, beta * v_art]`.
4. **Classification**. A linear head maps the fused feature to a single real/fake logit. The two AGID-side outputs are `stack([-logit, +logit], dim=1)` so the downstream metrics behave the same as for `npr` and `vib_net`.

The detector uses label `0` for real images and label `1` for generated images.

## Why this is considered an implementation of the paper

| Paper concept | AGID implementation |
|---|---|
| Frozen OpenCLIP semantic backbone | `open_clip.create_model_and_transforms(semantic_model_name, pretrained=semantic_pretrained)` with default `ViT-SO400M-14-SigLIP-384` / `webli` |
| Feature-space interpolation augmentation (Eq. 2) | `_mix_semantic_features` mixes `v_sem` and labels within batch using `delta ~ Beta(alpha, alpha)` for a `feat_interp_prob` fraction of samples |
| Frozen VAE reconstruction residual (Eq. 1) | `AutoencoderKL.from_pretrained(vae_id, subfolder=vae_subfolder)` with default `CompVis/stable-diffusion-v1-4`; residual `(x - VAE(x)) * artifact_residual_scale` |
| ResNet-50-style artifact encoder | `_ArtifactResNetStem` -- conv1 + bn1 + maxpool + layer1 (3 Bottleneck 64) + layer2 (4 Bottleneck 128) + avgpool -> 512-d, with Kaiming init |
| Two per-branch regulator coefficients `alpha`, `beta` | `sem_fc: d_sem -> 1` and `art_fc: 512 -> 1` |
| Coefficient dropout (never both) | `_apply_coeff_dropout` zeros at most one of `alpha`, `beta` per sample with probability `coeff_dropout` |
| Concatenation + linear classifier | `self.fc(torch.cat([alpha * v_sem, beta * v_art], dim=1))` |
| BCE loss with label smoothing | `_LabelSmoothingBCEWithLogits(smoothing = TrainerConfig.label_smoothing)` |

## Model details

The AGID model is implemented in `models/image/co_spy/co_spy.py` as `AGIDCoSpyModel`.

| Component | Setting |
|---|---|
| Semantic backbone | OpenCLIP visual encoder, frozen |
| Default semantic model | `ViT-SO400M-14-SigLIP-384`, `webli` |
| Semantic input | `[0, 1]` image resized to `384 x 384`, normalized with `mean = std = 0.5` (inside `forward`) |
| Artifact backbone | SD-v1.4 `AutoencoderKL`, frozen |
| Artifact input | `[0, 1]` image resized to `224 x 224` (inside `forward`) |
| Artifact encoder | ResNet-50 stem (`conv1 + bn1 + maxpool + layer1 + layer2 + avgpool`) -> 512-d |
| Regulators | `sem_fc(d_sem -> 1)`, `art_fc(512 -> 1)` |
| Classifier | `fc(d_sem + 512 -> 1)` |
| Coefficient dropout | `coeff_dropout` (default `0.3`) |
| Feature interpolation | `feat_interp_prob` (default `0.5`), `feat_interp_alpha` (default `0.2`) |
| Output (AGID metrics) | two-class real / fake logits = `stack([-logit, +logit], dim=1)` |

Trainable parameters: the artifact encoder, both regulators, and the final classifier head. The OpenCLIP backbone and the VAE remain frozen.

## Training setup

The CO-SPY configs live in:

```bash
conf/model/co_spy.yaml
conf/trainer/co_spy.yaml
```

The model config:

| Setting | Value |
|---|---:|
| `model_type` | `co_spy` |
| `semantic_model_name` | `ViT-SO400M-14-SigLIP-384` |
| `semantic_pretrained` | `webli` |
| `semantic_image_size` | `384` |
| `vae_id` | `CompVis/stable-diffusion-v1-4` |
| `vae_subfolder` | `vae` |
| `artifact_image_size` | `224` |
| `artifact_residual_scale` | `100 / 7` |
| `coeff_dropout` | `0.3` |
| `feat_interp_prob` | `0.5` |
| `feat_interp_alpha` | `0.2` |
| `freeze_backbone` | `true` |

The trainer config follows the paper's adapter-tuning regime: AdamW with `lr = 1e-4`, weight decay `0.0`, label smoothing `0.1`, cosine schedule with a short warmup, modest number of epochs.

A launcher is provided at:

```bash
experiments/co_spy.sh
```

It uses the default dataset (`genimage`) and a CO-SPY-style transform pipeline that produces raw `[0, 1]` images at `384 x 384`. Importantly, **the experiment script omits any `normalize_*` transform**, because CO-SPY applies per-branch normalization inside `forward` (SigLIP normalization for the semantic branch, raw `[0, 1]` for the artifact branch). Augmentations applied at the dataset side are `resize_short_384, gaussian_blur, jpeg_compression, random_crop_384, random_horizontal_flip, to_dtype`.

## Integration notes

The CO-SPY model is called through the shared `compute_loss(batch)` interface used by the AGID trainer:

| Key | Meaning |
|---|---|
| `loss` | scalar training objective (BCE-with-logits on the single fused logit; soft targets are used when feature interpolation is active) |
| `outputs` | two-class real / fake logits for AGID metrics (`stack([-logit, +logit], dim=1)`) |

For the trainer to instantiate CO-SPY, `AGIDCoSpyModel` is registered in `trainer.py` **before** the generic CLIP branch. CO-SPY does not use the `CLIP:` backbone prefix (it loads OpenCLIP directly), but registering early keeps the file in line with `fat_former` and avoids surprises if `backbone_name` ever leaks into the config.

The model holds large frozen sub-modules (the SigLIP encoder and the SD-v1.4 VAE). To reduce memory pressure, the default `experiments/co_spy.sh` uses a per-process batch size of `16`. Increase it on hardware with more GPU memory.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Single end-to-end training pipeline | The official repo offers `end2end`, `branch`, and `fusion` modes. AGID's trainer is single-pass, so we replicate the `end2end` variant. The two regulator heads, the residual scaling, the coefficient dropout, and the feature interpolation are all preserved. |
| Feature interpolation always on (controlled by `feat_interp_prob`) | The official code only enables feature interpolation in the `branch:semantic` stage. Since AGID trains all parts jointly, we apply it during end-to-end training as well; setting `feat_interp_prob=0.0` disables it. |
| Augmentations (`gaussian_blur`, `jpeg_compression`) come from AGID's transform pipeline | The upstream code applies blur and JPEG inside a `transforms.Lambda` on PIL images; AGID does the equivalent with `torchvision.transforms.v2.GaussianBlur` and `torchvision.transforms.v2.JPEG`. |
| Pretrained weights come from the standard HuggingFace / OpenCLIP caches | `HF_HOME` and `TORCH_HOME` are already set in `experiments/co_spy.sh`. |
| Uses Hydra configs, Accelerate, and AGID metrics | Keeps training, validation, testing, logging, and distributed support consistent with the rest of this project. |

These are integration differences rather than conceptual changes: the detector still uses a frozen SigLIP, a frozen SD-v1.4 VAE, the same artifact residual, a learnable ResNet-50 stem, two per-sample regulators, coefficient dropout, semantic feature interpolation, and the same BCE-with-logits objective.

## Core novelties

### 1. VAE reconstruction residual

Rather than relying on raw upsampling residuals (which JPEG erases), CO-SPY uses a pre-trained VAE that shares the decoding stage common to both diffusion models and GANs. The reconstruction residual `x - VAE(x)` therefore captures higher-level decoding artifacts that survive lossy compression much better than raw pixel-level cues.

### 2. SigLIP semantic backbone

CO-SPY swaps the older OpenAI CLIP for OpenCLIP `ViT-SO400M-14-SigLIP-384`, which was pre-trained on a larger and more diverse corpus with a sigmoid loss. This gives the semantic branch richer object/scene priors and better generalization to unseen content.

### 3. Feature-space interpolation

For half the batch on average, CO-SPY mixes the CLIP feature with a random partner using a coefficient sampled from `Beta(alpha, alpha)` (with small `alpha`, so mixes concentrate near hard labels but include occasional rich mixtures). The label becomes the same convex combination. This both enlarges the training set and discourages overfitting to hard real/fake boundaries.

### 4. Adaptive fusion with coefficient dropout

Two regulator heads emit per-sample scalar coefficients `alpha` and `beta`. The classifier sees `[alpha * v_sem, beta * v_art]`, so the model learns how strongly to trust each branch for each input. With probability `coeff_dropout`, one of the two coefficients is randomly zeroed (never both), which prevents the classifier from collapsing onto either branch and keeps both pathways informative.

## Summary

CO-SPY in AGID is a method-level reproduction of Cheng et al.'s CVPR 2025 paper. A frozen OpenCLIP SigLIP backbone supplies the semantic feature, a frozen SD-v1.4 VAE supplies a reconstruction-residual artifact feature, two learned regulator heads emit per-sample mixing coefficients, and a small linear classifier reads the fused representation. Training adds the paper's semantic feature interpolation and coefficient dropout. The implementation is adapted to AGID's model / config / trainer / metrics structure, and is built directly on the `open_clip_torch` and `diffusers` packages -- so it is not a line-by-line copy of the official repo, but it implements the central architecture and objective that define the paper.
