# SAFE: Simple Preserved and Augmented Features

## Short method explanation

**SAFE** is a lightweight synthetic-image detector from the paper **"Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective"** by Li et al., accepted to KDD 2025.

The paper argues that generalizable synthetic image detection is limited not only by model architecture, but also by the training image transformations. The authors identify two common problems:

1. **Weakened artifact features:** resizing and down-sampling can distort or suppress the low-level artifacts that distinguish synthetic images from real images.
2. **Overfitted artifact features:** detectors can overfit to color, semantic, or generator-specific shortcuts in the training data.

SAFE addresses these issues with a simple detector and a targeted transformation recipe. Instead of relying on a large pretrained representation, it uses local high-frequency artifact evidence and training augmentations that encourage the model to focus on robust image traces.

Official repo: https://github.com/Ouxiang-Li/SAFE  
Paper: https://huggingface.co/papers/2408.06741

## Pipeline

Given an input image:

1. Convert the image to a tensor in `[0, 1]`.
2. Apply a discrete wavelet transform.
3. Keep the high-high frequency sub-band from the first DWT level.
4. Resize the high-frequency map back to the input resolution.
5. Feed the transformed image into a lightweight ResNet-50-style binary classifier.
6. Output real/fake logits.

The model is trained as a binary classifier with real images as class `0` and generated images as class `1`.

## Training dataset

SAFE follows the CNNDetection training setup and trains on the **CNNDetection / ForenSynths** real-vs-fake dataset.

The official repository lists:

| Split | Source |
|---|---|
| Train | CNNDetection CVPR 2020 |
| Val | CNNDetection CVPR 2020 |
| Test Set 1 | CNNDetection CVPR 2020 |
| Test Set 2 | FreqNet |
| Test Set 3 | UniversalFakeDetect |
| Test Set 4 | GenImage |
| Test Set 5 | DiTFake |
| Test Set 6 | GPT-ImgEval |
| Test Set 7 | T2I-CoReBench |

In this repo, the SAFE pre-test launcher selects the CNNDetection config explicitly:

```bash
dataset@DatasetConfig=cnndetection
```

## Augmentation details

SAFE's main contribution is its image transformation recipe. The official training transform uses crop-based preprocessing, horizontal flip, 180-degree random rotation, color jitter, tensor conversion, and patch-based random masking.

| Component | Official SAFE setting | SynthScope SAFE setting |
|---|---|---|
| Input size | `256` | `256` |
| Train crop | Random crop with `pad_if_needed=True` | `random_crop_256_pad` |
| Eval crop | Center crop | `center_crop_256` |
| Horizontal flip | `p = 0.5` | `random_horizontal_flip` |
| Rotation | `RandomRotation(180)` | `random_rotation_180` |
| Color jitter | brightness, contrast, saturation `0.5` | `color_jitter_safe` |
| Tensor conversion | `ToTensor()` | `to_dtype` |
| Random mask | ratio `(0.00, 0.75)`, patch size `16`, `p = 0.5` | `random_mask_safe` |
| Normalization | None | None |

The current SAFE experiment launcher uses:

```bash
DatasetConfig.train_transform_list=[random_crop_256_pad,random_horizontal_flip,random_rotation_180,color_jitter_safe,to_dtype,random_mask_safe]
DatasetConfig.eval_transform_list=[center_crop_256,to_dtype]
```

This matches the official transform order closely. `to_dtype` plays the same role as `ToTensor()` in the official code: it converts the loaded image tensor to `float32` and scales values to `[0, 1]`.

## Model details

The official SAFE model is a ResNet-50-style detector with a DWT preprocessing step. The active forward path applies:

1. `DWTForward(J=1, mode="symmetric", wave="bior1.3")`
2. The HH sub-band: `Yh[0][:, :, 2, :, :]`
3. A small ResNet trunk using `conv1`, `layer1`, and `layer2`
4. Global average pooling
5. A binary linear classifier

The SynthScope implementation follows the same active path:

```python
self.dwt_filter = DWTForward(J=1, mode="symmetric", wave="bior1.3")
_, high_bands = self.dwt_filter(x)
high_high = high_bands[0][:, :, 2, :, :]
```

It then feeds the high-frequency image through the same ResNet-50-style stem, `layer1`, `layer2`, average pooling, and binary head.

## Training setup

The official SAFE training defaults include:

| Setting | Official SAFE | SynthScope SAFE pre-test |
|---|---:|---:|
| Model | SAFE / ResNet50-style | `ModelConfig.model_type=safe` |
| Per-GPU batch size | `64` | `64` |
| Epochs | `100` | `100` |
| Base learning rate | `5e-4` | `0.0005` |
| Optimizer | AdamW | AdamW |
| Weight decay | `0.05` | `0.05` |
| Label smoothing | `0.1` | `0.1` |
| Warmup | `20` epochs | `warmup_ratio=0.2` |
| Scheduler | Cosine-style decay | `cosine` scheduler |

The SynthScope launcher is:

```bash
experiments/safe_pretest.sh
```

It runs through Accelerate and the `SynthScope` conda environment.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

`SynthScope_train` = **retrained SAFE v2** (the full 40-epoch SAFE crop recipe, checkpoint `checkpoints_retrain/safe_full_v2/model_38.pth`; best-val epoch). This supersedes the earlier under-trained / resize-era column (ForenSynths 81.74). Traced to `repro/bench_results/safe_v2_*.json` (jobs 186232 / 186235).

> **Served checkpoint:** the web app / `synthscope.load("safe")` now serves this **SynthScope-trained** checkpoint (registry `weights="synthscope_trained"`, `ckpt_kind="synthscope"`, `safe.pth → safe_full_v2/model_38`), since the retrain reproduces the paper and beats the official checkpoint on GenImage and full-13 ForenSynths. The official SAFE weights remain available under `repro/official_ckpts/safe/checkpoint-best.pth`.

| Benchmark | Paper | SynthScope_official | SynthScope_train (retrain v2) | Status |
|---|---|---|---|---|
| ForenSynths-8 (Table 1, intra-architecture GANs) | map=98.8 / acc=96.2 | mAP=98.78 / acc=96.2 | mAP=97.79 / acc=95.1 | reproduced |
| Self-Synthesis / GANGen (Table 2, intra-archit) | map=99.8 / acc=99.0 | mAP=99.85 / acc=99.0 | not re-run (GANGen data offline on cn023) | official |
| Ojha / UniversalFakeDetect-8 (Table 3, cross-arch) | map=99.1 / acc=95.7 | mAP=99.1 / acc=95.71 | mAP=98.44 / acc=94.0 | reproduced |
| GenImage-8 (Table 4, cross-architecture DM+GAN) | map=99.3 / acc=95.6 | mAP=99.32 / acc=95.62 | mAP=99.51 / acc=95.9 | reproduced |
| DiTFake (Table 6, cross-architecture DiT) | map=99.9 / acc=99.4 | — | mAP=99.96 / acc=99.2 | reproduced |
| DiffusionForensics-8 | — | — | mAP=99.34 / acc=91.3 | reproduced |
| ForenSynths-13 (full set, incl. CRN/IMLE/face) | — | mAP=85.74 / acc=84.9 | mAP=87.31 / acc=83.9 | retrain ≥ official |

**reproduced** = SynthScope-trained checkpoint matches the paper · **official** = official weights loaded into the SynthScope class match · **retrain ≥ official** = on this protocol the retrain equals/exceeds the official checkpoint · **missing** = not yet evaluated.

> **ForenSynths-8 vs -13.** The paper's ForenSynths protocol is the **8-generator** subset (`progan, stylegan, stylegan2, biggan, cyclegan, stargan, gaugan, deepfake`). The full 13-category CNNDetection test set additionally contains CRN/IMLE (graphics-rendered) and `whichfaceisreal` (faces), where **both** the official checkpoint (85.74) and our retrain (87.31) drop — it is a generator-distribution effect, not a training failure. On the paper's own 8-subset the retrain reaches 97.79 mAP (deepfake AP recovered from 58.82 at epoch 10 → 86.74 at epoch 38 over the full schedule).

## Similarity to the official SAFE implementation

The SynthScope implementation is intentionally close to the official SAFE implementation in the parts that define the method:

| Area | Similarity |
|---|---|
| DWT preprocessing | Same `pytorch_wavelets.DWTForward`, same level, mode, wavelet, and HH sub-band |
| Architecture | Same lightweight ResNet-50-style active path using stem, `layer1`, `layer2`, average pooling, and binary classifier |
| Input scale | Same 256-pixel input size |
| Train augmentations | Same crop, flip, rotation, color jitter, tensor conversion, and random mask recipe |
| Dataset choice | Same CNNDetection / ForenSynths training source |
| Training hyperparameters | Same main defaults: batch size, epochs, learning rate, weight decay, label smoothing, warmup length |

The remaining differences are minor integration differences caused by adapting SAFE to this repo's trainer and config system:

| Difference | Why it exists |
|---|---|
| Classifier name is `head` instead of `fc1` | Matches the naming convention used by other SynthScope image models |
| DWT module is created once in `__init__` | Avoids recreating `DWTForward` every forward pass; the DWT settings are the same |
| DWT resize uses `F.interpolate` | Keeps resizing inside PyTorch tensor operations; it is equivalent in purpose to resizing the HH band back to input size |
| Training loop uses SynthScope + Accelerate | Keeps distributed training, metrics, and logging consistent with the rest of this project |
| Scheduler uses Hugging Face `get_scheduler` | Matches SynthScope's scheduler integration; it approximates SAFE's warmup plus cosine schedule |
| Metrics use SynthScope metrics | Current repo reports real/fake accuracy through `AIGIMetrics`; official SAFE also reports average precision in evaluation |

These differences should not change the conceptual method: the detector still trains on CNNDetection with SAFE's DWT high-frequency input and SAFE's transformation recipe.

## Core novelties

### 1. Crop instead of resize-heavy preprocessing

SAFE avoids relying on down-sampling as the main preprocessing step. The paper argues that down-sampling can weaken subtle artifact features, while crop-based preprocessing better preserves local traces.

### 2. DWT high-frequency artifact features

SAFE uses a wavelet transform and keeps the high-high frequency component. This emphasizes local high-frequency evidence where synthetic-image artifacts can appear.

### 3. Color and rotation augmentation

Color jitter and random rotation reduce shortcut learning from color discrepancies or semantic biases in the training set.

### 4. Patch-based random masking

Random masking forces the detector to use local regions instead of depending too strongly on one global cue. This is meant to improve local awareness and cross-generator generalization.

## Summary

SAFE is a compact synthetic-image detector built around high-frequency DWT features and a SID-specific augmentation recipe. In SynthScope, `SynthScopeSAFEModel` follows the official SAFE active model path and `experiments/safe_pretest.sh` uses the official-style CNNDetection training setup. The implementation is not a byte-for-byte copy of the official repo, but it is method-aligned: same core DWT feature, same lightweight classifier path, same 256-pixel transformation recipe, and same main training hyperparameters.
