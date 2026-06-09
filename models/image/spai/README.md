# SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning

## Short method explanation

**SPAI** is an AI-generated image detector from the paper **"Any-Resolution AI-Generated Image Detection by Spectral Learning"** by Karageorgiou et al., CVPR 2025.

The paper's key idea is that the **spectral distribution of real images is both an invariant and highly discriminative pattern**. A model trained only on real images to reconstruct missing frequencies (a self-supervised pretext task called *masked spectral learning* / Masked Frequency Modeling) learns this distribution well; AI-generated images deviate from it, and that deviation is the detection signal. Two contributions operationalize this:

1. **Spectral Reconstruction Similarity (SRS)** — measures how consistently the backbone represents an image, its low-frequency component, and its high-frequency component. On real images the three views stay coherent across the backbone's blocks; on generated images they diverge.
2. **Spectral Context Attention (SCA)** — a patch-based cross-attention that aggregates per-patch spectral features so the detector can process images of *any resolution* without resizing.

Official repo: https://github.com/mever-team/spai
Project page: https://mever-team.github.io/spai/
Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.pdf

> **Scope of this implementation.** AGID implements the **fixed-resolution** variant of SPAI (the official `MFViT` path): a single `224 x 224` crop, SRS + Spectral Context Vector + classifier. The any-resolution **Spectral Context Attention** wrapper (`PatchBasedMFViT`) is intentionally **not** included. Everything else matches the paper, and the implementation is verified numerically identical to the official model (see *Equivalence with the official model* below).

## Pipeline

Given an input image `x` in `[0, 1]^{3 x 224 x 224}` (no normalization at the dataset side):

1. **Frequency split**. A fixed circular low-pass mask `M` of radius `16`, applied to the center-shifted 2D DFT, splits `x` into a low-frequency image `x_low = IFFT(F(x) ⊙ M)` and a high-frequency image `x_high = IFFT(F(x) ⊙ (1 − M))`. Both are clamped to `[0, 1]`. (Same `filters.py` as upstream.)
2. **Backbone features**. The three views (`x`, `x_low`, `x_high`) are ImageNet-normalized **inside the model** and passed through a frozen MFM-pretrained ViT-B/16. The backbone returns the patch-token sequence of **all 12 transformer blocks** (CLS dropped): each view becomes a `B x 12 x 196 x 768` tensor.
3. **Per-block projection**. A separate projection head per block maps each block's tokens `768 -> 1024` (`LayerNorm -> Linear -> GELU -> Linear -> LayerNorm`, dropout `0.5`, no activation on the last layer). Applied identically to all three views.
4. **Spectral Reconstruction Similarity (SRS)**. For each block, cosine similarity over the 196 tokens is computed for the three view pairs — `(orig, low)`, `(orig, high)`, `(low, high)` — and summarized by its **mean and std** over the tokens. This yields `6 x 12 = 72` scalars `z_srs`.
5. **Spectral Context Vector (SCV)**. The projected original features are summarized per block by `[mean; std]` over tokens (`B x 12 x 2048`), projected to `B x 12 x 1024`, combined across blocks with a **learnable softmax importance map**, summed over blocks to `B x 1024`, and projected again. This is `z_src`.
6. **Classification**. `z = [z_srs; z_src]` (dimension `72 + 1024 = 1096`) is fed to a 3-layer MLP (`mlp_ratio = 3`, ReLU + dropout) producing a single real/fake logit. The AGID-side output is `stack([-logit, +logit], dim=1)`.

The detector uses label `0` for real images and label `1` for generated images.

## Why we vendor the official backbone

SPAI's backbone is a ViT-B/16 pre-trained with **Masked Frequency Modeling** (MFM); the checkpoint (`mfm_pretrain_vit_base.pth`) uses a BEiT-style layout (separate `q_bias` / `v_bias`, absolute position embeddings, no relative position bias). Stock `timm` ViT does **not** match these keys, so loading the weights into a `timm` model silently fails.

| Upstream file | Why we need it | What we do here |
|---|---|---|
| `spai/models/vision_transformer.py` | The exact ViT architecture the MFM checkpoint was trained with, plus its `return_features` path that exposes all intermediate blocks' patch tokens | Vendored **verbatim** as `vision_transformer.py`; instantiated with `return_features=True`, `intermediate_layers = range(12)` |
| `spai/models/filters.py` | The exact FFT high/low-frequency split and circular-mask convention | Vendored **verbatim** as `filters.py` |

The remaining logic (projectors, SRS, SCV, classifier) is re-implemented crisply in `spai.py`; it is **not** a copy of the upstream `sid.py` (which also carries any-resolution, ONNX export, and attention-export code we don't need).

## Equivalence with the official model

Sharing the same backbone weights and copying the same head weights into both models, the AGID `SPAI` and the official `MFViT` produce **bit-identical** logits in eval mode:

```text
mine     : [0.23217, 0.26906, 0.22517, 0.26105]
official : [0.23217, 0.26906, 0.22517, 0.26105]
max |diff| = 0.000e+00
```

So the fixed-resolution forward — frequency split, backbone, per-block projection, SRS, SCV, classifier — matches the official implementation exactly.

## Model details

The AGID model is implemented in `models/image/spai/spai.py` as `SPAI`.

| Component | Setting |
|---|---|
| Backbone | MFM-pretrained ViT-B/16, frozen |
| Backbone source | `mfm_pretrain_vit_base.pth` (loaded `strict=False`; the unused final `norm` is dropped since the model uses mean-pooling/intermediate features) |
| Features used | patch tokens of all 12 blocks, `B x 12 x 196 x 768` |
| Per-block projectors | `Projector` (LayerNorm + Linear + GELU + Linear + LayerNorm, dropout `0.5`), one per block, `768 -> 1024` |
| SRS | cosine mean/std over tokens for `(orig,low)`, `(orig,high)`, `(low,high)` -> `72` scalars |
| SCV | `SpectralContext`: per-block `[mean;std]` -> proj -> learnable softmax over blocks -> sum -> proj -> `1024` |
| Classifier | MLP `1096 -> 3288 -> 3288 -> 1` (`mlp_ratio = 3`), ReLU + dropout `0.5` |
| Frequency mask | circular low-pass, radius `mask_radius` (default `16`), input size `224` |
| Normalization | input expected in `[0, 1]`; ImageNet normalization applied **inside** `forward` to all three views |
| Output (AGID metrics) | two-class real / fake logits = `stack([-logit, +logit], dim=1)` |

Trainable parameters: the 12 per-block projectors, the spectral-context module, and the classifier head (~42M). The ViT backbone (~102M) stays frozen and runs under `torch.no_grad()`.

## Why this is considered an implementation of the paper

| Paper concept | AGID implementation |
|---|---|
| MFM-pretrained ViT-B/16 backbone | vendored `vision_transformer.VisionTransformer` + `mfm_pretrain_vit_base.pth` |
| Masking-radius frequency split (`r = 16`) | `filters.generate_circular_mask` + `filters.filter_image_frequencies` |
| Per-block, per-feature projection `P_n` | `nn.ModuleList` of `Projector` (one per intermediate block) |
| Spectral Reconstruction Similarity (Eq. SRS) | `_srs`: cosine mean/std over tokens for the three view pairs |
| Spectral Context Vector with learnable map `C` | `SpectralContext`: `softmax(alpha)`-weighted block aggregation of `[mean;std]` |
| 3-layer MLP classifier + BCE | `nn.Sequential(...)` head + `nn.BCEWithLogitsLoss` |

## Training setup

The SPAI configs live in:

```bash
conf/model/spai.yaml
conf/trainer/spai.yaml
```

The model config:

| Setting | Value |
|---|---:|
| `model_type` | `spai` |
| `backbone_name` | `ViT-B/16` |
| `input_size` | `224` |
| `mask_radius` | `16` |
| `projection_dim` | `1024` |
| `freeze_backbone` | `true` |

The trainer config follows the paper's regime: AdamW with `lr = 5e-4`, weight decay `0.05`, a cosine schedule with a ~5-epoch warmup, `35` epochs.

A launcher is provided at:

```bash
experiments/spai.sh
```

It uses the default dataset (`genimage`). **The transform lists end in `to_dtype` only and contain no `normalize_*` transform**, because SPAI normalizes internally; feeding pre-normalized images would corrupt the frequency split. Dataset-side augmentations are `resize_short_256, gaussian_blur, jpeg_compression, random_crop_224, random_horizontal_flip, to_dtype` (eval: `resize_short_256, center_crop_224, to_dtype`).

## Reproduced benchmark results

All benchmarks the paper reports, with our reproduced numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Metric | Paper | Ours | Status |
|---|---|---|---|---|
| SPAI main eval set (13 generators x 5 real sou | AUROC | auroc=91.0 | — | missing |
| GenImage (cross-dataset, AGID benchmark) | AP | — | 90.74 mAP (any-res) | reproduced |
| Synthbuster | AUROC | — | ~96.6 mAUROC (any-res) | reproduced |

## Integration notes

The SPAI model is called through the shared `compute_loss(batch)` interface used by the AGID trainer:

| Key | Meaning |
|---|---|
| `loss` | scalar training objective (BCE-with-logits on the single logit) |
| `outputs` | two-class real / fake logits for AGID metrics (`stack([-logit, +logit], dim=1)`) |

`SPAI` is registered in `trainer.py` under `model_type == "spai"`. The MFM checkpoint path is set at the top of `spai.py` (`MFM_CKPT`); update it if the weights live elsewhere.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| No any-resolution Spectral Context Attention (`PatchBasedMFViT`) | We implement only the fixed-resolution `MFViT` path; a single `224 x 224` crop is used instead of patch-based attention over arbitrary-resolution images. |
| No multi-view (`AUGMENTED_VIEWS`) training | The upstream training samples several augmented patches per image; AGID trains on one crop per image. |
| Only `vision_transformer.py` and `filters.py` are vendored | The rest of the pipeline is re-implemented crisply and verified bit-identical to upstream `MFViT`, so vendoring the full `sid.py` would just duplicate unused code (ONNX/attention export, any-resolution). |
| Uses Hydra configs, Accelerate, and AGID metrics | Keeps training, validation, testing, logging, and distributed support consistent with the rest of this project. |

These are scope/integration differences rather than conceptual changes: the detector still uses the frozen MFM backbone, the same frequency split, the same per-block projection, the same SRS statistics, the same learnable spectral-context aggregation, and the same BCE objective.

## Core novelties

### 1. Masked spectral learning backbone

The backbone is pre-trained only on real images to reconstruct masked frequencies, so it internalizes the spectral distribution of real images. Deviations from this learned distribution are what expose generated images.

### 2. Spectral Reconstruction Similarity (SRS)

Comparing the backbone's representations of an image, its low-pass version, and its high-pass version — via cosine similarity summarized across blocks — quantifies how spectrally self-consistent the image is. Real images stay coherent; generated images do not.

### 3. Spectral Context Vector (SCV)

A learnable softmax importance map weights the per-block `[mean; std]` statistics before summing them, letting the model decide which transformer blocks carry the most spectral evidence rather than fixing this a priori.

## Summary

SPAI in AGID is a method-level reproduction of Karageorgiou et al.'s CVPR 2025 paper, restricted to its fixed-resolution variant. A frozen MFM-pretrained ViT-B/16 supplies multi-block patch features for an image and its low/high-frequency components; Spectral Reconstruction Similarity measures their cross-view coherence, a learnable Spectral Context Vector summarizes the per-block statistics, and a small MLP reads the combined representation to produce a real/fake logit. The backbone (`vision_transformer.py`) and frequency split (`filters.py`) are vendored verbatim from the official repo so the MFM checkpoint loads exactly; the rest is re-implemented and verified numerically identical to the official `MFViT`. The any-resolution Spectral Context Attention is deliberately omitted.
