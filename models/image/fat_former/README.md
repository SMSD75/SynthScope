# FatFormer: Forgery-Aware Adaptive Transformer for AI-Generated Image Detection

## Short method explanation

**FatFormer** is an AI-generated image detector based on the paper **"Forgery-Aware Adaptive Transformer for Generalizable Synthetic Image Detection"** by Liu et al., CVPR 2024.

The paper argues that frozen CLIP features carry strong, generic image understanding but are not specialized to the kinds of artifacts left by image generators. Two issues follow:

1. **Authentication-irrelevant features dominate**: CLIP was trained for image-text alignment, not real-vs-fake discrimination, so its CLS feature mixes content semantics with subtle forgery cues.
2. **Single-cue classifiers overfit to one generator family**: a head trained directly on CLIP CLS tends to latch onto generator-specific shortcuts and generalizes poorly.

FatFormer addresses both with a frozen CLIP backbone plus three trainable components:

- A **Forgery-Aware Adapter (FAA)** that adds a wavelet-domain, intra-band/inter-band self-attention branch into selected visual transformer blocks. This injects high-frequency, forgery-sensitive information into the residual stream.
- A **Language-Guided Alignment (LGA)** module that learns class-conditional prompt embeddings ("real" / "fake") and refines them with a patch-based enhancer cross-attended against the image's patch tokens.
- A **Text-Guided Interactor (TGI)** that cross-attends patch tokens against the LGA-produced class text features, producing an augmented image representation used for a second set of classification logits.

The two logit branches are summed for the final real/fake prediction.

Official repo: https://github.com/Michel-liu/FatFormer
Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.pdf

## Pipeline

Given an input image:

1. Encode the image with a CLIP visual backbone whose selected residual blocks have a Forgery-Aware Adapter attached. The encoder returns the **full** patch-token sequence (CLS + grid×grid patches), not just the CLS token.
2. Run the Language-Guided Alignment module: learnable context tokens are refined by cross-attention against the patch tokens, then assembled with `SOS` and class-name token embeddings into class-specific prompts.
3. Encode each class prompt with the (frozen) CLIP text encoder to obtain a text feature.
4. Compute `logits = cosine(CLS_image, text_features) * exp(logit_scale)`.
5. Run the Text-Guided Interactor: patch tokens cross-attend over the text features, the result is averaged into an augmented image feature, and a second set of logits `aug_logits` is computed against the same text features.
6. Return `logits + aug_logits` as two-class real / fake logits.

The detector uses label `0` for real images and label `1` for generated images.

## Why no vendored CLIP fork

The official repository ships a custom copy of CLIP (`models/clip/`) because the authors needed three things on top of stock OpenAI CLIP:

| Upstream change | Reason | What we do here |
|---|---|---|
| `Transformer.forward` returns `(per_layer_dict, x)` instead of `x` | Bookkeeping for debugging | Cosmetic — stock CLIP returns `x`, and the upstream code only consumes `x`. We use stock `Transformer` unchanged. |
| `VisionTransformer.forward(x, return_full=True)` | Expose all patch tokens | We provide a 10-line `_visual_forward(visual, x, return_full)` helper that re-runs stock CLIP's vision forward and exposes the full token sequence. No new weights involved. |
| `ForgeryAwareAdapterLayer` injected into selected blocks | **Core method change** | We attach an `ForgeryAwareAdapterLayer` to selected `ResidualAttentionBlock`s and rebind their `forward` to include the adapter in the residual path. |

Result: this implementation depends only on the already-installed `clip` package and `pytorch_wavelets`, both of which the AGID repo already requires. No vendored `clip.py` / `model.py` / `simple_tokenizer.py` / BPE file.

## Forgery-Aware Adapter (FAA)

The FAA module is added to a fixed subset of the visual transformer's residual blocks (default: every `layers / num_vit_adapter`-th block). For each adapted block, the forward becomes:

```text
x = x + Attention(LN1(x))
x = x + MLP(LN2(x)) + FAA(x)
```

Inside `FAA(x)`:

1. The patch tokens are reshaped to a 2D feature map.
2. A single-level Haar Discrete Wavelet Transform (DWT) produces one low-pass and three high-pass sub-bands.
3. The four sub-bands are normalized with GroupNorm and processed by an **intra-band** and an **inter-band** multi-head self-attention, plus an FFN.
4. An inverse DWT folds the sub-bands back to spatial features, which are concatenated with a zero CLS token.
5. In parallel, a tiny `Conv1d`-bottleneck branch processes the original token sequence.
6. The two branches are summed: `bottleneck_branch * adapter_scalar + wavelet_branch * freq_scale`, where `freq_scale` is a learnable scalar initialized to `0` so the adapter starts as a no-op.

This is implemented verbatim from the FatFormer source.

## Model details

The AGID model is implemented in `models/image/fat_former/fat_former.py` as `AGIDFatFormerModel`.

| Component | Setting |
|---|---|
| Backbone | CLIP visual encoder + FAA |
| Default backbone | `CLIP:ViT-L/14` |
| Backbone training | frozen by default; only the injected FAA modules remain trainable inside the CLIP backbone |
| Trainable head modules | FAA layers, LGA (context vectors + patch enhancer + FFN), TGI (cross-attention + FFN), normalization layers |
| Number of FAA blocks | `num_vit_adapter` (default `8`, must divide visual layers `24` for ViT-L/14) |
| Number of context tokens | `num_context_embedding` (default `8`) |
| Context initialization | random by default, or any text string via `init_context_embedding` |
| Output | two-class real / fake logits (`logits + aug_logits`) |

## Why this is considered an implementation of the paper

| Paper concept | AGID implementation |
|---|---|
| Frozen CLIP visual backbone with adapters | `clip.load(backbone_name)` + parameter freezing that keeps only `forgery_aware_adapter.*` trainable |
| Forgery-Aware Adapter on selected blocks | `ForgeryAwareAdapterLayer` attached to `num_vit_adapter` evenly spaced visual blocks |
| Wavelet-domain attention branch | Haar DWT + intra/inter-band MultiheadAttention + FFN + inverse DWT |
| Language-Guided Alignment | `LanguageGuidedAlignment` with learnable context vectors and a patch-based enhancer |
| Text-Guided Interactor | `MultiheadAttention(d_model, 12)` cross-attending patch tokens over text features, plus FFN |
| Two-branch real/fake logits | `forward` returns `logits + aug_logits` as two-class scores |
| Cross-entropy training | `nn.CrossEntropyLoss` with optional `label_smoothing` from `TrainerConfig` |

The implementation is therefore best understood as a **paper-faithful AGID reproduction** of FatFormer's architecture and objective. The only structural difference from the official code is the absence of a vendored CLIP fork — we obtain the same effective network by attaching FAAs to stock CLIP and exposing the patch-token sequence via a small re-implemented `_visual_forward`.

## Training setup

The FatFormer configs live in:

```bash
conf/model/fat_former.yaml
conf/trainer/fat_former.yaml
```

The model config:

| Setting | Value |
|---|---:|
| `model_type` | `fat_former` |
| `backbone_name` | `CLIP:ViT-L/14` |
| `num_vit_adapter` | `8` |
| `num_context_embedding` | `8` |
| `init_context_embedding` | `""` |
| `num_classes` | `2` |
| `freeze_backbone` | `true` |
| `force_fp32` | `true` |

The trainer config follows the paper's adapter-tuning setup: AdamW, lr `1e-4`, small weight decay, cosine schedule with short warmup, modest number of epochs. `force_fp32=true` makes the loaded CLIP backbone run in fp32 for stable Accelerate training (the upstream code keeps CLIP in fp16 on GPU). This can be disabled if you want to reproduce the upstream half-precision setup.

A launcher is provided at:

```bash
experiments/fat_former.sh
```

It uses the standard CLIP-style preprocessing pipeline (`resize_short_256`, `center_crop_224` / `random_crop_224`, `to_dtype`, `normalize_clip`), matching FatFormer's input format.

## Integration notes

The FatFormer model is called through the shared `compute_loss(batch)` interface used by the AGID trainer:

| Key | Meaning |
|---|---|
| `loss` | scalar training objective (cross-entropy on the summed two-class logits) |
| `outputs` | two-class real/fake logits for AGID metrics |

For the trainer to instantiate FatFormer, `AGIDFatFormerModel` is registered in `trainer.py` **before** the generic CLIP branch. FatFormer uses a `CLIP:` backbone name, so if the generic CLIP branch came first it would accidentally instantiate `AGIDCLIPModel` instead. (This is the same lesson noted in `models/image/vib_net/README.md`.)

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| No vendored CLIP fork — FAA is attached to stock CLIP via `forward`-rebinding | The only real architectural change is FAA itself; the other upstream changes are cosmetic or trivially re-derivable, so vendoring CLIP would just duplicate code |
| `force_fp32` defaults to `true` | Stable training under Accelerate; upstream relies on fp16 on GPU |
| Single CE loss on the summed logits | Matches AGID's `compute_loss(batch) -> {loss, outputs}` API; the official code also classifies via the two-class output and uses CE |
| Uses Hydra configs and AGID metrics | Keeps training, validation, testing, logging, and distributed support consistent with the rest of this project |
| Pretrained CLIP weights come from the standard `clip` cache (`~/.cache/clip`) | Avoids the hardcoded `pretrained/ViT-L-14.pt` path used by the upstream `clip.py` |

These are integration differences rather than conceptual changes: the detector still uses a frozen CLIP, FAA-injected residual blocks, LGA prompt learning, a text-guided interactor, and the same two-branch real/fake logits.

## Core novelties

### 1. Forgery-Aware Adapter (FAA)

The FAA brings explicit high-frequency, wavelet-domain processing into the residual stream of selected CLIP visual blocks. This injects forgery-sensitive cues that frozen CLIP features otherwise underuse.

### 2. Language-Guided Alignment (LGA)

Rather than handcrafting prompts, LGA learns class-conditional context tokens and refines them against the image's own patch tokens, so the text branch becomes image-aware.

### 3. Text-Guided Interactor (TGI)

After LGA produces class-specific text features, TGI lets the patch tokens attend over those text features. The averaged result feeds a second logits branch, so the final score combines both directions of image↔text grounding.

### 4. Adapter-tuning regime

CLIP stays frozen. Only FAA, LGA, TGI, and the small classifier-side norms / FFNs train. This keeps the parameter count and compute footprint small while still adapting the representation to the real-vs-fake task.

## Summary

FatFormer in AGID is a method-level reproduction of Liu et al.'s CVPR 2024 FatFormer paper. It uses frozen CLIP image features, attaches Forgery-Aware Adapters into selected visual blocks, learns class-conditional prompts with Language-Guided Alignment, and combines two real/fake logit branches via a Text-Guided Interactor. The implementation is adapted to AGID's model, config, trainer, and metrics structure, and is built directly on the standard `clip` package rather than vendoring a custom CLIP fork — so it is not a line-by-line copy of the official repo, but it implements the central architecture and objective that define the paper.
