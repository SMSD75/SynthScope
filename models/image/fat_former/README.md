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

Result: this implementation depends only on the already-installed `clip` package and `pytorch_wavelets`, both of which the SynthScope repo already requires. No vendored `clip.py` / `model.py` / `simple_tokenizer.py` / BPE file.

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

The SynthScope model is implemented in `models/image/fat_former/fat_former.py` as `SynthScopeFatFormerModel`.

| Component | Setting |
|---|---|
| Backbone | CLIP visual encoder + FAA |
| Default backbone | `CLIP:ViT-L/14` |
| Backbone training | frozen by default; only the injected FAA modules remain trainable inside the CLIP backbone |
| Trainable head modules | FAA layers, LGA (context vectors + patch enhancer + FFN), TGI (cross-attention + FFN), normalization layers |
| Number of FAA blocks | `num_vit_adapter` (config default `3` = the released checkpoint, blocks 7/15/23; the official argparse default is 8; must divide visual layers `24` for ViT-L/14) |
| Number of context tokens | `num_context_embedding` (default `8`) |
| Context initialization | random by default, or any text string via `init_context_embedding` |
| Output | two-class real / fake logits (`logits + aug_logits`) |

## Why this is considered an implementation of the paper

| Paper concept | SynthScope implementation |
|---|---|
| Frozen CLIP visual backbone with adapters | `clip.load(backbone_name)` + parameter freezing that keeps only `forgery_aware_adapter.*` trainable |
| Forgery-Aware Adapter on selected blocks | `ForgeryAwareAdapterLayer` attached to `num_vit_adapter` evenly spaced visual blocks |
| Wavelet-domain attention branch | Haar DWT + intra/inter-band MultiheadAttention + FFN + inverse DWT |
| Language-Guided Alignment | `LanguageGuidedAlignment` with learnable context vectors and a patch-based enhancer |
| Text-Guided Interactor | `MultiheadAttention(d_model, 12)` cross-attending patch tokens over text features, plus FFN |
| Two-branch real/fake logits | `forward` returns `logits + aug_logits` as two-class scores |
| Cross-entropy training | `nn.CrossEntropyLoss` with optional `label_smoothing` from `TrainerConfig` |

The implementation is therefore best understood as a **paper-faithful SynthScope reproduction** of FatFormer's architecture and objective. The only structural difference from the official code is the absence of a vendored CLIP fork — we obtain the same effective network by attaching FAAs to stock CLIP and exposing the patch-token sequence via a small re-implemented `_visual_forward`.

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
| `num_vit_adapter` | `3` |
| `num_context_embedding` | `8` |
| `init_context_embedding` | `""` |
| `num_classes` | `2` |
| `freeze_backbone` | `true` |
| `force_fp32` | `true` |

The trainer config (`conf/trainer/image/fat_former.yaml`) follows the paper's recipe (Sec. 4.1; the authors released no training code): Adam (AdamW with weight decay 0), lr `4e-4`, 25 epochs, total batch 256, lr x0.9 every 10 epochs (`lr_step` schedule, no warmup, no gradient clipping), no augmentation beyond resize 256 / random crop 224 / horizontal flip. Training data is the released 4-class ProGAN setting (car, cat, chair, horse; 144k images). `force_fp32=true` keeps the loaded CLIP backbone in fp32 (the upstream code keeps CLIP in fp16 on GPU).

The launcher is `experiments/image/fat_former.sh` (total batch 256 = GPUs x micro-batch x gradient accumulation; default 4 x 32 x 2 on 24 GB GPUs). It saves `model_last.pth` (taken as the released model) and `model_best.pth` (best validation accuracy, for reference).

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

Values are `mAP / acc`. **SynthScope_official** = the authors' released weights run through the SynthScope class / eval. **SynthScope_train** = the SynthScope class trained from scratch by us with the paper's recipe (below); the headline checkpoint is `model_last` (epoch 24), the final model, which is what the paper and the official code evaluate.

| Benchmark | Paper | SynthScope_official | SynthScope_train (model_last) | Status |
|---|---|---|---|---|
| GANs — 8 GANs (4-class ProGAN) | 99.7 / 98.4 | 99.7 / 98.39 | 99.56 / 97.29 | reproduced (mAP -0.14, acc -1.1) |
| GANs — 8 GANs (2-class ProGAN) | 99.2 / 95.8 | — | — | missing |
| Diffusion (DMs) — 10 subsets | 98.8 / 95.0 | 98.6 / 93.72 (8 UFD subsets) | 98.88 / 92.09 (8 UFD subsets) | official (the 8-subset mean is not the paper's 10-subset mean, see below) |
| UniversalFakeDetect-8 | — | 98.60 / 93.7 | 98.88 / 92.09 | official |
| DiffusionForensics-8 | — | 94.66 / 79.0 | — | official |

Checkpoint choice. `model_best` is chosen by validation accuracy with a strict `>`; validation accuracy was already 1.0000 after epoch 0, so `model_best` **is the epoch-0 checkpoint** and never changes (later epochs reach 1.0000 again but do not beat it). It is reported for reference only: GANs-8 99.66 / 95.71 and UFD-8 98.67 / 90.89 (mAP / acc). It has the same mAP as `model_last` but lower accuracy at the fixed 0.5 threshold (StyleGAN 87.03 acc), so training for the full 25 epochs helps calibration, not ranking.

Per-subset results for `model_last` (acc / AP): GANs-8 ProGAN 99.90/100.0, StyleGAN 94.73/99.72, StyleGAN2 94.08/98.79, BigGAN 99.45/99.93, CycleGAN 99.62/100.0, StarGAN 99.97/100.0, GauGAN 99.59/100.0, Deepfake 90.96/98.02 (paper: 99.9/97.2/98.8/99.5/99.3/99.8/99.4/93.2 acc). The gap to the paper is concentrated in StyleGAN (-2.5 acc), StyleGAN2 (-4.7) and Deepfake (-2.2).

DMs. The paper's DM table has 10 subsets (PNDM and VQ-Diffusion from DIRE, which we do not have); we evaluate the 8 UFD subsets. Averaging the paper's own per-subset numbers over these same 8 gives 98.54 AP / 93.80 acc, so on a like-for-like basis the trained model is +0.3 mAP and -1.7 acc (92.09 vs 93.80); the official weights give 98.60 / 93.72. The weakest subsets are Guided (76.10 acc, 93.82 AP; paper 76.1 / 92.0), LDM_200_cfg (91.65) and Glide_100-27 (91.40). The 10-subset mean cannot be formed without PNDM and VQ-Diffusion, so the row is not marked reproduced.

Training run and protocol. Recipe from the paper (Sec. 4.1; no official training code): 4-class ProGAN (car, cat, chair, horse; 144k), resize 256, random crop 224, flip, Adam lr 4e-4, weight decay 0, total batch 256 (1 GPU x micro 32 x accumulation 8), 25 epochs, lr x0.9 every 10 epochs, no gradient clipping; training job 277158 (RTX 3090, finished 2026-11-07). Evaluation: `benchmark/bench_eval.py --model fat_former --ckpt <model_last|model_best>.pth`, ForenSynths test with the 8 subsets (progan, stylegan, stylegan2, biggan, cyclegan, stargan, gaugan, deepfake) and the 8 UFD diffusion sets, the same command and data as the official-weights runs; Resize((256,256)) -> CenterCrop(224) -> ImageNet normalization. Both checkpoints load strictly into the served spec (num_vit_adapter=3: 719 tensors, 0 missing, 0 unexpected). Eval job 66626 ran on the DAS6 cluster (node409, RTX A6000) at commit a2a4d2b; tags `fat_former_trained_{last,best}_{forensynths,ufd}` in `repro/bench_results/` and `repro/MASTER_RESULTS.jsonl`. Checkpoints are not uploaded.

**reproduced** = SynthScope-trained checkpoint matches the paper · **official** = official weights loaded into the SynthScope class match (the SynthScope-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Integration notes

The FatFormer model is called through the shared `compute_loss(batch)` interface used by the SynthScope trainer:

| Key | Meaning |
|---|---|
| `loss` | scalar training objective (cross-entropy on the summed two-class logits) |
| `outputs` | two-class real/fake logits for SynthScope metrics |

For the trainer to instantiate FatFormer, `SynthScopeFatFormerModel` is registered in `trainer.py` **before** the generic CLIP branch. FatFormer uses a `CLIP:` backbone name, so if the generic CLIP branch came first it would accidentally instantiate `SynthScopeCLIPModel` instead. (This is the same lesson noted in `models/image/vib_net/README.md`.)

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| No vendored CLIP fork — FAA is attached to stock CLIP via `forward`-rebinding | The only real architectural change is FAA itself; the other upstream changes are cosmetic or trivially re-derivable, so vendoring CLIP would just duplicate code |
| `force_fp32` defaults to `true` | Stable training under Accelerate; upstream relies on fp16 on GPU |
| Single CE loss on the summed logits | Matches SynthScope's `compute_loss(batch) -> {loss, outputs}` API; the official code also classifies via the two-class output and uses CE |
| Uses Hydra configs and SynthScope metrics | Keeps training, validation, testing, logging, and distributed support consistent with the rest of this project |
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

FatFormer in SynthScope is a method-level reproduction of Liu et al.'s CVPR 2024 FatFormer paper. It uses frozen CLIP image features, attaches Forgery-Aware Adapters into selected visual blocks, learns class-conditional prompts with Language-Guided Alignment, and combines two real/fake logit branches via a Text-Guided Interactor. The implementation is adapted to SynthScope's model, config, trainer, and metrics structure, and is built directly on the standard `clip` package rather than vendoring a custom CLIP fork — so it is not a line-by-line copy of the official repo, but it implements the central architecture and objective that define the paper.
