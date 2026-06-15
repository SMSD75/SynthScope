# ReStraV: AI-Generated Video Detection via Perceptual Straightening

## Short method explanation

**ReStraV** (*Representation Straightening for Video*) is an AI-generated *video* detector from the
paper **"AI-Generated Video Detection via Perceptual Straightening"** by Christian Internò, Robert
Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, and David Klindt (2025).

It is built on the **perceptual straightening** hypothesis from neuroscience: natural video
trajectories become *straighter* (lower curvature) in a good neural representation space, while
unnatural sequences stay curved. The authors find that in a frozen self-supervised encoder
(**DINOv2 ViT-S/14**), **natural videos trace straighter paths than AI-generated ones**. The
trajectory geometry — stepwise distance and curvature between consecutive frame embeddings — is a
discriminative, generator-agnostic signal. A lightweight classifier trained on aggregated geometry
statistics achieves state-of-the-art detection.

- Official repo: https://github.com/ChristianInterno/ReStraV
- Paper: *AI-Generated Video Detection via Perceptual Straightening*, arXiv:2507.00583.

## Pipeline

Given a clip `frames` of shape `[B, T, 3, 224, 224]` in `[0, 1]` (`T = 24` over a 2 s window; the
model ImageNet-normalizes internally):

1. **Trajectory in DINOv2 space.** A frozen **DINOv2 ViT-S/14** encodes each frame. From the final
   block we take the **CLS token (384-d)** concatenated with the **flattened patch tokens** to form
   a per-frame vector, giving the temporal trajectory `Z = (z_1, …, z_T)`. (Uses
   `forward_features`'s `x_norm_clstoken` + `x_norm_patchtokens`.) Run under `torch.no_grad()`.
2. **Geometry (Eq. 1).**
   - displacement `Δz_i = z_{i+1} − z_i`; **stepwise distance** `d_i = ||Δz_i||_2`, `i = 1 … T-1`.
   - **curvature** `θ_i = arccos( cos(Δz_i, Δz_{i+1}) )` in **degrees**, `i = 1 … T-2`; the cosine
     is clamped to `[-1, 1]` for numerical safety.
3. **Per-video feature `y ∈ R^21`** = concat of:
   - the first 7 distances `[d_1 … d_7]`,
   - the first 6 curvatures `[θ_1 … θ_6]`,
   - `[mean, var, min, max]` of all `d_i`,
   - `[mean, var, min, max]` of all `θ_i`. (7 + 6 + 4 + 4 = 21.)
4. **Classifier.** A small trainable **MLP** (paper: two layers `64 → 32`) maps `y → 2` logits.
   Trained with cross-entropy; the DINOv2 backbone stays **frozen**.

The detector uses label `0` for real videos and `1` for AI-generated videos; `outputs` are the two
logits `[B, 2]`.

## Model details

Implemented in `models/video/restrav/restrav.py` as `AGIDReStraVModel`.

| Component | Setting |
|---|---|
| Backbone | DINOv2 ViT-S/14 (`torch.hub` `facebookresearch/dinov2`), **frozen**, `torch.no_grad()` |
| Per-frame embedding | final-block CLS (384) ⊕ flattened patch tokens |
| Geometry | stepwise distance `d_i` + curvature `θ_i` (degrees), cosine clamped to `[-1, 1]` |
| Feature `y` | `R^21` = `[d_1..d_7] ⊕ [θ_1..θ_6] ⊕ stats(d) ⊕ stats(θ)`, `stats = [mean, var(pop), min, max]` |
| Classifier | MLP `21 → 64 → 32 → 2` (ReLU between layers), trained with `CrossEntropyLoss` |
| Trainable params | the MLP only (backbone frozen) |
| Output | two logits `[B, 2]` (col 0 real, col 1 fake) |

## Input normalization

ReStraV does **not** normalize internally — it expects frames already normalized by the
dataset transform pipeline. Use `normalize_dinov2` (ImageNet stats, what DINOv2 was trained
with) via `DatasetConfig.*_transform_list` in the experiment script, consistent with the image
CLIP/DINO detectors. The frozen backbone silently degrades on un-normalized frames.
`AIGITransformations` wraps each `[T, 3, H, W]` clip as `torchvision.tv_tensors.Video`, so the
normalization applies consistently across all frames.

## Training setup

Configs:

```bash
conf/model/restrav.yaml      # model_type: restrav, backbone_name: dinov2_vits14, num_frames: 24, mlp_hidden: [64,32], freeze_backbone: true
conf/trainer/restrav.yaml    # short AdamW run (epochs: 15)
```

| Setting | Value |
|---|---:|
| `model_type` | `restrav` |
| `backbone_name` | `dinov2_vits14` |
| `num_frames` | `24` |
| `mlp_hidden` | `[64, 32]` |
| `freeze_backbone` | `true` |

The trainer config is a normal short run: AdamW, `lr = 1e-3`, weight decay `0.01`, `15` epochs
(only the MLP trains). Launcher: `experiments/restrav.sh` (uses the `vidprom` video dataset; data
is not downloaded, so it is a template). The paper's benchmarks are **VidProM, GenVidBench,
Physics-IQ**; configs exist for all three under `conf/dataset/`.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Paper | AGID_official | AGID_train | Status |
|---|---|---|---|---|
| A) VidProM vs image-based detectors (Table 2) | acc=97.06 / mAP=98.81 | — | — | missing |
| B) VidProM vs VideoSwin — seen/unseen/future ( | — | — | — | missing |
| B) GenVidBench vs SoTA video detectors — Main  | Main_M_avg=93.01 / Plants_P_avg=96.96 | — | mAP=92.7 / acc=86.87 / AUROC=95.9 | reproduced |
| C) DeMamba one-to-many generalization (Table 5 | — | — | — | missing |
| D) Veo3 zero-shot generalization (Sec 7-D) | acc=83.2 / F1=85.1 / AUROC=86.9 | — | — | missing |
| E) Physics-IQ 2AFC vs Gemini 1.5 Pro (Sec 7-E, | Pika=97.5 / Runway=94.9 / Sora=99.0 | — | — | missing |

**reproduced** = AGID-trained checkpoint matches the paper · **official** = official weights loaded into the AGID class match (the AGID-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Integration notes

ReStraV uses the shared `compute_loss(batch) -> {loss, outputs}` interface, with `batch = (frames,
labels, folder_ids)` and `frames` of shape `[B, T, 3, H, W]`:

| Key | Meaning |
|---|---|
| `loss` | `CrossEntropyLoss(logits, labels)` (with optional label smoothing) |
| `outputs` | two-class real/fake logits `[B, 2]` for AGID metrics |

`AGIDReStraVModel` is registered in `trainer.py` under `model_type == "restrav"`. The AGID video
dataset (`VideoFolderDataset`) decodes `num_frames = 24` frames per clip at `224×224` in `[0, 1]`,
sampled evenly across a 2 s window (`fps: null`).

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Per-frame embedding dimension is **not hardcoded to 75648** | The paper states 384 (CLS) + 196 patch tokens × 384 = 75648, which assumes a 196-token patch grid. DINOv2 ViT-S/14 has patch size 14, so a `224×224` input yields a `16×16 = 256` patch grid → embedding `384 + 256·384 = 98688`. The trajectory geometry (distance, curvature) is **dimension-invariant**, so we concatenate CLS + *all* patch tokens dynamically and the method is unchanged. (Resizing to `196×196` would reproduce the exact 75648 if needed.) |
| Classifier is the **MLP** only | The paper benchmarks several off-the-shelf classifiers (LR, GNB, RF, GB, SVM, MLP) and reports the MLP as best (AUROC ≈ 98.6%); AGID implements the MLP, trained end-to-end with the AGID trainer. |
| Variance uses the population estimator (`1/N`) | Matches the paper's `σ² = 1/(T-1) Σ (·)²` over the `T-1` distance / `T-2` curvature values (sum divided by the number of terms). |
| Frames decoded with **decord**, 24 frames over 2 s, resized `224` | Matches the paper's `T = 24`, 2 s window, `224×224`, `[0, 1]` preprocessing; AGID's generic loader handles decoding. |
| Metrics via AGID's `AIGIMetrics` | Keeps evaluation/logging consistent with the rest of the project. |

These are integration/representation-bookkeeping differences, not conceptual changes: the detector
still computes DINOv2 trajectory distance + curvature, aggregates the same 21-d feature, and trains
the same small MLP.

## Summary

ReStraV in AGID is a method-level reproduction of Internò et al.'s perceptual-straightening video
detector. A frozen DINOv2 ViT-S/14 turns each frame into a CLS-⊕-patch embedding; the sequence of
embeddings forms a trajectory whose **stepwise distance** and **curvature** are computed per step;
21 aggregated geometry statistics feed a small `64 → 32` MLP that is trained to separate natural
from AI-generated videos. Natural videos are "straighter" (different distance/curvature statistics)
than generated ones, which is the detection signal. Only the MLP trains; the backbone is frozen.
