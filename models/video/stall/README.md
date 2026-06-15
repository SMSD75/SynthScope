# STALL: Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods

## Short method explanation

**STALL** is a **training-free, zero-shot** AI-generated *video* detector from **"Training-free
Detection of Generated Videos via Spatial-Temporal Likelihoods"** (Ben Hayun, Betser, Levi, Kassel,
Gilboa; CVPR 2026). It measures how *anomalous* a video's spatial and temporal statistics are relative
to a calibration set of **real** videos, under two isotropic-Gaussian likelihood branches. No training,
no fake videos, no labels.

- Official repo: https://github.com/OmerBenHayun/STALL
- Paper: arXiv:2603.15026

## Pipeline

Per-frame features come from a frozen **DINOv3 ViT-L/16** (D = 1024). For a clip `[B, T, 3, 224, 224]`:

1. **Spatial branch.** Whiten each frame embedding with a PCA-whitening transform fit on real videos:
   `z = (F − μ_s)·W_s`, then `LL_spat = N(0,I).logpdf(z) = −½(D·log2π + ‖z‖²)`. → `[B, T]`.
2. **Temporal branch.** `T_i = L2normalize(F_{i+1} − F_i)`; whiten with `(μ_t, W_t)`; same Gaussian LL.
   → `[B, T-1]`. Identical consecutive frames (zero diff) are excluded.
3. **Aggregate.** Spatial = **max** over frames; temporal = **min** over frame-pairs.
4. **Calibrate.** Convert each aggregate to a **percentile** against the real-video calibration LLs.
   `final = ½(pct_spat + pct_temp)` — **higher ⇒ more real**, so `P(fake) = 1 − final`.

Whitening = PCA decorrelation + scaling, so `‖z‖²` is the **Mahalanobis distance** to the real-feature
distribution; real frames/motions land near the mean (low distance, high likelihood), AI-generated ones
are anomalous (high distance, low likelihood).

## Model details

Implemented in `models/video/stall/stall.py` as `AGIDSTALLModel`.

| Component | Setting |
|---|---|
| Backbone | frozen **DINOv3 ViT-L/16** (`dinov3_vitl16`), D = 1024 |
| Calibration params | `μ_s, W_s, μ_t, W_t, calib_ll_*` — shipped `stall_params_vatex_dino_v3.npz` (fit on VATEX) |
| Aggregation | spatial `max`, temporal `min` |
| Score | `1 − ½(pct_spat + pct_temp)` (calibrated percentiles); `forward` returns `[real, fake]` |
| Trainable params | one dummy `_noop` scalar (training-free, like D3) |

## Calibration — bring your own real videos

The shipped params are computed on **VATEX**. To calibrate on your own real videos (matching your test
distribution), set `ModelConfig.recompute_params=true` and point `DatasetConfig` at a real-video set; the
model extracts features through its own DINOv3 backbone (`extract_embeddings`), fits the whitening +
calibration LLs (`fit_stall_params`, mirroring the official `create_params.build_params`), and saves to
`stall_params_path`. Knobs: `calib_split` (default `train`), `calib_real_only`, `calib_max_videos`.

## Input normalization

STALL does **not** normalize internally — frames are decoded square at `frame_size=224` and normalized
by the dataset transform pipeline with **ImageNet** statistics (`normalize_dinov2`). This matches the
official DINOv3 transform, which uses ImageNet mean/std — **not** the DINOv3-paper stats, so use
`normalize_dinov2`, *not* `normalize_dinov3`.

## Backbone weights (gated)

DINOv3 ViT-L/16 is gated. Place `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` and a local clone of
`github.com/facebookresearch/dinov3`, then set `DINO_V3_WEIGHTS` / `DINO_V3_REPO_DIR` (or
`ModelConfig.dino_weights` / `dino_repo`). The calibration npz ships with the model.

## Training setup

```bash
conf/model/video/stall.yaml     # model_type: stall, backbone, spat_agg/temp_agg, stall_params_path, recompute_params
conf/trainer/video/stall.yaml   # epochs: 1 (training-free, like D3)
experiments/video/stall.sh      # runs on the genvideo video dataset; set DINO_V3_WEIGHTS
```

Registered in `trainer.py` under `model_type == "stall"`. Threshold-free **AP/AUROC** are the
meaningful metrics (the percentile score is calibrated, but accuracy@0.5 depends on the calibration set).

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Paper | AGID_official | AGID_train | Status |
|---|---|---|---|---|
| GenVideo (DeMamba GenVideo-Val, 10 generators  | avg_AUC=0.8 / avg_AP=0.8 / per_gen_AUC_range=0.72-0.85 / per_gen_AP_range=0.75-0.84 | — | mAP=82.38 / acc=56.88 / AUROC=83.51 | reproduced |
| VideoFeedback (TIGER-Lab, 11 generators + DiDe | avg_AUC=0.83 / avg_AP=0.85 / per_gen_AUC_range=0.78-0.93 / per_gen_AP_range=0.80-0.94 | — | mAP=83.07 / acc=62.12 / AUROC=85.1 | reproduced |
| ComGenVid (paper's new benchmark — Sora + VEO3 | avg_AUC=0.852 / avg_AP=0.858 | — | mAP=82.59 / acc=57.34 / AUROC=84.0 | below |

**reproduced** = AGID-trained checkpoint matches the paper · **official** = official weights loaded into the AGID class match (the AGID-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Faithful re-implementation of the scoring math | Verified **bit-exact** against the official `STALL._scores_from_embs` on identical embeddings (0 difference). |
| `fit_stall_params` reimplements `create_params.build_params` | Lets the model recompute calibration in-framework; verified to give identical final scores to the official `WhiteningTransform`. |
| Decode via the AGID video dataset (decord) | The official uses torchcodec with a center 2 s window; AGID's loader samples evenly. Minor preprocessing detail; the calibration is recomputable to match whatever decode is used. |
| One dummy `_noop` parameter | Required so the AGID `AdamW` trainer does not crash on a training-free model. |
