# SPLIT (`split`)

**SPLIT: Training-Free AI-Generated and Partially Edited Video Detection via Spatial Patch-Level Incoherence and
Temporal Roughness** (Jongyeop Hyun, Hyounghun Kim; ECCV 2026) — [paper (arXiv 2607.02886)](https://arxiv.org/abs/2607.02886) ·
[official code](https://github.com/mldljyh/SPLIT)

A training-free video detector. Real motion is smooth in a frozen encoder's patch-feature space and moves coherently
across neighbouring patches; generated (or partially edited) video jitters from frame to frame and moves incoherently.
SPLIT measures both on patch tokens and multiplies them. The paper evaluates it where deployment needs it: fake recall
at a fixed, very low false-positive rate on real videos (0.1 / 1 / 5%), with the threshold calibrated on real videos only.

## Pipeline

```
video ─ 16 frames at 8 fps (8 for clips under 2 s), BGR, 10% crop of the longer side, resize 224, ImageNet norm
      → frozen X-CLIP B/16 vision tower → patch tokens P[t, n] (CLS dropped)
      → TTR  = mean over patches of log2(L1 / L2)          L1 = Σ‖P[t+1] − P[t]‖ (1-frame steps)
                                                           L2 = Σ‖P[t+2] − P[t]‖ / 2 · (T−1)/(T−2) (2-frame steps)
      → LSMI = mean ‖spatial difference of the motion field P[t+1] − P[t] on the patch grid‖ (x and y averaged)
      → score = TTR ** 8 · LSMI          (higher = more likely generated)
```

## SynthScope adaptations

| choice | reason |
|---|---|
| `SynthScopeSPLITModel` is SynthScope's own implementation of the paper's equations | The official repository has no license, so no code is copied. It returns the official `SPLIT_model.forward_score` on the same frames (relative difference ≤ 7e-6, job 269966). |
| Registry entry reuses D3's video preprocessing | SPLIT's released code reads frames exactly as D3 does (the same `video2frame` / `read_video`): 16 frames at 8 fps, BGR, 10% crop of the long side, resize 224, ImageNet normalization. |
| Frames are decoded directly (decord; PIL for GIFs), not written to JPEG by ffmpeg and read back | The official pipeline's lossy JPEG round-trip is the one remaining difference from the official code, and it changes the numbers a lot (see Results). |
| Clips longer than 3 s start at 0 | The official extraction picks a random start in a multiprocessing pool, so its frames are not reproducible anyway. |
| `split_calibration.json` (P(fake) for the web app) | The raw score spans many orders of magnitude, so it is calibrated on its log, by the paper's real-only rule (see below). |
| One dummy `_noop` parameter | Lets the SynthScope trainer run on a training-free model; it has no effect. |

## Results

Paper metric: **fake recall (%) at FPR 0.1 / 1 / 5%**, the threshold being the (1 − FPR) quantile of the real scores
(official `eval.py`). GenVideo-Val: its 10,000 MSR-VTT real videos calibrate the threshold, and all 8,584 fakes of the
10 generators are scored (the official reader drops 11 clips with fewer than 8 frames) (paper Table 18, "thresholds calibrated using the MSR-VTT real set only").

Two columns besides the paper: **official code, our data** runs the official frame extraction (ffmpeg JPEG frames),
reader, `SPLIT_model` and `evaluate_split` end to end on our copy of GenVideo-Val
(`repro/official_evals/split_official_eval.py`, job 270323); **SynthScope** is this implementation
(`benchmark/bench_eval_video.py -m split --dataset genvideo`, job 270384).

| GenVideo (fake recall @ FPR 0.1 / 1 / 5%) | Paper | Official code, our data | SynthScope |
|---|---|---|---|
| **Overall** | 82.13 / 91.69 / 97.20 | **82.22 / 90.17 / 96.89** | 98.87 / 99.86 / 99.98 |
| Crafter | 88.84 / 96.78 / 99.64 | 89.99 / 96.21 / 99.64 | 100 / 100 / 100 |
| Gen2 | 53.26 / 77.61 / 94.71 | 56.16 / 76.30 / 94.64 | 100 / 100 / 100 |
| HotShot | 99.86 / 100 / 100 | 99.86 / 100 / 100 | 100 / 100 / 100 |
| Lavie | 92.50 / 96.36 / 98.79 | 93.21 / 96.07 / 98.71 | 99.93 / 99.93 / 100 |
| ModelScope | 86.71 / 93.57 / 97.86 | 87.29 / 93.14 / 97.86 | 99.57 / 99.86 / 100 |
| MoonValley | 49.36 / 76.36 / 95.37 | 52.88 / 74.76 / 95.37 | 100 / 100 / 100 |
| MorphStudio | 94.71 / 98.43 / 99.71 | 95.00 / 98.00 / 99.71 | 100 / 100 / 100 |
| Show_1 | 96.86 / 99.86 / 100 | 97.86 / 99.71 / 100 | 100 / 100 / 100 |
| Sora | 87.50 / 98.21 / 100 | 91.07 / 98.21 / 100 | 100 / 100 / 100 |
| WildScrape | 71.66 / 79.76 / 85.89 | 73.27 / 79.74 / 86.42 | 89.94 / 98.92 / 99.78 |
| D3 Overall (the paper's baseline row) | 3.00 / 55.75 / 87.42 | — | 0.24 / 63.32 / 96.41 |

- **The official code reproduces the paper on our data**: Overall within 1.6 points at every FPR, every generator within
  0.6 at FPR 5%, and the larger 0.1% differences (Gen2, MoonValley, Sora with 56 fakes) are the random clip start the
  official extraction draws.
- **SynthScope scores higher because it skips the JPEG round-trip.** The official `video2frame.py` writes every frame
  with ffmpeg's default JPEG settings and reads it back; the compression noise differs from frame to frame, which adds
  temporal roughness to real videos and stretches the real-score tail that sets the threshold. On the same videos,
  decoded directly, SPLIT separates real and generated almost perfectly (AP 99.79, AUROC 99.97 averaged over generators).
  The **official code, our data** column is the like-for-like comparison with the paper.
- AP / AUROC / balanced accuracy at the calibrated threshold (SynthScope, mean over generators): 99.79 / 99.97 / 99.41.
  Per-generator numbers: `repro/bench_results/split_genvideo_fpr.json`, `split_official_genvideo.json`.
- Not run: FakeParts and ViF-Bench (the paper's other two benchmarks; not in SynthScope yet), and the cross-real
  threshold transfer to ROVI (needs FakeParts' real set).

## Probability calibration (`split_calibration.json`)

The raw score (TTR⁸ · LSMI) has a median of about 1e-10 on real GenVideo videos and about 1 on generated ones, so the engine maps
its logarithm to P(fake) = sigmoid((log score − threshold) / scale). Following the paper's real-only protocol, the
threshold is the log score at FPR 1% on 2,000 GenVideo-Val real videos (P(fake) = 0.5 there), and the scale puts the
median real video at P(fake) = 0.05 (`interface/scripts/calibrate_split.py`, through the engine's own preprocessing; on
1,000 random GenVideo fakes the recall at that threshold is 100%). A `split_calibration.json` in `$SYNTHSCOPE_CKPT_DIR`
overrides the shipped one. AP, AUROC and fake recall at fixed FPR do not depend on it.

## Usage

```bash
synthscope detect path/to/video.mp4 -m split
python benchmark/bench_eval_video.py -m split --dataset genvideo --tag split_genvideo_fpr
```

`bench_eval_video.py` reports fake recall at FPR 0.1 / 1 / 5% (per generator, their mean, and pooled over all
generators) for training-free detectors that expose a raw `score`.

## License

SynthScope's implementation is Apache-2.0. The official SPLIT repository has no license; no code from it is included.
The X-CLIP weights (`microsoft/xclip-base-patch16`) are MIT.
