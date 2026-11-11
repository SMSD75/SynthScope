# NSG-VD (`nsg_vd`, `nsg_vd_pika`, `nsg_vd_unbalance`)

**Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection** (Zhang, Lian, Yang, Li, Pang, Liu, Han, Li,
Tan; NeurIPS 2025 Spotlight) — [paper (arXiv 2510.08073)](https://arxiv.org/abs/2510.08073) ·
[official code](https://github.com/ZSHsh98/NSG-VD) (Apache-2.0)

A physics-motivated video detector. Under probability-flow conservation, a real video's density changes over time only
by moving probability mass around; the paper's statistic, the **Normalized Spatiotemporal Gradient** (NSG), is the ratio
of the spatial score (estimated by a pre-trained diffusion model) to the temporal change of the frames. Generated videos
violate the conservation and their NSG drifts. A video is scored by a deep-kernel **Maximum Mean Discrepancy** between
its NSG features and those of a fixed set of real reference videos: above 1 it is called generated.

Three released checkpoints are registered, the paper's main method (MMD-MP, Eq. 13) in its three tables:

| key | checkpoint (official `ckpts/`) | trained on | paper table |
|---|---|---|---|
| `nsg_vd` (default) | `standard-SEINE-mp.pth` (`nsg_vd.pth`) | Kinetics-400 + SEINE | Table 2 |
| `nsg_vd_pika` | `standard-Pika-mp.pth` (`nsg_vd_pika.pth`) | Kinetics-400 + Pika | Table 1 |
| `nsg_vd_unbalance` | `unbalance-SEINE-mp.pth` (`nsg_vd_unbalance.pth`) | Kinetics-400 + 1,000 SEINE | Table 3 |

**Status (details in Results below).** Exact parity with the official code at its own batching; the authors' released-CSV
protocol reproduces for all three keys (official code on our data and SynthScope); the paper's Tables 1-3 are NOT reproduced
(their ~120-per-generator subsets are unspecified); Sora's AUROC is below the released CSV for every checkpoint, cause unknown;
the ffmpeg build matters (pinned to 4.2). Not on the Space (see Availability); the reference file is MSR-VTT-derived (research use).

Which tables use which checkpoint (arXiv 2510.08073 v2): the paper's method trains the deep kernel with the
multi-population objective ("we adopt a multi-population aware optimization for the kernel training", Sec. 3.3, Eq. 13),
i.e. MMD-MP, and the captions of its three result tables name the training data: Table 1 "10,000 real and generated
videos from Kinetics-400 and Pika", Table 2 "... Kinetics-400 and SEINE", Table 3 "10,000 real and 1,000 generated
videos from Kinetics-400 and SEINE" (data-imbalanced). No table or ablation reports an MMD-D model. The released files
map onto them as above (by name; the repo does not say which file produced which table).

The default is `standard-SEINE-mp`: MMD-MP is the paper's method and the variant the authors' README recommends ("Use
MMD-MP for multiple generators or unbalanced data"), and SEINE-mp has the best AUROC of the three MMD-MP checkpoints in
the released results (`assets/nsg-vd-test_results`, average 97.68, against 96.80 Pika-mp and 96.95 unbalance-mp; its
accuracy, 88.97, is the lowest of the three). The three MMD-D checkpoints (`*-d.pth`) appear only in the released
results files, not in the paper, and are not registered (standard-SEINE-d has the best released numbers overall, 94.00
accuracy / 98.38 AUROC); the parity check below covers them too.

## Pipeline (official, reproduced exactly)

```
video ─ moviepy frame count n = int(fps * duration); k = max(1, n // 8)          (data/video_frames_dataset.py)
  → ffmpeg -vf select='not(mod(n\,k))',setpts=N/FRAME_RATE/TB -vframes 8 → frame1..8.jpg  (fewer than 8: video skipped)
  → each JPEG: PIL resize (224, 224) LANCZOS, /255                       (score_feature_dataset.py)
  → guided-diffusion 256x256 uncond (fp16), t = (5 + 1) / 1000: s = -eps(2x - 1, 999 t) / std(t), first 3 channels
  → NSG = s / (<s, central frame difference> + 1e-10) per frame; first and last frame x100     [8, 3, 224, 224]
  → TALL single Swin block (8 frames tiled 1x8, resized to 224) → 300-d feature
  → MMD_batch2 vs the 100 reference videos (deep kernel, trained sigma / sigma0 / epsilon); fake if > 1
```

The reference set is the official one: the first 100 ids of `split/real/MSR-VTT/val_ids.txt` (released
`assets/split.zip`; `test.yaml`: `ref_model: MSR-VTT`, `ref_mode: val`, `ref_load_len: 100`). Its NSG features
(100 x 8 x 3 x 224 x 224 float32, 482 MB) are precomputed once with `build_reference.py` and shipped as
`nsg_vd_reference_msrvtt_val100.safetensors`, the same precedent as STALL's calibration statistics. The reference
discriminator features depend on the checkpoint and are computed from the file at the first score.

Files in `$SYNTHSCOPE_CKPT_DIR`:

| file | what | source |
|---|---|---|
| `nsg_vd.pth`, `nsg_vd_pika.pth`, `nsg_vd_unbalance.pth` | the released checkpoints, unchanged (a `deep_MMD` state dict, 47 tensors) | official repo `ckpts/` |
| `256x256_diffusion_uncond.pt` | OpenAI guided-diffusion (MIT), 2.2 GB, md5 (base64) `/Z3SM1uHNtUh3grtVL2Qyg==` | openaipublic.blob.core.windows.net/diffusion/jul-2021/ |
| `nsg_vd_reference_msrvtt_val100.safetensors` | the reference NSG features | `build_reference.py` from the official split + MSR-VTT clips |

## SynthScope adaptations

| choice | reason |
|---|---|
| `SynthScopeNSGVDModel` holds the official `deep_MMD` as `mmd` (`registry.remap_nsg_vd` adds `mmd.`) and the diffusion UNet as `unet` (loaded in `__init__` from its own file, declared in `pretrained_prefixes`) | The checkpoints hold only the kernel parameters and the discriminator; strict load of all 47 tensors. |
| `official_tall.py`, `official_mmd.py`, `guided_diffusion/` vendored unchanged (only the needed parts; see headers); `official_nsg.py` ports `_get_score_features` / `_get_score_velocity` / `get_score_fn` line for line | `tests/test_nsg_vd_smoke.py` [D] checks them bit-identical against the official modules. |
| reference features shipped as a file, built by `build_reference.py` | The official test recomputes them from the 100 videos on every run; a deterministic port needs them fixed. |
| `media.ffmpeg_frame_count` re-implements moviepy's frame count; `media.ffmpeg_select_frames` runs the official ffmpeg command | SynthScope does not depend on moviepy; both checked against the official code (counts and decoded frames identical). |
| a video that gives fewer than 8 frames is a `VideoDecodeError` | The official dataset silently drops it. |
| the UNet's convolutions are converted to fp16 before each diffusion pass | The official `convert_to_fp16`; the engine casts models to fp32 after loading. On a CPU the UNet runs in fp32 (`cpu_fp32`), a deviation. |
| P(fake) = sigmoid((MMD - 1) / `prob_scale`), `prob_scale` 0.1; verdict = MMD > 1 | The official output is the raw MMD thresholded at 1; the probability is for display (0.5 at the threshold). The verdict follows the official rule exactly (`raw_predict > 1`: a score of exactly 1 is real, although its P(fake) is 0.5). The engine reports the raw statistic as `logit`. |
| ffmpeg pinned per detector: `ffmpeg_version="4.2"`, binary from `$SYNTHSCOPE_FFMPEG_NSG_VD` (else the global `$SYNTHSCOPE_FFMPEG` / PATH / imageio-ffmpeg) | The official code uses whatever ffmpeg is installed, and the build changes the frames (below). Each result records the ffmpeg path and version used (`ffmpeg`) and carries a `warning` (also logged, once per binary) when it is not 4.2.x; `bench_eval_video.py`'s cache identity includes the ffmpeg build and the sha256 of the diffusion model and the reference file. Other detectors are unaffected. |
| no training (`compute_loss` raises) | Use the official `train_dMMD.py`. |

## Fidelity check

The unmodified official code (`ZSHsh98/NSG-VD` @ `9a60f57`: `get_ref_dataloader` + `get_ref_features`,
`get_score_datasets`, the body of `test_dMMD.test_dMMD`) against SynthScope's full pipeline (raw `.mp4` ->
`media.ffmpeg_select_frames` -> `nsgvd_frame_224` -> engine-built model, strict load -> MMD against the shipped
reference file), all six released checkpoints, on 75 GenVideo-Val test videos in the official test order (15 MSR-VTT
reals from index 200 of the test ids, 15 ModelScope and 5 each of MorphStudio, MoonValley, HotShot, Show-1, Gen2,
Crafter, LaVie, Sora, WildScrape), RTX A5000, same environment (torch 2.11), same ffmpeg (static 4.2.2) on both
sides. Record: `repro/nsg_vd_official_parity_genvideo75.json` (per-video scores; jobs 286253 official, 286389 SynthScope).

| check | result |
|---|---|
| state dict | all 47 tensors of each of the 6 checkpoints load strictly (0 missing / 0 unexpected) and equal the file |
| reference set | the shipped file (`build_reference.py`) is bit-identical to the official `ref_data` (100 x 8 x 3 x 224 x 224, max abs diff 0.0), same 100 ids in the same order (none dropped) |
| frames | 75/75 videos: the 8 decoded frames identical to the official `video_frames/` JPEGs |
| frame count | `media.ffmpeg_frame_count` == moviepy 2.2.1 `int(fps * duration)` on all 18,228 val + test videos of the split (0 mismatches; job 286254, `repro/nsg_vd_framecount_genvideo_val.json`) |
| NSG features, official grouping (3 videos per diffusion pass, the official order) | bit-identical on 75/75 videos |
| scores, official grouping + the official MMD batches | **bit-identical (max abs diff 0.0) for all 6 checkpoints** |
| scores, the served engine path (1 video per pass, 1 per MMD batch) | max abs diff 0.038 / 0.024 / 0.022 (Pika-mp / SEINE-mp / unbalance-mp), 0.015 / 0.002 / 0.008 (the -d ones); official verdict (> 1) reproduced on 75/75 for every checkpoint |
| the official code against itself: its own NSG features scored one at a time, or in reshuffled batches (`test_dMMD` uses `shuffle=True`) | max spread 0.037 / 0.035 / 0.022 (mp), 0.0001 / 0.0015 / 0.005 (d) |

So the port is exact; the engine's per-video score differs from a given official run only through fp16 / batch numerics
(which videos share a diffusion pass and an MMD batch), by at most 0.038, about as much as the official test itself
varies with the batch a video lands in for the MMD-MP checkpoints (for the -d ones the diffusion batch dominates). The statistic is sensitive to them because
the trained kernel width `sigma0` is tiny (5e-7 to 1e-6 for the MMD-MP checkpoints).

**Frame extraction (ffmpeg build).** The official code shells out to whatever `ffmpeg` is installed; its version is not
pinned, and it matters. On the same 75 videos (frames: `repro/nsg_vd_ffmpeg_builds_genvideo75.json`, job 286412;
scores: `alt_ffmpeg` in `repro/nsg_vd_official_parity_genvideo75.json`, job 286389):

| ffmpeg | videos giving 8 frames | frames vs 4.2.2 | score vs the official run with 4.2.2 (SEINE-mp / Pika-mp / unbalance-mp) |
|---|---|---|---|
| 4.2.2 (static, imageio-ffmpeg 0.4.2) — used for parity | 75/75 | — | (exact, above) |
| 9.0.2 (static, imageio-ffmpeg 0.8.1) | 75/75 | same frames, re-encoded: model inputs 0.83 grey levels apart on average (max 1.16 per video), none identical | mean abs diff 0.19 / 0.07 / 0.04, max 0.79 / 0.83 / 0.44; verdicts agree 70 / 70 / 73 of 75 |
| 7.0.2 (static, the repo's VINA / D3 build) | 68/75: 6 frames on all 5 MoonValley clips, 7 on 2 of 5 Sora clips | on the 68: 7 identical, the others select different frames (25.3 grey levels apart on average, max 114.5) | mean abs diff 0.13 / 0.17 / 0.12 on the 68; verdicts agree 59 / 57 / 59 |

The frame counts (`n`, hence the step `k`) are the same under all three builds; 9.0.2 changes only the JPEG encoding,
which the NSG amplifies; 7.0.x drops and shifts the selected frames and is unusable for this protocol. The registry pins
4.2 and warns otherwise; which build the authors used is unknown.

## Results

**Verdict. The authors' released-CSV protocol REPRODUCES for all three registered keys, both with the unmodified official
code on our data and with SynthScope: on every generator, for every key and both paths, accuracy, recall, F1 and
precision are within 0.43 points of the released CSVs, and the 10-generator average accuracy within 0.25.
The paper's Tables 1-3 are NOT reproduced: the roughly 120-per-generator subsets they were computed on are not specified
and cannot be reconstructed (below), and even the authors' own released CSVs differ from those tables. One discrepancy is
unexplained: Sora's AUROC is below the CSV for every checkpoint.**

Metrics are the official ones (`test_dMMD.py`, recomputed here from per-video statistics by
`repro/nsg_vd_bench_compare.py`): the positive class is fake; a video is fake iff the MMD statistic is greater than 1;
recall, precision and F1 are on the fake class; **accuracy is pooled over real + fake** (it is not the balanced
accuracy that `bench_eval_video`'s own JSONs report: the two differ for Sora, which has 56 reals and 51 fakes); AUROC is
computed on the raw statistic; "Avg." is the mean over the 10 generators. Every number below is in
`repro/nsg_vd_benchmark_genvideo_val.json` (per video: `..._per_video.json`).

### The protocols

| name | what | used for |
|---|---|---|
| released-CSV protocol ("L500") | `test.yaml` as released: the first 500 `.mp4` test ids per generator; the reals are 500 MSR-VTT test videos, from index 200 of the released `split.zip` test ids (the first 200 are not used); **Sora: 51 fakes** (the whole released Sora test split) and **56 reals** (`test_dMMD.py` hard-codes load_len 56 for Sora) | the authors' released result CSVs (`assets/nsg-vd-test_results`) reproduce on exactly these counts (98.04 = 50/51, 86.92 = 93/107) |
| paper Tables 1-3 ("~120 per generator") | NSG-VD rows: the percentages imply 120 reals + 120 fakes per generator (68.33 = 82/120, 81.67 = 196/240) and 56 + 56 for Sora (94.64 = 53/56); the baselines in the same tables use 500 per generator (DeMamba HotShot recall 40.60 = 203/500) | the paper's numbers |
| L120 (our approximation) | the first 120 of each official list (Sora 51 + 56) | **a labelled approximation of the paper's subsets, not the paper's protocol** |

The paper's subsets cannot be reconstructed: Appendix C gives no counts and no selection rule, `test.yaml` uses 500, the
released val splits do not have 120 ids either (62 to 140 per generator), the paper's Sora column has 56 fakes against
51 in the released test split, and the checkpoint path in `test.yaml` is dated 2025-07-01. The released CSVs themselves differ from the paper's
tables (CSV minus paper, 10-generator average, the largest metric per key: SEINE-mp accuracy +2.92; Pika-mp recall
+7.84; unbalance-SEINE-mp recall +5.23). So the released checkpoints with the released protocol give the CSV numbers,
not the paper's; the paper's Table 5 reports a 3-seed spread of up to 3.07 AUROC (SEINE, ModelScope), so its rows are probably
other training runs (not verifiable).

### Released-CSV protocol: the default key `nsg_vd` (standard-SEINE-mp), per generator

"official code, our data" = the unmodified `test_dMMD.main` (Hydra entry point, the README's command, seed 42, shuffled
24-video batches, 3 videos per diffusion pass) on our copy of GenVideo-Val; "SynthScope" = `benchmark/bench_eval_video.py`
(the served path: raw file, one video per pass); the paper row is Table 2 with its unspecified ~120-per-generator subsets
and is shown for scale only. The ModelScope .. WildScrape columns use 500 + 500 videos, Sora 56 reals + 51 fakes.

**Accuracy (pooled over real + fake)**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper Table 2 (~120 per generator, subset unspecified) | 82.50 | 88.33 | 89.58 | 84.58 | 86.25 | 87.08 | 86.67 | 87.92 | 89.29 | 78.33 | 86.05 |
| released CSV (authors) | 88.40 | 89.70 | 89.90 | 89.90 | 89.90 | 89.20 | 89.80 | 89.40 | 86.92 | 86.60 | 88.97 |
| official code, our data | 88.70 | 89.90 | 90.20 | 90.20 | 90.20 | 89.50 | 90.10 | 89.70 | 86.92 | 86.80 | 89.22 |
| SynthScope | 88.40 | 89.70 | 89.90 | 89.90 | 89.90 | 89.20 | 89.80 | 89.40 | 86.92 | 86.50 | 88.96 |

**Recall (fake class)**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper Table 2 (~120 per generator, subset unspecified) | 91.67 | 100.00 | 100.00 | 100.00 | 100.00 | 98.33 | 100.00 | 97.50 | 94.64 | 89.17 | 97.13 |
| released CSV (authors) | 97.00 | 99.60 | 100.00 | 100.00 | 100.00 | 98.60 | 99.80 | 99.00 | 98.04 | 93.40 | 98.54 |
| official code, our data | 97.00 | 99.40 | 100.00 | 100.00 | 100.00 | 98.60 | 99.80 | 99.00 | 98.04 | 93.20 | 98.50 |
| SynthScope | 97.00 | 99.60 | 100.00 | 100.00 | 100.00 | 98.60 | 99.80 | 99.00 | 98.04 | 93.20 | 98.52 |

**F1 (fake class)**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper Table 2 (~120 per generator, subset unspecified) | 83.97 | 89.55 | 90.57 | 86.64 | 87.91 | 88.39 | 88.24 | 88.97 | 89.83 | 80.45 | 87.45 |
| released CSV (authors) | 89.32 | 90.63 | 90.83 | 90.83 | 90.83 | 90.13 | 90.73 | 90.33 | 87.72 | 87.45 | 89.88 |
| official code, our data | 89.57 | 90.78 | 91.07 | 91.07 | 91.07 | 90.38 | 90.98 | 90.58 | 87.72 | 87.59 | 90.08 |
| SynthScope | 89.32 | 90.63 | 90.83 | 90.83 | 90.83 | 90.13 | 90.73 | 90.33 | 87.72 | 87.35 | 89.87 |

**AUROC**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper Table 2 (~120 per generator, subset unspecified) | 90.67 | 97.62 | 98.38 | 95.88 | 96.69 | 97.87 | 97.64 | 95.09 | 96.14 | 88.65 | 95.46 |
| released CSV (authors) | 96.16 | 98.34 | 99.73 | 99.22 | 98.34 | 97.76 | 99.13 | 96.81 | 96.92 | 94.41 | 97.68 |
| official code, our data | 96.04 | 98.26 | 99.71 | 99.16 | 98.20 | 97.66 | 99.08 | 96.68 | 96.04 | 94.30 | 97.51 |
| SynthScope | 95.99 | 98.26 | 99.73 | 99.18 | 98.19 | 97.64 | 99.08 | 96.65 | 96.08 | 94.25 | 97.50 |

**Precision (fake class)**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| released CSV (authors) | 82.76 | 83.14 | 83.19 | 83.19 | 83.19 | 83.00 | 83.17 | 83.05 | 79.37 | 82.22 | 82.63 |
| official code, our data | 83.19 | 83.53 | 83.61 | 83.61 | 83.61 | 83.42 | 83.58 | 83.47 | 79.37 | 82.62 | 83.00 |
| SynthScope | 82.76 | 83.14 | 83.19 | 83.19 | 83.19 | 83.00 | 83.17 | 83.05 | 79.37 | 82.19 | 82.63 |

Per-generator tables for `nsg_vd_pika`, `nsg_vd_unbalance` and the three unregistered MMD-D checkpoints are in the JSON
(`checkpoints.<name>.L500`).

### Released-CSV protocol: averages for all three registered keys

10-generator averages. The paper rows are the Avg. column of Tables 1-3 (~120 per generator, unspecified subsets; precision
is not printed).

| key = checkpoint | source | accuracy | recall | F1 | AUROC | precision |
|---|---|---:|---:|---:|---:|---:|
| `nsg_vd` = standard-SEINE-mp | paper Table 2 (~120/gen) | 86.05 | 97.13 | 87.45 | 95.46 | - |
|  | released CSV | 88.97 | 98.54 | 89.88 | 97.68 | 82.63 |
|  | official code, our data | 89.22 | 98.50 | 90.08 | 97.51 | 83.00 |
|  | SynthScope | 88.96 | 98.52 | 89.87 | 97.50 | 82.63 |
| `nsg_vd_pika` = standard-Pika-mp | paper Table 1 (~120/gen) | 91.46 | 88.02 | 90.87 | 96.14 | - |
|  | released CSV | 89.34 | 95.86 | 89.90 | 96.80 | 84.73 |
|  | official code, our data | 89.42 | 95.84 | 89.97 | 96.69 | 84.87 |
|  | SynthScope | 89.15 | 95.84 | 89.74 | 96.75 | 84.46 |
| `nsg_vd_unbalance` = unbalance-SEINE-mp | paper Table 3 (~120/gen) | 89.16 | 93.21 | 89.48 | 94.91 | - |
|  | released CSV | 90.37 | 98.44 | 91.04 | 96.95 | 84.68 |
|  | official code, our data | 90.37 | 98.44 | 91.04 | 96.83 | 84.68 |
|  | SynthScope | 90.47 | 98.46 | 91.12 | 96.90 | 84.82 |

| key = checkpoint | path | max per-generator gap to the CSV: accuracy / recall / F1 / precision | generators with identical accuracy | Avg. accuracy gap | Avg. AUROC gap |
|---|---|---|---|---|---|
| `nsg_vd` = standard-SEINE-mp | SynthScope | 0.10 / 0.20 / 0.10 / 0.03 | 9 of 10 | -0.01 | -0.18 |
| | official code, our data | 0.30 / 0.20 / 0.25 / 0.43 | 1 of 10 | +0.25 | -0.17 |
| `nsg_vd_pika` = standard-Pika-mp | SynthScope | 0.30 / 0.20 / 0.28 / 0.34 | 1 of 10 | -0.19 | -0.05 |
| | official code, our data | 0.30 / 0.40 / 0.28 / 0.36 | 3 of 10 | +0.08 | -0.11 |
| `nsg_vd_unbalance` = unbalance-SEINE-mp | SynthScope | 0.30 / 0.40 / 0.29 / 0.21 | 2 of 10 | +0.10 | -0.05 |
| | official code, our data | 0.10 / 0.20 / 0.11 / 0.02 | 8 of 10 | 0.00 | -0.12 |

Neither path is uniformly closer to the CSV: SynthScope matches it on 9 of 10 generators for SEINE-mp, the official
code on 8 of 10 for unbalance-SEINE-mp. AUROC agrees to within 0.17 on every generator except Sora (below).

### Sora AUROC: below the CSV for every checkpoint (UNEXPLAINED)

Sora's recall and accuracy equal the CSV (the same 56 + 51 videos), but its AUROC does not:

| key = checkpoint | released CSV | official code, our data | SynthScope |
|---|---:|---:|---:|
| `nsg_vd` | 96.92 | 96.04 (-0.88) | 96.08 (-0.84) |
| `nsg_vd_pika` | 96.99 | 96.53 (-0.46) | 96.67 (-0.32) |
| `nsg_vd_unbalance` | 95.87 | 95.03 (-0.84) | 95.50 (-0.37) |
| standard-SEINE-d (not registered) | 97.72 | 96.85 (-0.87) | - |
| standard-Pika-d (not registered) | 96.39 | 95.55 (-0.84) | - |
| unbalance-SEINE-d (not registered) | 94.19 | 93.35 (-0.84) | - |

The gap is 0.84 to 0.88 below the CSV for five of the six checkpoints on the official path, and 0.46 for standard-Pika-mp;
SynthScope's gaps are 0.84 (SEINE-mp), 0.32 (Pika-mp) and 0.37 (unbalance-SEINE-mp). **We do not know the cause.** Candidates,
none verified: the authors' ffmpeg build (with 9.0.2 instead of 4.2.2 the SEINE-mp Sora AUROC is 97.72, above the CSV's 96.92,
against 96.08 with 4.2.2); a different copy or encode of the Sora files (they are long, high-resolution clips, and
GenVideo has several copies); a different reference or score cache state. Counter-evidence against a global build effect: under the
pinned 4.2.2 the other nine generators agree with the CSV's AUROC to within 0.17, and with 9.0.2 ModelScope moves away from the CSV, not towards it (AUROC 96.39
against 95.99 with 4.2.2 and 96.16 in the CSV; recall 97.2 against 97.0 in both the CSV and the 4.2.2 run). A
Sora-specific data or encoding difference is at least as plausible as a build difference.

### Paper Tables 1-3: NOT reproduced; L120 approximation

L120 is our approximation of the paper's ~120-per-generator subsets (the first 120 of each official list; Sora 51 + 56).
It does not reproduce the tables either: the worst cells are far off (recall on ModelScope 91.67 in Table 2 against 99.17;
Table 1 recall on ModelScope 68.33 against 90.00; Table 3 recall on Sora 82.14 against 100.00), and its differences from the paper are of the same order as those of the released CSVs. Averages
(10 generators):

| key | source | accuracy | recall | F1 | AUROC | precision |
|---|---|---:|---:|---:|---:|---:|
| `nsg_vd` | paper Table 2 | 86.05 | 97.13 | 87.45 | 95.46 | - |
|  | official code, L120 | 86.90 | 98.72 | 88.23 | 97.16 | 79.76 |
|  | SynthScope, L120 | 86.52 | 98.72 | 87.93 | 97.18 | 79.28 |
| `nsg_vd_pika` | paper Table 1 | 91.46 | 88.02 | 90.87 | 96.14 | - |
|  | official code, L120 | 89.71 | 96.67 | 90.31 | 96.88 | 84.80 |
|  | SynthScope, L120 | 89.75 | 96.75 | 90.35 | 97.01 | 84.81 |
| `nsg_vd_unbalance` | paper Table 3 | 89.16 | 93.21 | 89.48 | 94.91 | - |
|  | official code, L120 | 91.27 | 98.67 | 91.82 | 96.26 | 85.89 |
|  | SynthScope, L120 | 91.23 | 98.58 | 91.78 | 96.39 | 85.88 |

L120 minus paper for the default key (average, official code / SynthScope): accuracy +0.85 / +0.47, recall +1.59 / +1.59,
F1 +0.78 / +0.48, AUROC +1.70 / +1.72. Per generator, `nsg_vd` (paper Table 2 against the approximation):

**Accuracy (pooled over real + fake)**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper Table 2 | 82.50 | 88.33 | 89.58 | 84.58 | 86.25 | 87.08 | 86.67 | 87.92 | 89.29 | 78.33 | 86.05 |
| official code, L120 approximation | 87.08 | 87.50 | 87.50 | 87.50 | 87.50 | 86.25 | 87.50 | 87.08 | 86.92 | 84.17 | 86.90 |
| SynthScope, L120 approximation | 86.67 | 87.08 | 87.08 | 87.08 | 87.08 | 85.83 | 87.08 | 86.67 | 86.92 | 83.75 | 86.52 |
| released CSV (L500, for scale) | 88.40 | 89.70 | 89.90 | 89.90 | 89.90 | 89.20 | 89.80 | 89.40 | 86.92 | 86.60 | 88.97 |

**Recall (fake class)**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper Table 2 | 91.67 | 100.00 | 100.00 | 100.00 | 100.00 | 98.33 | 100.00 | 97.50 | 94.64 | 89.17 | 97.13 |
| official code, L120 approximation | 99.17 | 100.00 | 100.00 | 100.00 | 100.00 | 97.50 | 100.00 | 99.17 | 98.04 | 93.33 | 98.72 |
| SynthScope, L120 approximation | 99.17 | 100.00 | 100.00 | 100.00 | 100.00 | 97.50 | 100.00 | 99.17 | 98.04 | 93.33 | 98.72 |
| released CSV (L500, for scale) | 97.00 | 99.60 | 100.00 | 100.00 | 100.00 | 98.60 | 99.80 | 99.00 | 98.04 | 93.40 | 98.54 |

**AUROC**

| source | ModelScope | MorphStudio | MoonValley | HotShot | Show-1 | Gen2 | Crafter | LaVie | Sora | WildScrape | **Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper Table 2 | 90.67 | 97.62 | 98.38 | 95.88 | 96.69 | 97.87 | 97.64 | 95.09 | 96.14 | 88.65 | 95.46 |
| official code, L120 approximation | 96.42 | 98.35 | 99.78 | 98.93 | 97.84 | 96.29 | 98.74 | 96.10 | 96.04 | 93.06 | 97.16 |
| SynthScope, L120 approximation | 96.44 | 98.40 | 99.79 | 98.98 | 97.88 | 96.28 | 98.82 | 96.11 | 96.08 | 93.01 | 97.18 |
| released CSV (L500, for scale) | 96.16 | 98.34 | 99.73 | 99.22 | 98.34 | 97.76 | 99.13 | 96.81 | 96.92 | 94.41 | 97.68 |


### The served path against the official batching

The port is bit-identical to the official code at the official grouping (Fidelity check above). SynthScope's served path
scores one video per diffusion pass and one per MMD batch, while the official test runs 3 videos per fp16 UNet pass
and shuffled batches of 24 for the net and the MMD, so a video's statistic differs slightly between the two. Over the 9,107
(generator, video) pairs of the released-CSV protocol, per checkpoint:

| checkpoint | per-video max abs diff | mean abs diff | verdicts that differ | Avg. metric difference, SynthScope minus official |
|---|---:|---:|---:|---|
| standard-SEINE-mp | 0.0964 | 0.0066 | 28 of 9,107 | accuracy -0.26, recall +0.02, F1 -0.21, AUROC -0.01, precision -0.37 |
| standard-Pika-mp | 0.2671 | 0.0078 | 33 of 9,107 | accuracy -0.27, recall 0.00, F1 -0.23, AUROC +0.06, precision -0.41 |
| unbalance-SEINE-mp | 0.0693 | 0.0041 | 32 of 9,107 | accuracy +0.10, recall +0.02, F1 +0.08, AUROC +0.07, precision +0.14 |

* **The largest differences are near the kernel's ceiling.** The 0.267 for Pika-mp is WildScrape video D87: official
  1.146, SynthScope 1.413, which is the highest statistic any video reached in these runs (the ceiling); it is fake on both paths. The largest
  gap of any pair is on that same video for all three checkpoints (SEINE-mp 1.276 against 1.372, unbalance 1.220 against 1.290).
* **Near the threshold the gap is small.** Where both paths score within 0.1 of s = 1, the largest difference is 0.027 (SEINE-mp),
  0.029 (Pika-mp) and 0.017 (unbalance-SEINE-mp); within 0.2 of 1 it is 0.041 / 0.067 / 0.028. Every verdict flip is within
  0.017 of the threshold. About three videos flip per generator, mostly the same few MSR-VTT reals, which is why every generator's accuracy moves by
  about 0.3 together.
* **Cause: the diffusion step's batch grouping** (3 videos per pass in the official code, 1 in the engine), by elimination: the
  official code against itself, with the same cached NSG features but other shuffles (seeds 1 and 2), moves per-video statistics by at most
  0.041 / 0.061 / 0.030 with 0 / 2-3 / 0 flips; a seed-42 rerun is bit-identical (max 0.0), and re-batching the same
  videos (L500 against L120) moves at most 0.033 with 0-1 flips. The SynthScope-against-official gap (28-33 flips) is larger than that
  batch-shuffle noise (0-3 flips). The 75-video parity run (max 0.038) agrees; over 9,107 pairs the tails are longer.
* The seed runs bound the official numbers' own noise: the 10-generator averages move by at most 0.02 across seeds.

### ffmpeg build sensitivity (9.0.2 against the pinned 4.2.2)

The official code uses whatever ffmpeg is on the PATH; the authors' build is unknown. The pin (`$SYNTHSCOPE_FFMPEG_NSG_VD`, 4.2.x)
is ours. 7.0.2 is unusable: it drops frames and selects different ones (68 of 75 parity videos give 8 frames, and on those
the frames differ by 25 grey levels on average; `repro/nsg_vd_ffmpeg_builds_genvideo75.json`). 9.0.2 gives the same frames
re-encoded, but the NSG amplifies the JPEG difference. SynthScope, `nsg_vd`, job 286453 (the only generators scored with 9.0.2):

| generator (released-CSV protocol) | accuracy 9.0.2 / 4.2.2 | recall | AUROC | per-video mean / max abs diff | verdict flips |
|---|---|---|---|---|---|
| ModelScope (500 + 500) | 88.50 / 88.40 | 97.20 / 97.00 | 96.39 / 95.99 | 0.073 / 0.743 | 51 of 1,000 |
| Sora (56 + 51) | 87.85 / 86.92 | 98.04 / 98.04 | 97.72 / 96.08 | 0.076 / 0.430 | 7 of 107 |

On the L120 ModelScope subset, accuracy is 87.50 against 86.67 and recall 97.50 against 99.17, with 18 of 240 verdicts flipped.
The build changes per-video statistics about ten times more than the official-against-served batching does, and moves a
generator's metrics by 0.1 to 1.7 points (most flips cancel). Run the detector with the 4.2 build; the registry warns, and
every result records the ffmpeg path and version.

### Zero-shot reference rows (NOT paper reproductions; `nsg_vd` only)

The paper reports none of these sets. `benchmark/bench_eval_video.py --model nsg_vd --dataset <name> --limit 1000` (the
first 1,000 clips per class per generator, as for WaveRep and D3), ffmpeg 4.2.2. **The reference set stays the shipped 100
MSR-VTT val videos for every dataset: nothing is re-referenced to the target domain**, and these are zero-shot rows
for a detector whose reference statistics come from one real-video domain. The real pools are MSR-VTT clips: VidProM and
VideoPhy use their own `0_real` (MSR-VTT, the first 1,000 by file name); EvalCrafter has no reals of its own and uses
GenVideo-Val's. Metrics as above (pooled accuracy, fake iff s > 1; the bench JSONs report balanced accuracy and AP instead:
mAUROC 95.49 / 97.88 / 97.79, mAP 93.63 / 90.91 / 94.71). Averages over the generators; per-generator rows in the JSON
(`zero_shot`).

| dataset (jobs) | generators | reals + fakes per generator | accuracy | recall | precision | AUROC | AUROC range over generators |
|---|---:|---|---:|---:|---:|---:|---|
| VidProM (286717) | 4 | 1,000 + 1,000 (Pika 998: 2 empty files) | 86.80 | 95.80 | 81.16 | 95.49 | 92.77 (ModelScope) to 98.60 (Pika) |
| VideoPhy (286718) | 8 | 1,000 + 304 to 344 | 83.17 | 99.23 | 59.87 | 97.88 | 95.66 (Gen-2) to 99.23 (Pika) |
| EvalCrafter (286719) | 14 | 1,000 + 598 to 700 | 86.47 | 99.12 | 75.35 | 97.79 | 95.34 (modelscope) to 99.41 (MoonValley) |

* **The false-positive rate at s > 1 is 22.2% on every pool** (real accuracy 77.8), so pooled accuracy and especially precision
  look low where the fakes are few (VideoPhy), while AUROC stays high: the threshold 1 is calibrated on the GenVideo-Val reals,
  and these sets are not it.
* **12 of the 100 reference videos are in each real pool**, and are scored against a set that contains them. The JSON also
  gives every row without those 12: accuracy, recall and AUROC change by at most 0.03; precision and F1 by up to 0.21 and 0.17
  (VideoPhy precision 59.87 to 60.09, F1 74.67 to 74.84).

### Deviations and provenance

* **Frames were extracted outside the main official run.** The official `VideoFramesDataset` uses moviepy readers in long-lived
  worker processes that were OOM-killed here (jobs 286429, 286438 and 286449); the frames were extracted beforehand by the official `process_video`
  with ffmpeg 4.2.2 (job 286532: 5,151 videos, all give 8 frames, none dropped), and the official dataset then re-extracted nothing.
* **All official numbers are warm-cache runs**, in the order of the official features: the NSG features were extracted by the official
  `ScoreFeaturesDataset` (3 videos per pass, list order) and cached; a first "cold" run (job 286533) died with rc 137 (OOM, 16
  DataLoader workers forked from the model-heavy process) after the extraction, so every reported run (jobs 286533,
  12 runs; 286716, 9 runs) started from the cached features. The exact permutations of an authors' cold run depend on RNG
  draws that cannot be recovered; the seed runs above bound the effect.
* **Seed 42.** The runs use `test.yaml`'s seed (cudnn deterministic); seeds 1 and 2 and a seed-42 rerun are in the JSON (`official_seed_runs`).
* **pandas shim.** The official code calls `DataFrame.applymap`, which is gone from the pandas of this environment; the driver aliases it to
  `DataFrame.map` only when it is missing. It runs after the metrics are computed, and the metrics `test_dMMD` logged equal the ones recomputed
  from the recorded per-video statistics (max difference 0.0 in all 21 runs).
* **The official modules are unmodified**: the staged copy was checked against the repository sources in every job (the `.py` files
  are identical; only `__pycache__` differs). The wrappers only record (sampler indices, the statistic per id); a toy check and the
  bit-identical seed-42 rerun show they change nothing.
* **Drivers and node-local files.** The driver scripts (official run and per-video recorder, frame extraction, list farm,
  SynthScope batch scripts) live in a private working directory and are identified here by job id: official frames 286532,
  official runs 286533 / 286716, SynthScope runs 286440 / 286441 / 286442 (nsg_vd / pika / unbalance), ffmpeg 9.0.2 run 286453,
  zero-shot 286717-286719, ffmpeg builds 286412, parity 286253 / 286389. The `data_root` / `cache_dir` paths in `repro/bench_results/nsg_vd*.json`
  point at node-local scratch (`.../_tmp/nsgvd/...`) of ivi-cn029, which is not part of the repository; the comparison JSON is
  regenerated from the per-video caches by `repro/nsg_vd_bench_compare.py`. Logs: `repro/logs/nsgvd*`.
* **Result files.** `repro/bench_results/nsg_vd{,_pika,_unbalance}_genvideo_split{500,120}.json` (the `split120` files are the L120 approximation of
  the same cache, not the paper's protocol), `..._ffmpeg902.json`, `nsg_vd_{vidprom,videophy,evalcrafter}.json`; website data in
  `repro/benchmark_specs/nsg_vd*.json`.

## Availability

**Not on the Space, not in `deploy/hf-space/fetch_weights.py`, not in the model zoo.** Serving it needs the user's approval
and: a check of the MSR-VTT terms for the shipped reference file (derived from 100 MSR-VTT videos, released for research
use); the 2.2 GB OpenAI diffusion model plus the 482 MB reference file on the hub (a hub upload; `synthscope push` uploads
no aux files for it, `AUX_FILES` lists only D3's and STALL's); ffmpeg 4.2 in the image (the Dockerfile installs the distribution's ffmpeg with apt, which is not
a 4.2 build; 7.0.x is unusable); and CPU time. Measured here
(`nsg_vd`, one 4-second 256x256 clip, fp32 UNet on CPU, warm model, engine path): 91 s with 2 threads, 21 s with 16 threads
(a shared login node, so an upper bound); on an RTX A5000 the benchmark runs took 0.45 to 0.67 s per video. With the model resident (about 1.1 GB UNet plus the
482 MB reference) a clip is therefore a minute-scale request on a small CPU Space. CPU scores also differ in the second decimal between thread counts (1.106 with 2 threads,
1.094 with 16 for that clip), against the statistic's tiny kernel width, which is another reason to treat a CPU verdict near 1 as noise.
To run it yourself, put the three checkpoints, `256x256_diffusion_uncond.pt` and
`nsg_vd_reference_msrvtt_val100.safetensors` (built with `build_reference.py` from the official split and local MSR-VTT clips) in
`$SYNTHSCOPE_CKPT_DIR` and point `$SYNTHSCOPE_FFMPEG_NSG_VD` at an ffmpeg 4.2 binary.

## License

Code and released checkpoints: Apache-2.0 (the official repository's `LICENSE`). The vendored guided-diffusion code and the
256x256 unconditional diffusion model are MIT (OpenAI; `guided_diffusion/LICENSE_GUIDED_DIFFUSION`). `official_tall.py` derives
from the TALL / Swin Transformer code (MIT, Microsoft). **The reference-feature file is derived from MSR-VTT videos, which
Microsoft released for research use: keep it to research use and do not redistribute it without checking the MSR-VTT terms.**
Citation (official):

```
@inproceedings{zhang2025NSGVD,
  title={Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection},
  author={Zhang, Shuhai and Lian, Zihao and Yang, Jiahao and Li, Daiyuan and Pang, Guoxuan and Liu, Feng and Han, Bo and Li, Shutao and Tan, Mingkui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
