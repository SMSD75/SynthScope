# D3: Training-Free AI-Generated Video Detection Using Second-Order Features

## Short method explanation

**D3** (*Detection by Difference of Differences*) is a **training-free** AI-generated *video*
detector from the paper **"D3: Training-Free AI-Generated Video Detection Using Second-Order
Features"** by Chende Zheng, Ruiqi Suo, Chenhao Lin, Zhengyu Zhao, Le Yang, Shuai Liu, Minghui
Yang, Cong Wang, and Chao Shen (ICCV 2025).

The key idea comes from second-order dynamical analysis under Newtonian mechanics: a faithful
model of the real world should reproduce the **second-order (acceleration-like) temporal
features** of real video. The authors show that real videos exhibit large *volatility* in these
second-order features, while AI-generated videos are comparatively **flat** (generators fail to
replicate second-order dynamics). D3 quantifies that volatility with a single number — the
standard deviation of the second-order features — and uses it as the detection score. No training
and no generated videos are required.

- Official repo: https://github.com/Zig-HS/D3
- Paper: *D3: Training-Free AI-Generated Video Detection Using Second-Order Features*, ICCV 2025.

## Pipeline

Given a clip `frames` of shape `[B, T, 3, 224, 224]` in `[0, 1]` (decoded by the SynthScope video
dataset and normalized by its transform pipeline):

1. **Zero-order features.** A frozen pretrained visual encoder embeds each frame:
   `F0[b, t] = encoder(frame_{b,t}) ∈ R^N`. Run under `torch.no_grad()`. Default encoder is
   **CLIP ViT-B/16** (HF `CLIPVisionModel`, `openai/clip-vit-base-patch16`, per-frame
   `pooler_output`); DINOv2 and ResNet-18 are also selectable via `backbone_name`.
2. **First-order features.** Inter-frame change between consecutive frames:
   `F1[b, k] = dist(F0[b, k], F0[b, k+1]) / dt`, `k = 0 … T-2`, where `dist` is the **L2 distance**
   (default, `first_order: l2`) or **cosine similarity** (`first_order: cos`).
3. **Second-order features (central difference).** `F2[b, k] = (F1[b, k] − F1[b, k-1]) / dt`,
   `k = 1 … T-2` (a length `T-2` series).
4. **Score.** `s[b] = std(F2[b, :])` over the time axis. The unbiased std over the `T-2`
   second-order values equals `sqrt( 1/(T-3) · Σ (F2 − mean)^2 )`, i.e. Eq. (8) of the paper.
   Real videos → **high** `s`; AI-generated → **low** `s`.
5. **Two-class output for SynthScope metrics.** `outputs = stack([s − tau, tau − s], dim=1)` where
   `tau` is `threshold` (default `0.0`). Column 0 = real, column 1 = fake, so
   `softmax(outputs)[:, 1]` (the fake score) ranks AI-generated videos by **low** `s`.

The detector uses label `0` for real videos and `1` for AI-generated videos.

## Important: D3 is training-free

D3 has **no learnable detection parameters**. The threshold-free metrics — **Average Precision
(AP)** and **AUROC** — computed on `softmax(outputs)[:, 1]` are the *meaningful* metrics (and are
exactly what the paper reports). The argmax-based real/fake **accuracy** depends entirely on the
chosen `threshold` (`tau`); with the default `tau = 0` it reflects the sign of `s`, which is not
calibrated. Use AP/AUROC to judge D3.

To satisfy the SynthScope trainer (its `AdamW` is built over *trainable* parameters and would crash on
an empty list), the module carries a single dummy parameter `self._noop = nn.Parameter(zeros(1))`
and returns `loss = (self._noop * 0).sum()` — a true no-op with a valid gradient. "Training" runs
for zero effect; evaluation (the val/test passes the trainer runs every epoch, including epoch 0)
produces the real result. The trainer config uses `epochs: 1`.

## Model details

Implemented in `models/video/d3/d3.py` as `SynthScopeD3Model`.

| Component | Setting |
|---|---|
| Visual encoder | frozen, run under `torch.no_grad()` |
| `backbone_name` | `xclip` (the paper's encoder; what SynthScope serves) / `clip` (config default for training-style runs) / `dinov2` / `resnet18` |
| First-order `dist` | `l2` (default) or `cos` |
| Score | `std` of second-order central-difference features over time (Eq. 8) |
| `threshold` (`tau`) | decision threshold for argmax accuracy only (default `0.0`) |
| `dt` | sampling-interval scale (default `1.0`; constant — does not change the std ranking) |
| Normalization | done outside the model: the served D3 (registry, web app, `bench_eval_video.py`) uses the official ImageNet statistics on BGR frames; Hydra dataset configs use `normalize_clip` / `normalize_dinov2` |
| Trainable params | one dummy `_noop` scalar (no-op) |
| Output | `stack([s − tau, tau − s], dim=1)` (col 0 real, col 1 fake) |

### Backbone notes

D3 is encoder-agnostic; every backbone is used as a **frozen, per-frame** feature extractor
(one global vector per frame — the inter-frame *differences* carry the temporal signal).

- **CLIP ViT-B/16** (default) — HF `CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")`,
  per-frame `pooler_output` (768-d), matching the official D3 `CLIP-16`. `clip-32` selects the patch32 variant.
- **DINOv2** (`dinov2-base` / `dinov2-large`) — HF `AutoModel`, per-frame `pooler_output`, matching the
  official D3 `DINO-base` / `DINO-large`.
- **ResNet-18** via torchvision (`fc` replaced with `Identity`, 512-d global-avg-pooled feature) — matches
  the official D3 CNN path (`children()[:-1]`).

All transformer backbones use the HF `pooler_output` (the global pooled per-frame vector), exactly
as the official `models/D3_model.py` does — **not** dense/patch tokens.

> **X-CLIP-B/16** (`backbone_name=xclip`, HF `microsoft/xclip-base-patch16`) is the paper's encoder and what SynthScope
> serves. It is used exactly as the official `models/D3_model.py` uses it (frames flattened to `B·T` images, per-frame
> `pooler_output`); on the same input frames SynthScope's scores equal the official model's (max |diff| 0). Clip lengths
> of 8 or 16 frames satisfy X-CLIP's 8-frame message-token grouping.

## Input normalization

D3 does **not** normalize internally — it expects frames already normalized by the dataset
transform pipeline, matched to the chosen backbone (consistent with the image CLIP/DINO
detectors). Use `normalize_clip` for the `clip` backbone and `normalize_dinov2`
(ImageNet stats) for `dinov2`/`resnet18`, set via `DatasetConfig.*_transform_list` in the
experiment script. The served X-CLIP D3 (registry `d3`) follows the official code instead: ImageNet statistics on frames
in BGR order (see "Differences" below). The frozen encoder silently degrades on un-normalized or mismatched
frames. `AIGITransformations` wraps each `[T, 3, H, W]` clip as `torchvision.tv_tensors.Video`,
so the normalization (and any spatial transform) applies consistently across all frames.

## Training setup

Configs:

```bash
conf/model/d3.yaml      # model_type: d3, backbone_name, first_order, threshold, dt, num_frames, freeze_backbone
conf/trainer/d3.yaml    # epochs: 1, tiny lr, warmup off (it does not really train)
```

| Setting | Value |
|---|---:|
| `model_type` | `d3` |
| `backbone_name` | `clip` |
| `first_order` | `l2` |
| `threshold` | `0.0` |
| `dt` | `1.0` |
| `num_frames` | `16` |
| `freeze_backbone` | `true` |

Launcher: `experiments/d3.sh` (uses the `genvideo` video dataset; data is not downloaded, so it
is a template). The paper's benchmarks are **GenVideo, VideoPhy, EvalCrafter, VidProM**; configs
exist for all four under `conf/dataset/`.

## Integration notes

D3 uses the shared `compute_loss(batch) -> {loss, outputs}` interface, with `batch = (frames,
labels, folder_ids)` and `frames` of shape `[B, T, 3, H, W]`:

| Key | Meaning |
|---|---|
| `loss` | no-op scalar `(_noop * 0).sum()` (training-free) |
| `outputs` | two-class real/fake scores `stack([s − tau, tau − s], dim=1)` for SynthScope metrics |

`SynthScopeD3Model` is registered in `trainer.py` under `model_type == "d3"`. The SynthScope video dataset
(`VideoFolderDataset`) decodes `num_frames` frames per clip at `224×224` in `[0, 1]`; D3 samples
at ~8 fps over up to 2 s (`num_frames: 16`, `fps: 8` in the dataset config).

## Reproduced benchmark results

The paper's protocol (official `eval.py`): the first 1000 clips per class per generator, and AP with **real videos as the
positive class** (`average_precision_score(1 - y, std)`). Fake-only sets (EvalCrafter, VideoPhy, VidProM) are paired with
GenVideo-Val's MSR-VTT reals, as the official README does. Two columns: **SynthScope** is this implementation
(`benchmark/bench_eval_video.py --model d3 --dataset NAME --limit 1000`), **official code** is the D3 repository's own
frame extraction, reader and model run end to end on the same data (`repro/official_evals/d3_official_eval.py`).

| Benchmark (AP, real positive) | Paper | Official code, our data | SynthScope | Status |
|---|---:|---:|---:|---|
| GenVideo-Val (Table 1, 10 generators) | 98.46 | 97.93 | **99.14** | reproduced |
| EvalCrafter (Table 2, 14 generators) | 98.87 | 98.70 | **99.30** | reproduced |
| VideoPhy (Table 3; 8 of 10 generators still downloadable) | 99.16 (8 shown: 99.22) | 98.84 | **99.45** | reproduced |
| VidProM (Table 4; 4 of 6 generators, first 1000 clips each) | 88.46 (those 4: 84.95) | 84.91 | 92.15 | reference (subsample) |

- **The official code reproduces the paper on our data**: 9 of 10 GenVideo generators and all 14 EvalCrafter generators
  are within 0.25 AP; VidProM's four generators average 84.91 vs 84.95 (Text2Video-Zero 46.36 vs 45.11, the paper's
  noted failure case). The one outlier is GenVideo's WildScrape (92.50 vs 96.49): its 282 GIFs cannot be timed by
  decord, so their frames start at 0 instead of the official random offset.
- **SynthScope's D3 is the official model**: on the frames the official code extracts it returns identical scores
  (max |diff| 0). Its numbers differ from the official column only because the official pipeline writes frames to JPEG
  with ffmpeg and reads them back; SynthScope decodes frames directly. That matters most for low-quality clips
  (VidProM Text2Video-Zero: 71.28 in SynthScope vs 46.36 with the official JPEG round-trip).
- AUROC and balanced accuracy (at the calibrated threshold) for SynthScope: GenVideo 98.46 / 96.15, EvalCrafter
  98.62 / 96.59, VideoPhy 98.09 / 95.19, VidProM 90.86 / 84.86. Per-generator numbers are in
  `repro/benchmark_specs/d3.json` and `repro/bench_results/d3_*.json`.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Frames are decoded directly (decord), not written to JPEG by ffmpeg and read back | Avoids a lossy round-trip and an ffmpeg dependency; it is the only remaining difference from the official pipeline (see above). |
| Clips longer than 3 s start at 0 | The official extraction picks a random start in a multiprocessing pool, so its frames are not reproducible anyway. |
| Metrics via SynthScope's evaluators | `bench_eval_video.py` reports per-generator AP (both positive classes), AUROC and accuracy. |
| One dummy `_noop` parameter | Required so the SynthScope `AdamW` trainer does not crash on a training-free model; it has zero effect. |

Matched to the official code (2026-10-11): X-CLIP-B/16 per-frame features with L2 first-order distances; 16 frames at 8 fps,
or 8 frames for clips under 2 s (earlier versions repeated frames up to 16, which broke 1 s generators such as HotShot);
frames taken at multiples of 1/8 s like ffmpeg's `fps` filter; 10% cropped from each end of the longer side; resize to
224; ImageNet normalization; and BGR channel order, because the official reader loads frames with `cv2.imread` and never
converts them to RGB.

## Summary

D3 in SynthScope is the official training-free detector of Zheng et al. (ICCV 2025): a frozen X-CLIP-B/16 turns each frame
into a feature, the L2 distance between consecutive frames gives first-order features, their central difference gives
second-order features, and the standard deviation of those is the score. Real videos are volatile (high std),
AI-generated ones are flat (low std). There is nothing to train.

## Probability calibration (`d3_calibration.json`)

D3 is training-free and outputs a raw score: the volatility of second-order frame-feature differences (real videos
average about 11.3, generated ones about 1.8 on GenVideo-Val). Turning that score into P(fake) needs a threshold and a
scale, which ship with the model in `d3_calibration.json` (fitted by `interface/scripts/calibrate_d3.py`, through the same
preprocessing the engine uses, on 200 real + 200 fake GenVideo-Val clips; balanced accuracy 94.8%). Without it every video would get P(fake) of about 0.01, i.e. be
called "real"; AP/AUROC are unaffected because they depend only on the ranking. A `d3_calibration.json` in
`$SYNTHSCOPE_CKPT_DIR` overrides the shipped one.

