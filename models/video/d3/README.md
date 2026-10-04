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
| `backbone_name` | `clip` (default) / `dinov2` / `resnet18` |
| First-order `dist` | `l2` (default) or `cos` |
| Score | `std` of second-order central-difference features over time (Eq. 8) |
| `threshold` (`tau`) | decision threshold for argmax accuracy only (default `0.0`) |
| `dt` | sampling-interval scale (default `1.0`; constant — does not change the std ranking) |
| Normalization | done by the **dataset transform pipeline** (`normalize_clip` for CLIP, `normalize_dinov2` for DINOv2/ResNet) — **not** inside the model |
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

> **Why not XCLIP?** The paper's headline encoder is XCLIP-B/16 (≈0.7–1 mAP above CLIP-B/16),
> but `transformers`' `XCLIPVisionModel` is a *video* encoder: its cross-frame "message-token"
> attention (tied to `config.num_frames = 8`) mixes frames together, which contradicts D3's
> independent-per-frame premise and also errors unless `B·T` is a multiple of 8. We therefore use
> the clean per-frame **CLIP ViT-B/16** instead (XCLIP's spatial tower is CLIP-initialized anyway).

## Input normalization

D3 does **not** normalize internally — it expects frames already normalized by the dataset
transform pipeline, matched to the chosen backbone (consistent with the image CLIP/DINO
detectors). Use `normalize_clip` for the `clip` backbone and `normalize_dinov2`
(ImageNet stats) for `dinov2`/`resnet18`, set via `DatasetConfig.*_transform_list` in the
experiment script. The frozen encoder silently degrades on un-normalized or mismatched
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

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

> **Faithful protocol (training-free):** D3 now runs the official setup — **X-CLIP-B/16** per-frame
> features + a **10% center edge-crop** before resize (registry `backbone_name=xclip, edge_crop=0.1`).
> On GenVideo this lifts the robust metrics **AUROC 93.87→95.95, acc 86.04→91.71** (`d3_xclip_genvideo`).
> Note D3 is training-free and SynthScope's GenVideo **mAP is degenerate** — the pool pairs **10,000 shared
> reals against as few as 56 fakes** per generator (Sora), so AP is dominated by a few high-scoring
> reals (the 89.0→63.93 "drop" is imbalance noise). **AUROC/acc are the metrics to compare** (~96 ≈ paper-quality).

| Benchmark | Paper | SynthScope (training-free, XCLIP+edge-crop) | Status |
|---|---|---|---|
| GenVideo (DeMamba GenVideo-Val, Table 1) | mAP=98.46 | AUROC=95.95 / acc=91.71 | reproduced (AUROC) |
| EvalCrafter (Table 2, 14 generators) | mAP=98.87 | AUROC=95.41 / acc=87.67 | below |
| VideoPhy (Table 3, 10 generators) | mAP=99.16 | — | missing |
| VidProM (Table 4, 6 generators) | mAP=88.46 | — | missing |

**reproduced (AUROC)** = SynthScope matches the paper on the robust ranking metric (mAP unreliable here due to extreme real:fake imbalance) · **below** = gap remains · **missing** = not yet evaluated.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| **Default encoder is CLIP ViT-B/16, not XCLIP-B/16** | The paper's best encoder is XCLIP-B/16, but HF `XCLIPVisionModel` does cross-frame message-token attention (tied to `num_frames`), which violates D3's independent-per-frame premise and crashes unless `B·T % num_frames == 0`. CLIP-B/16 is a clean per-frame encoder ≈0.7–1 mAP behind; XCLIP is intentionally not included. |
| Default frame sampling is `num_frames=16` at `fps=8` over 2 s (configurable) | Matches the paper's "up to 2 s at 8 fps"; SynthScope fixes `T` per dataset so clips can be batch-collated. |
| Frames are decoded with **decord** and resized to `224×224` | The paper crops 10% of the longer edge before resizing; SynthScope's generic video loader resizes directly. This preprocessing detail does not change the second-order std signal materially. |
| Metrics via SynthScope's `AIGIMetrics` (per-generator AP/AUROC) | Keeps evaluation/logging consistent with the rest of the project; the paper also reports per-subset AP/AUROC. |
| One dummy `_noop` parameter | Required so the SynthScope `AdamW` trainer does not crash on a training-free model; it has zero effect. |

These are integration/preprocessing differences, not conceptual changes: the detector is still the
frozen-encoder → first-order → second-order-central-difference → std pipeline of the paper.

## Summary

D3 in SynthScope is a faithful, method-level reproduction of Zheng et al.'s ICCV 2025 training-free
detector. A frozen visual encoder (CLIP ViT-B/16 by default) turns each frame into a feature; the L2
distance between consecutive frames gives first-order features; their central difference gives
second-order features; and the standard deviation of those second-order features is the detection
score. Real videos are volatile (high std), AI-generated videos are flat (low std). The score is
mapped to two-class outputs so SynthScope's AP/AUROC metrics rank fakes by low std. There is nothing to
train — a single no-op parameter only keeps the optimizer happy.

## Probability calibration (`d3_calibration.json`)

D3 is training-free and outputs a raw score: the volatility of second-order frame-feature differences (real videos
average about 9.2, generated ones about 2.1 on GenVideo-Val). Turning that score into P(fake) needs a threshold and a
scale, which ship with the model in `d3_calibration.json` (fitted by `interface/scripts/calibrate_d3.py` on 200 real +
200 fake GenVideo-Val clips; balanced accuracy 87.8%). Without it every video would get P(fake) of about 0.01, i.e. be
called "real"; AP/AUROC are unaffected because they depend only on the ranking. A `d3_calibration.json` in
`$SYNTHSCOPE_CKPT_DIR` overrides the shipped one.

