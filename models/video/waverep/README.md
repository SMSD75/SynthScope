# WaveRep (`waverep`, `waverep_g4`)

**Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation** (Corvi,
Cozzolino, Prashnani, De Mello, Nagano, Verdoliva; NeurIPS 2025) —
[paper (arXiv 2506.16802)](https://arxiv.org/abs/2506.16802) ·
[official code](https://github.com/grip-unina/WaveRep-SyntheticVideoDetection) ·
[project page](https://grip-unina.github.io/WaveRep-SyntheticVideoDetection/)

A video detector built from an image backbone. A DINOv2 ViT-B/14 with registers is fine-tuned end to end on single
video frames. The method's contribution is the training data and an augmentation, not the architecture: in a wavelet
decomposition, a fake frame's low-pass and horizontal / vertical bands are replaced by those of a real frame, which steers
the detector towards the diagonal mid-high frequencies that generators leave behind and that survive compression. The
paper trains on ONE generator (Pyramid Flow) and tests on 15 unseen ones.

Two released checkpoints, both registered:

| key | checkpoint | trained on |
|---|---|---|
| `waverep` | `weights_dinov2_G1.ckpt` (`waverep.pth`) | "a single generator" (official demo README); we infer Pyramid Flow, the paper's detector in Tables 1-2 |
| `waverep_g4` | `weights_dinov2_G4.ckpt` (`waverep_g4.pth`) | "four generators" (official demo README; the demo's default); we infer Pyramid Flow, CogVideoX 1.5, Allegro and Open-Sora Plan, the Table 5 set |

## Pipeline (official, reproduced exactly)

```
video ─ OpenCV VideoCapture, every frame in order, BGR -> RGB      (official demo/utils.py ReadVideoIteratorCV)
  → the first 64 frames                                               (paper, App. D; the demo's --limit 64)
  → per frame: CenterCrop(504) on the PIL frame (no resize; a smaller frame is zero-padded), ToTensor, ImageNet norm
  → DINOv2 ViT-B/14 + 4 registers (timm vit_base_patch14_reg4_dinov2, img_size 504) → one logit per frame
  → video logit = mean of the frame logits;  P(fake) = sigmoid(logit)  (logit > 0: synthetic)
```

Weights: `weights_dinov2_G{1,4}.ckpt` from grip.unina.it (md5 `bc9852a7…` / `8bf19e6f…`, as listed in the demo
README; sha256 `1c7566b2…a897` / `50d63904…5d53`). Each is a flat timm state dict of the full model, loaded strictly.

## SynthScope adaptations

| choice | reason |
|---|---|
| `SynthScopeWaveRepModel` holds the timm model as `self.model`; `registry.remap_waverep` prefixes the checkpoint keys with `model.` | The official `create_model` builds exactly this timm model and loads the checkpoint into it strictly. The remap also accepts the Lightning `{"state_dict": {"model.*"}}` form the official loader handles. |
| `pretrained_backbone: false` | The official code first downloads the timm DINOv2 weights, then overwrites every one of them with the checkpoint; building the model empty gives the same tensors without the download. |
| `official_utils.py` = the official `demo/utils.py`, verbatim | Reference for the tests and the parity check (GRIP license: nonprofit use; notices kept). SynthScope's reader `media.cv2_frames` is the same loop as `ReadVideoIteratorCV`, written out so that the package does not import pandas. |
| video sampler `cv2_frames`, `num_frames=64` | The paper scores the first 64 frames (App. D, Fig. 9). The demo scores every frame unless `--limit` is given; `dataclasses.replace(spec, num_frames=0)` reproduces that default. |
| frames scored `frame_batch` (16) at a time; the mean is taken in float64 | The demo scores one frame at a time and averages the float32 logits with pandas (float64). Batching changes the CPU logits by at most ~1e-5 (measured below) and bounds memory on long videos (`Engine.video_frame_logits` streams). |
| new transform step `center_crop_504` | torchvision `CenterCrop(504)` on the PIL frame: bit-exact with the official `CenterCrop((504, 504))`, zero padding included. |
| `compute_loss` = BCE with logits on single frames | The training code is not released (only the augmentation); no training recipe is reproduced. |

## Fidelity check (identical to the official model)

The unmodified official demo (`demo/main_avideo.py` @ `0fd6010`, CPU) against SynthScope's full pipeline (raw file ->
`media.cv2_frames` -> registry transform -> engine-built model, strict load -> `Engine.video_frame_logits` /
`predict_video_clip`), both checkpoints, same environment (torch 2.11, timm 1.0.27, OpenCV 4.13), on 15 videos:
GenVideo-Val (MSR-VTT reals 298×224, Sora 1280×720 × 600 frames, WildScrape `.mov` 2160×544 and `.gif`), EvalCrafter
(Hot Shot 672×384 × 8, MoonValley 1184×672 at 50 fps), VideoPhy (Dream Machine 1360×752 × 121), VidProM (ModelScope
256×256 MPEG-4 part 2, Text2Video-Zero 512×512), the demo's own two videos, and three re-encodes (VP9 `.webm`, MJPEG
`.avi`, HEVC `.mkv`); codecs H.264, MPEG-4 part 2, GIF, VP9, MJPEG, HEVC; 8 to 600 frames.

| check | result |
|---|---|
| state dict | 177/177 tensors identical (sha256), 0 missing / 0 unexpected, no transformers key translation (timm model), both checkpoints |
| model inputs (every frame of every video) | bit-exact: same frame count, every 504×504 input tensor identical (max abs diff 0.0) |
| per-frame logits, every frame (demo default) | max abs diff 4.8e-6 (frames batched 16 at a time; the demo runs one at a time) |
| video logit / P(fake), every frame | max abs diff 6.0e-7 / 2.8e-8 |
| served protocol (first 64 frames) vs the demo with `--limit 64` | max abs logit diff 2.7e-7 |
| `frame_batch=1` (the demo's batch) vs the demo | per-frame max abs diff 1.8e-15: identical; the 4.8e-6 above is CPU batching only |

Gate check on the demo's own videos (official demo output, logit > 0 = synthetic): `opensoraplan_-J24pksGK64_2.mp4`
4.80 (G1) / 9.33 (G4), `real_-D8sIkdnSkM_0.mp4` -9.03 / -8.49.

**Frame decoder.** SynthScope reads frames with OpenCV exactly as the demo, so there is no decoder difference. Measured
for reference (first 64 frames, both checkpoints): the demo's alternative PyAV reader (`ReadVideoIteratorAV`) gives the
same logits on all 15 videos (diff 0.0); decord (SynthScope's sampler for image detectors on video) gives the same
logits on 13, fails on the GIF, and on the WildScrape `.mov` returns 58 frames where OpenCV and PyAV decode 52 (OpenCV
reports 58 in its header but stops reading after 52), which moves the G4 logit by 0.26 (P(fake) by 0.064). Protocol
matters too: on the 600-frame Sora clip the G4 logit is -2.61 over the first 64 frames and -1.20 over every frame.

## Results

**Verdict: faithful port (exact parity with the official demo, below); the paper's own benchmark is not held, so the only
paper-comparable numbers are the paper's 8 GenVideo columns, a PARTIAL comparison. `waverep` (G1, the paper's model)
reproduces 5 of those 8 cells to within about 2 points and does NOT reproduce HotShot (60.1 vs 81.4 bAcc), Show-1 (79.7
vs 92.1) or LaVie (92.1 vs 95.5). Every other row is a zero-shot reference on a SynthScope benchmark, not a paper
reproduction.**

Metrics: balanced accuracy at threshold 0.5 (logit 0; mean of real and fake accuracy) and AUC, the paper's metrics
(Tables 1-2: bAcc, Table 7: AUC). Protocol for every number below: the official OpenCV reader, the first 64 frames
(paper App. D), 504 centre crop, mean frame logit, fp32 (no TF32), `benchmark/bench_eval_video.py --model waverep|waverep_g4`.
SynthScope_train is "-": the paper's training code is not released (only the augmentation), so there is no SynthScope-trained model.

### What is and is not the paper's benchmark

| | status |
|---|---|
| The paper's own test set: 7 generators (Allegro, CogVideoX 1.5, Open-Sora Plan, Mochi-1, NOVA, Sora, FLUX), 300 prompts each (2,100 fakes), against 300 Panda-70M YouTube clips | **NOT HELD.** The fakes are downloadable from grip.unina.it, but the reals are YouTube videos that need the authors' downloader script (`yt-dlp`, cookies); we did not scrape YouTube. Without the reals there is no bAcc or AUC, so a fakes-only run would not be the paper's benchmark. |
| The 8 GenVideo generators among the paper's columns in Tables 1-2 and 7 (Crafter, Gen2, HotShot, LaVie, ModelScope, MorphStudio, Show-1, MoonValley) | **PARTIAL comparison**, below. The paper says it uses "the publicly available GenVideo dataset" and does not say which real videos it paired with each generator; SynthScope scores every generator against GenVideo-Val's own 10,000 MSR-VTT reals. |
| GenVideo-Val extras (Sora, WildScrape), VideoPhy, VidProM, EvalCrafter | zero-shot references; the paper reports none of them (its "Sora" column is its own web set, not GenVideo's) |

### Partial comparison: the paper's 8 GenVideo generators (GenVideo-Val, full data)

Full GenVideo-Val per generator (10,000 reals + all fakes of that generator), 64 frames. Paper = "Ours" in Table 1 (bAcc) and
Table 7 (AUC), the paper's single-generator detector (Pyramid Flow) = `waverep` (G1; the release says only "a single generator", so this identity is our inference). **G4 has no paper cell in these columns**
(Table 5 trains it on four generators and reports other test sets); its columns are for reference only, and G4 is trained on four
generators, so its numbers are not single-generator generalization.

| Generator (n fake) | paper bAcc | G1 bAcc | G4 bAcc | paper AUC | G1 AUC | G4 AUC |
|---|---:|---:|---:|---:|---:|---:|
| Crafter (1398) | 98.3 | 98.99 | 98.90 | 99.8 | 99.90 | 99.98 |
| Gen2 (1380) | 98.8 | 99.49 | 98.90 | 100.0 | 99.99 | 99.99 |
| HotShot (700) | 81.4 | 60.10 | 83.40 | 91.0 | 75.60 | 94.15 |
| LaVie (1400) | 95.5 | 92.06 | 97.33 | 97.7 | 97.03 | 99.26 |
| ModelScope (700) | 97.1 | 95.24 | 97.54 | 99.4 | 99.31 | 99.69 |
| MorphStudio (700) | 97.0 | 97.74 | 98.61 | 99.3 | 99.74 | 99.94 |
| Show-1 (700) | 92.1 | 79.67 | 96.83 | 97.1 | 94.27 | 99.43 |
| MoonValley (626) | 98.4 | 99.12 | 98.89 | 99.9 | 99.97 | 99.98 |
| **8-generator mean** | **94.83** | **90.30** | 96.30 | **98.03** | **95.73** | 99.05 |

Source: `repro/bench_results/waverep_genvideo_full.json` and `waverep_g4_genvideo_full.json` (`per_generator` `acc` = balanced accuracy
at 0.5, `auroc`); the paper's means are the means of its printed values.

* **Reproduced (within about 2 points of the paper, G1):** Crafter, Gen2, MorphStudio, MoonValley (within 0.75 bAcc and 0.5 AUC), ModelScope (-1.86 bAcc, -0.09 AUC).
* **Not reproduced (G1):** HotShot -21.3 bAcc / -15.4 AUC, Show-1 -12.4 / -2.8, LaVie -3.4 / -0.7. HotShot: real accuracy 99.2, but only
  21.0% of the fakes are called fake at P >= 0.5 (Show-1: 60.1%, LaVie: 84.9%), and the AUC drops too (75.6 vs 91.0), so it is not a
  threshold or calibration offset. **The cause is untested.** A different real pool or fake subset in the paper (unstated) is a possible
  cause; we cannot check it. SynthScope equals the official demo on every file compared (below), so it is not the code path.
* G4 is above the paper's G1 on all 8 columns in bAcc (+0.1 to +4.7; Show-1 96.83, HotShot 83.4) and above or equal in AUC except Gen2
  (99.99 vs 100.0).
* Sora (56 fakes; G1 67.46 bAcc / 68.54 AUC, G4 76.65 / 83.61) and WildScrape (924; G1 78.87 / 86.92, G4 85.77 / 90.83) are GenVideo-Val extras with no paper cell.

### Zero-shot reference rows (NOT paper reproductions)

Benchmarks the paper does not report. Both keys are zero-shot on these sets, but G4 was trained on four generators (Table 5); do not
read the G4 rows as the paper's single-generator generalization. mAP (fake as positive) on full GenVideo-Val is low only because of the
10,000 : 56-1400 class imbalance; AUROC and mAP (real as positive) are not affected the same way. Means over the generators of each set.

| Set (protocol) | gens | key | bAcc@0.5 | mAP (fake +) | mAP (real +) | mAUROC |
|---|---:|---|---:|---:|---:|---:|
| GenVideo-Val full (10,000 reals + all fakes) | 10 | `waverep` | 86.87 | 79.85 | 99.24 | 92.13 |
|  |  | `waverep_g4` | 93.28 | 88.88 | 99.69 | 96.69 |
| GenVideo-Val `--limit 1000` | 10 | `waverep` | 86.62 | 89.81 | 93.96 | 92.09 |
|  |  | `waverep_g4` | 92.65 | 94.47 | 97.33 | 96.53 |
| VideoPhy `--limit 1000` | 8 | `waverep` | 90.00 | 93.21 | 98.44 | 96.54 |
|  |  | `waverep_g4` | 95.73 | 98.20 | 99.74 | 99.29 |
| VidProM `--limit 1000` | 4 | `waverep` | 89.85 | 98.11 | 97.94 | 98.06 |
|  |  | `waverep_g4` | 95.80 | 99.29 | 99.19 | 99.25 |
| EvalCrafter (fakes vs the first 1000 GenVideo-Val MSR-VTT reals) | 14 | `waverep` | 92.84 | 96.93 | 97.39 | 97.17 |
|  |  | `waverep_g4` | 96.55 | 99.19 | 99.35 | 99.29 |

Caveats:

* `--limit 1000`: the first 1000 clips per class per generator (the D3 protocol), so exactly 1000 reals and up to 1000 fakes; the full-data GenVideo-Val row has 10,000 reals.
* **EvalCrafter has no reals of its own.** Its fakes (up to 700 per generator, 14 generators) are scored against the first 1000 GenVideo-Val MSR-VTT reals, the same for every generator; its HotShot, ModelScope, Show-1, MorphStudio and MoonValley cells equal the GenVideo `--limit 1000` cells (same videos).
* VideoPhy: 8 of its 10 generators are online (dead links skipped). VidProM: ModelScope, Pika, Text2Video-Zero and VideoCrafter2 (the first 1000 clips each; the full set is terabytes); non-commercial licence.
* **2 empty VidProM fakes are excluded** (Pika `00642_pika-9e93dcfc...`, `00891_pika-e025f9ee...`: 0-byte files in our VidProM copy, so OpenCV decodes no frame; the official demo would write NaN for them), so Pika has 998 fakes. All other sets: 0 decode failures. Per key the four per-set caches hold 37,859 entries (GenVideo-Val 18,584, VideoPhy 3,674, VidProM 5,000, EvalCrafter 10,601), i.e. 37,857 scored videos plus the 2 undecodable ones; this is a sum over sets, not a unique count (the 1000 EvalCrafter reals are the first 1000 GenVideo-Val reals).
* Weakest cells (G1): VideoPhy OpenSora 62.15 and ZeroScope 63.75 bAcc, VidProM ModelScope 71.55, GenVideo HotShot 60.10 and Sora 67.46. G4 is better on every one of these.

Per generator (bAcc at 0.5 with real / fake accuracy in brackets; AUC):

**GenVideo-Val full (10,000 reals + all fakes)**

| Generator | n real / n fake | G1 bAcc (real / fake acc) | G1 AUC | G4 bAcc (real / fake acc) | G4 AUC |
|---|---|---:|---:|---:|---:|
| Crafter | 10000 / 1398 | 98.99 (99.2 / 98.8) | 99.90 | 98.90 (97.9 / 99.9) | 99.98 |
| Gen2 | 10000 / 1380 | 99.49 (99.2 / 99.8) | 99.99 | 98.90 (97.9 / 99.9) | 99.99 |
| HotShot | 10000 / 700 | 60.10 (99.2 / 21.0) | 75.60 | 83.40 (97.9 / 68.9) | 94.15 |
| LaVie | 10000 / 1400 | 92.06 (99.2 / 84.9) | 97.03 | 97.33 (97.9 / 96.7) | 99.26 |
| ModelScope | 10000 / 700 | 95.24 (99.2 / 91.3) | 99.31 | 97.54 (97.9 / 97.1) | 99.69 |
| MoonValley | 10000 / 626 | 99.12 (99.2 / 99.0) | 99.97 | 98.89 (97.9 / 99.8) | 99.98 |
| MorphStudio | 10000 / 700 | 97.74 (99.2 / 96.3) | 99.74 | 98.61 (97.9 / 99.3) | 99.94 |
| Show-1 | 10000 / 700 | 79.67 (99.2 / 60.1) | 94.27 | 96.83 (97.9 / 95.7) | 99.43 |
| Sora | 10000 / 56 | 67.46 (99.2 / 35.7) | 68.54 | 76.65 (97.9 / 55.4) | 83.61 |
| WildScrape | 10000 / 924 | 78.87 (99.2 / 58.5) | 86.92 | 85.77 (97.9 / 73.6) | 90.83 |
| **mean** | | **86.87** | 92.13 | **93.28** | 96.69 |

**VideoPhy `--limit 1000`**

| Generator | n real / n fake | G1 bAcc (real / fake acc) | G1 AUC | G4 bAcc (real / fake acc) | G4 AUC |
|---|---|---:|---:|---:|---:|
| Dream-Machine | 1000 / 339 | 99.50 (99.3 / 99.7) | 99.98 | 98.40 (96.8 / 100.0) | 100.00 |
| Gen-2 | 1000 / 312 | 99.65 (99.3 / 100.0) | 99.99 | 98.40 (96.8 / 100.0) | 100.00 |
| LaVie | 1000 / 343 | 98.48 (99.3 / 97.7) | 99.88 | 98.40 (96.8 / 100.0) | 99.96 |
| OpenSora | 1000 / 344 | 62.15 (99.3 / 25.0) | 88.55 | 83.43 (96.8 / 70.1) | 96.20 |
| Pika | 1000 / 304 | 99.32 (99.3 / 99.3) | 99.94 | 98.40 (96.8 / 100.0) | 99.99 |
| SVD | 1000 / 344 | 98.34 (99.3 / 97.4) | 99.67 | 98.40 (96.8 / 100.0) | 99.99 |
| VideoCrafter2 | 1000 / 344 | 98.78 (99.3 / 98.3) | 99.91 | 98.25 (96.8 / 99.7) | 99.96 |
| ZeroScope | 1000 / 344 | 63.75 (99.3 / 28.2) | 84.42 | 92.15 (96.8 / 87.5) | 98.22 |
| **mean** | | **90.00** | 96.54 | **95.73** | 99.29 |

**VidProM `--limit 1000`**

| Generator | n real / n fake | G1 bAcc (real / fake acc) | G1 AUC | G4 bAcc (real / fake acc) | G4 AUC |
|---|---|---:|---:|---:|---:|
| ModelScope | 1000 / 1000 | 71.55 (99.3 / 43.8) | 94.52 | 89.40 (96.8 / 82.0) | 97.59 |
| Pika | 1000 / 998 | 99.35 (99.3 / 99.4) | 99.97 | 98.35 (96.8 / 99.9) | 99.98 |
| Text2Video-Zero | 1000 / 1000 | 91.55 (99.3 / 83.8) | 98.10 | 97.35 (96.8 / 97.9) | 99.53 |
| VideoCrafter2 | 1000 / 1000 | 96.95 (99.3 / 94.6) | 99.66 | 98.10 (96.8 / 99.4) | 99.91 |
| **mean** | | **89.85** | 98.06 | **95.80** | 99.25 |

**EvalCrafter (reals: the first 1000 GenVideo-Val MSR-VTT)**

| Generator | n real / n fake | G1 bAcc (real / fake acc) | G1 AUC | G4 bAcc (real / fake acc) | G4 AUC |
|---|---|---:|---:|---:|---:|
| MoonValley | 1000 / 626 | 99.17 (99.3 / 99.0) | 99.97 | 98.32 (96.8 / 99.8) | 99.98 |
| floor33 | 1000 / 700 | 97.79 (99.3 / 96.3) | 99.74 | 98.04 (96.8 / 99.3) | 99.93 |
| gen2 | 1000 / 681 | 99.50 (99.3 / 99.7) | 99.99 | 98.25 (96.8 / 99.7) | 99.99 |
| gen2_december | 1000 / 699 | 99.58 (99.3 / 99.9) | 99.99 | 98.40 (96.8 / 100.0) | 100.00 |
| hotshot | 1000 / 700 | 60.15 (99.3 / 21.0) | 75.95 | 82.83 (96.8 / 68.9) | 93.75 |
| lavie-base | 1000 / 700 | 98.58 (99.3 / 97.9) | 99.83 | 98.26 (96.8 / 99.7) | 99.94 |
| lavie-interpolation | 1000 / 700 | 85.65 (99.3 / 72.0) | 94.30 | 95.26 (96.8 / 93.7) | 98.46 |
| mix-sr | 1000 / 700 | 99.58 (99.3 / 99.9) | 99.99 | 98.40 (96.8 / 100.0) | 100.00 |
| modelscope | 1000 / 700 | 95.29 (99.3 / 91.3) | 99.31 | 96.97 (96.8 / 97.1) | 99.61 |
| pika | 1000 / 697 | 96.57 (99.3 / 93.8) | 99.11 | 97.97 (96.8 / 99.1) | 99.92 |
| pika_v1_december_1 | 1000 / 700 | 97.58 (99.3 / 95.9) | 99.42 | 98.19 (96.8 / 99.6) | 99.91 |
| show_1 | 1000 / 700 | 79.72 (99.3 / 60.1) | 94.35 | 96.26 (96.8 / 95.7) | 99.35 |
| videocrafter-v1.0 | 1000 / 700 | 98.51 (99.3 / 97.7) | 99.81 | 98.26 (96.8 / 99.7) | 99.96 |
| zeroscope | 1000 / 598 | 92.12 (99.3 / 85.0) | 98.68 | 96.31 (96.8 / 95.8) | 99.32 |
| **mean** | | **92.84** | 97.17 | **96.55** | 99.29 |

The `--limit 1000` GenVideo-Val per-generator values are in `repro/bench_results/waverep_genvideo.json` / `waverep_g4_genvideo.json`.

### Official-demo parity on benchmark files

The unmodified official `demo/main_directory.py` (`--limit 64`, GPU) against the SynthScope scores of the benchmark run, both
checkpoints, 180 GenVideo-Val files (the first 60 Crafter reals and the first 12 fakes of each of the 10 generators). Ours = the cached
scores of the benchmark run (RTX A5000 / RTX 6000, rounded to 6 decimals); official = DAS6 A6000. The differences are cross-device
fp32 noise, not a protocol difference. File: `repro/waverep_official_parity_genvideo180.json`.

| checkpoint | files compared | NaN | max abs logit diff | mean abs logit diff | max abs P(fake) diff | sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 180/180 | 0 | 1.02e-05 | 1.7e-06 | 2.1e-06 | 180/180 |
| G4 | 180/180 | 0 | 1.05e-05 | 1.4e-06 | 1.0e-06 | 180/180 |

Together with the 15-video, 6-codec check above (model inputs bit-exact, video logit max diff 6.0e-7), SynthScope's scores are the
official demo's scores, so a difference from the paper is not a difference of this port; its cause (the unstated real pairing, the
fake subset, or the paper's evaluation details) is untested.

### Jobs and files

Scores come from a resumable per-video cache (`bench_eval_video.py --cache-dir`, with `cache_meta.json` recording the checkpoint
sha256 `1c7566b2...a897` (G1) / `50d63904...5d53` (G4) and the preprocessing; the cache refuses a run whose setup differs). Four
sets x two keys, scored in shards: IVI 285376_0 (cn027), 285390_3 and 285390_4 (cn031), 285389_2 and its remainder 285886_2, DAS6
67328 and its remainder 67404 (node409); full GenVideo-Val took about a day over three IVI GPUs and one DAS6 node. Aggregation: IVI
286053, re-run as 286059 after the cache-identity fix with identical means (CPU, `--cache-only`). Official-demo parity: DAS6 67291.
Task 1 parity: IVI 285339 (official side) and 285342. Logs: `repro/logs/waverep_*`. Result JSONs: `repro/bench_results/waverep[_g4]_{genvideo_full,genvideo,videophy,vidprom,evalcrafter}.json`;
website data: `repro/benchmark_specs/waverep.json`, `waverep_g4.json`.

## Availability

Space: served (both keys), under the GRIP non-profit licence below: the free, non-commercial demo credits the authors and links
the paper and repository (maintainer approval 2026-11-11). The checkpoints sit in the private weights repo the Space reads
(`waverep.pth` = `weights_dinov2_G1.ckpt`, `waverep_g4.pth` = `weights_dinov2_G4.ckpt`, sha256 1c7566b2... / 50d63904...) and are
fetched by `deploy/hf-space/fetch_weights.py`. A clip costs about 30 s on the Space's CPU (up to 64 ViT-B/14 forwards at 504 px).
Locally: place the checkpoints under `$SYNTHSCOPE_CKPT_DIR` as `waverep.pth` / `waverep_g4.pth`.

## License

GRIP-UNINA license (`LICENSE.md` of the official repository): free for informational and nonprofit purposes only,
original notices kept, and the authors credited whenever results are made public. The checkpoints (distributed by GRIP
alongside the code) are treated as covered by the same terms, as for B-Free. Citation (official, from the project page):

```
@InProceedings{corvi2025seeing,
    title = {Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation},
    author = {Riccardo Corvi and Davide Cozzolino and Ekta Prashnani and Shalini De Mello and Koki Nagano and Luisa Verdoliva},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year = {2025},
    url = {https://openreview.net/forum?id=dOGXKBL7IE}
}
```
