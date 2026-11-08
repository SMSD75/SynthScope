# VINA (`vina`, `vina_video`)

**Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection** (arXiv 2605.21977) —
[paper](https://arxiv.org/abs/2605.21977) · [official code](https://github.com/mgiant/VINA) ·
[checkpoint](https://huggingface.co/mgiant/VINA_checkpoints)

VINA detects AI-generated images and AI-generated videos with one model. A DINOv3 ViT-L/16 is fine-tuned end to end
with a linear head on its CLS output, on images and video frames together: the frames of real and generated videos
act as a natural augmentation (compression, motion blur, varied content) that makes the image detector generalize,
and the same model then scores video frames. The released checkpoint `VINA_DINOv3-L_140k+DDA` is the one the official
README points to for both the image and the video benchmarks.

## Pipeline (official, reproduced exactly)

```
image ─ Resize(256, bicubic) on the short side ─ CenterCrop(256) ─ ToImage, ToDtype(float32) ─ ImageNet normalization
      → DINOv3 ViT-L/16 (eager attention, fp32) → pooler output (CLS after the final norm) [1024]
      → Linear(1024, 1) → logit;  P(fake) = sigmoid(logit)                 (official test.py / util.eval_model)

video ─ ffmpeg -vf fps=2 → JPEG frames 1.jpg, 2.jpg, ...                  (official preprocess/video2frame.py)
      ─ the central select_frame_nums = 1 frame (index (n - 1) // 2)       (configs/VINA_DINOv3.yaml, dataloader.py)
      ─ the image preprocessing above → logit per frame
      → mean of the frame logits → P(fake) = sigmoid(mean)
```

In SynthScope the two paths are two registry entries that share the model and the checkpoint:

* `vina` (image detector): the image path. Like every SynthScope image detector it also accepts a video, scored by the
  engine's generic frame sampling (16 frames spread over the clip, mean P(fake)), which is *not* the paper's protocol.
* `vina_video` (video detector): the official video protocol above, run by the engine's video path
  (`media.ffmpeg_fps_frames` + the image transform, sigmoid of the mean frame logit); `benchmark/bench_eval_video.py
  --model vina_video` uses it. It needs an ffmpeg binary, found as `$SYNTHSCOPE_FFMPEG`, else `ffmpeg` on the PATH,
  else the static build of imageio-ffmpeg. The SynthScope environment ships none: `pip install imageio-ffmpeg` or set
  `SYNTHSCOPE_FFMPEG=/path/to/ffmpeg`. Without one, `vina_video` is reported as unavailable. The parity below used the
  static ffmpeg 7.0.2 (the imageio-ffmpeg build); the whole clip is extracted because ffmpeg's JPEG rate control runs
  across frames, so other ffmpeg versions may encode the frames slightly differently.

Weights: `VINA_DINOv3-L_140k+DDA.pth` (HF `mgiant/VINA_checkpoints`, 1.21 GB, sha256 `1c46c9d8…c1d59d`) holds the
full model (all 415 backbone tensors + the head) and loads key-for-key as `vina.pth`; the gated DINOv3 weights are not
needed for inference.

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official `DINOv3` class of `models/DINOv3.py` | `SynthScopeVINAModel` subclasses it, so the checkpoint loads strictly; attribution and licence status in the header. |
| backbone built from its config (`DINOV3_VITL16`), not `from_pretrained('weights/dinov3-vitl16-pretrain-lvd1689m')` | the checkpoint already holds the backbone weights; `backbone_path` still loads a pretrained HF-format DINOv3 (for training). |
| ViT block prefix renamed on load (`backbone.layer` ↔ `backbone.model.layer`) | the checkpoint was saved with transformers 4.x; transformers 5 moved the blocks. |
| `remap_vina` unwraps `model_state_dict` and drops `module.` | the official `util.load_checkpoint` rule. |
| `forward` averages the logits of a `[B, T, 3, H, W]` clip | the official `util.eval_model` / `test.py` aggregation, so a clip is one call. |
| new transform steps `resize_short_256_bicubic` and `to_image` | the official eval is torchvision v2 `Resize(256, interpolation=3)` then `ToImage` + `ToDtype(float32, scale=True)`; `to_tensor`-style conversion differs from it by one float ulp, so the v2 order is kept for bit-exact inputs. |
| the video protocol uses the frame-directory path, not the `_mp4` path | the official benchmark keys (`genvideo`, `magic`, `reward`, ...) read ffmpeg frame directories; the `_mp4` keys decode with decord and, at test time too, pick a frame at `random.randint` (unseeded), so they are not reproducible. |
| image files of every format PIL reads | the official test loader only accepts `.jpg`, `.jpeg`, `.png`, `.webp` (other files become a black image); the pixels of the accepted formats are the same. |
| `compute_loss` = BCE with logits, `outputs` = `[-logit, logit]` | the official `util._classification_loss` for one output; the two columns let the trainer's metrics (argmax, softmax[:, 1]) read the single logit, as DDA / D3QE / B-Free. The training code is not released. |

## Fidelity check

The official code (`vina_ref` at commit 225d331, in a separate process with transformers 4.57.1, `test.py`'s
`build_inference_loader` + `infer_inputs` and configs/VINA_DINOv3.yaml) against SynthScope's full pipeline, both with
the same static ffmpeg 7.0.2:

* **images** (24: PNG, JPEG, WebP, BMP; 128×128 to 2048×1365, square and non-square): raw file →
  `bench_eval._open_rgb` → registry transform → engine-built model with `vina.pth` loaded by the engine → sigmoid;
* **videos** (8 GenVideo-Val clips: 3 MSR-VTT reals, Sora, Gen2, Crafter, Lavie, MorphStudio; 3 to 61 frames at
  2 fps): the official `preprocess/video2frame.py` then `test.py` on the frame directories, against SynthScope's
  `vina_video` on the mp4 (`engine.video_clip` → `engine._video_fake_prob`), with the served `select_frame_nums = 1`
  and with 8 (exercising the window, the last-frame padding of 3- to 5-frame clips and the logit mean).

Results (CPU, commit d0b87f6; logs `repro/logs/vinapar_cpu_{official,ours}_d0b87f6.log`):

* state dict: **all 417 tensors identical** (sha256 of the bytes; the only difference is the block-prefix name),
* model inputs: identical on all 24 images and on all 8 clips for both frame counts,
* P(fake) and logits: **max |diff| 0.0, 0 flips** on the 23 images the official loader reads (against its batch-1 run;
  1.2e-7 against its default batch-64 run, the official batch-64 vs batch-1 difference itself) and on all 8 videos for
  T = 1 and T = 8. The BMP file, which the official loader replaces by a black image, equals the official transform
  applied directly (max |diff| 0.0).

The official `_mp4` path (decord, a random frame) gives different scores on every run for the same clip (e.g. Sora
P(fake) 0.995963 / 0.997796 for two seeds vs 0.999641 from the frame directory); SynthScope follows the frame-directory
protocol.

Memory and speed (CPU, 16 threads on a shared login node, 256×256 input, fp32): 276 ms per image at batch 1 (3.6
img/s), 4.4 img/s at batch 8, peak RSS 2.96 GiB (`repro/logs/vinapar_cpu_speed_d0b87f6.log`); a `vina_video` clip
(ffmpeg extraction + one frame) took 0.80 s on average over the 8 parity clips.

## Training data and the checkpoint

The paper (section 4.1) trains three video-image combinations; the released checkpoint is the third, `140k+DDA`: the
**140k video set** (from Qwen2.5-ViT, the paper's reference [20]) together with the image data of **DDA** (the paper's
reference [9]). The paper samples one frame per video clip when it trains, so "140k" counts clips (videos), not frames.
The paper's training recipe: DINOv3-L, all parameters tuned end to end, 256×256 input (short side resized to 256, then a
256×256 crop), random horizontal flip as the only extra augmentation, AdamW for 10 epochs at a fixed learning rate of
1e-6, batch size 64, early stopping on validation loss, and a cross-modal supervised contrastive loss with weight 0.05
(temperature 0.07) next to the BCE loss. The official repository releases no training code, so SynthScope does not
train VINA (SynthScope_train "—"); `compute_loss` is the BCE term only. That the released file is the paper's
`140k+DDA` model is taken from the official README (it points to this checkpoint for the image and the video
benchmarks) and is supported by the digit-level agreement below on four benchmarks.

## Results

Target: the paper's tables for **VINA (140k+DDA)**, the released checkpoint. SynthScope_official = the released
checkpoint run through SynthScope (`bench_eval.py --model vina`, `bench_eval_video.py --model vina_video`);
SynthScope_train = "—" (see above).

**Metric.** The paper's means are the official code's: for every image benchmark, per-subset **pooled accuracy**
(`accuracy_score` over the subset's reals and fakes together, threshold 0.5) and average precision, averaged over the
subsets (`dataset/metrics.py`). The SynthScope columns use that metric: they are the `<tag>_offmetric` results,
SynthScope's per-file scores passed through the official metric functions (each call asserts it returns what the
official function returns). `bench_eval`'s own `acc` is a different number: the mean of per-subset *balanced* accuracy
(real and fake accuracy averaged). The two differ only where a subset is unbalanced (ForenSynths deepfake and SAN,
SynthWildX, the WildRF Twitter subset, and RRDataset's redigital subset); they are shown as "bAcc" where it matters.
The paper's Table 2 is headed "balanced ACC", but its official code computes the pooled accuracy; both are given for
those sets. GenVideo-Val is scored as the paper does: overall pooled accuracy and AP over all reals and all fakes, and
per generator recall, F1 and AP (Table 17).

| Benchmark | Paper ACC / AP | SynthScope_official ACC / AP | SynthScope_train | Status |
|---|---:|---:|:-:|---|
| **ForenSynths** (CNNDetection test, the paper's 11 subsets) | 86.99 / 97.78 (T12) | **86.93 / 97.80** | — | **reproduced** (-0.06 / +0.02); see the note on SAN below |
| **UniversalFakeDetect (LDM)** (8 sets) | 94.84 / 99.01 (T13) | **94.84 / 99.01** | — | **reproduced** (all 8 subsets equal; ldm_200 AP 100.00 against the paper's 99.99, within 0.01) |
| **DiTFake** (3 generators) | 97.98 / 99.95 (T14) | **97.98 / 99.95** | — | **reproduced** (all 3 subsets equal) |
| **ARForensics** (the 6 subsets that both our copy and the paper have) | 96.36 / 99.54 (mean of the paper's six per-subset values; its 7-subset mean with MAR is 96.74 / 99.59, T15) | **96.36 / 99.54** | — | **reproduced** (all 6 subsets equal); our copy has no MAR and has Switti, see below |
| **GenImage** (8 subsets, full val split, 100,000 images) | 94.74 (T11) | **94.74 / 99.05** | — | **reproduced** (all 8 subsets equal); AP is not in the paper |
| **GenVideo-Val** (10 generators, 18,584 videos, overall) | 96.9 / 99.26 (T1, T10) | **96.86 / 99.26** | — | **reproduced**; per-generator recall, F1 and AP equal Table 17 (see below) |
| SynthWildX (dalle3 / firefly / midjourney_v5) | 94.5 / 89.2 / 94.7 (T2) | 92.80 / 89.50 / 93.16 (mean 91.82 / 96.36; bAcc 91.76) | — | **below the paper** (-1.70 / +0.30 / -1.54); data, not code (see below) |
| WildRF (facebook / reddit / twitter) | 96.6 / 96.7 / 98.1 (T2) | 96.25 / 94.73 / 97.39 (mean 96.12 / 99.41; bAcc 95.94) | — | **below the paper** (-0.35 / -1.97 / -0.71); data, not code (see below) |
| AIGIBench (SocialRF / CommunityAI) | 90.9 / 93.9 (T2) | 89.72 / 95.73 (mean 92.72 / 97.33) | — | **differs from the paper** (-1.18 / +1.83); data, not code (see below) |
| RRDataset (mean of original / transfer / redigital) | 82.7 (T2) | 82.89 / 92.77 (bAcc 82.90) | — | **close** (+0.19); the paper gives the mean only |
| Chameleon | 91.4 (T2) | — | — | missing (available by request only) |
| Magic Videos · GenBuster++ · DeepTraceReward | 77.1 · 90.1 · 97.4 (T1) | — | — | missing (no data or loader in SynthScope) |

"T" is the table of the paper (arXiv 2605.21977). Table 2 prints the SynthWildX, WildRF and AIGIBench values per subset
and no mean; the means in the SynthScope column are ours. The mean of the paper's printed values is 92.8 (SynthWildX),
97.13 (WildRF) and 92.4 (AIGIBench); these are derived here, not printed in the paper.

**Verdict.** SynthScope's VINA is identical to the official code (see the fidelity check), and the released checkpoint
reproduces the paper's tables on the benchmarks whose data is the paper's: UniversalFakeDetect, DiTFake, GenImage (full
val split), ARForensics (six shared subsets) and GenVideo-Val in every subset, and ForenSynths in 10 of 11 (SAN differs
by 0.7, see below). Three of the in-the-wild sets differ from the paper by up to 2 points; the unmodified official
`eval.py` gives the same per-file scores as SynthScope on those files, so the gap is in the data (dataset version or
file list), not in the code.

### Per subset

ForenSynths (CNNDetection test; pooled ACC / AP; the paper's Table 12):

| Subset | Paper | SynthScope_official |
|---|---:|---:|
| progan | 95.21 / 99.35 | 95.21 / 99.35 |
| cyclegan | 93.15 / 99.52 | 93.15 / 99.52 |
| biggan | 95.05 / 99.81 | 95.05 / 99.81 |
| stylegan | 93.42 / 97.84 | 93.42 / 97.84 |
| gaugan | 96.06 / 99.96 | 96.06 / 99.96 |
| stargan | 98.20 / 99.97 | 98.20 / 99.97 |
| deepfake | 63.16 / 90.77 | 63.16 / 90.77 |
| seeingdark (SITD) | 85.28 / 95.43 | 85.28 / 95.43 |
| san | 83.79 / 95.16 | 83.05 / 95.44 |
| crn | 76.83 / 99.60 | 76.83 / 99.60 |
| imle | 76.79 / 98.12 | 76.79 / 98.12 |
| **mean** | **86.99 / 97.78** | **86.93 / 97.80** |

* Ten of the 11 subsets are equal to the second decimal. The two quirks of the paper's protocol reproduce: the official
  subset rule is a substring match on the path, so StyleGAN2 is pooled into StyleGAN and whichfaceisreal (no match) is
  left out, and deepfake's accuracy is the pooled 63.16 (its balanced accuracy is 63.10).
* **SAN.** The SAN real set holds 19 `.bmp` files. The official CSV builder (`preprocess/image2csv.py`) lists only
  `.jpg`, `.jpeg`, `.png` and `.webp`, so the headline excludes them: 90,310 files, SAN 83.05 / 95.44, mean **86.93 /
  97.80** (SynthScope's own balanced mean on those files: 86.98 / 97.80). The paper's SAN value (83.79 / 95.16) is
  matched exactly by a *reconstruction*: if the 19 `.bmp` reals are kept and each is scored as the black 256×256 image
  the official dataset class turns an unreadable file into (P(fake) 0.000884, the value measured for the official code in
  `repro/logs/vina_bmpblack_variant.log`), the SAN row is 83.79 / 95.16 and the mean is 86.99 / 97.77
  (`vina_cnndetection_bmpblack_offmetric`). That the authors' CSV listed those files is an inference from the digits, not
  something we observed, so it is not the headline.
* For all 13 CNNDetection folders (stylegan2 and whichfaceisreal separate), `bench_eval`'s balanced mean is 88.27 / 97.95
  (`vina_cnndetection`).

UniversalFakeDetect (LDM) (pooled ACC / AP; Table 13): the SynthScope numbers equal the paper's in every cell except
ldm_200's AP (100.00 against 99.99); the official `eval.py` also prints 100.00 for it.

| Subset | Paper | SynthScope_official |
|---|---:|---:|
| guided | 75.80 / 93.98 | 75.80 / 93.98 |
| ldm_200 | 99.30 / 99.99 | 99.30 / 100.00 |
| ldm_200_cfg | 99.30 / 99.99 | 99.30 / 99.99 |
| ldm_100 | 99.30 / 99.99 | 99.30 / 99.99 |
| glide_100_27 | 94.45 / 99.30 | 94.45 / 99.30 |
| glide_50_27 | 96.10 / 99.45 | 96.10 / 99.45 |
| glide_100_10 | 96.55 / 99.59 | 96.55 / 99.59 |
| dalle | 97.95 / 99.81 | 97.95 / 99.81 |
| **mean** | **94.84 / 99.01** | **94.84 / 99.01** |

Real pairing: guided against the ImageNet reals, the others against the LAION reals (Ojha's pairing, as `bench_eval`'s
`ufd` profile).

DiTFake (Table 14): FLUX.1-schnell 97.11 / 99.95, PixArt-Sigma-XL-2-1024-MS 100.00 / 100.00, stable-diffusion-3-medium
96.84 / 99.90, mean 97.98 / 99.95; SynthScope gives the same value in each of the 8 cells.

ARForensics (pooled ACC / AP; Table 15). **Our copy of ARForensics has 7 subsets: Infinity, Janus_Pro, LlamaGen,
Open_MAGVIT2, RAR, Switti and VAR.** The paper's seven are the same without Switti and with MAR, which our copy lacks.
The official metric configuration lists MAR and not Switti, so the `_offmetric` run scores the six shared subsets
(12,000 Switti files fall in no subset).

| Subset | Paper | SynthScope_official |
|---|---:|---:|
| Infinity | 99.84 / 100.00 | 99.84 / 100.00 |
| Janus_Pro | 99.90 / 100.00 | 99.90 / 100.00 |
| LlamaGen | 98.07 / 99.77 | 98.07 / 99.77 |
| MAR | 99.01 / 99.95 | not in our copy |
| Open_MAGVIT2 | 84.08 / 97.56 | 84.08 / 97.56 |
| RAR | 98.17 / 99.96 | 98.17 / 99.96 |
| VAR | 98.11 / 99.92 | 98.11 / 99.92 |
| mean of the six shared subsets | 96.36 / 99.54 (computed from the paper's values) | **96.36 / 99.54** |
| mean of the paper's seven (with MAR) | 96.74 / 99.59 | — |

Switti (not in the paper): 99.95 / 100.00 (`bench_eval`'s balanced accuracy). `bench_eval`'s own mean over our seven rows
is 96.88 / 99.60 (`vina_arforensics`); it averages a different set of subsets from the paper's 96.74 / 99.59, so the
two are not to be compared.

GenImage (Table 11; ACC only in the paper): equal in all 8 subsets on the full val split (100,000 images,
`vina_genimage_val100k`).

| Subset | Paper ACC | SynthScope_official ACC / AP |
|---|---:|---:|
| Midjourney | 96.11 | 96.11 / 99.79 |
| SD v1.4 | 99.89 | 99.89 / 100.00 |
| SD v1.5 | 99.85 | 99.85 / 100.00 |
| ADM | 76.23 | 76.23 / 93.21 |
| GLIDE | 94.65 | 94.65 / 99.67 |
| Wukong | 99.83 | 99.83 / 100.00 |
| VQDM | 94.98 | 94.98 / 99.77 |
| BigGAN | 96.38 | 96.38 / 99.92 |
| **mean** | **94.74** | **94.74 / 99.05** |

`vina_genimage_full` is a different, smaller run: the **test bundle** with 1500 images per class and subset (33,000 images;
SD v1.4 has 6000), an earlier 1,500-per-class subsample run. It gives 93.29 / 98.45 and is not the
paper's GenImage number (ADM 70.10, GLIDE 92.73, VQDM 92.07, BigGAN 95.37, Midjourney 96.43, SD v1.4 99.89, SD v1.5
99.90, Wukong 99.83).

In-the-wild sets (Table 2; pooled ACC, with the balanced accuracy in brackets):

| Set · subset | Paper | SynthScope_official ACC / AP (bAcc) |
|---|---:|---:|
| SynthWildX · dalle3 | 94.5 | 92.80 / 95.89 (92.96) |
| SynthWildX · firefly | 89.2 | 89.50 / 96.03 (89.32) |
| SynthWildX · midjourney_v5 | 94.7 | 93.16 / 97.17 (93.00) |
| WildRF · facebook | 96.6 | 96.25 / 99.43 (96.25) |
| WildRF · reddit | 96.7 | 94.73 / 99.09 (94.73) |
| WildRF · twitter | 98.1 | 97.39 / 99.73 (96.85) |
| AIGIBench · SocialRF | 90.9 | 89.72 / 95.51 (89.72) |
| AIGIBench · CommunityAI | 93.9 | 95.73 / 99.14 (95.72) |
| RRDataset · original | — | 94.08 / 98.57 (94.08) |
| RRDataset · transfer | — | 84.87 / 94.95 (84.87) |
| RRDataset · redigital | — | 69.73 / 84.80 (69.74) |
| RRDataset · mean | 82.7 | 82.89 / 92.77 (82.90) |

SynthWildX shares one real set (413 images) between its three fake sets. Chameleon (paper 91.4) is not held. RRDataset's
redigital scenario is weak: real accuracy 97.75, fake accuracy 41.72.

**Why SynthWildX, WildRF and AIGIBench differ from the paper.** The unmodified official `eval.py` was run on the same
files (SynthWildX and WildRF on an RTX A5000, AIGIBench on an RTX A6000) and its per-file P(fake) was compared with
SynthScope's:

| Set | files | max abs diff | mean abs diff | decision flips at 0.5 | official-code means (ACC / AP) | SynthScope means (ACC / AP) |
|---|---:|---:|---:|---:|---:|---:|
| SynthWildX | 1,713 | 1.7e-05 | 2.2e-07 | 0 | 91.82 / 96.36 | 91.82 / 96.36 |
| WildRF | 2,241 | 1.5e-05 | 2.3e-07 | 0 | 96.12 / 99.41 | 96.12 / 99.41 |
| AIGIBench | 18,000 | 5.8e-05 | 5.9e-07 | 0 | 92.72 / 97.33 | 92.72 / 97.33 |

The official code and SynthScope agree to within 6e-05 on every file, and every per-subset number is equal, so SynthScope
is not the cause of the gap to the paper; the cause is the data (the dataset version or the file list). Our SynthWildX
copy holds 1,713 of the 2,000 images (413 reals and 1,300 fakes; the paper's Table 9 lists 500 and 1.5K), our WildRF
copy has 1,054 reals and 1,187 fakes after leaving out 262 `.jfif` files (Table 9 lists 500 and 500), and AIGIBench is
the Hugging Face `HorizonTEL/AIGIBench` release with the paper's count (9k reals and 9k fakes), probably another release
or file list. We did not find which file lists the authors used.

GenVideo-Val (`vina_video`, the official protocol: ffmpeg at 2 fps, the central frame, 18,584 videos = 10,000 real
MSR-VTT clips and 8,584 fakes; Table 17, recall / F1 / AP):

| Generator | Paper | SynthScope_official |
|---|---:|---:|
| sora | 75.00 / 29.79 / 56.82 | 75.00 / 29.79 / 56.82 |
| morphstudio | 99.86 / 88.31 / 99.74 | 99.86 / 88.31 / 99.74 |
| gen2 | 100.00 / 93.75 / 99.96 | 100.00 / 93.75 / 99.96 |
| hotshot | 94.29 / 85.49 / 95.90 | 94.29 / 85.49 / 95.90 |
| lavie | 97.71 / 92.68 / 99.03 | 97.71 / 92.68 / 99.03 |
| show1 | 96.29 / 86.52 / 97.53 | 96.29 / 86.52 / 97.53 |
| moonvalley | 100.00 / 87.19 / 99.97 | 100.00 / 87.19 / 99.97 |
| crafter | 99.79 / 93.72 / 99.87 | 99.79 / 93.72 / 99.87 |
| modelscope | 98.71 / 87.75 / 99.08 | 98.71 / 87.75 / 99.08 |
| wildscrape | 70.35 / 73.95 / 80.42 | 70.35 / 73.95 / 80.42 |
| mean over generators | 93.20 / 81.52 / 92.83 | 93.20 / 81.91 / 92.83 |
| overall pooled ACC / AP | 96.9 / 99.26 | 96.86 / 99.26 |

The paper's printed mean F1 (81.52) is not the mean of its own column (81.92); ours is the mean of ours (81.91). The
official `eval.py` run on GPU (RTX A5000) gives the same numbers as SynthScope, with a maximum per-file difference of
4.5e-05 and no flipped decision on all 18,584 videos. `bench_eval_video`'s own per-generator mean (balanced accuracy
over each generator against the shared reals) is 95.68 / 92.83 (`vina_video_genvideo`).

The UniversalFakeDetect (LDM) run was compared with the official `eval.py` as well (16,000 files, max abs diff 4.9e-05,
0 flips; official means 94.84 / 99.01).

### Protocol and runs

* **Model and input.** `VINA_DINOv3-L_140k+DDA.pth` as `vina.pth`, fp32, the registered eval transform (short side to
  256 bicubic, center crop 256, ImageNet normalization), no test-time JPEG; P(fake) = sigmoid(logit) thresholded at 0.5.
* **Files.** The official CSV builder lists `.jpg`, `.jpeg`, `.png` and `.webp` only, and every run uses that file set
  (other extensions removed; counts are in the logs). Subsets follow the official rule (the first subset name, longest
  first, that is a substring of the file's path).
* **Videos.** `video2frame.py`'s frame extraction (ffmpeg 7.0.2 static build, 2 fps, JPEG at ffmpeg defaults) and the
  central frame, as the paper ("we select the middle frame of each video"); the WildScrape set is the 924-video set
  (the paper's recall 70.35 = 650/924).
* **Scoring.** `bench_eval.py --model vina` and `bench_eval_video.py --model vina_video` on the full data (wrapped by a
  recorder that stores the per-file P(fake)); the `_offmetric` tags apply the official metric functions to those scores.
  Tags map to columns as follows: `vina_<bench>` = `bench_eval`'s own `acc` (mean balanced accuracy) with mAP and
  mAUROC; `vina_<bench>_offmetric` = the official pooled accuracy and AP, the numbers compared with the paper above.
  `vina_genimage_full` is the 1500-per-class test bundle; the full split is `vina_genimage_val100k`.
* **Official code on the same files.** The official `eval.py` was run on SynthWildX, WildRF, AIGIBench, UniversalFakeDetect
  and GenVideo-Val; `repro/bench_results/vina_official_code/*.json` hold the per-file comparison (agreement figures and the
  means of both). The official per-file CSVs are not in the repository (the largest are 1.4 to 1.7 MB); the JSONs name them
  by paths relative to the shared working directory (`vina_official_out/...`) and they are kept outside the repo. It was not
  run on ForenSynths, DiTFake, ARForensics, GenImage or RRDataset; for those the evidence is the digit-level agreement with
  the paper (and, for ARForensics, GenImage and ForenSynths, the file-level parity above).
* **Code and jobs.** The CPU runs are on commit 0bd0597 and the GPU runs on 0954488 (0 uncommitted tracked files in every
  log header). Logs are `repro/logs/vinabench_<job>.out`.

| job | where | what |
|---|---|---|
| 281654 | cn023, CPU | UniversalFakeDetect (8 sets), DiTFake, AIGIBench |
| 281655 | cn023, CPU | ForenSynths (all 13 CNNDetection folders) |
| 281656 | cn023, CPU | GenVideo-Val (`vina_video`) |
| 281831 | cn023, CPU | GenImage test bundle (1500 per class) |
| 281657 | IVI cn029, RTX A5000 | SynthWildX, WildRF, RRDataset; official `eval.py` on SynthWildX and WildRF |
| 66630 | DAS6, RTX A6000 | ARForensics, GenImage full val split (100,000 images) |
| 66627 | DAS6, RTX A6000 | official `eval.py` on AIGIBench and UniversalFakeDetect |
| 281662 | IVI cn028, RTX A5000 | official GenVideo-Val pipeline (`video2frame.py` + `eval.py`) |

The AIGIBench, UniversalFakeDetect and GenVideo-Val comparisons above are SynthScope CPU scores (cn023) against official
GPU scores (max abs diff 5.8e-05, 0 flips), so they also check CPU against GPU.

### Where it is weak

* ForenSynths: deepfake 63.16 (fake accuracy 26.32), CRN and IMLE 76.8 (real accuracy 53.65, fake accuracy about 100),
  SAN and SITD 83 to 85.
* UniversalFakeDetect: guided 75.80 (fake accuracy 51.6).
* GenImage: ADM 76.23 (fake accuracy 52.63); on the 1500-per-class bundle ADM is 70.10.
* ARForensics: Open_MAGVIT2 84.08 (fake accuracy 68.23).
* GenVideo-Val: Sora (75.00 recall on 56 clips) and WildScrape (70.35); the overall accuracy hides them because the
  reals dominate.
* In the wild: RRDataset redigital 69.73 (fake accuracy 41.72), SynthWildX firefly 89.50 (real accuracy 86.68).

## Caveats

* **Licence unclear.** The official repository (commit 225d331) has no LICENSE file and the README states no licence;
  the Hugging Face checkpoint repository has no licence tag. VINA is therefore not redistributed: not in the SynthScope
  model zoo, not on the Space, and never served. `LICENSES.md` records it as "unclear". The vendored `official_arch.py` is
  kept with attribution for research use.
* **The checkpoint embeds the gated DINOv3 ViT-L/16 weights** (DINOv3 License). Inference needs only `vina.pth`; the gated
  download is needed only to train.
* **Training is not reproduced.** The training code is not released; the numbers are for the released checkpoint.
* **Not reproduced:** Chameleon, Magic Videos, GenBuster++ and DeepTraceReward (no data or loader in SynthScope).
* **Data differences.** SynthWildX, WildRF and AIGIBench use the dataset versions we hold (see above); ARForensics has
  Switti and no MAR.
* **`vina` on a video.** Like every image detector, `vina` scores a video by averaging 16 frames spread over the clip, which
  is not the paper's protocol; use `vina_video` for it.
* **Memory and speed** (forward pass only, fp32, 256×256, no decoding, from the probe at the start of each GPU job):

| GPU | batch 1 | batch 32 | peak allocated (batch 1 / 32) |
|---|---:|---:|---:|
| RTX A5000 24 GB (IVI cn028, cn029) | 23.0 ms (43.4 img/s) | 501.6 ms (63.8 img/s) | 1.157 / 1.677 GiB |
| RTX A6000 48 GB (DAS6) | 17.4 ms (57.2 to 57.5 img/s) | 376 to 377 ms (84.9 to 85.1 img/s) | 1.157 / 1.677 GiB |

  Whole-benchmark throughput including decoding: 14 to 26 img/s on the A5000 (SynthWildX 17.97, WildRF 13.87, RRDataset
  26.46) and 29.7 to 35.0 img/s on the A6000 (ARForensics 34.97, GenImage 29.74); 1.27 to 1.66 img/s on 10 CPUs of cn023 (fp32).
  `vina_video` end to end (24 clips): 1.28 to 2.38 clips/s on the A5000 (the spread is not explained in the logs), 2.25
  to 2.31 on the A6000, peak allocated 2.290 GiB; 1.36 clips/s on the CPU.

## Usage

```bash
synthscope detect photo.png --model vina          # needs vina.pth (= VINA_DINOv3-L_140k+DDA.pth) in $SYNTHSCOPE_CKPT_DIR
synthscope detect clip.mp4 --model vina_video     # the paper's video protocol; needs ffmpeg (see above)
python benchmark/bench_eval.py --model vina --benchmark pergen --data /data/GenImage_val_full/test --bs 32
python benchmark/bench_eval_video.py --model vina_video --dataset genvideo --root /data/GenVideo
```
