# PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection

## Short method explanation

**PGC** is a **trained** AI-generated *image* detector from **"PGC: Peak-Guided Calibration for
Generalizable AI-Generated Image Detection"** (Zhou, Fei, Yu, Xie, Cheng, Xia; 2026). It fuses two
streams and calibrates a global decision with a **peak-focusing** mechanism that emphasizes the most
discriminative local evidence.

- Official repo: https://github.com/xiaoyu6868/PGC
- Paper: arXiv:2605.21207

> **Correction (2026-11-09).** From 2026-06-15 to 2026-11-09 the served PGC (`pgc.pth` = the official `PGC_train_progan_ckpt.pth`)
> did not load the official checkpoint's `rgb_stream.*` tensors: the checkpoint prefix was not remapped (the official network keeps
> the DINOv2 backbone under `rgb_stream.`, the SynthScope class under `backbone.`) and the load did not fail. Only the 28 head
> tensors loaded and every trained LoRA tensor was dropped, so the model was the official network without its LoRA. On the
> ProGAN-trained benchmarks this collapsed accuracy and left mAP near-saturated: UniversalFakeDetect accuracy was 83.57 instead of
> 94.19 and mAP 99.17 instead of 99.35. The load is fixed (`remap_pgc` in `interface/backend/registry.py`; the SynthScope class now
> matches the official code to a max probability difference of 0.0 on 24 images, `tests/test_pgc_load.py`), and every affected number
> below was rescored (tags with the suffix `_fixed`; the old runs are in `repro/bench_results/_superseded/`). SynthScope-trained
> checkpoints (`backbone.*` keys) were never affected. The effect was not one-directional: on the generator benchmarks (FLUX.2 klein,
> Qwen-Image, Ideogram) the broken model had a higher AUROC than the fixed one (FLUX.2 klein 4B 97.33 against 90.70 now), while its
> accuracy and fake-detection rate were lower on Qwen-Image and Ideogram (about equal on FLUX.2 klein). The earlier "known
> model-forward difference" and the seeingdark gap of about 10 points were this
> bug, not a model difference. See `repro/investigate/pgc_gap_investigation.md`.

## Pipeline

Input is a normalized RGB image `[B, 3, 224, 224]`; the model builds the 6-channel `[RGB ‖ residual]`
tensor **internally** (see "SynthScope adaptations"):

1. **RGB stream** — a **LoRA-tuned, frozen DINOv2-Large** ViT (HF `AutoModel`). Returns the CLS token
   `[B, D]` and patch tokens `[B, N, D]`. Only the injected LoRA adapters (rank 8) on the MLP fc1/fc2
   are trainable (as in the released checkpoints, whose attention target names do not match HF DINOv2); the
   backbone is frozen.
2. **Residual stream** — a **YCbCr quantization residual** (BT.601 round-trip rounding error, paper
   Eq. 1), fed through a 3-block stride-2 CNN → feature map `[B, 256, Hr, Wr]`, global-avg-pooled to
   `[B, 256]`.
3. **PGCM (peak-guided calibration)** — per-patch scores (`Linear`) and per-location residual scores
   (`1×1 Conv`) are aggregated with a smooth-max **log-sum-exp** "peak" operator
   `z = τ·(logsumexp(s/τ) − log N)`, giving a local bias `z_local = z_res + λ_rgb · z_rgb` (Eq. 4).
4. **Global head** — `z_global = Linear(L2norm([CLS ‖ pooled-residual]))`.
5. **Decision** — `y = z_global + z_local` (single logit, Eq. 5). BCE-with-logits; label `1 = fake`.

## Model details

Implemented in `models/image/pgc/pgc.py` as `SynthScopePGCModel`.

| Component | Setting |
|---|---|
| RGB backbone | `facebook/dinov2-large` (HF), frozen + **LoRA** (rank 8, α 1.0, dropout 0.1) |
| LoRA targets | `mlp.fc1`, `mlp.fc2` only (`lora_targets` in `conf/model/image/pgc.yaml`; the released checkpoints carry LoRA on the MLP only) |
| Residual | YCbCr quantization residual, normalized mean 0 / std 0.5 |
| Residual stream | Conv(3→64→128→256), stride 2, BN+ReLU |
| PGCM | per-patch + per-location score heads, log-sum-exp peak aggregation, learnable `λ_rgb` |
| `tau_rgb`, `tau_res` | 0.5, 0.5 |
| Output | single logit `z_global + z_local`; `compute_loss` returns `stack([-logit, logit])` |

## SynthScope adaptations

The model is a pure `nn.Module` (`forward`/`compute_loss`) like every SynthScope detector, with two
adaptations vs the official code (both preserve the method):

1. **Residual is computed inside the model.** The official builds the 6-channel input in the dataset
   transform; here the dataset feeds a normal 3-channel ImageNet-normalized image (`normalize_imagenet`)
   and `forward` de-normalizes to `[0,1]`, computes the YCbCr residual, and concatenates — so the
   dataset/transform pipeline stays standard.
2. **Backbone loads by Hugging Face name** (auto-download) instead of a local checkpoint path.

The residual is resolution-sensitive, so the experiment uses **crop-only** transforms (no resize):
`[random_crop_224, …]` train / `[center_crop_224, …]` eval.

## Training setup

```bash
conf/model/image/pgc.yaml     # model_type: pgc, backbone_name, lora_*, tau_rgb/res, label smoothing 0.1
conf/trainer/image/pgc.yaml   # AdamW 5e-5, wd 0.05, 32 x 4 accumulation, clip 1.0, step budget (checkpoint every 100)
experiments/image/pgc.sh      # the official ProGAN recipe (all 20 classes); see "Training in SynthScope" below
```

Registered in `trainer.py` under `model_type == "pgc"`. Trainable params: LoRA adapters + residual
stream + PGCM + classifier (DINOv2 backbone frozen). The paper's benchmarks are CommGen15, GenImage,
AIGI, and UniversalFakeDetect; CommGen15 (the paper's new dataset) is optional — the SynthScope integration
trains on the existing ProGAN/GenImage data.

## Training in SynthScope

`experiments/image/pgc.sh` runs the official ProGAN recipe (`train.py` + `engine/trainer.py`, README 3.2) through
SynthScope's trainer: all 20 ProGAN classes (720k images, = the UniversalFakeDetect train set); flip → ColorJitter
(0.15, 0.15, 0.15, 0.05) p 0.2 → pad + RandomCrop 224 → ImageNet normalization, residual computed in the model; AdamW
5e-5, weight decay 0.05, label smoothing 0.1, micro-batch 32 × 4 accumulation, gradient clipping 1.0, seed 42, one GPU.
The authors' released checkpoint is only **~400 steps** in, because DINOv2's prior overfits ProGAN within the first
epoch, so the run is a budget of 2000 micro-batches (trainer `step_budget`) with a checkpoint every 100, and the early
curve is evaluated. (The official cosine schedule spans 100 epochs and is stepped per optimizer step, so the learning
rate stays at 5e-5 here.) The run takes 38 minutes on one A5000 (job 266276).

ForenSynths with the same `bench_eval` command for every checkpoint (job 266282; FS-8 = the paper's 8 GANs, hard-3 =
deepfake, SAN, SeeingDark). The official row is the corrected rescore (`pgc_official_forensynths13_fixed`, DAS6 job 66717, 2026-11-09);
the earlier row (FS-8 96.41 / 76.34, hard-3 90.87, FS-13 93.80) had the LoRA dropped. The SynthScope-trained rows load correctly and are unchanged:

| checkpoint | FS-8 mAP | FS-8 acc | hard-3 AUROC | FS-13 mAP |
|---|---|---|---|---|
| official `PGC_train_progan_ckpt.pth` | 97.83 | 89.17 | 94.34 | 96.25 |
| step 300 | 97.60 | 90.82 | 92.88 | 95.07 |
| **step 500** | **97.41** | **85.89** | **93.93** | **95.78** |
| step 900 | 97.82 | 85.71 | 93.88 | 97.12 |
| step 1300 | 97.63 | 85.61 | 92.76 | 97.03 |
| step 1700 | 97.21 | 84.06 | 91.03 | 96.69 |
| step 1900 | — | 74.49 (FS-13) | — | 95.97 |

The paper reports no standalone ForenSynths number, so the reference is the official checkpoint. With the load fixed, the
SynthScope-trained checkpoints land at or just below it: FS-8 mAP 97.21 to 97.82 against 97.83 (step 900 is 0.01 below), accuracy
84.06 to 90.82 against 89.17 (only step 300 is above), and hard-3 AUROC 91.03 to 93.93 against 94.34 (every step below). On the full 13-generator set the picture differs: FS-13 mAP is
above the official 96.25 for steps 700 to 1700 (96.67, 97.12, 96.74, 97.03, 96.85, 96.69; `pgc_trained_step*_forensynths13.json`) and
below it at steps 300, 500 and 1900 (95.07, 95.78, 95.97), while FS-13 accuracy (74.18 to 82.98) stays below the official 83.76. Transfer to
the hard generators peaks around steps 500-900 and then erodes, as the authors describe. **Step 500**, the saved checkpoint nearest
the authors' ~400 steps (picked by that rule, not by test score), is the SynthScope-trained number in the table below; on
UniversalFakeDetect, which the paper does report, it gives 98.99 mAP / 91.96 acc (paper 98.0, reproduced; the official weights
give 99.35 / 94.19 on the corrected rescore, `pgc_official_ufd8_fixed`, so step 500 is 0.36 mAP and 2.2 acc below them). The earlier
claim here that every checkpoint from step 300 on "matches or exceeds" the official one compared against the broken load
(96.41 / 76.34 / 90.87) and is withdrawn. The GenImage, AIGIBench and CommGen15 SynthScope_train entries are
from an earlier SynthScope recipe (2 GPUs, full cosine over epochs) and have not been rerun with this one.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| GenImage | map=100.0 / acc=98.3 | mAP=100.0 / acc=99.49 | mAP=97.62 / acc=78.6 | official |
| UniversalFakeDetect (UFD) | map=98.0 | mAP=99.35 / acc=94.19 | **mAP=98.99 / acc=91.96** | **reproduced** (step 500) |
| ForenSynths-8 (CNNDetection test) | — (not reported separately) | mAP=97.83 / acc=89.17 | mAP=97.41 / acc=85.89 | reference; step 500 is slightly below the official checkpoint |
| AIGIBench (AIGI) | map=95.0 / acc=86.6 | mAP=95.36 / acc=88.33 | mAP=86.88 | official |
| CommGen15 (commercial in-the-wild) | acc=97.9 | mAP=99.26 / acc=97.39 | mAP=98.61 / acc=66.66 | official |

SynthScope_official uses the release that matches each benchmark's training setting (`repro/benchmark_specs/pgc.json`): `PGC_train_sdv1_4_ckpt.pth` for GenImage and CommGen15, `PGC_train_progan_ckpt.pth` (the served `pgc.pth`) for UFD and ForenSynths, `PGC_train_progan_sdv1_4_ckpt.pth` for AIGIBench. The step-500 numbers are from jobs 266660 and 266282 (PadCenterCrop eval); the UFD and ForenSynths-8 SynthScope_official numbers are the corrected rescore of the served weights with the same eval (`pgc_official_ufd8_fixed`, `pgc_official_forensynths13_fixed`, DAS6 job 66717, 2026-11-09). GenImage, AIGIBench and CommGen15 are official-code numbers: 2026-06-04 runs of the official PGC code (`repro/official_evals/pgc_official_eval.py`, PadCenterCrop). The same official-code runs gave UFD 99.34 / 94.16 and FS-13 96.25 / 83.73; the SynthScope class on the same weights now gives 99.35 / 94.19 and 96.25 / 83.76 (UFD / FS-13), i.e. the official code within 0.03 on every comparable cell. (Before 2026-11-09 it gave 99.17 / 83.57 and 93.80 / 70.58, which was the dropped-LoRA load, not a model-forward difference.)

> **ForenSynths hard generators — reproduced via early-stopping (resolved 2026-06-19).** The released ProGAN checkpoint is trained for only **~400 steps (< 1 epoch)** — confirmed by the authors on our issue. DINOv2's strong priors overfit ProGAN within a few epochs (loss → floor), which keeps easy generators high but **erodes hard-generator transfer**; our earlier epoch-level checkpoints (ep10+) all overshot the peak. Re-running the official `train.py` verbatim with fine-grained eval shows the hard-3 (deepfake/san/seeingdark) **mean AUROC climb to 94.56 at step ~900 ≈ official 94.35**, then decay with longer training. So PGC **does** reproduce from the published recipe+data with early-stopping inside the first epoch. (Served PGC uses the official weights.) See `repro/investigate/pgc_gap_investigation.md`.

> **Eval transform — `PadCenterCrop(224)`, no resize (verified 2026-06-19).** PGC's YCbCr residual is resolution-sensitive, so the eval transform is center-crop-224 with **no resize** (= the official `PadCenterCrop`; byte-identical to the genuine repo's). The earlier `Resize→CenterCrop` registry transform wiped the forensic signal on super-res/low-light generators (san 73.3, seeingdark 72.5 — same eval-transform family as the SAFE/AIDE/SPAI fixes). The numbers quoted on 2026-06-19 for this fix (ForenSynths-8 96.39 mAP with the official weights, 96.05 with the official code retrained, seeingdark about 10 pt below the genuine repo, 77.9 vs 87.49) were scored with the dropped-LoRA load (see the correction at the top) and are superseded. With the load fixed: the official weights give FS-8 97.83 mAP / 98.14 mAUROC and the hard-3 AUROC deepfake 96.48 / san 99.06 / seeingdark 87.49, equal to the genuine repo's evaluator, so **the seeingdark gap was the load bug and is closed**. The official `train.py` retrained here and stopped at step 900 (`pgc_evalfix_step900_fixed`) gives FS-8 98.01 mAP, FS-13 97.34 mAP / 80.68 acc, hard-3 AUROC deepfake 93.31 / san 99.25 / seeingdark 91.13 (mean 94.56), equal to the official evaluator's step-900 values on the same checkpoint (93.29 / 99.25 / 91.13); the paper reports no ForenSynths number, and the official PGC code's own FS-13 mean for the released checkpoint is 96.25.

**reproduced** = SynthScope-trained checkpoint matches the paper · **official** = official weights loaded into the SynthScope class match (the SynthScope-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Residual built in-model (not in the dataset transform) | Keeps the SynthScope dataset/transform pipeline standard (3-channel in). Same residual math (BT.601 quantization, Eq. 1) — `_quantization_residual` snaps `round(x01*255)` to the uint8 grid so the residual is **bit-exact** to the official 6-channel transform despite the in-model ImageNet de-normalization (verified zero-diff, `repro/_residual_path_diff.py`). |
| DINOv2-Large loaded via HF name | No local-checkpoint requirement; LoRA targets use the HF DINOv2 module names. |
| Training budget | The official checkpoint is ~400 steps in; `experiments/image/pgc.sh` saves every 100 micro-batches up to 2000 instead of training 100 epochs, which overfits ProGAN. |
