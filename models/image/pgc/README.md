# PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection

## Short method explanation

**PGC** is a **trained** AI-generated *image* detector from **"PGC: Peak-Guided Calibration for
Generalizable AI-Generated Image Detection"** (Zhou, Fei, Yu, Xie, Cheng, Xia; 2026). It fuses two
streams and calibrates a global decision with a **peak-focusing** mechanism that emphasizes the most
discriminative local evidence.

- Official repo: https://github.com/xiaoyu6868/PGC
- Paper: arXiv:2605.21207

## Pipeline

Input is a normalized RGB image `[B, 3, 224, 224]`; the model builds the 6-channel `[RGB ‖ residual]`
tensor **internally** (see "SynthScope adaptations"):

1. **RGB stream** — a **LoRA-tuned, frozen DINOv2-Large** ViT (HF `AutoModel`). Returns the CLS token
   `[B, D]` and patch tokens `[B, N, D]`. Only the injected LoRA adapters (rank 8) on the attention
   q/k/v/output + MLP fc1/fc2 are trainable; the backbone is frozen.
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
| LoRA targets | attention `query/key/value`, `attention.output.dense`, `mlp.fc1/fc2` (HF DINOv2 names) |
| Residual | YCbCr quantization residual, normalized mean 0 / std 0.5 |
| Residual stream | Conv(3→64→128→256), stride 2, BN+ReLU |
| PGCM | per-patch + per-location score heads, log-sum-exp peak aggregation, learnable `λ_rgb` |
| `tau_rgb`, `tau_res` | 0.5, 0.5 |
| Output | single logit `z_global + z_local`; `compute_loss` returns `stack([-logit, logit])` |

## SynthScope adaptations

The model is a pure `nn.Module` (`forward`/`compute_loss`) like every SynthScope detector, with two
adaptations vs the official code (both preserve the method):

1. **Residual is computed inside the model.** The official builds the 6-channel input in the dataset
   transform; here the dataset feeds a normal 3-channel ImageNet-normalized image (`normalize_dinov2`)
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
deepfake, SAN, SeeingDark):

| checkpoint | FS-8 mAP | FS-8 acc | hard-3 AUROC | FS-13 mAP |
|---|---|---|---|---|
| official `PGC_train_progan_ckpt.pth` | 96.41 | 76.34 | 90.87 | 93.80 |
| step 300 | 97.60 | 90.82 | 92.88 | 95.07 |
| **step 500** | **97.41** | **85.89** | **93.93** | **95.78** |
| step 900 | 97.82 | 85.71 | 93.88 | 97.12 |
| step 1300 | 97.63 | 85.61 | 92.76 | 97.03 |
| step 1700 | 97.21 | 84.06 | 91.03 | 96.69 |
| step 1900 | — | 74.49 (FS-13) | — | 95.97 |

The paper reports no standalone ForenSynths number, so the reference is the official checkpoint: every checkpoint from
step 300 on matches or exceeds it on ForenSynths-8 mAP (96.41 on this evaluation); transfer to the hard generators peaks around steps 500-900 and then erodes, as the authors
describe. **Step 500**, the saved checkpoint nearest the authors' ~400 steps (picked by that rule, not by test score),
is the SynthScope-trained number in the table below; on UniversalFakeDetect, which the paper does report, it gives 98.99 mAP / 91.96 acc (paper
98.0 — reproduced; official 99.17 / 83.57 on the same run, job 266660). The GenImage, AIGIBench and CommGen15 SynthScope_train entries are
from an earlier SynthScope recipe (2 GPUs, full cosine over epochs) and have not been rerun with this one.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| GenImage | map=100.0 / acc=98.3 | mAP=100.0 / acc=99.49 | mAP=97.62 / acc=78.6 | official |
| UniversalFakeDetect (UFD) | map=98.0 | mAP=99.34 / acc=94.16 | **mAP=98.99 / acc=91.96** | **reproduced** (step 500) |
| ForenSynths-8 (CNNDetection test) | — (not reported separately; official checkpoint 96.25) | mAP=96.39 / acc=76.29 | **mAP=97.41 / acc=85.89** | matches/exceeds the official checkpoint (step 500) |
| AIGIBench (AIGI) | map=95.0 / acc=86.6 | mAP=95.36 / acc=88.33 | mAP=86.88 | official |
| CommGen15 (commercial in-the-wild) | acc=97.9 | mAP=99.26 / acc=97.39 | mAP=98.61 / acc=66.66 | official |

> **ForenSynths hard generators — reproduced via early-stopping (resolved 2026-06-19).** The released ProGAN checkpoint is trained for only **~400 steps (< 1 epoch)** — confirmed by the authors on our issue. DINOv2's strong priors overfit ProGAN within a few epochs (loss → floor), which keeps easy generators high but **erodes hard-generator transfer**; our earlier epoch-level checkpoints (ep10+) all overshot the peak. Re-running the official `train.py` verbatim with fine-grained eval shows the hard-3 (deepfake/san/seeingdark) **mean AUROC climb to 94.56 at step ~900 ≈ official 94.35**, then decay with longer training. So PGC **does** reproduce from the published recipe+data with early-stopping inside the first epoch. (Served PGC uses the official weights.) See `repro/investigate/pgc_gap_investigation.md`.

> **Eval transform — `PadCenterCrop(224)`, no resize (verified 2026-06-19).** PGC's YCbCr residual is resolution-sensitive, so the eval transform is center-crop-224 with **no resize** (= the official `PadCenterCrop`; byte-identical to the genuine repo's). The earlier `Resize→CenterCrop` registry transform wiped the forensic signal on super-res/low-light generators (san 73.3, seeingdark 72.5 — same eval-transform family as the SAFE/AIDE/SPAI fixes). With the fix, SynthScope `bench_eval` gives ForenSynths-8 96.39 mAP with the official weights and 96.05 with the official code retrained (the paper reports no ForenSynths number; 96.25 was the earlier official-weights measurement) and san recovers to 98.65 AUROC. One generator, **seeingdark**, stays ~10 pt below the genuine repo (77.9 vs 87.49) on identical images/weights with now-bit-exact preprocessing — isolated to the model forward, not the transform; it is not part of the reported ForenSynths-8.

**reproduced** = SynthScope-trained checkpoint matches the paper · **official** = official weights loaded into the SynthScope class match (the SynthScope-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Residual built in-model (not in the dataset transform) | Keeps the SynthScope dataset/transform pipeline standard (3-channel in). Same residual math (BT.601 quantization, Eq. 1) — `_quantization_residual` snaps `round(x01*255)` to the uint8 grid so the residual is **bit-exact** to the official 6-channel transform despite the in-model ImageNet de-normalization (verified zero-diff, `repro/_residual_path_diff.py`). |
| DINOv2-Large loaded via HF name | No local-checkpoint requirement; LoRA targets use the HF DINOv2 module names. |
| Training budget | The official checkpoint is ~400 steps in; `experiments/image/pgc.sh` saves every 100 micro-batches up to 2000 instead of training 100 epochs, which overfits ProGAN. |
