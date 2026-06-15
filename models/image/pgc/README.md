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
tensor **internally** (see "AGID adaptations"):

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

Implemented in `models/image/pgc/pgc.py` as `AGIDPGCModel`.

| Component | Setting |
|---|---|
| RGB backbone | `facebook/dinov2-large` (HF), frozen + **LoRA** (rank 8, α 1.0, dropout 0.1) |
| LoRA targets | attention `query/key/value`, `attention.output.dense`, `mlp.fc1/fc2` (HF DINOv2 names) |
| Residual | YCbCr quantization residual, normalized mean 0 / std 0.5 |
| Residual stream | Conv(3→64→128→256), stride 2, BN+ReLU |
| PGCM | per-patch + per-location score heads, log-sum-exp peak aggregation, learnable `λ_rgb` |
| `tau_rgb`, `tau_res` | 0.5, 0.5 |
| Output | single logit `z_global + z_local`; `compute_loss` returns `stack([-logit, logit])` |

## AGID adaptations

The model is a pure `nn.Module` (`forward`/`compute_loss`) like every AGID detector, with two
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
conf/model/image/pgc.yaml     # model_type: pgc, backbone_name, lora_*, tau_rgb/res
conf/trainer/image/pgc.yaml   # Adam, lr 5e-5, weight_decay 0.05
experiments/image/pgc.sh      # trains on ProGAN (CNNDetection); swap dataset for GenImage SDv1.4
```

Registered in `trainer.py` under `model_type == "pgc"`. Trainable params: LoRA adapters + residual
stream + PGCM + classifier (DINOv2 backbone frozen). The paper's benchmarks are CommGen15, GenImage,
AIGI, and UniversalFakeDetect; CommGen15 (the paper's new dataset) is optional — the AGID integration
trains on the existing ProGAN/GenImage data.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Paper | AGID_official | AGID_train | Status |
|---|---|---|---|---|
| GenImage | map=100.0 / acc=98.3 | mAP=100.0 / acc=99.49 | mAP=97.62 / acc=78.6 | official |
| UniversalFakeDetect (UFD) | map=98.0 | mAP=99.34 / acc=94.16 | mAP=99.01 | reproduced |
| ForenSynths (CNNDetection test) | — | mAP=96.25 / acc=83.73 | mAP=94.57 / acc=83.41 | reference |
| AIGIBench (AIGI) | map=95.0 / acc=86.6 | mAP=95.36 / acc=88.33 | mAP=86.88 | official |
| CommGen15 (commercial in-the-wild) | acc=97.9 | mAP=99.26 / acc=97.39 | mAP=98.61 / acc=66.66 | official |

> **ForenSynths caveat:** ForenSynths AGID_train mAP (94.57) is the 13-generator mean and is close to AGID_official because the 10 easy generators dominate; on the three hard generators (deepfake/san/seeingdark) AGID_train reaches only 75.87 mAUROC vs 94.35 for official weights. Running the authors' train.py verbatim reproduces the same plateau, so the released ProGAN checkpoint is not reproducible from the published recipe+data — see repro/investigate/pgc_gap_investigation.md.

**reproduced** = AGID-trained checkpoint matches the paper · **official** = official weights loaded into the AGID class match (the AGID-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Residual built in-model (not in the dataset transform) | Keeps the AGID dataset/transform pipeline standard (3-channel in). Same residual math (BT.601 quantization, Eq. 1). |
| DINOv2-Large loaded via HF name | No local-checkpoint requirement; LoRA targets use the HF DINOv2 module names. |
| Reduced training schedule for the bundled checkpoint | The shipped checkpoint is trained for turnaround on a ProGAN subset; raise epochs / use GenImage for a full reproduction. |
