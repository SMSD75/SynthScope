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
| ForenSynths-8 (CNNDetection test) | mAP=96.25 | mAP=96.39 / acc=76.29 | mAP=96.05 / acc=71.12 | reproduced |
| AIGIBench (AIGI) | map=95.0 / acc=86.6 | mAP=95.36 / acc=88.33 | mAP=86.88 | official |
| CommGen15 (commercial in-the-wild) | acc=97.9 | mAP=99.26 / acc=97.39 | mAP=98.61 / acc=66.66 | official |

> **ForenSynths hard generators — reproduced via early-stopping (resolved 2026-06-19).** The released ProGAN checkpoint is trained for only **~400 steps (< 1 epoch)** — confirmed by the authors on our issue. DINOv2's strong priors overfit ProGAN within a few epochs (loss → floor), which keeps easy generators high but **erodes hard-generator transfer**; our earlier epoch-level checkpoints (ep10+) all overshot the peak. Re-running the official `train.py` verbatim with fine-grained eval shows the hard-3 (deepfake/san/seeingdark) **mean AUROC climb to 94.56 at step ~900 ≈ official 94.35**, then decay with longer training. So PGC **does** reproduce from the published recipe+data with early-stopping inside the first epoch. (Served PGC uses the official weights.) See `repro/investigate/pgc_gap_investigation.md`.

> **Eval transform — `PadCenterCrop(224)`, no resize (verified 2026-06-19).** PGC's YCbCr residual is resolution-sensitive, so the eval transform is center-crop-224 with **no resize** (= the official `PadCenterCrop`; byte-identical to the genuine repo's). The earlier `Resize→CenterCrop` registry transform wiped the forensic signal on super-res/low-light generators (san 73.3, seeingdark 72.5 — same eval-transform family as the SAFE/AIDE/SPAI fixes). With the fix, AGID `bench_eval` reproduces ForenSynths-8 (96.39 mAP official / 96.05 from-scratch vs paper 96.25) and san recovers to 98.65 AUROC. One generator, **seeingdark**, stays ~10 pt below the genuine repo (77.9 vs 87.49) on identical images/weights with now-bit-exact preprocessing — isolated to the model forward, not the transform; it is not part of the reported ForenSynths-8.

**reproduced** = AGID-trained checkpoint matches the paper · **official** = official weights loaded into the AGID class match (the AGID-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.

## Differences from the paper / official code

| Difference | Reason |
|---|---|
| Residual built in-model (not in the dataset transform) | Keeps the AGID dataset/transform pipeline standard (3-channel in). Same residual math (BT.601 quantization, Eq. 1) — `_quantization_residual` snaps `round(x01*255)` to the uint8 grid so the residual is **bit-exact** to the official 6-channel transform despite the in-model ImageNet de-normalization (verified zero-diff, `repro/_residual_path_diff.py`). |
| DINOv2-Large loaded via HF name | No local-checkpoint requirement; LoRA targets use the HF DINOv2 module names. |
| Reduced training schedule for the bundled checkpoint | The shipped checkpoint is trained for turnaround on a ProGAN subset; raise epochs / use GenImage for a full reproduction. |
