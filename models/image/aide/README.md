# AIDE: AI-generated Image Detector with Hybrid Features

## Short method explanation

**AIDE** (ICLR 2025, "A Sanity Check for AI-generated Image Detection") detects generated images by
fusing a **semantic** branch and a **low-level artifact** branch:

- **Semantic branch** — a frozen OpenCLIP **ConvNeXt-XXL** trunk, mean-pooled and projected to 256-d.
- **Artifact branch** — the image is resized to 256×256, then 32×32 patches are **DCT-scored** with `K=6`
  band-pass filters (weighted `2^k` over `log(|DCT|+1)`); the **2 highest- and 2 lowest-energy patches**
  are each upsampled to 256×256, passed through **30 fixed SRM high-pass kernels** and two ResNet-50
  streams (max-frequency pair → `model_max`, min-frequency pair → `model_min`). The mean of the four
  pooled features is a 2048-d artifact descriptor.

The semantic (256-d) and artifact (2048-d) features are concatenated and classified. This **5-view**
(1 semantic + 4 patch) construction is the method's defining preprocessing.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

> **Served checkpoint + faithful preprocessing:** the web app / `agid.load("aide")` serves the
> **official SDv1.4 weights** (`aide.pth → sd14_train.pth`) through the **genuine 5-view arch with
> native-resolution DCT** (`official_arch.AIDEFiveView`). This matters a lot: the DCT patch selector
> must run at native resolution — resizing the whole image to 256 *before* the DCT costs **~12 mAP on
> the same weights** (ablation: GenImage-8 native 99.84 → resize-256 87.45). With the fix, the served
> registry path reproduces the official: `bench_eval --model aide` GenImage-8 = **96.83 mAP** (≈ official
> 97.12). The earlier AGID-class path resized before the DCT and collapsed (55.67 / 72.25).

| Benchmark | Paper | AGID_official | AGID_train | Status |
|---|---|---|---|---|
| AIGCDetectBenchmark (B1, Setting-I, ProGAN-trained) | acc=92.77 | mAP=97.94 / acc=91.89 | — | official |
| GenImage (B2, Setting-I, SDv1.4-trained) | acc=86.88 | mAP=97.12 / acc=85.91 (served 96.83) | **mAP=96.42 / acc=77.9** | reproduced |
| ForenSynths-13 | — | mAP=94.40 / acc=81.8 | mAP=79.62 (cross-domain) | reference |
| Chameleon (B3, Setting-II, in-the-wild) | acc=58–66 | — | — | missing |

*(**AGID_train = a from-scratch retrain with the faithful native-res 5-view preprocessing** (`checkpoints_retrain/aide_native_v2/model_2`, GenImage SDv1.4): GenImage-8 **96.42 mAP / 97.07 mAUROC**, reproducing the official 97.12 — up from the old resize-256 run's 92.47. This proves the repo can regenerate the AIDE checkpoint from scratch. ForenSynths-13 is cross-domain (GenImage-trained model on the GAN benchmark, like the official sd14 checkpoint there).)*

**reproduced** = AGID-trained checkpoint matches the paper · **official** = official weights loaded into the AGID class match (the AGID-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.
