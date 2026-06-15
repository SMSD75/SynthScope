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

| Benchmark | Paper | AGID_official | AGID_train | Status |
|---|---|---|---|---|
| AIGCDetectBenchmark (B1, Setting-I, ProGAN-trained) | acc=92.77 | mAP=97.94 / acc=91.89 | mAP=72.25 / acc=61.64 | official |
| GenImage (B2, Setting-I, SDv1.4-trained) | acc=86.88 | mAP=97.12 / acc=85.91 | mAP=92.47 / acc=73.5 | official |
| ForenSynths-13 | — | mAP=94.40 / acc=81.8 | mAP=74.92 (cross-domain) | reference |
| Chameleon (B3, Setting-II, in-the-wild) | acc=58–66 | — | — | missing |

**reproduced** = AGID-trained checkpoint matches the paper · **official** = official weights loaded into the AGID class match (the AGID-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.
