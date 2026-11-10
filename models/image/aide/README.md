# AIDE: AI-generated Image Detector with Hybrid Features

## Short method explanation

**AIDE** (ICLR 2025, "A Sanity Check for AI-generated Image Detection") detects generated images by
fusing a **semantic** branch and a **low-level artifact** branch:

- **Semantic branch** — a frozen OpenCLIP **ConvNeXt-XXL** trunk, mean-pooled and projected to 256-d by a linear layer that is frozen too (the official forward computes it under `torch.no_grad()`, so it keeps its random initialization; training updates only the two ResNet-50 streams and the MLP head).
- **Artifact branch** — the image is resized to 256×256, then 32×32 patches are **DCT-scored** with `K=6`
  band-pass filters (weighted `2^k` over `log(|DCT|+1)`); the **2 highest- and 2 lowest-energy patches**
  are each upsampled to 256×256, passed through **30 fixed SRM high-pass kernels** and two ResNet-50
  streams (max-frequency pair → `model_max`, min-frequency pair → `model_min`). The mean of the four
  pooled features is a 2048-d artifact descriptor.

The semantic (256-d) and artifact (2048-d) features are concatenated and classified. This **5-view**
(1 semantic + 4 patch) construction is the method's defining preprocessing.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

> **Served checkpoint + faithful preprocessing:** the web app / `synthscope.load("aide")` serves the
> **official SDv1.4 weights** (`aide.pth → sd14_train.pth`) through the **genuine 5-view arch with
> native-resolution DCT** (`official_arch.AIDEFiveView`). This matters a lot: the DCT patch selector
> must run at native resolution — resizing the whole image to 256 *before* the DCT costs **~12 mAP on
> the same weights** (ablation: GenImage-8 native 99.84 → resize-256 87.45). With the fix, the served
> registry path reproduces the official: `bench_eval --model aide` GenImage-8 = **96.83 mAP** (≈ official
> 97.12). The earlier SynthScope-class path resized before the DCT and collapsed (55.67 / 72.25).

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| AIGCDetectBenchmark (B1, Setting-I, ProGAN-trained) | acc=92.77 | mAP=97.94 / acc=91.89 (official `progan_train.pth`) | — | official |
| GenImage (B2, Setting-I, SDv1.4-trained) | acc=86.88 | mAP=97.12 / acc=85.91 (served 96.83) | **mAP=93.90 / acc=80.73** | official; trained: below the paper (acc -6.15, mAP -3.22 vs the official weights) |
| ForenSynths-13 | — | mAP=94.40 / acc=81.8 (official `progan_train.pth`) | mAP=80.42 / acc=62.5 (cross-domain) | reference |
| Chameleon (B3, Setting-II, in-the-wild) | acc=58–66 | — | — | missing |

*(**SynthScope_train = the official training recipe run in SynthScope** (`bash experiments/image/aide.sh`, train job 281041, `model_best` = epoch 4 of 5, validation accuracy 0.9984), evaluated with `bench_eval --model aide --ckpt model_best.pth`: GenImage-8 on the full test set (tag `aide_trained_genimage8`) **93.90 mAP / 91.66 mAUROC / 80.73 acc**; on the first 300 images per class and generator (the protocol of the served row and of the previous retrain; tag `aide_trained_genimage8_l300`) 92.57 / 89.63 / 80.69. It stays outside the 2-point tolerance of the paper (86.88 acc, -6.15) and below the official weights (97.12 mAP, -3.22): SD v1.4, SD v1.5, Wukong and GLIDE match them (acc 99.97 / 100.0 / 99.93 / 92.00 against 99.76 / 99.80 / 99.13 / 90.83), but ADM (58.63 against 76.60), Midjourney (74.60 against 83.70), VQDM (69.83 against 76.80) and BigGAN (50.87 against 60.63) transfer worse; real accuracy is 99.93-100 on every subset, so the loss is fake recall at the 0.5 threshold. The previous retrain (`aide_native_v2/model_2`: native-resolution 5-view, but the earlier recipe with lr 1e-4, batch 32, horizontal flip, no perturbations, no label smoothing, DeepSpeed gradient clipping 1.0 and the epoch picked by validation loss; 300 images per class) had 96.42 mAP / 77.87 acc: the faithful recipe gains 2.8 points of accuracy and loses 3.9 mAP on the same images. The cause of the remaining gap is not identified. ForenSynths-13 (tag `aide_trained_forensynths`, 300 per class) is cross-domain for a GenImage-trained model (80.42 mAP / 62.5 acc); the official number there is the ProGAN-trained `progan_train.pth`, not the served `sd14_train.pth`.)*

### Training recipe (`experiments/image/aide.sh`)

The recipe follows the official code (`shilinyan99/AIDE`, `main_finetune.py`, `data/datasets.py`) and the paper's GenImage setting:

- **Data**: GenImage SD v1.4 train (162,000 ImageNet reals + 162,000 SD v1.4 fakes); validation = the SD v1.4 val split (6,000 + 6,000).
- **Trainable set**: the two ResNet-50 streams and the MLP head (322 tensors, 49,547,778 values). The OpenCLIP ConvNeXt-XXL trunk and its projection stay frozen, the ResNets start from random weights (the official code never loads its ImageNet state dict).
- **Transforms** (native resolution, no flip, no crop): `aide_blur` (Gaussian blur, 3x3 kernel, sigma 0.1-3.0, p 0.1) then `aide_jpeg` (JPEG quality 30-100, p 0.1; kornia 0.7.2, images whose sides are not multiples of 16 are edge-padded for the codec and cropped back), then the 5-view DCT patch selection (`aide_five_view_train`). Validation images get the same perturbations, as in the official run. The one deviation is a 1024-pixel cap on the long side before the DCT (dataloader memory); inference never caps.
- **Optimisation**: AdamW, lr 5e-4, weight decay 0, 5 epochs, per-iteration cosine decay to 1e-6, no warm-up, label smoothing 0.1, fp32, no gradient clipping, no EMA; plain DDP (no DeepSpeed).
- **Batch**: effective batch 256 = 4 GPUs x micro-batch 8 x gradient accumulation 8, with SyncBatchNorm over the 4 GPUs so each BatchNorm layer sees 32 images, as on each of the official 8 GPUs (batch 32 each).
- **Selection**: `model_best.pth` = the epoch with the highest validation accuracy (the official `max_accuracy`); every epoch is kept as `model_<epoch>.pth`. Validation accuracy rose every epoch (0.9913, 0.9956, 0.9973, 0.9979, 0.9984), so epoch 4 is also the last.
- **Evaluate**: `python benchmark/bench_eval.py --model aide --ckpt <save_path>/model_best.pth --benchmark pergen --data <GenImage test root>`.

**reproduced** = SynthScope-trained checkpoint matches the paper · **official** = official weights loaded into the SynthScope class match (the SynthScope-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.
