# DDA (`dda`)

**Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable** (NeurIPS 2025 Spotlight) —
[paper (arXiv 2505.14359)](https://arxiv.org/abs/2505.14359) · [official code](https://github.com/roy-ch/Dual-Data-Alignment)

Detectors trained on biased data learn shortcuts: content, format and, as DDA shows, frequency content. Reconstructing
fakes from reals with a VAE aligns the content, but the reconstructions restore high-frequency detail that the real
photos had lost, so a detector learns "rich high frequencies = fake". DDA aligns the real and synthetic images in
**both pixel and frequency space** and trains a single detector **only on DDA-aligned MSCOCO**. The paper reports
**90.7 average accuracy over 11 benchmarks**, and it is the first detector above 80 on Chameleon.

## Pipeline (official, reproduced exactly)

```
image ─ CenterCrop(336) (no resize) → ToTensor → Normalize(CLIP mean/std)
      ─ DINOv2 ViT-L/14 + LoRA (rank 8, alpha 1) on every attn.qkv / attn.proj / mlp.fc1 / mlp.fc2
      ─ x_norm_clstoken                                  [B, 1024]
      ─ Linear(1024, 1)                                  → logit;  P(fake) = sigmoid(logit)
```

`dda.pth` is the official `DDA_ckpt.pth` (Hugging Face `Junwei-Xi/Dual-Data-Alignment`) unchanged: the full model
(537 tensors, 307.5M parameters) under its `model` key, so the DINOv2 weights are not downloaded for inference
(`torch.hub` still fetches the DINOv2 code once).

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official `Inference/models/lora.py`, verbatim | The checkpoint loads key-for-key; Apache-2.0 attribution in the header. |
| `SynthScopeDDAModel` keeps the official `base_model.model.*` / `base_model.fc.*` names | Same state-dict layout as `DINOv2ModelWithLoRA`. |
| `ModelConfig.pretrained_backbone=false` in the registry | The checkpoint holds the full backbone, so the DINOv2 weights are not fetched just to be overwritten. |
| `compute_loss` = BCE on the single logit, LoRA + head trainable | The official trainable set. DDA's data construction (VAE reconstruction, frequency fusion, online JPEG) is not part of this class, so SynthScope does not reproduce DDA's training. |
| `bench_eval --jpeg-fakes 96` | DDA's protocol JPEG-compresses only the synthetic images of GenImage, ForenSynths and AIGCDetect (quality 96) to remove the PNG-fake / JPEG-real format bias. The round trip matches the official `JPEG_Compression`. |

## Fidelity check (identical to the official model)

The official `DINOv2ModelWithLoRA` (from the official repo) and SynthScope's class, both loaded from `DDA_ckpt.pth`:
**537/537 state-dict tensors identical (max |diff| 0.0)**, the registry transform equals the official test transform
pixel for pixel, and **max |logit diff| 0.0 with 0 verdict flips** over 40 images (EvalGEN Flux real/fake, WildRF
Reddit/Twitter, SynthWildX Firefly). Run on CPU (job 264307).

## Results

Paper metric: accuracy at 0.5, mean ± std over each benchmark's subsets (official README table). SynthScope numbers
are mean per-subset balanced accuracy with the official checkpoint; ± is the sample std over subsets.

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| GenImage (1G + 7D, fakes JPEG q96) | 91.7 ± 7.8 | queued | — | |
| DRCT-2M (16D) | 98.1 ± 1.4 | — | — | missing (not in SynthScope yet) |
| DDA-COCO (5D) | 92.2 ± 10.6 | — | — | missing (not in SynthScope yet) |
| EvalGEN (3D + 2AR, fake-only accuracy) | 97.2 ± 4.2 | queued | — | |
| Synthbuster (9D) | 90.1 ± 5.6 | — | — | missing (RAISE-1k reals not downloaded yet) |
| ForenSynths (11G, fakes JPEG q96) | 81.4 ± 13.9 | queued | — | |
| AIGCDetectBenchmark (7G + 10D, fakes JPEG q96) | 87.8 ± 12.6 | queued | — | |
| Chameleon | 82.4 | — | — | missing (available by request only) |
| **SynthWildX** (1713 of 2000 images still online on X) | 90.9 ± 3.1 | **90.80 ± 3.13** (mAP 96.47, AUROC 96.40) | — | **reproduced** |
| **WildRF** (full test split, 1251 real / 1252 fake) | 90.3 ± 3.5 | **90.20 ± 3.45** (mAP 96.76, AUROC 96.14) | — | **reproduced** |
| BFree-Online (807 of the 1111 listed images still online) | 95.1 | 94.98 (mAP 99.41, AUROC 98.86) | — | close (not the paper's full set) |

Per subset: SynthWildX DALL-E 3 92.28 · Firefly 87.21 · Midjourney v5 92.92 (real acc 88.38); WildRF Facebook 93.13 ·
Reddit 86.40 · Twitter 91.07; BFree-Online real acc 96.65, fake acc 93.31 (tags `dda_synthwildx`, `dda_wildrf`,
`dda_bfree_online`, job 264310). These three in-the-wild sets get no JPEG step in the paper's protocol.

The authors note that results depend slightly on the JPEG decoder and recommend Pillow 12.3.0 with libjpeg-turbo
3.1.4.1; these runs used Pillow 12.1.1 with libjpeg-turbo 3.1.3.

## Usage

```bash
synthscope detect photo.png --model dda           # needs dda.pth (= DDA_ckpt.pth) in $SYNTHSCOPE_CKPT_DIR
python benchmark/bench_eval.py --model dda --dataset wildrf --root /data/WildRF
python benchmark/bench_eval.py --model dda --dataset aigcdetect --root /data/AIGCDetect --jpeg-fakes 96
```

The input is center-cropped to 336 without resizing, as in the official inference script, so images smaller than
336 px are padded by the crop.
