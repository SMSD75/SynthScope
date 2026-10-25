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
| GenImage (1G + 7D, fakes JPEG q96) | 91.7 ± 7.8 | 89.37 ± 11.08 (mAP 97.74, AUROC 98.21) | — | **below** by 2.3 pt (see below) |
| DRCT-2M (16D) | 98.1 ± 1.4 | — | — | missing (not in SynthScope yet) |
| DDA-COCO (5D) | 92.2 ± 10.6 | — | — | missing (not in SynthScope yet) |
| **EvalGEN** (3D + 2AR, all 55,298 fakes, fake-only accuracy) | 97.2 ± 4.2 | **96.86 ± 4.62** (mAP 99.93, AUROC 99.85) | — | **reproduced** |
| Synthbuster (9D) | 90.1 ± 5.6 | — | — | missing (RAISE-1k reals not downloaded yet) |
| **ForenSynths** (all 13 test sets, fakes JPEG q96) | 81.4 ± 13.9 ("11G") | **81.43 ± 13.84** (mAP 91.13, AUROC 92.11) | — | **reproduced** |
| **AIGCDetectBenchmark** (7G + 10D, full data, fakes JPEG q96) | 87.8 ± 12.6 | **87.84 ± 12.56** (mAP 96.35, AUROC 96.65) | — | **reproduced** |
| Chameleon | 82.4 | — | — | missing (available by request only) |
| **SynthWildX** (1713 of 2000 images still online on X) | 90.9 ± 3.1 | **90.80 ± 3.13** (mAP 96.47, AUROC 96.40) | — | **reproduced** |
| **WildRF** (full test split, 1251 real / 1252 fake) | 90.3 ± 3.5 | **90.20 ± 3.45** (mAP 96.76, AUROC 96.14) | — | **reproduced** |
| BFree-Online (807 of the 1111 listed images still online) | 95.1 | 94.98 (mAP 99.41, AUROC 98.86) | — | close (not the paper's full set) |
| RealChain (7 generators, full data) | — | 67.23 (mAP 76.46, AUROC 74.97) | — | reference (not in the paper) |
| RRDataset (original / transfer / redigital) | — | 75.99 (mAP 85.18, AUROC 81.74) | — | reference (not in the paper) |

Of the paper's 11 benchmarks we can run 7: **5 are reproduced** (EvalGEN, ForenSynths, AIGCDetect, SynthWildX,
WildRF), BFree-Online is close on the images still online, and GenImage is 2.3 pt below. AIGCDetect and ForenSynths
match the paper's mean *and* std to the printed digit. Not run: DRCT-2M and DDA-COCO (not in SynthScope yet),
Synthbuster (its RAISE-1k reals are not downloaded yet) and Chameleon (by request only).

**JPEG-on-fakes protocol.** GenImage, ForenSynths and AIGCDetect are scored with `--jpeg-fakes 96`, as in the paper.
Without it the same checkpoint scores lower: GenImage 86.27, ForenSynths 79.71, AIGCDetect 85.35 (tags without the
`_jpeg96` suffix). The other benchmarks get no JPEG step.

**ForenSynths.** The paper labels the set "11G" but does not list the generators. All 13 CNNDetection test sets give
81.43 ± 13.84, the paper's number, so the paper most likely averaged the same 13. Our guess of the 11 UnivFD GAN subsets
(without StyleGAN2 and WhichFaceIsReal) gives 83.30 ± 11.51, which does not match. Per set (q96): BigGAN 90.98 ·
CRN 86.98 · CycleGAN 72.67 · Deepfake 76.53 · GauGAN 92.74 · IMLE 89.71 · ProGAN 92.80 · SAN 94.75 · SeeingDark 58.61 ·
StarGAN 72.74 · StyleGAN 87.80 · StyleGAN2 90.17 · WhichFaceIsReal 52.10.

**GenImage is 2.3 pt below.** Per subset (q96): ADM 85.40 · BigGAN 82.77 · GLIDE 88.17 · Midjourney 95.17 ·
SD-v1.4 98.72 · SD-v1.5 98.87 · VQDM 67.03 · Wukong 98.80; VQDM and BigGAN pull the mean down and widen the spread
(± 11.1 vs 7.8). The same JPEG step reproduces AIGCDetect and ForenSynths exactly, so the gap more likely comes from the
GenImage data (we use the GenImage test release in `genimage_full`) than from the pipeline. Not resolved yet.

**EvalGEN.** EvalGEN ships fakes only, and the paper reports the accuracy on the fakes: Flux 88.68 · GoT 99.14 ·
Infinity 97.73 · NOVA 99.53 · OmniGen 99.20, mean 96.86. With the MSCOCO val2017 reals that SynthScope pairs it with
(real acc 98.98), the balanced accuracy is 97.92.

**In the wild.** SynthWildX DALL-E 3 92.28 · Firefly 87.21 · Midjourney v5 92.92 (real acc 88.38); WildRF Facebook
93.13 · Reddit 86.40 · Twitter 91.07; BFree-Online real acc 96.65, fake acc 93.31. RealChain: Flux1 58.60 ·
Hunyuan3 59.80 · NanoBanana 57.55 · QwenImage 74.60 · SDv3.5 70.05 · Seedream4 75.00 · i2i 75.00. RRDataset: original
91.84 · transfer 73.26 · redigital 62.88.

Runs: job 264310 (SynthWildX, WildRF, BFree-Online, on CPU), 264311 (EvalGEN, GenImage), 264312 (AIGCDetect,
ForenSynths), 264323 (RealChain, RRDataset); results in `repro/bench_results/dda_*.json`.

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
