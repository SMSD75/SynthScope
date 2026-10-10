# CuRe (`cure`)

**Learning Continuous Source Responses for Generalizable AI-Generated Image Detection** —
[paper (arXiv 2609.14316)](https://arxiv.org/abs/2609.14316) · [official code](https://github.com/manic-cui/CuRe)

CuRe reframes backbone adaptation as **regressing real↔generated mixing ratios** instead of binary
classification (finer supervision), with a compact *source-response subspace* that suppresses nuisance
variation and shortcut cues. The released detector is trained **only on GenImage SD‑v1.4** and
reports **89.7 % average balanced accuracy over ten public benchmarks** (+5.2 pts over the second best).

## Pipeline (official, reproduced exactly)

```
image ─ PNG? → JPEG q96 round-trip ─ SmallestMaxSize(390) → CenterCrop(390) → Resize(336)  (cubic)
      ─ JPEG q70 (cv2, on the RGB array) ─ (x − 127.5)/127.5
      ─ PE-Core L/14 @336 (frozen) + Q/V LoRA   → mean of the 576 patch tokens   [B, 1024]
      ─ (feat − mu) @ W                          → [B, 128]    (source-response subspace)
      ─ Linear(128, 2)                           → logits;  P(fake) = softmax(logits)[:, 1]
```

`cure.pth` is the official **25 MB adapter** (`cure_adapter.pt`, SHA-256 checked) unchanged: LoRA
updates, `mu`, `W` and the head. The frozen base (`facebook/PE-Core-L14-336`, 2.7 GB, Apache-2.0) is
fetched from the Hugging Face Hub on first use (or `ModelConfig.base_checkpoint` / `$SYNTHSCOPE_PE_CORE_CKPT`).

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official backbone, **byte-identical body** | The adapter loads key-for-key; Apache-2.0 attribution in the header. |
| Preprocessing in cv2/PIL (`preprocess.py`), not albumentations | No new dependency. Equal to the official pipeline up to **1 float32 ulp (5.96e-8)** on 104 PNG/JPEG images, 200 px–4256 px, square and non-square (`repro/cure_preproc_equiv.py`). |
| **Source-format hint** `img.info["synthscope_src_format"]` | Official PNG sources get an extra JPEG‑q96 round-trip, decided by the file suffix. `bench_eval` and the web decoder now record the original format on the PIL image; without the hint an image is treated as non-PNG. |
| fp16 autocast on CUDA inside `forward` | The released results use fp16 AMP; CPU runs in fp32 (scores differ in the 3rd decimal). |
| `compute_loss` = plain cross-entropy | **The paper's two-stage mixing-ratio-regression training recipe is not in the official release.** `compute_loss` only satisfies SynthScope's trainer interface (LoRA + head trainable); it does **not** reproduce the paper's training. The SynthScope-trained row below is therefore empty. |

## Fidelity check (identical to the official model)

`repro/cure_verify_load.py` loads the official `cure` package and SynthScope's class with the same adapter
and base, on an A5000: all **395 state-dict tensors identical**, all 100 adapter tensors consumed
(`unexpected=0`), **max |logit diff| 4.8e-6**, **max |P(fake) diff| 5.4e-7, 0 verdict flips** over 60
images (official preprocessing vs SynthScope preprocessing, each end to end). Log: `repro/logs/cure_verify_245812.out`.

## Results

Summary in the repo's **Paper / SynthScope_official / SynthScope_train** schema (`acc` = mean per-generator balanced accuracy at 0.5):

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| AIGIBench (paper's 20 full-image-synthesis subsets, full test data) | acc=92.8 / AUC=97.5 | mAP=97.00 / acc=92.85 / AUROC=97.48 | — | **reproduced** (every subset within 0.10 pt) |
| UniversalFakeDetect (UFD-8) | acc=82.0 (UnivFD, aggregation unspecified) | mAP=99.19 / acc=93.94 | — | reference |
| ForenSynths (13) | (GAN half of UnivFD) | mAP=94.05 / acc=81.95 | — | reference |
| GenImage-8 (its training set) | — | mAP=97.30 / acc=85.18 | — | reference |
| AIGCDetectBenchmark | acc=89.3 | — | — | missing |
| RealChain (7 generators, 7000 real / 7000 fake, full data) | acc=79.7 | mAP=87.39 / acc=79.69 / AUROC=85.47 | — | **reproduced** |
| EvalGEN (5 generators, about 55k fakes, MSCOCO val2017 reals) | acc=98.1 | mAP=99.86 / acc=98.14 / AUROC=99.66 | — | **reproduced** (our real pool; see below) |
| SynthWildX (3 generators; 1713 of 2000 images still online on X) | acc=92.7 | mAP=95.61 / acc=92.69 / AUROC=96.50 | — | **reproduced** (on the images still online) |
| WildRF (Facebook, Reddit, Twitter test sets, 1251 real / 1252 fake) | acc=95.8 | mAP=99.13 / acc=95.89 / AUROC=99.14 | — | **reproduced** |
| BFree-Online, RRDataset, Chameleon | 90.9, 81.5, 93.8 | — | — | missing |

`SynthScope_train` is empty: the paper's training recipe is not released. Details and per-generator tables below
(official weights through SynthScope; balanced accuracy at 0.5 unless noted).

Paper: **89.7 avg over ten benchmarks** — AIGCDetect 89.3 · AIGIBench 92.8 · UnivFD 82.0 · EvalGEN 98.1 ·
SynthWildX 92.7 · WildRF 95.8 · BFree-Online 90.9 · RRDataset 81.5 · RealChain 79.7 · Chameleon 93.8.

| Benchmark (SynthScope) | generators | acc (mean of per-gen balanced) | mAP | mAUROC | Paper | Status |
|---|---|---:|---:|---:|---|---|
| UFD-8 (diffusion/DALL·E half of UnivFD) | 8 | **93.94** | 99.19 | 98.98 | UnivFD 82.0 | **not comparable** (see below) |
| ForenSynths-13 (GAN half) | 13 | **81.95** | 94.05 | 95.02 | UnivFD 82.0 | **not comparable** |
| ForenSynths-8 ∪ UFD-8 (16-gen "UnivFD-style") | 16 | 89.27 | – | – | UnivFD 82.0 | aggregation differs from the paper |
| GenImage-8 (`--limit 1500`/class; CuRe's *training* set) | 8 | 85.18 | 97.30 | 97.00 | not in the paper's ten | sanity (in-distribution) |
| **AIGIBench — the paper's 20 full-image-synthesis subsets, FULL test data** (168,002 images) | 20 | **92.85** | 97.00 | 97.48 | **92.8** (AUC 97.5) | **reproduced** (mean abs diff 0.03; max 0.10) |
| same 20 subsets, `--limit 1500`/class (earlier run) | 20 | 93.13 | 97.14 | 97.59 | 92.8 | within 2.1 pt per subset; the cap caused the residual |
| AIGIBench, all 25 test generators (adds 5 face-swap subsets the paper excludes) | 25 | 85.95 | 90.72 | 91.43 | – | not the paper's protocol |
| AIGCDetect (17 gens) | 17 | *not run* | | | 89.3 | data lives only on cn023 (its GPUs/CPUs were fully booked) |
| **RealChain** (`--dataset realchain`, full data) | 7 | **79.69** | 87.39 | 85.47 | **79.7** | **reproduced** |
| **EvalGEN** (`--dataset evalgen`, full data, MSCOCO val2017 reals) | 5 | **98.14** | 99.86 | 99.66 | **98.1** | **reproduced** |
| **SynthWildX** (`--dataset synthwildx`; 1713/2000 images still on X) | 3 | **92.69** | 95.61 | 96.50 | **92.7** | **reproduced** |
| **WildRF** (`--dataset wildrf`, full test split) | 3 | **95.89** | 99.13 | 99.14 | **95.8** | **reproduced** |
| BFree-Online, RRDataset, Chameleon | – | *not in SynthScope* | | | see above | datasets not available here |

**Honest reading.** The port is verified identical to the official model (above), and the in-distribution
behaviour is exactly what the paper's training implies — SD‑v1.4 **99.73**, SD‑v1.5 **99.80**, Wukong
**99.87** — while other families are weaker at the fixed 0.5 threshold, with high AP/AUROC (e.g. GenImage
ADM/BigGAN/VQDM fake-recall 22–65 %, AUROC 87–99).

**RealChain and EvalGEN are reproduced** (jobs 251412, logs `repro/logs/evalds_251412.out`).
RealChain: the release holds one image set, all JPEG q60, with 7000 unpaired reals that the loader splits into 7 disjoint
slices, so the mean of the per-generator balanced accuracies equals the pooled 7000-vs-7000 one the paper reports
(real acc 94.79, fake acc 64.60). Per generator: Flux1 90.75, Hunyuan3 91.65, QwenImage 89.10, NanoBanana 81.80,
SDv3.5 79.10, Seedream4 63.62, i2i 61.90. EvalGEN ships fakes only; CuRe pairs it with an unspecified "shared MSCOCO
real pool", and we use MSCOCO val2017 (5000 images; real acc 99.72, mean fake acc 96.56). Per generator: Flux 93.37,
GoT 99.71, Infinity 98.55, NOVA 99.32, OmniGen 99.76.

**SynthWildX and WildRF are reproduced** (jobs 251435 and 251440). SynthWildX is released as URLs of posts on X; 287 of
the 2000 have been deleted since, so we score the 1713 still online (413 real; DALL-E 3 94.83, Firefly 88.35,
Midjourney v5 94.88). WildRF uses its full test split, one subset per platform (Facebook 95.31, Reddit 96.60,
Twitter 95.75). Both are the unweighted mean over subsets, as in the paper.

**AIGIBench is reproduced.** The paper evaluates "the 20 subsets corresponding to full-image synthesis" and
reports "the unweighted macro-average over its evaluated subsets" (Table 2): ProGAN, R3GAN, StyleGAN3,
StyleGAN‑XL, StyleSwim, WFIR, DALL·E 3, FLUX.1‑dev, GLIDE, Imagen 3, Midjourney, SD3, SDXL, BLIP‑Diffusion,
Infinite‑ID, InstantID, IP‑Adapter, PhotoMaker, CommunityAI, SocialRF. On exactly those 20 with the **full test data**
(168,002 images), SynthScope's port of the official adapter gives **92.85 balanced accuracy / 97.48 AUROC vs the
paper's 92.8 / 97.5**, and **every one of the 20 subsets is within 0.10 pt of the paper's Table 2** (mean |diff|
0.03, largest WFIR +0.10; ProGAN, StyleGAN3, StyleGAN‑XL, SD3 match to the printed digit). An earlier run on the
first 1,500 images per class gave 93.13 / 97.59 (per-subset mean |diff| 0.76, max 2.13), so the cap, not the port,
caused the residual. The five face-swap/editing subsets in SynthScope's 25-generator
test split (BlendFace, FaceSwap, InSwap, SimSwap, E4S; mean 57.2) are excluded by the paper — including them is
what produced my first, misleading 85.95. (Paper values were read from the arXiv
HTML Table 2 via a text extraction, not from the PDF.)

**Still NOT confirmed:** only 3 of the paper's 10 benchmarks exist in SynthScope. UnivFD: the paper's aggregation is not
specified (SynthScope's UFD-8 and ForenSynths-13 splits give 93.9 and 82.0, the 16-generator union 89.3, vs 82.0).
AIGCDetect was not run (its data lives only on cn023, fully booked). The other 7 benchmarks are not available here.

Per-generator tables (balanced acc / AP / AUROC / real-acc / fake-acc), from `repro/bench_results/cure_repro_*.json`:

**UFD-8** — dalle 97.45 / 99.73 / 99.66 · glide_100_10 95.25 / 99.30 / 99.10 · glide_100_27 93.20 / 99.17 / 99.01 ·
glide_50_27 92.65 / 98.93 / 98.62 · guided 79.25 / 96.68 / 95.80 · ldm_100 97.55 / 99.89 / 99.89 ·
ldm_200 97.95 / 99.91 / 99.92 · ldm_200_cfg 98.20 / 99.88 / 99.86

**ForenSynths-13** — progan 95.20 · stylegan 87.15 · stylegan2 79.11 · biggan 93.50 · cyclegan 70.36 ·
stargan 92.90 · gaugan 94.21 · deepfake 64.46 · crn 72.34 · imle 74.62 · san 75.11 · seeingdark 72.78 ·
whichfaceisreal 93.60 (the low accuracies are fake-recall: cyclegan 41.8, deepfake 29.7; AP/AUROC are 95–99 on those)

**GenImage-8** — sdv1.4 99.73 · sdv1.5 99.80 · wukong 99.87 · midjourney 85.83 · glide 85.33 · vqdm 82.43 ·
biggan 67.80 · adm 60.63

## Usage

```bash
synthscope detect photo.png --model cure          # needs cure.pth in $SYNTHSCOPE_CKPT_DIR (official adapter) + HF access
python benchmark/bench_eval.py --model cure --benchmark ufd --data <UFD/test> --tag cure_repro_ufd
```

`cure.pth` = copy of the official `weights/cure_adapter.pt`. Use the registered transform (`CuReTransform`):
the PNG→JPEG96 step and the final JPEG‑q70 pass are part of the method.
