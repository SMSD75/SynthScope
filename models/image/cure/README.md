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
| Training recipe implemented from the paper | **The official release has inference code only.** SynthScope implements the paper's three stages (see *Training recipe* below); without `training_stage` `compute_loss` stays a plain cross-entropy that only satisfies the trainer interface. |

## Fidelity check (identical to the official model)

`repro/cure_verify_load.py` loads the official `cure` package and SynthScope's class with the same adapter
and base, on an A5000: all **395 state-dict tensors identical**, all 100 adapter tensors consumed
(`unexpected=0`), **max |logit diff| 4.8e-6**, **max |P(fake) diff| 5.4e-7, 0 verdict flips** over 60
images (official preprocessing vs SynthScope preprocessing, each end to end). Log: `repro/logs/cure_verify_245812.out`.

## Results

Summary in the repo's **Paper / SynthScope_official / SynthScope_train** schema (`acc` = mean per-generator balanced accuracy at 0.5):

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status (official adapter) |
|---|---|---|---|---|
| AIGIBench (paper's 20 full-image-synthesis subsets, full test data) | acc=92.8 / AUC=97.5 | mAP=97.00 / acc=92.85 / AUROC=97.48 | mAP=92.69 / acc=85.05 / AUROC=91.84 (**not reproduced, -7.8 acc**) | **reproduced** (every subset within 0.10 pt) |
| UniversalFakeDetect (UFD-8) | acc=82.0 (UnivFD, aggregation unspecified) | mAP=99.19 / acc=93.94 | mAP=95.26 / acc=82.69 / AUROC=93.85 (**reference**) | reference |
| ForenSynths (13) | (GAN half of UnivFD) | mAP=94.05 / acc=81.95 | mAP=94.24 / acc=75.83 / AUROC=93.56 (**reference**) | reference |
| GenImage-8 (its training set) | — | mAP=97.30 / acc=85.18 | mAP=89.41 / acc=77.70 / AUROC=87.88 (**reference**; SD v1.4 99.97, SD v1.5 99.93, Wukong 99.93) | reference |
| AIGCDetectBenchmark (17 generators, full data, 76.3k real / 76.3k fake) | acc=89.3 / AUC=98.0 | mAP=98.08 / acc=89.30 / AUROC=98.02 | mAP=93.52 / acc=81.55 / AUROC=92.22 (**not reproduced, -7.8 acc**) | **reproduced** |
| RealChain (7 generators, 7000 real / 7000 fake, full data) | acc=79.7 | mAP=87.39 / acc=79.69 / AUROC=85.47 | mAP=79.60 / acc=63.33 / AUROC=76.50 (**not reproduced, -16.4 acc**) | **reproduced** |
| EvalGEN (5 generators, about 55k fakes, MSCOCO val2017 reals) | acc=98.1 | mAP=99.86 / acc=98.14 / AUROC=99.66 | mAP=98.30 / acc=90.65 / AUROC=95.96 (**not reproduced, -7.5 acc**) | **reproduced** (our real pool; see below) |
| SynthWildX (3 generators; 1713 of 2000 images still online on X) | acc=92.7 | mAP=95.61 / acc=92.69 / AUROC=96.50 | mAP=88.83 / acc=76.11 / AUROC=86.84 (**not reproduced, -16.6 acc**) | **reproduced** (on the images still online) |
| WildRF (Facebook, Reddit, Twitter test sets, 1251 real / 1252 fake) | acc=95.8 | mAP=99.13 / acc=95.89 / AUROC=99.14 | mAP=95.00 / acc=85.80 / AUROC=94.59 (**not reproduced, -10.0 acc**) | **reproduced** |
| BFree-Online (807 of the 1111 listed links still online; the paper had about 1400 images) | acc=90.9 | mAP=99.06 / acc=91.64 / AUROC=98.08 | mAP=93.37 / acc=66.64 / AUROC=85.77 (**not reproduced, -24.3 acc; fewer images online**) | close (about 58% of the paper's images) |
| RRDataset (original / transfer / redigital, public test archive, 25.5k real / 25.5k fake) | acc=81.5 | mAP=90.40 / acc=82.04 / AUROC=88.88 | mAP=80.40 / acc=66.36 / AUROC=76.87 (**not reproduced, -15.1 acc**) | **reproduced** within 0.6 pt (the paper counts 30k/30k) |
| Chameleon | 93.8 | — | — | missing (available by request only) |

`SynthScope_train` = the checkpoint trained by `experiments/image/cure.sh` (the recipe below, DAS6 job 66658), scored with `bench_eval --ckpt` and the same flags as the official rows (tags `cure_trained_<bench>`, DAS6 job 66721, see *Trained checkpoint* below). The verdict in that column uses the repo's tolerance (ours >= paper - 2.0 acc, - 2.5 AUROC): **the recipe does not reproduce the paper**; every comparable benchmark is 7.5-24 acc points below it. Details and per-generator tables below
The tables below are the official weights through SynthScope (balanced accuracy at 0.5 unless noted); the trained checkpoint has its own section after the recipe.

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
| **AIGCDetect** (`--dataset aigcdetect`, all 17 generators, full data) | 17 | **89.30** | 98.08 | 98.02 | **89.3** (AUC 98.0) | **reproduced** |
| **RealChain** (`--dataset realchain`, full data) | 7 | **79.69** | 87.39 | 85.47 | **79.7** | **reproduced** |
| **EvalGEN** (`--dataset evalgen`, full data, MSCOCO val2017 reals) | 5 | **98.14** | 99.86 | 99.66 | **98.1** | **reproduced** |
| **SynthWildX** (`--dataset synthwildx`; 1713/2000 images still on X) | 3 | **92.69** | 95.61 | 96.50 | **92.7** | **reproduced** |
| **WildRF** (`--dataset wildrf`, full test split) | 3 | **95.89** | 99.13 | 99.14 | **95.8** | **reproduced** |
| BFree-Online (`--dataset bfree_online`; 269 real / 538 fake still online) | 1 | **91.64** | 99.06 | 98.08 | 90.9 | close; not the paper's full set |
| **RRDataset** (`--dataset rrdataset`, public test archive) | 3 | **82.04** | 90.40 | 88.88 | **81.5** | **reproduced** within 0.6 pt |
| Chameleon | – | *not in SynthScope* | | | see above | released by email request only |

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

**BFree-Online is close but not the same data** (job 251449). It is released as URLs with md5s; only 807 of the 1111
listed images are still online (269 real, 538 fake), against about 1400 in the paper, so 91.64 (real acc 96.65,
fake acc 86.62) is not a like-for-like comparison with 90.9.

**RRDataset is reproduced within 0.6 pt** (job 251476): original 91.08, transfer 85.52, redigital 69.51, mean 82.04
vs 81.5. We use the public test archive (25.5k real / 25.5k fake); the paper's dataset table counts 30k/30k, which
probably includes the separate train/val archive, so the evaluated sets may differ slightly.

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

**AIGCDetectBenchmark is reproduced** (job 251523): 89.30 balanced acc and 98.02 AUROC on all 17 generators with the
full test data (76,299 real / 76,299 fake), vs 89.3 / 98.0. Per generator: ADM 71.13, DALLE2 82.45, Glide 88.07,
Midjourney 82.62, VQDM 89.32, BigGAN 93.55, CycleGAN 70.36, GauGAN 94.23, ProGAN 95.20, SD-XL 99.45, SD-v1.4 99.77,
SD-v1.5 99.73, StarGAN 92.82, StyleGAN 87.12, StyleGAN2 79.11, WhichFaceIsReal 93.50, Wukong 99.65.

**Still NOT confirmed:** UnivFD, whose aggregation the paper does not specify (SynthScope's UFD-8 and ForenSynths-13
splits give 93.9 and 82.0, the 16-generator union 89.3, vs 82.0), and Chameleon, which is released by request only.
The other 8 of the paper's 10 benchmarks are reproduced above (BFree-Online only on the images still online).

Per-generator tables (balanced acc / AP / AUROC / real-acc / fake-acc), from `repro/bench_results/cure_repro_*.json`:

**UFD-8** — dalle 97.45 / 99.73 / 99.66 · glide_100_10 95.25 / 99.30 / 99.10 · glide_100_27 93.20 / 99.17 / 99.01 ·
glide_50_27 92.65 / 98.93 / 98.62 · guided 79.25 / 96.68 / 95.80 · ldm_100 97.55 / 99.89 / 99.89 ·
ldm_200 97.95 / 99.91 / 99.92 · ldm_200_cfg 98.20 / 99.88 / 99.86

**ForenSynths-13** — progan 95.20 · stylegan 87.15 · stylegan2 79.11 · biggan 93.50 · cyclegan 70.36 ·
stargan 92.90 · gaugan 94.21 · deepfake 64.46 · crn 72.34 · imle 74.62 · san 75.11 · seeingdark 72.78 ·
whichfaceisreal 93.60 (the low accuracies are fake-recall: cyclegan 41.8, deepfake 29.7; AP/AUROC are 95–99 on those)

**GenImage-8** — sdv1.4 99.73 · sdv1.5 99.80 · wukong 99.87 · midjourney 85.83 · glide 85.33 · vqdm 82.43 ·
biggan 67.80 · adm 60.63

**Rescore note.** CuRe's rows in `benchmark/generator_runs/*` and `repro/ideogram_bench*` were first scored without its
official PNG to JPEG q96 step (the scorers did not pass the source-format hint then). They were rescored with the step on
2026-11-08 (AUROC changes between -2.1 and +0.9, ranks unchanged); see `docs/BENCHMARKS.md`.

## Training recipe (from the paper; no official training code)

`experiments/image/cure.sh` runs the three stages of the paper (Sec. 3, 4.1-4.2, Appendix A) on GenImage SD v1.4 train
(162,000 ImageNet reals + 162,000 SD v1.4 images) and writes `cure_trained.pth` with exactly the released adapter's 100
tensors, so it is scored like the official weights: `bench_eval.py --model cure --ckpt <SAVE>/stage23/cure_trained.pth`.

| stage | what (paper) | where |
|---|---|---|
| 1 | x_a = (1 - a) x_r + a x_g with independent real / generated endpoints, a ~ U(0, 1) per pair; linear ratio head + sigmoid on the adapted feature, BCE with soft target a; LoRA on Q and V of all 24 layers (rank 64, alpha 128, dropout 0); AdamW lr 1e-5, weight decay 0.1 (LoRA) / 0.01 (head); 5 epochs, per-GPU batch 64, 2 warmup epochs then cosine to 1e-6 | `main.py` with `trainer/image/cure.yaml`, `dataset/image/cure_train.yaml`, `ModelConfig.training_stage=ratio` (`dataset.CuReMixDataset`, `SynthScopeCuReModel.compute_loss`) |
| 2 | frozen encoder; 20,000 endpoint pairs; RF / RR / FF trajectories on {0, .2, .4, .6, .8, 1}; slopes (Eq. 6); C_F, C_M, C_S (Eq. 7-9) trace-normalised; generalised eigenproblem (Eq. 10), top k = 128 -> W, mu = mean RF trajectory feature | `train_stages.py stage2`, `subspace.py` |
| 3 | only Linear(128, 2) on W^T (z - mu): dropout 0.05, CE, AdamW lr 2e-4, weight decay 0.1, batch 256, 5 epochs | `train_stages.py stage3` (features cached once) |

**Choices the paper leaves open** (each a documented assumption; the released `W` / `mu` allow checking stage 2):

| open point | SynthScope choice | reason |
|---|---|---|
| training preprocessing | each endpoint goes through the released test pipeline (`cure_eval`: PNG -> JPEG q96, 390 resize / crop, 336, JPEG q70, mean = std = 0.5), then the mixture is formed | the paper re-encodes the PNG fakes as JPEG q96 (the pipeline's first step); the subspace and head are used with this pipeline at test time; no augmentation is mentioned. Check (consistency, not proof): with the released encoder, stage 2 on this preprocessing gives a mu with cosine 0.999998 to the released mu (relative distance 0.19 %, 20,000 pairs per type on the full data, job 66720); the same run without the final JPEG q70 pass (one control) gives cosine 0.982 / distance 19 % and a 17x larger off-diagonal ratio, so the mu match does discriminate that choice, whereas the subspace overlap (0.89 vs 0.86) does not. It is therefore consistent with the release's stage 2 having used this pipeline, but it does not establish it (mixing before q70, and stage 1's own preprocessing, are untested; stage 1 cannot be checked this way). See *Trained checkpoint* |
| where mixing happens | on the preprocessed tensors | normalisation is affine, so this equals the pixel mixture of the preprocessed images |
| pairing / epoch | one item per real image (162k per epoch), its generated partner and a drawn at random per item (with replacement), fresh every epoch | "independently drawn" endpoints, "each training pair receives one independently sampled mixing ratio" |
| warmup / cosine shape | linear from 0 per optimizer step over 2 epochs to 1e-5, then cosine over the remaining 3 epochs to 1e-6 (`timm_cosine.warmup_prefix`) | the usual meaning of "N warmup epochs and cosine decay to a minimum lr" (MAE-style) |
| GPU count / global batch | per-GPU 64 as stated; the global batch is NGPU x 64 | not stated |
| precision, clipping, betas, seed | fp16 autocast with loss scaling, no clipping, torch AdamW betas, seed 0 | the released inference is fp16; nothing else stated |
| stage-1 model selection | the last epoch | not stated |
| ratio-head weight decay | 0.01 on all head parameters (bias too) | "weight decay of 0.1 and 0.01, respectively" |
| pairs per trajectory type | 20,000 for each of RF, RR, FF (360k forwards) | "20,000 sampled endpoint pairs" |
| lambda_s, lambda_m, eps | 1, 1, 1e-3 (`subspace.py` defaults; `--lambda-s/--lambda-m/--eps`) | not stated. Stage 2 with `--reference-adapter` reports the overlap with the released W and fits the (lambda_s, lambda_m, eps) under which the released W is B-orthonormal. With the **released encoder** on 2500 pairs per type of a 2572 + 2572 subset of the training set (job 283104): fit lambda_s 0.90, lambda_m 1.02, eps 1.05e-3 (relative residual 0.09); with 1, 1, 1e-3 the solved W overlaps the released W at 0.74 (random 0.125), its columns have norms 9.0-9.6 (released 6.8-8.8), and the released W is nearly a generalised eigenbasis of our C_F (off-diagonal ratio 0.03). eps 1e-4 gave overlap 0.19 and 6x larger columns (job 283072, 1000 pairs). Final check on the full data (released encoder, 20,000 pairs per type, job 66720): overlap 0.89, fit lambda_s 0.96, lambda_m 1.02, eps 1.00e-3 (residual 0.036), off-diagonal ratio 0.016, so 1, 1, 1e-3 stand |
| eigenvector scale | B-orthonormal (W^T B W = I), as LAPACK / scipy generalised solvers return | the released W is not unit-norm (column norms 6.8-8.8) |
| stage-3 data and schedule | all 324k clean training images, constant lr, last epoch | not stated |

**Memory.** On a 24 GB A5000, stage 1 at micro-batch 64 needs gradient checkpointing (`ModelConfig.grad_checkpointing`,
6.3 GiB peak, 32 images/s); without it micro-batch 32 takes 14.1 GiB (43 images/s), so a 48 GB GPU can run 64 without
checkpointing, or use `MICRO=32` (gradient accumulation 2, same per-GPU batch of 64). Checkpointing gives the same features and
gradients (`tests/test_cure_recipe.py`).

**Smoke run** (job 283104, one A5000, log `repro/logs/curesmoke_283104.out`; a 2572 + 2572 subset of the training set,
every 63rd image of each class; not a result). Stage 1, 200 optimizer steps at per-GPU batch 64 with checkpointing: the
validation BCE on 256 fixed mixtures falls 0.738, 0.714, 0.698, 0.683, 0.674 over the 5 epochs (0.5 is the floor for
a ~ U(0, 1)). Stage 2 on 200 pairs per type, stage 3 on the 5144 subset images (eval-mode accuracy real 0.63 / fake 0.66
after 5 epochs of 21 steps), export: 100 tensors, reloaded logits equal the trained head's (max |diff| 3.6e-6), and
`synthscope.load("cure", checkpoint=...)` loads it with the same status as the released adapter (`loaded 100 tensors,
missing=295` frozen backbone keys, `unexpected=0`). Stage-1 lr (job 283137, `repro/logs/curesmlr_283137.out`): all 45 logged steps equal the intended schedule (linear from 0, 1.000e-05 at the start of epoch 2, cosine to 1.002e-06 at the last step).

**Tests.** `tests/test_cure_recipe.py` (CPU): the mixing items, the BCE soft-target loss, the trainable set and weight-decay
groups, gradient checkpointing, the trainer's logged lr at every step against the intended schedule, Eq. 6-10 against numpy
/ scipy, and the export format (the released adapter's keys, shapes and dtypes).

## Trained checkpoint (SynthScope_train): does the recipe reproduce the paper?

**No.** The checkpoint trained with the recipe above (`experiments/image/cure.sh`, code 2966f64, DAS6 job 66658 on 3x A6000: global batch 192, stage 1 about 1.5 h; sha256 of `cure_trained.pth` 38390ca9a7024ec3..., never uploaded) learns the training task and the training distribution, but it does **not** reach the paper's numbers on any benchmark the official adapter reproduces. It was scored with the official rows' exact protocol (`bench_eval --ckpt`, `--bs 16`, same data roots, AIGIBench `--select` of the paper's 20 subsets on full data, no `--limit`; DAS6 job 66721, tags `cure_trained_<bench>`, logs `repro/logs/cure_trained_<bench>_das6_66721.full.log`). Verdict by the repo's tolerance (reproduced = acc >= paper - 2.0 and AUROC >= paper - 2.5):

| Benchmark | Paper acc / AUROC | Official adapter acc / AUROC | **Trained acc / AUROC** | Trained - paper (acc) | Trained real acc / fake acc (mean per generator) | Verdict |
|---|---|---|---|---|---|---|
| AIGIBench (20 subsets) | 92.8 / 97.5 | 92.85 / 97.48 | **85.05 / 91.84** | -7.8 | 98.0 / 72.1 (official 93.8 / 91.9) | not reproduced |
| AIGCDetect (17) | 89.3 / 98.0 | 89.30 / 98.02 | **81.55 / 92.22** | -7.8 | 99.6 / 63.5 (official 98.7 / 79.9) | not reproduced |
| EvalGEN (5) | 98.1 / - | 98.14 / 99.66 | **90.65 / 95.96** | -7.4 | 99.8 / 81.5 (official 99.7 / 96.6) | not reproduced |
| SynthWildX (3) | 92.7 / - | 92.69 / 96.50 | **76.11 / 86.84** | -16.6 | 96.6 / 55.6 (official 93.0 / 92.4) | not reproduced |
| WildRF (3) | 95.8 / - | 95.89 / 99.14 | **85.80 / 94.59** | -10.0 | 98.7 / 72.9 (official 93.6 / 98.2) | not reproduced |
| BFree-Online (807 images) | 90.9 / - | 91.64 / 98.08 | **66.64 / 85.77** | -24.3 | 98.9 / 34.4 (official 96.7 / 86.6) | not reproduced |
| RRDataset (3) | 81.5 / - | 82.04 / 88.88 | **66.36 / 76.87** | -15.1 | 99.0 / 33.7 (official 94.4 / 69.7) | not reproduced |
| RealChain (7) | 79.7 / - | 79.69 / 85.47 | **63.33 / 76.50** | -16.4 | 98.9 / 27.8 (official 94.8 / 64.6) | not reproduced |
| UFD-8 (diffusion half) | UnivFD 82.0 (aggregation unspecified) | 93.94 / 98.98 | 82.69 / 93.85 | n/a | 100.0 / 65.4 (official 99.6 / 88.2) | reference (not comparable) |
| ForenSynths-13 | - | 81.95 / 95.02 | 75.83 / 93.56 | n/a | 99.5 / 52.2 (official 96.2 / 67.7) | reference |
| GenImage-8 (`--limit 1500`) | - | 85.18 / 97.00 | 77.70 / 87.88 | n/a | 99.9 / 55.5 (official 99.7 / 70.7) | reference |

All numbers are from `repro/bench_results/cure_trained_*.json` (each checked against its log's RESULT line); the official columns are the existing `cure_*` rows. The unweighted mean of the eight paper-comparable benchmarks is 76.9 acc for the trained checkpoint against 90.1 for the paper's eight numbers (the paper reports 89.7 over ten).

What the numbers say:
- **It is a conservative detector.** Real accuracy stays at 98-100 % everywhere, fake recall is what collapses (BFree-Online 34 %, RRDataset 34 %, RealChain 28 %, SynthWildX 56 %); the official adapter has real/fake accuracy balanced at 0.5. Part of the gap is therefore threshold calibration, but AUROC is also 3.7-12.3 points lower than the official adapter's (e.g. SynthWildX 86.8 vs 96.5, RRDataset 76.9 vs 88.9), so the ranking is worse too, not only the operating point.
- **In-distribution it is fine.** SD v1.4 99.97, SD v1.5 99.93, Wukong 99.93 on GenImage (official 99.73 / 99.80 / 99.87); stage-3 training accuracy 99.8 %; stage-1 validation BCE 0.5099 (floor 0.5 for a ~ U(0, 1); epoch 4, `repro/logs/curetrain_das6_66658.out`). The failure is generalisation to other generators and to in-the-wild images, which is exactly what the paper's training is supposed to buy.
- **Where the loss comes from** is examined by the stage-2 parity check and the hybrid-checkpoint diagnostic below.

**Stage-2 parity check on the full data (review item M2).** Stage 2 run with the **released** encoder (`cure.pth` LoRA), 20,000 pairs per trajectory type on all 162k + 162k training images, once with the recipe's endpoint preprocessing and once with ONE control (the final JPEG q70 pass removed; same seed, same pairs). DAS6 job 66720 (`repro/logs/curepar_std_das6_66720.full.log`, `curepar_noq70_das6_66720.full.log`):

| | recipe preprocessing | control (no q70) | random |
|---|---|---|---|
| overlap of the solved W with the released W | 0.892 | 0.864 | 0.125 |
| cosine of mu to the released mu | 0.999998 | 0.982 | |
| relative distance of mu | 0.19 % | 19.2 % | |
| off-diagonal ratio of the released `W^T C_F W` | 0.016 | 0.276 | |
| fitted (lambda_s, lambda_m, eps) / residual | 0.96, 1.02, 1.0e-3 / 0.036 | 1.03, 0.80, 1.06e-3 / 0.103 | |

The subspace overlap barely separates the two pipelines (0.89 vs 0.86, both far above random), so an overlap alone does **not** identify the release's preprocessing. The mu distance (100x) and the off-diagonal ratio (17x) do, and the recipe's pipeline is the one that matches: this is the evidence that the released stage 2 used the endpoint preprocessing assumed here (the earlier subset claim was too strong). Mixing before the q70 step was not tested. With the fitted values the lambdas are 1, 1 and eps 1e-3 as assumed. Even with the released encoder the full-data W overlaps the released one at 0.89, not 1.0: the pair sampling and other unstated details of stage 2 are not pinned down. For the trained encoder stage 2 gave overlap 0.685 with the released W, mu cosine 0.696, and one dominant generalised eigenvalue (2.856, then 0.017) in `curetrain_das6_66658.out`; the released encoder's own spectrum is 2.157, then 0.039.

HYBRID_SECTION

## Usage

```bash
synthscope detect photo.png --model cure          # needs cure.pth in $SYNTHSCOPE_CKPT_DIR (official adapter) + HF access
python benchmark/bench_eval.py --model cure --benchmark ufd --data <UFD/test> --tag cure_repro_ufd
```

`cure.pth` = copy of the official `weights/cure_adapter.pt`. Use the registered transform (`CuReTransform`):
the PNG→JPEG96 step and the final JPEG‑q70 pass are part of the method.
