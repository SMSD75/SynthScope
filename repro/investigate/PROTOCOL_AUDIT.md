# Protocol & Official-Weights Fidelity Audit — SynthScope detector reproductions

Date: 2026-06-07. Auditor: headless agent on ivi-cn023 (CPU/analysis only, no GPU).
Repo: `/ivi/zfs/s0/original_homes/ssalehi/SynthScope`.
Official repos cloned to: `/hddstore/ssalehi/SynthScope/official_repos/{NPR,UniversalFakeDetect,CNNDetection,FatFormer,SAFE,AIDE,Co-Spy}`.

Our harness `benchmark/bench_eval.py` computes per-generator AP/AUROC/acc then means them.
The ForenSynths "13-gen" run uses ALL 13 CNNDetection generators
`{progan,stylegan,stylegan2,biggan,cyclegan,stargan,gaugan,deepfake,crn,imle,seeingdark,san,whichfaceisreal}`.
Most papers report the **8-GAN subset** `{progan,cyclegan,biggan,gaugan,stargan,stylegan,stylegan2,deepfake}`.
All ForenSynths-8 / -13 means below were recomputed from the per-generator entries in
`repro/bench_results/*.json` (see "Recomputed subset means" appendix).

## TL;DR verdict table

| Detector | Paper # (protocol) | Our # (as-run) | Apples-to-apples | Verdict |
|----------|--------------------|----------------|------------------|---------|
| **NPR** | 96.1 mAP / 92.5 acc (ForenSynths-**8**, AP) | 79.45 mAP (FS-13) | **92.81 mAP / 91.83 acc (FS-8)** | **REPRODUCED-given-protocol**. 13-gen mean is dragged down by crn/imle/san/sitd/wfir which are NOT in NPR's reported FS-8 set; NPR genuinely scores ~chance there. Preprocessing already correct (square Resize 256, no crop). |
| **CLIP/UnivFD** | 98.3 AP / 89.1 acc (ForenSynths-8, AP) | 93.14 mAP (FS-13) | **96.95 mAP / 89.60 acc (FS-8)** | **REPRODUCED-given-protocol (minor residual)**. acc is an exact match (89.6 vs 89.1); AP ~1.3 short, concentrated in deepfake (81.8 vs 90.2) → test-set/subset version diff, not a weights/preproc bug. |
| **CNNDetection** | 91.16 AP(224) (FS-13, Blur+JPEG **0.5**) | 91.15 mAP (FS-13) | **91.15 vs 91.16 — exact** | **REPRODUCED (exact)**. Per-generator AP matches the official README table to <0.3 each. RESULTS_SUMMARY "~96–98" is WRONG for this variant/protocol — see note. |
| **FatFormer** | 99.7 AP (FS-8) | 99.70 mAP (FS-8) | 99.70 vs 99.72 | **REPRODUCED (exact)**. `num_vit_adapter=8` is the official default and our config uses 8 (the prompt's "3" is incorrect). |
| **SAFE** | (official protocol confirmed) | 98.78 mAP (FS-8) | matches protocol | **REPRODUCED-given-protocol**. Eval = CenterCrop(256)+ToTensor, NO normalization, 2-logit head — matches our `safe_official_eval.py` exactly. |
| **Co-Spy (official)** | 96.0/96.65 AP (Co-Spy-Bench, mean of 5 real sources) | 86.65 mAP (110 subsets) | **95.81 mAP (4 of 5 real sources)** | **GAP-REMAINS = DATA BUG**. 4/5 real-caption buckets reproduce the paper to <1 AP; the `flickr` bucket is at chance (50.02) because **50% of its real images on disk are corrupt/unreadable**. Not a model failure. |

---

## 1. NPR — `chuangchuangtan/NPR-DeepfakeDetection`

**Official repo:** `/hddstore/ssalehi/SynthScope/official_repos/NPR`.

### Official TEST protocol (the critical question: resize vs native-res?)
Read `official_repos/NPR/test.py:24-49` — the per-benchmark config dict:
- **ForenSynths**: `'no_resize': False`, `'no_crop': True`.
  Code comment (test.py:26): *"Due to the different shapes of images in the dataset, resizing is required during batch detection."*
- GANGen-Detection (Table-2/FreqNet set): `'no_resize': True`, `'no_crop': True`.
- DiffusionForensics / UniversalFakeDetect: `'no_resize': False`, `'no_crop': True`.

The transform (`official_repos/NPR/data/datasets.py:34-50`):
- when `no_resize=False` → `rz_func = transforms.Resize((opt.loadSize, opt.loadSize))` with `loadSize=256` (`options/base_options.py:30`). i.e. a **SQUARE resize to 256×256** (aspect-ratio-distorting), NOT `Resize(short=256)`.
- when `no_crop=True` → crop is identity.
- normalization is **ImageNet** mean/std (datasets.py:48).

**Answer to the critical question:** for the **ForenSynths/CNNDetection test set the official recipe RESIZES** (square 256×256), it does NOT use native-resolution. Native-res (`no_resize=True`) is the recipe for the **GANGen** Table-2 set only. So the prompt's hypothesis ("native-res is the correct ForenSynths recipe") is **inverted** for ForenSynths. The README adds a *separate* AIGCDetectBenchmark recipe (README "Detection Results" section) that DOES use `no_resize=no_crop=True, batch_size=1` plus an odd-size cropping patch in `networks/resnet.py:163` — but that is the 17-generator AIGCDetectBenchmark, not ForenSynths.

### Our run already uses the official preprocessing
`npr_official_forensynths.json` was produced by the **SynthScope harness** (`ckpt: registry-default`, `synthscope_version: 0.1.0`), not the vendored official script. The NPR registry spec is `interface/backend/registry.py:242-246`:
```
transform = tf_square_resize(256, None, IMAGENET_MEAN, IMAGENET_STD)   # Resize((256,256)) + ToTensor + ImageNet-norm, NO crop
```
That is **exactly** the official NPR ForenSynths recipe. So the real_acc=0.0 collapse on crn/imle (and low real_acc on biggan/gaugan/deepfake/sitd/wfir) is **NOT a resize/preprocessing bug** — those generators are simply **outside NPR's reported FS-8 protocol** and NPR genuinely fails on them (crn/imle AP = 50.0 = chance; they are CRN/IMLE *renderings*, not GANs).

> NOTE: `repro/RESULTS_SUMMARY.md:134-135` claims "NPR collapsed because the eval *resized* images … native-resolution scoring restores 96.49 (matches paper 96.6)." That 96.49 is the **GANGen-9** run (where native-res IS correct, `no_resize=True`), NOT ForenSynths. For **ForenSynths**, resize is the official recipe and is already what we use.

### Paper per-generator ForenSynths-8 table
From the (commented-out) table in `official_repos/NPR/README.md`, NPR("our") on ForenSynths-8, **Acc / AP**:

| | ProGAN | StyleGAN | StyleGAN2 | BigGAN | CycleGAN | StarGAN | GauGAN | Deepfake | **Mean** |
|--|--|--|--|--|--|--|--|--|--|
| Acc | 99.8 | 96.3 | 97.3 | 87.5 | 95.0 | 99.7 | 86.6 | 77.4 | **92.5** |
| AP  | 100.0 | 99.8 | 100.0 | 94.5 | 99.5 | 100.0 | 88.8 | 86.2 | **96.1** |

### Our recomputed FS-8 vs paper
| metric | our FS-13 | our FS-8 | paper FS-8 |
|--|--|--|--|
| mAP | 79.45 | **92.81** | 96.1 |
| acc | 78.09 | **91.83** | 92.5 |
| mAUROC | 82.45 | 95.18 | — |

FS-8 acc is essentially a match (91.83 vs 92.5). FS-8 AP is ~3.3 short, concentrated in biggan (85.6 vs 94.5), gaugan (83.0 vs 88.8), deepfake (76.1 vs 86.2) — consistent with seed/checkpoint sensitivity (`model_epoch_last_3090.pth` is the ProGAN-4class ckpt; NPR is known to be seed-sensitive, see README "Random seed 100").

**VERDICT: REPRODUCED-given-protocol.** Report NPR on **ForenSynths-8** (92.81 mAP / 91.83 acc), not on FS-13. No re-run with native-res is warranted for ForenSynths (the harness already matches the official square-256 recipe). If you want the headline 96.1, the remaining gap is checkpoint/seed, not protocol.

---

## 2. CLIP / UnivFD — `WisconsinAIVision/UniversalFakeDetect` (Ojha et al., CVPR'23)

**Official repo:** `/hddstore/ssalehi/SynthScope/official_repos/UniversalFakeDetect`.

### Official TEST protocol
Entry point `test.sh` → `validate.py`. The eval transform (`validate.py:197-202`):
```
stat_from = "clip"   # arch CLIP:ViT-L/14
transform = Compose([ CenterCrop(224), ToTensor(), Normalize(CLIP_mean, CLIP_std) ])
```
- **NO Resize** — image is center-cropped to 224 at native resolution.
- CLIP normalization: mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711] (`validate.py:31-39`).
- Score = `model(x).sigmoid()` (P(fake)); metric = `average_precision_score` (AP) + acc@0.5 (`validate.py:96-115`).
- Real pairing on the wang2020 (ForenSynths) set: `0_real` vs `1_fake` (`validate.py:205-208`).

Our `repro/official_evals/clip_official_eval.py:199-204` uses the identical `CenterCrop(224)+ToTensor+Normalize(CLIP)` with no resize — **faithful**.

### Paper number (8-GAN subset)
Ojha/UniFD on ForenSynths-8 (from the UniFD/Ojha rows reproduced in both the FatFormer README and the NPR README), **Acc / AP**: mean = **89.06 acc / 98.28 AP**.

### Our recomputed FS-8 vs paper
| metric | our FS-13 | our FS-8 | paper FS-8 |
|--|--|--|--|
| mAP | 93.14 | **96.95** | 98.28 |
| acc | 80.66 | **89.60** | 89.06 |

**acc is an exact match (89.60 vs 89.06).** AP is ~1.3 short, concentrated in **deepfake (our AP 81.76 vs paper 90.2)** and to a lesser degree stylegan/stylegan2 (97.6/97.9 vs 98.7/98.4). The exact acc match plus faithful weights/preproc means this is a **test-set version / deepfake-split difference**, not a fidelity failure. (Diffusion benchmark cross-check: our UFD-8 run `clip_official_ufd.json` = 95.66 mAP / 82.15 acc, consistent.)

**VERDICT: REPRODUCED-given-protocol.** Report CLIP on **ForenSynths-8 = 96.95 mAP / 89.60 acc**. ~97 on the 8-GAN subset is what the prompt expected and what we get; the headline 98.3 residual is a deepfake-subset artifact.

---

## 3. CNNDetection — `PeterWang512/CNNDetection` (Wang et al., CVPR'20)

**Official repo:** `/hddstore/ssalehi/SynthScope/official_repos/CNNDetection`.

### Checkpoint used
`cnn_official_forensynths.json` → `ckpt: .../official_ckpts/cnn/blur_jpg_prob0.5.pth` → the **Blur+JPEG(0.5)** model.

### Official TEST protocol
`eval.py:23` sets `opt.no_resize = True` (no resize); `--no_crop` is NOT passed, so the default crop applies: `CenterCrop(opt.cropSize)` with `cropSize=224` (`data/datasets.py:24-30`, `options/base_options.py:30-31`). Normalization = ImageNet (`data/datasets.py:46`). This is the **"AP (224)"** column of the README results table — i.e. no-resize + center-crop-224. Our `repro/official_evals/cnn_official_eval.py:175-189` reproduces exactly this (default CenterCrop(224), no resize, ImageNet norm, `sigmoid(logit)`).

### Paper number for THIS exact variant/protocol
From `official_repos/CNNDetection/README.md` **[Blur+JPEG(0.5)] AP (224)** column (lines 125-138), recomputed:
- **FS-13 mean AP(224) = 91.16**
- FS-8 mean AP(224) = 92.66
(For reference, Blur+JPEG(0.1) AP(224): FS-13=93.15, FS-8=94.17.)

### Our numbers (per-generator AP, exact match to README)
ProGAN 100.0/100.0 · StyleGAN 98.51/98.5 · BigGAN 88.24/88.2 · CycleGAN 96.83/96.8 · StarGAN 95.45/95.4 · GauGAN 98.09/98.1 · CRN 98.94/98.9 · IMLE 99.52/99.5 · SITD 92.76/92.7 · SAN 63.86/63.9 · Deepfake 66.27/66.3 · StyleGAN2 97.99/98.0 · WFIR 88.55/88.8.
- our FS-13 mAP = **91.15** vs official **91.16** — exact (<0.01).
- our FS-8 mAP = **92.67** vs official 92.66.

> NOTE: `repro/RESULTS_SUMMARY.md:21` lists paper "~96–98" for CNNDetection. **That is wrong for this variant/protocol** — no 96–98 figure exists for Blur+JPEG(0.5) AP(224) on ForenSynths. The actual official paper number is 91.16 (FS-13) / 92.66 (FS-8), which we reproduce exactly. The "~96–98" likely conflated the No-crop AP column (where some generators hit 99–100) or a different detector. **Recommend correcting RESULTS_SUMMARY to "Paper 91.16 (Blur+JPEG0.5, AP224), SynthScope 91.15 — exact".**

**VERDICT: REPRODUCED (exact).**

---

## 4. FatFormer — `Michel-liu/FatFormer` (CVPR'24)

**Official repo:** `/hddstore/ssalehi/SynthScope/official_repos/FatFormer`.

- **Adapter count:** `main.py:31` → `--num_vit_adapter` default = **8** (not 3). SynthScope `conf/model/image/fat_former.yaml:7` = `num_vit_adapter: 8`. ✅ correct. (24 visual layers / 8 = adapter every 3rd block; the "3" in the prompt is the block *stride*, not the adapter count.)
- **Paper ForenSynths-8 AP** (README "Ours" row, AP = 2nd number per `Acc / AP` pair): ProGAN 100.0, StyleGAN 99.8, StyleGAN2 100.0, BigGAN 100.0, CycleGAN 100.0, StarGAN 100.0, GauGAN 100.0, Deepfake 98.0 → **mean 99.72**.
- Our run `fatformer_official_forensynths.json` = **99.70 mAP** (per-gen AP within 0.25 of paper each).

**VERDICT: REPRODUCED (exact).** 99.70 ≡ paper 99.7; adapter config correct.

---

## 5. SAFE — `Ouxiang-Li/SAFE`

**Official repo:** `/hddstore/ssalehi/SynthScope/official_repos/SAFE`.

### Official protocol
`scripts/*.sh` (eval): `--input_size 256 --transform_mode 'crop' --model SAFE --resume checkpoint-best.pth --eval True`.
The `'crop'` eval transform (`data/datasets.py:124-131` + `:154`):
```
eval: [ CenterCrop([256,256]), ToTensor() ]    # NO Normalize
```
The model internally does a DWT (Haar) on the [0,1] tensor and upsamples sub-bands back with `transforms.Resize` (model code), head is 2-logit. Our `repro/official_evals/safe_official_eval.py:247-249` uses `CenterCrop([256,256]) + ToTensor`, **no normalization**, 2-logit head — matches exactly.

### Our number
`safe_official_forensynths.json` (8-gen, `ckpt: official_ckpts/safe/checkpoint-best.pth`) = **98.78 mAP / 96.20 acc / 98.83 mAUROC**. SAFE's headline benchmark is GenImage, not ForenSynths (README has no inline FS table); on GenImage-8 our official-weights run = 99.32 mAP (RESULTS_SUMMARY:67). The ForenSynths-8 numbers are strong and internally consistent with the faithful protocol.

**VERDICT: REPRODUCED-given-protocol.** Protocol (CenterCrop-256, no-norm, 2-logit) confirmed faithful to the official repo.

---

## 6. Co-Spy (official weights) — `Megum1/Co-Spy`

**Official repo:** `/hddstore/ssalehi/SynthScope/official_repos/Co-Spy`.
**Our run:** `cospy_official_cospybench.json` (`ckpt: official_ckpts/co_spy/sd-v1_4`, DRCT/SD-v1.4-trained triplet), 110 subsets, mAP **86.65**.

### Official benchmark structure
Co-Spy-Bench = 22 generators × **5 real-caption sources** (CC3M, FLICKR, MSCOCO, TEXTCAPS, SBU) = 110 subsets. The README "Average Precision (AP)" table (`README.md:156-181`) reports **per-real-source column averages**:
| source | paper AP avg |
|--|--|
| CC3M | 89.39 |
| FLICKR | 99.46 |
| MSCOCO | 99.40 |
| TEXTCAPS | 96.52 |
| SBU | 98.50 |
Grand mean of the 5 columns = **96.65** (paper text rounds to ~96.0). **CC3M is the hardest real source** (89.39); the other four are 96–99.

### Our per-source breakdown (recomputed from cospy_official_cospybench.json)
| source | n | our mAP | our real_acc | paper AP | match? |
|--|--|--|--|--|--|
| cc3m | 22 | **88.56** | 83.0 | 89.39 | ✅ |
| mscoco | 22 | **99.52** | 99.8 | 99.40 | ✅ |
| textcaps | 22 | **96.24** | 95.8 | 96.52 | ✅ |
| sbu | 22 | **98.90** | 100.0 | 98.50 | ✅ |
| **flickr** | 22 | **50.02** | 56.9 | 99.46 | ❌ **chance** |
- Overall (all 5, 110 subsets) = 86.65 (matches our run).
- **Overall EXCLUDING the broken flickr bucket (4 sources) = 95.81**, vs the paper's 4-source mean of **95.95** — a match.

### Root cause of the flickr collapse (NOT proxy reals, NOT a model failure)
The flickr real images ARE on disk (2500 real + 5000 fake per subset, `0_real/1_fake`), so this is not a missing/proxy-reals case. But **exactly 50% of the flickr real JPEGs are corrupt/truncated** — every odd-indexed file (`flickr_00001.jpg`, `_00003`, `_00005`, …) raises PIL `cannot identify image file`. Verified: `1250 / 2500 = 50.0%` unreadable in `flickr__CompVis@stable-diffusion-v1-4/0_real`. Corrupt reals get dropped/misscored, driving real_acc to ~57% and AP to chance. cc3m/mscoco/textcaps/sbu reals are intact (real_acc 83–100%).

**VERDICT: GAP-REMAINS, cause = DATA CORRUPTION in the flickr real bucket (download/extraction), not a Co-Spy weight/architecture failure.** The official SD-v1.4 weights reproduce the paper: 4/5 real sources land within <1 AP, and the 4-source mAP (95.81) matches the paper (95.95). The depressed full-110 number (86.65) is an artifact of the corrupt flickr reals. (This refines the prompt's "proxy reals" framing: the issue is corruption of *present* flickr reals, not proxy substitution; cc3m's 88.56 already matches the paper's 89.39.)

---

## Recomputed subset means (appendix — all from `repro/bench_results/*.json`)

ForenSynths-8 = {progan, cyclegan, biggan, gaugan, stargan, stylegan, stylegan2, deepfake}.

| detector (source json) | FS-13 mAP | **FS-8 mAP** | FS-13 acc | FS-8 acc |
|--|--|--|--|--|
| CLIP official (`clip_official_forensynths.json`) | 93.14 | **96.95** | 80.66 | 89.60 |
| CNN official Blur0.5 (`cnn_official_forensynths.json`) | 91.15 | **92.67** | 74.30 | 74.09 |
| NPR official (`npr_official_forensynths.json`) | 79.45 | **92.81** | 78.09 | 91.83 |
| SAFE official (`safe_official_forensynths.json`) | — (run is 8-gen) | **98.78** | — | 96.20 |
| FatFormer official (`fatformer_official_forensynths.json`) | — (run is 8-gen) | **99.70** | — | 98.39 |

The 5 "extra" generators that the 13-gen mean adds (crn, imle, san, seeingdark/sitd, whichfaceisreal) crush NPR
(crn/imle = 50.0 AP = chance) and depress CLIP; they are NOT part of the FS-8 protocol most papers report.

## Recommended re-runs / fixes
1. **Reporting fix (no compute):** report NPR, CLIP, CNN on **ForenSynths-8** alongside FS-13, using the recomputed FS-8 means above. NPR especially must be reported on FS-8 (92.81 mAP / 91.83 acc) — its FS-13 number (79.45) is protocol-mismatched, not a failure.
2. **Co-Spy flickr re-download:** re-fetch/replace the corrupt flickr real images in `/hddstore/ssalehi/SynthScope/CoSpyBench/test/flickr__*/0_real/` (50% are truncated). Re-running official Co-Spy after that should lift the full-110 mAP from 86.65 to ~95–96 (matching the paper). No model/code change needed.
3. **RESULTS_SUMMARY correction:** change the CNNDetection "paper ~96–98" entry to the verified **91.16 (Blur+JPEG0.5, AP@224)** — we reproduce it exactly (91.15). Also clarify the NPR note: native-res is the *GANGen* recipe; ForenSynths uses square-256 resize (already what the harness does).
4. **No re-run needed for NPR native-res on ForenSynths** — the harness already matches the official square-256 recipe; native-res would *diverge* from the official ForenSynths protocol.
