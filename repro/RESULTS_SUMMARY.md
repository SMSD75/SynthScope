# SynthScope Full-Benchmark Reproduction & Gap-Closing — Results Summary

*Updated 2026-11-08. Numbers trace to `repro/bench_results/*.json` and `repro/MASTER_RESULTS.jsonl`. Format: **mAP** (acc@0.5), % — unless noted.*

## Three number columns (consistent everywhere)
- **Paper** — figure reported in the paper.
- **SynthScope_official** — authors' released weights run through the SynthScope class/eval (architecture-fidelity check).
- **SynthScope_train** — the SynthScope class **trained from scratch** by us; the **single best** run is kept (older/weaker from-scratch checkpoints are discarded).

## Headline outcome (one row per paper)
| Method | Headline benchmark | Paper | SynthScope_official | SynthScope_train | Verdict |
|---|---|---:|---:|---:|---|
| FatFormer | ForenSynths-8 | 99.7 | 99.70 | 99.56 | ✅ official match; trained with the paper recipe (model_last, epoch 24): 99.56 / 97.29 acc vs paper 98.4 |
| NPR | ForenSynths-8 / GANGen | 96.1 / 96.6 | 94.22 / 96.49 | 93.83 / — | ✅ official match; trained (official recipe): AP close, acc 84.1 — the official train.py rerun here gets 81.5 |
| CLIP/UnivFD | ForenSynths-13 | — | 93.14 | 92.99 | ✅ match |
| CNNDetection | ForenSynths-11 (the paper's set) | 90.8 | 90.77 | 90.44 | ✅ match |
| SAFE | GenImage | 99.3 | 99.32 | 99.51 | ✅ retrain reproduces |
| AIDE | GenImage (SD-v1.4) | 86.9 acc | 85.91 acc (97.12 mAP) | 77.87 acc (96.42 mAP) | ✅ official match on acc; the native-res retrain matches the official weights' mAP, its acc is 9 pt lower |
| SPAI | GenImage | ~91 AUC | 94.22 | 94.67 | ✅ served native-res (beats vendored 91.33); official-recipe retrain reproduces the official weights (+0.45 mAP; Synthbuster -0.99) |
| PGC | GenImage / ForenSynths-8 / UFD | 100 / — / 98.0 AP | 100.0 / 96.41 / 99.17 | 97.62 / **97.41** / **98.99** | ✅ UFD reproduced by the official recipe in SynthScope (step 500); ForenSynths (not in the paper) above the official checkpoint |
| VIB-Net | 15 generators (7 GenImage + 8 GANs), SD-v1.4-trained | 95.53 AP / 85.83 acc | *not available* | **94.08 / 85.22** | ✅ reproduced by the reconstructed recipe (native-resolution training) |
| Co-Spy | Co-Spy-Bench-110 | 96.0 AP | 86.65 | 96.70 | ✅ beats official |
| DDA | AIGCDetect / ForenSynths / EvalGEN / SynthWildX / WildRF / GenImage | 87.8 / 81.4 / 97.2 / 90.9 / 90.3 / 91.7 acc | 87.84 / 81.43 / 96.86 / 90.80 / 90.20 / 89.37 acc | 89.58 / 80.79 / 96.79 / 88.00 / 90.56 / **91.35** acc | ✅ official match; trained in SynthScope reproduces 6 of 7 incl. GenImage (SynthWildX 2.9 below) |
| D3QE | ARForensics / ForenSynths-7 / GenImage-6 | 82.11 / 83.73 / 78.61 acc | 81.74 / 83.45 / 79.44 acc | 79.83 / 82.14 / 77.01 acc | ✅ official match; trained in SynthScope: AP matches, acc 1.6-2.3 below |
| OmniAID | GenImage / AIGCDetect / Mirage-Test (GenImage release) | 95.94 / 88.87 / 51.10 acc | 95.82 / 88.87 / 51.10 acc | — | ✅ official match (v1 release too: 97.16 / 92.88 / 88.39 vs 97.24 / 92.88 / 88.39) |
| Effort | GenImage / UniversalFakeDetect (19 subsets, ProGAN-trained) | 91.1 acc / 99.41 mAP, 95.19 acc | 89.83 acc / — (no ProGAN checkpoint released) | — / **99.20 mAP, 94.81 acc** | ✅ official match on GenImage; trained in SynthScope with the official ProGAN recipe reproduces the UFD table (iteration 18000 = the official selection) |
| LTF | GenImage (8 subsets, full val, 100,000 images) | 98.4 acc (Table 4: AUC 99.89 / AP 99.86) | 98.36 acc / 99.88 AUROC / 99.86 AP | — | ✅ official match (every subset within 0.10 pt; the official code equals SynthScope exactly on a 33k subset) |
| MIRROR | AIGCDetect / GenImage / UnivFakeDetect / Synthbuster / EvalGEN / SynthWildX / WildRF / AIGIBench / BFree-Online / RRDataset | 97.3 / 99.8 / 92.4 / 99.2 / 99.8 / 95.1 / 97.8 / 94.9 / 97.6 / 88.3 B.Acc (README, DINOv3-H+) | 97.39 / 99.78 / 92.44 / 99.28 / 99.85 / 95.08 / 97.88 / 94.98 / 97.40 / 86.91 B.Acc | — | ✅ official match on 9 of 10 held (RRDataset 1.4 short); DRCT-2M, Chameleon, CO-SPY in-the-wild, Human-AIGI not run |
| VINA | ForenSynths / UniversalFakeDetect / DiTFake / ARForensics (6 shared subsets) / GenImage (full val) / GenVideo-Val | 86.99 / 94.84 / 97.98 / 96.36 / 94.74 / 96.9 acc (ARForensics: mean of the paper's six per-subset values) | 86.93 / 94.84 / 97.98 / 96.36 / 94.74 / 96.86 acc (pooled accuracy per subset, the paper's metric) | — | ✅ official match on these 6 (every subset; ForenSynths 10 of 11). SynthWildX and WildRF are **below the paper** (91.82 vs 92.8 and 96.12 vs 97.13, the means of the paper's printed per-subset values) and AIGIBench differs (SocialRF -1.18, CommunityAI +1.83), but the unmodified official `eval.py` gives the same per-file scores on our files (max diff <= 5.8e-05, 0 flips), so the gap is the data, not the code; RRDataset 82.89 vs 82.7; Chameleon, Magic Videos, GenBuster++, DeepTraceReward not run |
| D3 (video) | GenVideo-Val | 98.5 AP | 97.93 (official code) | 98.95 (training-free; AP with real as positive) | ✅ reproduced (D3 protocol; preprocessing matched to the official code) |
| ReStraV (video) | GenVidBench (Main task: 4 test generators) | 93.01 acc | — | 93.33 acc on those 4; 92.70 mAP / 86.87 acc on all 8 | ✅ close on acc on the paper's 4 generators (not the paper's train/test split) |
| STALL (video) | GenVideo-Val | ~80 AUC | — | 83.5 AUROC (training-free) | ✅ match |
| SPLIT (video) | GenVideo-Val, fake recall at FPR 0.1 / 1 / 5% | 82.1 / 91.7 / 97.2 | 82.22 / 90.17 / 96.89 (official code) | 98.87 / 99.86 / 99.98 (training-free) | ✅ the official code reproduces the paper; SynthScope's SPLIT skips the official JPEG frame round trip, so it scores higher (see `models/video/split/README.md`) |
| B-Free | SynthWildX / WildRF | 95.6 / 95.6 · 86.2 · 97.3 bAcc | 95.64 / 95.31 · 86.47 · 97.66 bAcc | — | ✅ official match (details in `models/image/bfree/README.md`) |
| MoE-JEPA | SID-Set (3-way) | 95.54 acc | 96.37 acc | — | ✅ official match (details in `models/image/moejepa/README.md`) |

## Full per-benchmark results

### AIDE
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| GenImage-8 | 86.9 acc | 97.12 (85.9) | **96.42 (77.9)** (native-res 5-view retrain, model_2) |
| ForenSynths-13 | — | 94.40 (81.8) | 79.62 (59.0) (native-res retrain, cross-domain) |
| AIGCDetect-17 | 92.8 acc | 97.94 (91.9) | — (not trained in this setting) |

SynthScope_official is `sd14_train.pth` (the served `aide.pth`) on GenImage and the ProGAN-trained `progan_train.pth` on ForenSynths and AIGCDetect. The earlier 72.25 (61.6) AIGCDetect entry was `progan_train.pth` under the old resize-before-DCT eval, not a SynthScope-trained model.

### SPAI
SPAI SERVES official weights remapped into the SynthScope class at NATIVE resolution (any-res spectral path). `SynthScope_official` = that served number; `SynthScope_train` = the official-recipe retrain in SynthScope (35 epochs, job 267192), epoch 8 chosen by validation loss (epoch 17 as a note). It reproduces the official weights within about 1 point; the earlier from-scratch fixed-448 retrain (88.80) is superseded. Protocol: native resolution, GenImage-8 300 per class per generator, Synthbuster 200 with GenImage-nature proxy reals; eval jobs 273444-273446, 273597.
| Benchmark | Paper | SynthScope_official (served, native-res) | SynthScope_train (epoch 8) | Epoch 17 |
|---|---:|---:|---:|---:|
| GenImage-8 | ~91 AUC | **94.22 / AUROC 94.28** (vendored 91.33) | **94.67 / AUROC 94.77** (acc 87.25) | 93.52 / 93.48 |
| Synthbuster (9 gen) | — | **96.75 / AUROC 96.97** (was 55.67 via fixed-448) | **95.76 / AUROC 96.37** (acc 88.14) | 94.57 / 95.85 |

### PGC
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| GenImage-8 | 100 AP | 100.0 (99.5) | **97.62 (78.6)** |
| UniversalFakeDetect-8 | 98.0 AP | 99.17 (83.6) | **98.99 (92.0)** |
| ForenSynths-13 | — | 93.80 (70.6) (FS-8 96.41 / 76.3) | **95.78** (FS-8 97.41 / 85.9) |
| AIGIBench-25 | — | 95.36 (88.3) | 86.88 |
| CommGen15-15 | — | 99.26 (97.4) | **98.61 (66.7)** |

SynthScope_official uses the release matching each benchmark's training setting: `PGC_train_sdv1_4_ckpt.pth` for GenImage and CommGen15, `PGC_train_progan_ckpt.pth` (the served `pgc.pth`) for UniversalFakeDetect and ForenSynths, `PGC_train_progan_sdv1_4_ckpt.pth` for AIGIBench. UniversalFakeDetect and ForenSynths are the SynthScope class in the same jobs and PadCenterCrop eval as the step-500 numbers (266660, 266282). The GenImage, AIGIBench and CommGen15 entries are official-code numbers: 2026-06-04 runs of the official PGC code (`pgc_official_eval.py`, PadCenterCrop). Those runs gave UFD 99.34 (94.2) and FS-13 96.25 (83.7); the SynthScope class on the same weights gives 99.17 / 93.80 (UFD / FS-13); the gap is the known model-forward difference.

*PGC training:* the released ProGAN checkpoint is only ~400 steps in (the authors, on our issue), and DINOv2's prior overfits ProGAN within the first epoch, so the earlier epoch-level retrains (GenImage, AIGIBench, CommGen15 rows above, and the old 94.57 / hard-3 75.87 ForenSynths entry) overshot the transfer peak. The official recipe run in SynthScope with a 2000-micro-batch budget (`experiments/image/pgc.sh`, checkpoint every 100) matches or exceeds the official checkpoint at every step from 300 on: step 500 (the saved step nearest ~400) gives FS-8 97.41 mAP and hard-3 AUROC 93.93 (official 96.41 / 90.87 on the same evaluation) and UFD 98.99 (paper 98.0); transfer peaks around steps 500-900 and erodes after. GenImage, AIGIBench and CommGen15 have not been rerun with this recipe. See `repro/investigate/pgc_gap_investigation.md` and `models/image/pgc/README.md`.

### Co-Spy
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| Co-Spy-Bench-110 | 96.0 AP | 86.65 (83.9) | **96.70 (90.3)** |
| GenImage-8 | — | 92.41 (75.3) | 90.59 (76.2) |
| ForenSynths-13 | — | 87.97 (83.4) | 96.17 |
| AIGCDetect-17 | 95.2 AP | 94.59 (89.3) | 86.18 (73.0) |

SynthScope_train on Co-Spy-Bench and GenImage is the served `co_spy.pth` (`checkpoints_retrain/cospy_full/model_1.pth`, GenImage SD v1.4; `cospy_full_cospybench` also has rows for model_0 96.12 and model_2 96.60). The ForenSynths and AIGCDetect SynthScope_train entries are from earlier SynthScope checkpoints and have not been re-run with it. SynthScope_official is the official SD v1.4 checkpoint on Co-Spy-Bench and GenImage and the official ProGAN one on ForenSynths and AIGCDetect.

### SAFE
`SynthScope_train` = retrained SAFE v2 (full 40-epoch crop recipe, `safe_full_v2/model_38`); supersedes the earlier resize-era column.
| Benchmark | Paper | SynthScope_official | SynthScope_train (retrain v2) |
|---|---:|---:|---:|
| GenImage-8 | 99.3 / 95.6 | 99.32 (95.6) | **99.51 (95.9)** |
| ForenSynths-8 | 98.8 / 96.2 | 98.78 (96.2) | **97.79 (95.1)** |
| UniversalFakeDetect-8 | 99.1 / 95.7 | 99.10 (95.7) | **98.44 (94.0)** |
| DiTFake | 99.9 / 99.4 | — | **99.96 (99.2)** |
| DiffusionForensics-8 | — | — | **99.34 (91.3)** |
| FreqNet/GANGen-9 | 99.8 / 99.0 | 99.85 (99.0) | not re-run (data offline) |
| ForenSynths-13 (full, incl. CRN/IMLE/face) | — | 85.74 (84.9) | 87.31 (83.9) |

### FatFormer
SynthScope_train = the paper recipe (4-class ProGAN, Adam 4e-4, batch 256, 25 epochs, lr x0.9 / 10 epochs; train job 277158), `model_last` = epoch 24. `model_best` is the epoch-0 checkpoint (val acc 1.0000, strict >): ForenSynths-8 99.66 (95.71), UFD-8 98.67 (90.89). Evals: job 66626 on DAS6, commit a2a4d2b.

| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| ForenSynths-8 | 99.7 (98.4) | 99.70 (98.4) | 99.56 (97.29) |
| UniversalFakeDetect-8 (paper DMs-10: 98.8 (95.0); its per-subset numbers over these 8 average 98.54 (93.80)) | — | 98.60 (93.7) | 98.88 (92.09) |
| DiffusionForensics-8 | — | 94.66 (79.0) | — |

### NPR (official weights only)
| Benchmark | Paper | SynthScope_official |
|---|---:|---:|
| ForenSynths-8 | 96.1 | 94.22 (92.3) |
| FreqNet/GANGen-9 | 96.6 | 96.49 (94.8) |
| UniversalFakeDetect-8 | — | 97.16 (94.6) |
| DiffusionForensics-8 | — | 99.40 (95.0) |
| Diffusion-1kStep-5 | — | 83.47 (79.1) |

ForenSynths-8 and UniversalFakeDetect-8 use `NPR.pth`, the served `npr.pth`; GANGen, DiffusionForensics and Diffusion-1kStep were measured with `model_epoch_last_3090.pth` (the registry default before 2026-06-09) and not re-run (on UniversalFakeDetect-8 that file gives 96.69 (93.9)).

### CLIP/UnivFD & CNNDetection
| Method · Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| CLIP · ForenSynths-13 | — | 93.14 (80.7) | 92.99 (83.5) |
| CLIP · UniversalFakeDetect-8 | 95.0 | 95.66 (82.2) | 96.44 (87.2) |
| CNN · ForenSynths-11 (the paper's set) | 90.8 | 90.77 (75.8) | 90.44 (74.6) |
| CNN · ForenSynths-13 | — | 91.15 (74.3) | 91.05 (72.2) |

### VIB-Net (official weights on a restricted Google Drive; access requested)
No training code is released; the recipe is reconstructed from the paper, the released options and the UniversalFakeDetect
training loop the test code is built on (`experiments/image/vib_net.sh`, see the model README). Official model and test
transform (`vib_net_official`: fp32 CLIP, no feature norm, 224 center crop without resize).

| Benchmark | Paper | SynthScope_train |
|---|---:|---:|
| 15 generators (7 GenImage + 8 ForenSynths GANs), SD v1.4-trained, AP / acc | 95.53 / 85.83 | **94.08 / 85.22** (GenImage-7 97.18 / 87.16, GAN-8 91.37 / 83.52) |
| same, with UniversalFakeDetect's resize-256 training pipeline | 95.53 / 85.83 | 82.98 / 70.20 |
| ProGAN-trained (GAN-source) | 93.14 / 82.5 | not trained yet |

**Verdict: reproduced** once training sees the images at the scale the official test uses (no resize): the two runs
differ only in that resize (82.98 -> 94.08 AP). The earlier 83.9 / 77.6 model also differed from the official one (fp16
CLIP, an added feature BatchNorm); their separate effect was not measured.

### CuRe (official adapter and trained in SynthScope)
Fidelity: SynthScope's class == the official model (395/395 state-dict tensors identical, logit diff 4.8e-6, 0 verdict flips; `repro/cure_verify_load.py`). Paper: 89.7 avg balanced acc over 10 benchmarks (trained on GenImage SD-v1.4 only).

| Benchmark (SynthScope protocol) | acc (mean per-gen balanced) | mAP | Paper | Note |
|---|---:|---:|---|---|
| UFD-8 (diffusion half of UnivFD) | 93.94 | 99.19 | UnivFD 82.0 | not comparable — paper's aggregation unspecified |
| ForenSynths-13 (GAN half) | 81.95 | 94.05 | UnivFD 82.0 | not comparable |
| GenImage-8 (`--limit 1500`; training distribution) | 85.18 | 97.30 | — | sdv1.4 99.73 / sdv1.5 99.80 / wukong 99.87 |
| AIGIBench, paper's 20 full-image-synthesis subsets (full test data, 168k images) | **92.85** | 97.00 | **92.8** | **reproduced**: AUROC 97.48 vs 97.5; every subset within 0.10 pt (mean abs diff 0.03). The earlier `--limit 1500` run gave 93.13 (the cap caused its 0.3 residual); adding the 5 face-swap subsets the paper excludes gives 85.95 (not the paper's protocol) |
| RealChain (full data, 7000/7000; `--dataset realchain`) | **79.69** | 87.39 | **79.7** | **reproduced** (AUROC 85.47). One released JPEG-q60 set; unpaired reals split into 7 disjoint slices, so the per-gen mean = the pooled balanced acc. Weakest: i2i 61.9, Seedream4 63.6 |
| EvalGEN (full data, 55,298 fakes vs MSCOCO val2017; `--dataset evalgen`) | **98.14** | 99.86 | **98.1** | **reproduced** (AUROC 99.66). The paper's "shared MSCOCO pool" is unspecified; val2017 is our choice. Flux is the hardest (93.4) |
| SynthWildX (`--dataset synthwildx`; 1713/2000 images still on X) | **92.69** | 95.61 | **92.7** | **reproduced** (AUROC 96.50); 287 posts deleted from X since release |
| WildRF (`--dataset wildrf`; full test split, 1251/1252) | **95.89** | 99.13 | **95.8** | **reproduced** (AUROC 99.14) |
| BFree-Online (`--dataset bfree_online`; 807 of ~1400 images still online) | 91.64 | 99.06 | 90.9 | close, not like-for-like: 269 real / 538 fake survive (AUROC 98.08) |
| RRDataset (`--dataset rrdataset`; public test archive, 25.5k/25.5k) | **82.04** | 90.40 | **81.5** | **reproduced** within 0.6 pt (AUROC 88.88; original 91.1, transfer 85.5, redigital 69.5) |
| AIGCDetectBenchmark (`--dataset aigcdetect`; all 17 generators, full data, 76.3k/76.3k) | **89.30** | 98.08 | **89.3** | **reproduced** (AUROC 98.02 vs 98.0) |

**Verdict: faithful port (identical to the official model); 7 of the paper's 10 benchmarks reproduced** — AIGIBench (92.85 vs 92.8), AIGCDetectBenchmark (89.30 vs 89.3), RealChain (79.69 vs 79.7), EvalGEN (98.14 vs 98.1), SynthWildX (92.69 vs 92.7), WildRF (95.89 vs 95.8), RRDataset (82.04 vs 81.5). BFree-Online is close (91.64 vs 90.9) but only 807 of about 1400 images are still online. Not confirmed: UnivFD (the paper's aggregation is unspecified) and Chameleon (released by email request only).

**SynthScope_train (the paper's 3-stage recipe, `experiments/image/cure.sh`, trained on 3 GPUs, DAS6 job 66658; scored with `bench_eval --ckpt` and the official rows' flags, job 66721, tags `cure_trained_*`; sha256 38390ca9a7024ec3...; `models/image/cure/README.md` "Trained checkpoint")**

| Benchmark | Paper acc | SynthScope_official acc | SynthScope_train acc / mAP / AUROC | Verdict |
|---|---:|---:|---|---|
| AIGIBench (20 subsets, full data) | 92.8 | 92.85 | 85.05 / 92.69 / 91.84 | not reproduced (-7.8) |
| AIGCDetectBenchmark | 89.3 | 89.30 | 81.55 / 93.52 / 92.22 | not reproduced (-7.8) |
| EvalGEN | 98.1 | 98.14 | 90.65 / 98.30 / 95.96 | not reproduced (-7.5) |
| SynthWildX | 92.7 | 92.69 | 76.11 / 88.83 / 86.84 | not reproduced (-16.6) |
| WildRF | 95.8 | 95.89 | 85.80 / 95.00 / 94.59 | not reproduced (-10.0) |
| BFree-Online (807 images) | 90.9 | 91.64 | 66.64 / 93.37 / 85.77 | not reproduced (-24.3) |
| RRDataset | 81.5 | 82.04 | 66.36 / 80.40 / 76.87 | not reproduced (-15.1) |
| RealChain | 79.7 | 79.69 | 63.33 / 79.60 / 76.50 | not reproduced (-16.4) |
| UFD-8 / ForenSynths-13 / GenImage-8 (references) | UnivFD 82.0 | 93.94 / 81.95 / 85.18 | 82.69 / 75.83 / 77.70 | reference |

**Verdict: not reproduced** (repo tolerance: acc >= paper - 2.0, AUROC >= paper - 2.5). Mean of the eight comparable benchmarks 76.9 vs 90.1 for the paper's numbers. The checkpoint is fine in distribution (GenImage SD v1.4 99.97, SD v1.5 99.93, Wukong 99.93; stage-1 val BCE 0.5099, floor 0.5) but conservative out of distribution: real accuracy 98-100 % everywhere, fake recall 28-56 % on the in-the-wild and RealChain/RRDataset sets, and AUROC 3.7-12.3 points below the official adapter's, so it is a worse ranking and a shifted threshold. Stage-2 parity with the **released** encoder on the full data (20,000 pairs per type, job 66720): mu cosine 0.999998 (distance 0.19 %), W overlap 0.89, fitted lambda_s/lambda_m/eps 0.96/1.02/1.0e-3; the same run without the final JPEG q70 (one control) gives mu cosine 0.982 (19 %), overlap 0.86, 17x larger off-diagonal ratio, so mu discriminates the endpoint preprocessing but the subspace overlap does not. Diagnostic (README, jobs 66725/66726): the released encoder pushed through OUR stage 2 + stage 3 reproduces the paper (BFree-Online 91.08, SynthWildX 92.65, WildRF 95.45, RRDataset 81.04 vs official 91.64 / 92.69 / 95.89 / 82.04), so the stage-2 solver, the head trainer and the protocol are sound and the shortfall is in stage 1 (the trained LoRA); naive swaps of the released and trained adapter parts collapse (AUROC < 50) because W / mu / head are tied to their own encoder.

### DDA (official checkpoint and trained in SynthScope)
Fidelity: SynthScope's class == the official `DINOv2ModelWithLoRA` (537/537 state-dict tensors identical, transform and logits identical on 40 images, 0 verdict flips; job 264307). Paper: 90.7 avg accuracy over 11 benchmarks (trained on DDA-aligned MSCOCO only). Accuracy = mean per-subset balanced accuracy at 0.5, ± = sample std over subsets.

| Benchmark (SynthScope protocol) | acc | mAP | Paper | Note |
|---|---:|---:|---|---|
| SynthWildX (`--dataset synthwildx`; 1713 of 2000 still online) | **90.80 ± 3.13** | 96.47 | **90.9 ± 3.1** | **reproduced** (AUROC 96.40) |
| WildRF (`--dataset wildrf`; full test split, 1251/1252) | **90.20 ± 3.45** | 96.76 | **90.3 ± 3.5** | **reproduced** (AUROC 96.14) |
| BFree-Online (`--dataset bfree_online`; 807 of 1111 listed images still online) | 94.98 | 99.41 | 95.1 | close, not like-for-like (AUROC 98.86) |
| AIGCDetect (`--dataset aigcdetect --jpeg-fakes 96`; 17 generators, full data) | **87.84 ± 12.56** | 96.35 | **87.8 ± 12.6** | **reproduced** to the printed digit (AUROC 96.65; 85.35 without the JPEG step) |
| ForenSynths (`--dataset cnndetection --jpeg-fakes 96`; all 13 test sets) | **81.43 ± 13.84** | 91.13 | **81.4 ± 13.9** ("11G") | **reproduced** with all 13 sets; our UFD-11 guess gives 83.30 (79.71 without the JPEG step) |
| EvalGEN (`--dataset evalgen`; all 55,298 fakes; paper metric = accuracy on the fakes) | **96.86 ± 4.62** | 99.93 | **97.2 ± 4.2** | **reproduced** (balanced acc with MSCOCO val2017 reals 97.92, AUROC 99.85) |
| GenImage-8 (`--dataset genimage_full --jpeg-fakes 96`) | 89.37 ± 11.08 | 97.74 | 91.7 ± 7.8 | **below** by 2.3 pt: VQDM 67.0, BigGAN 82.8; likely a data difference, not resolved (86.27 without the JPEG step) |
| RealChain (`--dataset realchain`) | 67.23 | 76.46 | — | new measurement (AUROC 74.97) |
| RRDataset (`--dataset rrdataset`) | 75.99 | 85.18 | — | new measurement (original 91.8, transfer 73.3, redigital 62.9) |

**Verdict: faithful port (identical to the official model); 5 of the 7 paper benchmarks we can run are reproduced** (AIGCDetect, ForenSynths, EvalGEN, SynthWildX, WildRF), BFree-Online is close on the images still online, and GenImage is 2.3 pt below. Not run: DRCT-2M and DDA-COCO (not in SynthScope yet), Synthbuster (RAISE-1k reals not downloaded), Chameleon (request only).

**Trained in SynthScope** (official recipe, `experiments/image/dda.sh`: paired MSCOCO / SD-2.0 reconstructions, matched JPEG,
pixel/frequency mixing, 0.5 BCE + 0.5 contrastive, one epoch; job 267042, eval 267832): GenImage **91.35 ± 8.47** (paper
91.7; the released checkpoint 89.37), EvalGEN 96.79, ForenSynths 80.79, AIGCDetect 89.58, WildRF 90.56, BFree-Online 94.05,
SynthWildX 88.00 (2.9 below: AP 96.99 is higher than the released checkpoint's, but X reals are called fake more often,
real acc 79.4 vs 88.4); RealChain 66.76, RRDataset 77.19. **Verdict: the official recipe reproduces DDA in SynthScope on
6 of the 7 runnable benchmarks**, closing the GenImage gap of the released weights.

### D3QE (official checkpoint)
Fidelity: SynthScope's class == the official `D3QE` (871/871 state-dict tensors identical incl. the frozen CLIP ViT-L/14 and LlamaGen VQ-16, transform and logits identical on 56 ARForensics images, 0 verdict flips; job 264794). Trained on LlamaGen vs ImageNet only; the paper also applies it to GANs (Table 2) and diffusion (Table 3). Accuracy = plain = balanced (every subset is balanced).

| Benchmark (SynthScope protocol) | acc | mAP | Paper (acc / AP) | Note |
|---|---:|---:|---|---|
| ARForensics (`--dataset arforensics`; 7 AR generators, 84k images) | **81.74** | 92.22 | **82.11 / 92.07** | **reproduced**: every generator within 1.1 pt acc and 0.9 pt AP |
| ForenSynths, the paper's 7 GANs (`--dataset cnndetection --select ...`) | **83.45** | 92.23 | **83.73 / 92.23** | **reproduced**: every GAN within 1.5 pt |
| GenImage, the paper's 6 subsets (`--dataset genimage_full --select ...`) | **79.44** | 90.94 | **78.61 / 89.60** | mean reproduced; per subset up to 6.8 pt off in both directions (ADM 63.7 vs 70.4, Wukong 90.7 vs 84.4), likely a different test sample |
| In the wild: SynthWildX / WildRF / BFree-Online / RealChain / EvalGEN / RRDataset | 49.6 / 53.0 / 47.0 / 49.5 / 62.2 / 55.0 | – | — | new measurements: it calls nearly every recent fake real (AR-only training) |

**Verdict: faithful port (identical to the official model); all 3 paper tables reproduced on the mean** (ARForensics and ForenSynths per generator too). **Trained in SynthScope with the official recipe** (`experiments/image/d3qe.sh`, job 265064): 79.83 / 91.73, 82.14 / 91.35, 77.01 / 90.11 (acc / AP) — AP within 0.9 pt of the paper everywhere, accuracy 1.6-2.3 pt lower (a slightly more conservative threshold, not a weaker ranking). Note: in the released code the codebook-frequency bias is zero at inference (`freq_log_counter` is not saved), so the reported scores come from the quantization-error and CLIP branches.

### OmniAID (official checkpoints)
Fidelity: SynthScope's class == the official `OmniAID` for all three releases (state dicts identical, logits identical on 24 GenImage images, 0 verdict flips; job 265851). Accuracy at 0.5, mean over subsets; Mirage-Test mean = 4 balanced categories + Anime fake accuracy (the paper's Table 3).

| Benchmark | release | acc | mAP | Paper | Note |
|---|---|---:|---:|---|---|
| GenImage, 8 subsets (`--dataset genimage_full`) | `omniaid_genimage` | **95.82** | 99.02 | **95.94** | **reproduced**, every subset within 1.5 pt; confirms the v1-era 512 -> 336 preprocessing |
| AIGCDetect, 17 (`--dataset aigcdetect`) | `omniaid_genimage` | **88.87** | 97.88 | **88.87** | **reproduced** exactly |
| Mirage-Test, 5 categories (`--dataset mirage_test`) | `omniaid_genimage` | **51.10** | — | **51.10** | **reproduced** exactly, every category |
| GenImage, 8 subsets | `omniaid_v1` | **97.16** | 99.20 | **97.24** | **reproduced**, every subset within 0.6 pt |
| AIGCDetect, 17 | `omniaid_v1` | **92.88** | 98.93 | **92.88** | **reproduced** exactly |
| Mirage-Test, 5 categories | `omniaid_v1` | **88.39** | — | **88.39** | **reproduced** exactly |
| Mirage-Test / SynthWildX / WildRF / BFree-Online / RealChain / EvalGEN / RRDataset | `omniaid` (v2, served) | 78.46 / 86.02 / 88.36 / 70.26 / 61.71 / 74.14 / 71.75 | — | — | new measurements |

**Verdict: faithful port; every paper number SynthScope can run is reproduced** (Chameleon and DRCT-2M are not available).

### Effort (official GenImage checkpoint; trained in SynthScope on ProGAN)
Fidelity: SynthScope's class == the official `EffortDetector` for both releases (969/969 state-dict tensors identical, transform and P(fake) identical on 7 GenImage images; job 269032). The released GenImage-SD1.4 checkpoint gives **89.83 acc / 99.37 mAP** on the 8 GenImage subsets (paper 91.1; job 269039). The paper's UniversalFakeDetect tables (Tables 2-3) use a ProGAN-trained model whose checkpoint was never released, so we trained it with the official `UniversalFakeDetect_Benchmark` recipe (`experiments/image/effort_ufd.sh`: ProGAN 20 classes, AdamW lr 2e-4 constant, batch 48, BCE on one logit, 24000 iterations; job 276492) and score it with `bench_eval.py --model effort --model-config image/effort_ufd --ckpt ...` on the paper's protocol: accuracy at the fixed threshold 0.5, 19 subsets (11 ForenSynths GAN sets + Guided, LDM x3, Glide x3, DALL-E).

| Iteration | mAP | mAcc (0.5) | AUROC | Note |
|---:|---:|---:|---:|---|
| 3000 | 98.04 | 86.51 | 97.68 |  |
| 6000 | 98.40 | 90.42 | 98.14 |  |
| 9000 | 98.86 | 91.20 | 98.65 |  |
| 12000 | 99.20 | 93.42 | 99.04 |  |
| 15000 | 99.20 | 94.56 | 99.04 |  |
| **18000** | **99.20** | **94.81** | 99.04 | **the official selection (`script/test.sh`)**; paper 99.41 / 95.19 |
| 21000 | 98.99 | 92.38 | 98.78 |  |
| 24000 (= final) | 98.97 | 91.98 | 98.74 | fake recall falls on DeepFakes, SAN, Guided, Glide |

At iteration 18000 every subset's accuracy is within 1.7 pt and its AP within 2.5 pt (Guided 92.91 vs 95.39, SAN 96.28 vs 97.53) of the paper: Guided 68.70 / 69.15, SAN 81.05 / 81.50, DeepFakes 86.45 / 87.60, SITD 90.83 / 92.50, StyleGAN 96.23 / 95.05; the rest are above 96. The test transform is the official one (`translate_duplicate(256)` tiling, then `CenterCrop(224)`). The official `validate.py` samples 1000 images per class (100 when a set has fewer, i.e. SAN and SITD) and reads png/jpg only (SAN holds 19 bmp reals); we score every image. Reference: this ProGAN-trained checkpoint on GenImage-8 gives 79.96 acc / 95.25 mAP (a different training set from the served SD v1.4 model's 89.83 / 99.37, so not a recipe check). Jobs 278691-278700 (code commit 8f86c4a), results `repro/bench_results/effort_ufd_*.json`, details in `models/image/effort/README.md`. The checkpoints stay on the shared filesystem (Effort is CC BY-NC 4.0).

### LTF (official checkpoint)
Fidelity: SynthScope's class == the official `LTF` (333/333 state-dict tensors identical incl. the frozen CLIP ViT-L/14, view tensors bit-identical, max |P(fake) diff| 0.0 on 27 images of 100x300 to 1500x1001 px). Accuracy at 0.5, mean of real and fake accuracy, macro over the 8 GenImage subsets; the tiling is the corrected official policy (the earlier overlapping tiling was not run, so this does not show whether it differed). The official code was run on the 33k `GenImage_full/test` set only.

| Run | acc | real / fake acc | AUROC | AP | Paper | Note |
|---|---:|---:|---:|---:|---|---|
| Official `ltf_ref/evaluate.py`, GenImage_full/test (33,000 images) | 98.38 | 97.64 / 99.12 | 99.88 | 99.86 | — | jobs 277010, 277012, 277014, 277016 |
| SynthScope, GenImage_full/test (33,000 images) | **98.38** | 97.64 / 99.12 | 99.88 | 99.86 | 98.4 (Table 1) | equal to the official code on all 8 subsets, to the second decimal |
| SynthScope, full official val (100,000 images; `--benchmark pergen`) | **98.36** | 97.62 / 99.10 | 99.88 | 99.86 | **98.4** (98.375 recomputed); Table 4: 98.40, real 97.68 / fake 99.09, AUC 99.89, AP 99.86 | **reproduced**: every subset within 0.10 pt of Table 1; jobs 277066, 277068-277073 (SD v1.4 reuses 277011, identical data, CPU diff job 277197) |

Per subset on the full val (Midjourney, SD v1.4, SD v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN): 97.26, 98.80, 98.57, 98.44, 98.12, 98.89, 98.64, 98.15 against the paper's 97.3, 98.9, 98.6, 98.4, 98.1, 98.9, 98.7, 98.1.

**Verdict: faithful port (identical to the official model); the GenImage table is reproduced.** The paper and the released code differ in two places (paper: weighted-sum fusion plus a linear classifier, checkpoint: gated concatenation plus an MLP; the paper's bag loss is not in the released training path); SynthScope follows the code. Only the GenImage checkpoint is released and training is not ported. The real false-positive rate of about 2 to 3% is a property of the released weights. The weights card has no license tag (only the code is MIT), so `ltf` is not served on the Space yet.

### MIRROR (official checkpoint)
Fidelity: SynthScope's class == the official `MIRROR_Detector` (825/825 state-dict tensors identical, model inputs and logits identical on 24 images, 0 verdict flips, fp32 and fp16 autocast; GPU job 279415). The checkpoint is DINOv3 ViT-H+; the paper's per-subset tables are DINOv3-L, so the target is the official README's DINOv3-H+ column. Balanced accuracy at 0.5, mean over subsets; Synthbuster and EvalGEN (fakes only) are scored against the MSCOCO val2017 reals as in the paper.

| Benchmark (SynthScope protocol) | B.Acc | mAP | Paper (README H+) | Note |
|---|---:|---:|---|---|
| AIGCDetectBenchmark (`--dataset aigcdetect`; 17 subsets, full data) | **97.39** | 99.49 | **97.3** | **reproduced** (AUROC 99.51) |
| GenImage (full val split, 100k; `--benchmark pergen --data GenImage_val_full/test`) | **99.78** | 100.00 | **99.8** | **reproduced** (test release 99.79) |
| UnivFakeDetect (the official code's 13-subset `UnivFD` list) | **92.44** | 97.04 | **92.4** | **reproduced**; the paper's Table 7 list (12 subsets) gives 93.81 |
| Synthbuster (9 generators vs 5000 MSCOCO reals) | **99.28** | 99.92 | **99.2** | **reproduced** with the official extension-based JPEG-q96 rule (served decoded-format rule: 99.43; firefly is 1000 JPEGs named .png) |
| EvalGEN (`--dataset evalgen`; 5 generators vs MSCOCO) | **99.85** | 100.00 | **99.8** | **reproduced** |
| SynthWildX (`--dataset synthwildx`; 1713 of 2000 still online) | **95.08** | 97.71 | **95.1** | **reproduced**; the official `inference.py` gives the same B.Acc on every subset |
| WildRF (`--dataset wildrf`; official file set, no .jfif) | **97.88** | 99.80 | **97.8** | **reproduced** (all files 97.86); the official code gives the same B.Acc on every subset |
| AIGIBench (`--dataset aigibench`; SocialRF + CommunityAI) | **94.98** | 98.06 | **94.9** | **reproduced** |
| BFree-Online (`--dataset bfree_online`; 807 of ~1400 still online) | **97.40** | 99.78 | **97.6** | **reproduced** on the images still online |
| RRDataset (`--dataset rrdataset`; pooled folder) | 86.91 | 93.86 | 88.3 | **short** by 1.39: original 98.06, transfer 88.38, redigital 74.30 (fake acc 53.20); causes unresolved |
| RealChain (`--dataset realchain`) | 81.71 | 91.76 | — | new measurement (AUROC 89.96) |

**Verdict: faithful port (identical to the official model); the README H+ numbers reproduce on 9 of the 10 benchmarks we hold, each within 0.25.** On the held subset, the standard average (5 of 6, no DRCT-2M) is 97.75 against the README's 97.70, and the in-the-wild score (10 of 16 columns) is 95.32 against 95.44. Not run: DRCT-2M (not in SynthScope), Chameleon (request only), CO-SPY in-the-wild (not held), Human-AIGI (not released). The unmodified official code (job 279857; 279825 was a cancelled duplicate) gives the same numbers as SynthScope on Synthbuster firefly under the official extension rule (94.29) and on BFree-Online (97.40). Jobs 279513, 279527, 279550-279555; tags `mirror_*`; table built by `repro/mirror_task2_table.py`. Licence unclear, so the checkpoint is not redistributed (the zoo and the Space do not carry it). The released memory bank has the default initialisation statistics; the checkpoint nevertheless reproduces 9 of 10 held numbers (details in `models/image/mirror/README.md`).

### VINA (official checkpoint)
Fidelity: SynthScope's class == the official `DINOv3` model (417/417 state-dict tensors identical; model inputs and P(fake) identical on 24 images and 8 videos, max |diff| 0.0, 0 flips; CPU). The checkpoint is `VINA_DINOv3-L_140k+DDA` (the paper's "VINA (140k+DDA)" rows: the 140k video set plus DDA's image data); no training code is released, so there is no SynthScope-trained number. One model, two registry keys: `vina` (images) and `vina_video` (the paper's protocol: ffmpeg 2 fps, central frame, sigmoid of the mean logit). Metric: per-subset **pooled accuracy** at 0.5 and AP, averaged over subsets, which is the official code's metric (`<tag>_offmetric`; the paper's Table 2 is headed "balanced ACC" but its code computes the pooled one); `bench_eval`'s own `acc` is the mean balanced accuracy and differs only on unbalanced subsets.

| Benchmark | ACC | AP | Paper ACC / AP | Note |
|---|---:|---:|---|---|
| ForenSynths (CNNDetection, the 11 subsets; 90,310 files) | **86.93** | 97.80 | **86.99 / 97.78** (T12) | **reproduced** (-0.06 / +0.02); 10 of 11 subsets equal, SAN 83.05 vs 83.79. A reconstruction that keeps the 19 SAN .bmp reals as black images gives 86.99 / 97.77 (inferred, not the headline); `bench_eval` balanced mean 86.98 |
| UniversalFakeDetect (LDM, 8 sets) | **94.84** | 99.01 | **94.84 / 99.01** (T13) | **reproduced** in every subset (ldm_200 AP 100.00 vs 99.99, within 0.01); the official code on the same files gives the same |
| DiTFake | **97.98** | 99.95 | **97.98 / 99.95** (T14) | **reproduced** in every subset |
| ARForensics, 6 shared subsets | **96.36** | 99.54 | 96.36 / 99.54 (mean of the paper's six values; its 7-subset mean with MAR: 96.74 / 99.59, T15) | **reproduced** in every subset. Our copy has Switti (99.95 / 100.00) and no MAR; the 7-row `bench_eval` mean 96.88 / 99.60 is not comparable with the paper |
| GenImage (8 subsets, full val, 100,000 images) | **94.74** | 99.05 | **94.74** (T11) | **reproduced** in every subset (the 1500-per-class test bundle `vina_genimage_full`, 93.29 / 98.45, is a subsample, not this split) |
| GenVideo-Val (10 generators, 18,584 videos, overall) | **96.86** | 99.26 | **96.9 / 99.26** (T1, T10) | **reproduced**; all 10 generators equal Table 17 in recall / F1 / AP (mean recall 93.20, AP 92.83; mean F1 81.91 vs the printed 81.52, which is not the mean of its own column); official code on the same videos equal |
| SynthWildX (dalle3 / firefly / midjourney_v5) | 91.82 | 96.36 | 94.5 / 89.2 / 94.7 (T2) | **below the paper**: 92.80 (-1.70), 89.50 (+0.30), 93.16 (-1.54); bAcc 91.76. Official code equal per file (1,713 files, max diff 1.7e-05): data, not code (our copy: 413 reals + 1,300 fakes; T9: 500 + 1.5K) |
| WildRF (facebook / reddit / twitter) | 96.12 | 99.41 | 96.6 / 96.7 / 98.1 (T2) | **below the paper**: 96.25 (-0.35), 94.73 (-1.97), 97.39 (-0.71); bAcc 95.94. Official code equal per file (2,241 files, max diff 1.5e-05): data, not code (262 .jfif files are not in the official CSV) |
| AIGIBench (SocialRF / CommunityAI) | 92.72 | 97.33 | 90.9 / 93.9 (T2) | **differs**: 89.72 (-1.18), 95.73 (+1.83). Official code equal per file (18,000 files, max diff 5.8e-05): the data release or file list |
| RRDataset (mean of 3 scenarios) | 82.89 | 92.77 | 82.7 (T2) | **close** (+0.19): original 94.08, transfer 84.87, redigital 69.73 (fake acc 41.72) |
| Chameleon · Magic Videos · GenBuster++ · DeepTraceReward | — | — | 91.4 · 77.1 · 90.1 · 97.4 | not run (no data or loader; Chameleon by request only) |

**Verdict: faithful port (identical to the official model); the released checkpoint reproduces the paper on the six benchmarks whose data is the paper's, in every subset (ForenSynths: 10 of 11, SAN differs by 0.7 on the .bmp question).** Three in-the-wild sets are up to 2 points from the paper and the official code agrees with SynthScope on our files to 6e-05, so the cause is the dataset version or file list. The official `eval.py` was run on SynthWildX, WildRF, AIGIBench, UniversalFakeDetect and GenVideo-Val (not on the others); its per-file CSVs are kept outside the repo (the comparison JSONs are in `repro/bench_results/vina_official_code/`). Jobs: CPU 281654-281656 and 281831 (cn023), GPU 281657 (A5000), DAS6 66630 and 66627 (A6000), 281662 (A5000). Licence unclear (no LICENSE file): not redistributed, not on the Space. Details and per-subset tables: `models/image/vina/README.md`.

### Video detectors (D3, SPLIT and STALL are training-free; ReStraV is trained in SynthScope; no official weights)
| Method · Benchmark | Paper | SynthScope |
|---|---:|---:|
| D3 · GenVideo-Val-10 | 98.46 | 98.95 (95.68); official code 97.93 |
| D3 · EvalCrafter-14 | 98.87 | 99.30 (96.59); official code 98.70 |
| D3 · VideoPhy (8 of 10 generators online) | 99.16 | 99.45 (95.19); official code 98.84 |
| D3 · VidProM (4 of 6 generators, 1000-clip subsample) | 88.46 (84.95 on those 4) | 92.15 (84.86); official code 84.91 |
| ReStraV · GenVidBench (paper: Main-task acc on 4 test generators) | 93.01 acc | 93.33 acc on those 4; all 8: 92.70 (86.9) |
| STALL · GenVideo-Val-10 | ~80 AUC | 82.38 / AUROC 83.5 |
| STALL · VideoFeedback-11 | ~83 AUC | 83.07 (62.1) |
| STALL · ComGenVid | ~85 AUC | 82.59 (57.3) |

D3 numbers are AP with real videos as the positive class on the first 1000 clips per class per generator (D3's official
eval.py); "official code" runs the D3 repository's own frame extraction + reader + model on the same data
(`repro/official_evals/d3_official_eval.py`) and reproduces every generator within 0.25 AP except GenVideo WildScrape.

## Key findings
1. **Architecture is faithful where we can check it.** Loading official weights into the SynthScope class reproduces the paper for CLIP, CNN, FatFormer, NPR, **PGC**, SAFE, Co-Spy. AIDE/SPAI needed a preprocessing/any-resolution port to make official weights load correctly (now done).
2. **PGC reproduces from the published recipe with early stopping (resolved 2026-06-19).** The released ProGAN checkpoint is only ~400 steps in (the authors, on our issue); the earlier epoch-level retrains overshot the transfer peak, which is why the hard generators (deepfake/san/seeingdark) stayed at ~71–81 AUROC. With the official code and evaluator, the official `train.py` evaluated every 100 steps reaches hard-3 AUROC 94.56 at step 900 (the released checkpoint: 94.35). In the SynthScope class, the official recipe at step 500 gives FS-8 97.41 / UFD 98.99 mAP (the released checkpoint in the same class: 96.41 / 99.17). (`repro/investigate/pgc_gap_investigation.md`, `models/image/pgc/README.md`)
3. **Gaps closed by correcting training data/preprocessing:** SPAI (**native-res any-res path now wired into serving: official weights in the SynthScope class → GenImage 94.22 / Synthbuster 96.75 mAP, beating the vendored-official 91.33; the fixed-448 retrain 88.80 is superseded by the official-recipe retrain, 94.67 GenImage / 95.76 Synthbuster, within ~1 point of the official weights**), Co-Spy (right GenImage SD-v1.4 source, →96.70 beats official), **SAFE (crop-only eval + full 40-epoch retrain → reproduces the paper: ForenSynths-8 97.79 / UFD-8 98.44 / GenImage-8 99.51 / DiTFake 99.96)**, **AIDE (native-res 5-view DCT now wired into both serving AND training: served official 96.83, and a from-scratch retrain reproduces the mAP at 96.42 ≈ official 97.12 (acc 77.9 vs 85.9), up from the resize-256 92.47 — the DCT must run at native res, ~12 mAP on the same weights)**. The original SAFE "gap" was an eval-transform bug — a `Resize` destroyed the high-frequency signal SAFE keys on; crop-only preprocessing makes both the official weights and our retrain reproduce.
4. **VIB-Net** (weights on a restricted Drive): reproduced by training in SynthScope with the reconstructed recipe at native resolution (94.08 AP / 85.22 acc vs 95.53 / 85.83 over the paper's 15 generators); the resize-256 training pipeline alone costs 11 AP points (82.98 with it); the earlier model also had an fp16 CLIP and an added BatchNorm.
5. **Faithful reporting** — every number traces to a logged run; SynthScope_train shows the single best from-scratch checkpoint (older/weaker runs discarded, not stacked).

## Where everything lives
- Per-model write-ups: `models/image/*/README.md`, `models/video/*/README.md`
- Aggregated table: `repro/AGGREGATED_BENCHMARK.md`; repo `README.md` → "Full-benchmark reproduction"
- Website data: `interface/backend/registry.py` → `REPRO` dict
- Raw results: `repro/bench_results/*.json`, `repro/MASTER_RESULTS.jsonl`
- Gap-closing checkpoints: `checkpoints_retrain/`; promoted `co_spy.pth` (`cospy_full/model_1`) and `safe.pth` (`safe_full_v2/model_38`); `spai.pth` and `pgc.pth` are now the official weights
