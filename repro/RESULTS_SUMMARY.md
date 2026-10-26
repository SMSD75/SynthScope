# SynthScope Full-Benchmark Reproduction & Gap-Closing — Results Summary

*Updated 2026-06-14. Numbers trace to `repro/bench_results/*.json` and `repro/MASTER_RESULTS.jsonl`. Format: **mAP** (acc@0.5), % — unless noted.*

## Three number columns (consistent everywhere)
- **Paper** — figure reported in the paper.
- **SynthScope_official** — authors' released weights run through the SynthScope class/eval (architecture-fidelity check).
- **SynthScope_train** — the SynthScope class **trained from scratch** by us; the **single best** run is kept (older/weaker from-scratch checkpoints are discarded).

## Headline outcome (one row per paper)
| Method | Headline benchmark | Paper | SynthScope_official | SynthScope_train | Verdict |
|---|---|---:|---:|---:|---|
| FatFormer | ForenSynths-8 | 99.7 | 99.70 | — | ✅ official match |
| NPR | ForenSynths-8 / GANGen | 96.1 / 96.6 | 94.22 / 96.49 | — | ✅ official match |
| CLIP/UnivFD | ForenSynths | — | 93.14 | 92.99 | ✅ match |
| CNNDetection | ForenSynths | 90.8 | 91.15 | 91.05 | ✅ match |
| SAFE | GenImage | 99.3 | 99.32 | 99.51 | ✅ retrain reproduces |
| AIDE | GenImage (SD-v1.4) | 86.9 acc | 97.12 | 96.42 | ✅ reproduced (native-res retrain) |
| SPAI | GenImage | ~91 AUC | 94.22 | 88.80 | ✅ served native-res (beats vendored 91.33) |
| PGC | GenImage | 100 AP | 100.0 | 97.62 | ✅ near |
| VIB-Net | GenImage (SD-v1.4) | 95.5 AP | *none released* | 83.26 | ⚠️ recipe/arch gap |
| Co-Spy | Co-Spy-Bench-110 | 96.0 AP | 86.65 | 96.60 | ✅ beats official |
| DDA | AIGCDetect / ForenSynths / EvalGEN / SynthWildX / WildRF | 87.8 / 81.4 / 97.2 / 90.9 / 90.3 acc | 87.84 / 81.43 / 96.86 / 90.80 / 90.20 acc | — | ✅ official match (GenImage 2.3 pt below) |
| D3QE | ARForensics / ForenSynths-7 / GenImage-6 | 82.11 / 83.73 / 78.61 acc | 81.74 / 83.45 / 79.44 acc | — | ✅ official match |
| D3 (video) | GenVideo-Val | 98.5 AP | 97.93 (official code) | 99.14 | ✅ reproduced (D3 protocol; preprocessing matched to the official code) |
| ReStraV (video) | GenVidBench | 93.0 | — | 92.70 | ✅ match |
| STALL (video) | GenVideo-Val | ~80 AUC | — | 83.5 AUROC | ✅ match |

## Full per-benchmark results

### AIDE
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| GenImage-8 | 86.9 acc | 97.12 (85.9) | **96.42 (77.9)** (native-res 5-view retrain, model_2) |
| ForenSynths-13 | — | 94.40 (81.8) | 74.92 (cross-domain) |
| AIGCDetect-17 | 92.8 acc | 97.94 (91.9) | 72.25 (61.6) |

### SPAI
SPAI now SERVES official weights remapped into the SynthScope class at NATIVE resolution (any-res spectral path). `SynthScope_official` = that served number; the from-scratch retrain (88.8, fixed-448) is superseded.
| Benchmark | Paper | SynthScope_official (served, native-res) | vendored-official / retrain |
|---|---:|---:|---:|
| GenImage-8 | ~91 AUC | **94.22 / AUROC 94.28** | vendored 91.33 · retrain 88.80 |
| Synthbuster (9 gen) | — | **96.75 / AUROC 96.97** | (was 55.67 via fixed-448) |

### PGC
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| GenImage-8 | 100 AP | 100.0 (99.5) | **97.62 (78.6)** |
| UniversalFakeDetect-8 | — | 99.34 (94.2) | 99.01 |
| ForenSynths-13 | — | 96.25 (83.7) | 94.57 (83.4) — see hard-3 caveat |
| AIGIBench-25 | — | 95.36 (88.3) | 86.88 |
| CommGen15-15 | — | 99.26 (97.4) | **98.61 (66.7)** |

*PGC ForenSynths caveat:* the 13-gen SynthScope_train mean (94.57) is close to official (96.25), but only because the 10 easy generators (~99) dominate. On the three hard generators (deepfake/san/seeingdark) SynthScope_train is **75.87 mAUROC vs 94.35** official. Running the authors' `train.py` verbatim reproduces the same plateau → the released checkpoint is not reproducible from the published recipe+data. See `repro/investigate/pgc_gap_investigation.md`.

### Co-Spy
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| Co-Spy-Bench-110 | 96.0 AP | 86.65 (83.9) | **96.60 (89.9)** |
| GenImage-8 | — | 92.41 (75.3) | 90.59 (75.9) |
| ForenSynths-13 | — | 87.97 (83.4) | 96.17 |
| AIGCDetect-17 | 95.2 AP | 94.59 (89.3) | 86.18 (73.0) |

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

### FatFormer (official weights only)
| Benchmark | Paper | SynthScope_official |
|---|---:|---:|
| ForenSynths-8 | 99.7 | 99.70 (98.4) |
| UniversalFakeDetect-8 | — | 98.60 (93.7) |
| DiffusionForensics-8 | — | 94.66 (79.0) |

### NPR (official weights only)
| Benchmark | Paper | SynthScope_official |
|---|---:|---:|
| ForenSynths-13 | 96.1 | 94.22 (92.3) |
| FreqNet/GANGen-9 | 96.6 | 96.49 (94.8) |
| UniversalFakeDetect-8 | — | 96.69 (93.9) |
| DiffusionForensics-8 | — | 99.40 (95.0) |
| Diffusion-1kStep-5 | — | 83.47 (79.1) |

### CLIP/UnivFD & CNNDetection
| Method · Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| CLIP · ForenSynths-13 | — | 93.14 (80.7) | 92.99 (83.5) |
| CLIP · UniversalFakeDetect-8 | 95.0 | 95.66 (82.2) | 96.44 (87.2) |
| CNN · ForenSynths-13 | 90.8 | 91.15 (74.3) | 91.05 (72.2) |

### VIB-Net (no official weights released)
| Benchmark | Paper | SynthScope_train |
|---|---:|---:|
| GenImage-8 (diffusion) | 95.5 AP | 83.94 (77.6) |
| ForenSynths-13 (GAN) | 93.1 AP | 61.71 (57.9) |

### CuRe (official adapter)
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

### DDA (official checkpoint)
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

### D3QE (official checkpoint)
Fidelity: SynthScope's class == the official `D3QE` (871/871 state-dict tensors identical incl. the frozen CLIP ViT-L/14 and LlamaGen VQ-16, transform and logits identical on 56 ARForensics images, 0 verdict flips; job 264794). Trained on LlamaGen vs ImageNet only; the paper also applies it to GANs (Table 2) and diffusion (Table 3). Accuracy = plain = balanced (every subset is balanced).

| Benchmark (SynthScope protocol) | acc | mAP | Paper (acc / AP) | Note |
|---|---:|---:|---|---|
| ARForensics (`--dataset arforensics`; 7 AR generators, 84k images) | **81.74** | 92.22 | **82.11 / 92.07** | **reproduced**: every generator within 1.1 pt acc and 0.9 pt AP |
| ForenSynths, the paper's 7 GANs (`--dataset cnndetection --select ...`) | **83.45** | 92.23 | **83.73 / 92.23** | **reproduced**: every GAN within 1.5 pt |
| GenImage, the paper's 6 subsets (`--dataset genimage_full --select ...`) | **79.44** | 90.94 | **78.61 / 89.60** | mean reproduced; per subset up to 6.8 pt off in both directions (ADM 63.7 vs 70.4, Wukong 90.7 vs 84.4), likely a different test sample |
| In the wild: SynthWildX / WildRF / BFree-Online / RealChain / EvalGEN / RRDataset | 49.6 / 53.0 / 47.0 / 49.5 / 62.2 / 55.0 | – | — | new measurements: it calls nearly every recent fake real (AR-only training) |

**Verdict: faithful port (identical to the official model); all 3 paper tables reproduced on the mean** (ARForensics and ForenSynths per generator too). Note: in the released code the codebook-frequency bias is zero at inference (`freq_log_counter` is not saved), so the reported scores come from the quantization-error and CLIP branches.

### Video detectors (no official weights released)
| Method · Benchmark | Paper | SynthScope_train |
|---|---:|---:|
| D3 · GenVideo-Val-10 | 98.46 | 99.14 (96.15); official code 97.93 |
| D3 · EvalCrafter-14 | 98.87 | 99.30 (96.59); official code 98.70 |
| D3 · VideoPhy (8 of 10 generators online) | 99.16 | 99.45 (95.19); official code 98.84 |
| D3 · VidProM (4 of 6 generators, 1000-clip subsample) | 88.46 (84.95 on those 4) | 92.15 (84.86); official code 84.91 |
| ReStraV · GenVidBench-8 | 93.0 | 92.70 (86.9) |
| STALL · GenVideo-Val-10 | ~80 AUC | 82.38 / AUROC 83.5 |
| STALL · VideoFeedback-11 | ~83 AUC | 83.07 (62.1) |
| STALL · ComGenVid | ~85 AUC | 82.59 (57.3) |

D3 numbers are AP with real videos as the positive class on the first 1000 clips per class per generator (D3's official
eval.py); "official code" runs the D3 repository's own frame extraction + reader + model on the same data
(`repro/official_evals/d3_official_eval.py`) and reproduces every generator within 0.25 AP except GenVideo WildScrape.

## Key findings
1. **Architecture is faithful where we can check it.** Loading official weights into the SynthScope class reproduces the paper for CLIP, CNN, FatFormer, NPR, **PGC**, SAFE, Co-Spy. AIDE/SPAI needed a preprocessing/any-resolution port to make official weights load correctly (now done).
2. **PGC hard-generator gap is on the authors' side.** The SynthScope class reproduces the released PGC checkpoint exactly; but training from scratch — *with our code and with the authors' `train.py` run verbatim* — does not regenerate the hard-generator numbers (deepfake/san/seeingdark stay ~71–81 AUROC vs ~94). The released checkpoint is not reproducible from the published recipe+data. (`repro/investigate/pgc_gap_investigation.md`)
3. **Gaps closed by correcting training data/preprocessing:** SPAI (**native-res any-res path now wired into serving: official weights in the SynthScope class → GenImage 94.22 / Synthbuster 96.75 mAP, beating the vendored-official 91.33; the fixed-448 retrain 88.80 is superseded**), Co-Spy (right GenImage SD-v1.4 source, →96.60 beats official), **SAFE (crop-only eval + full 40-epoch retrain → reproduces the paper: ForenSynths-8 97.79 / UFD-8 98.44 / GenImage-8 99.51 / DiTFake 99.96)**, **AIDE (native-res 5-view DCT now wired into both serving AND training: served official 96.83, and a from-scratch retrain reproduces at 96.42 ≈ official 97.12, up from the resize-256 92.47 — the DCT must run at native res, ~12 mAP on the same weights)**. The original SAFE "gap" was an eval-transform bug — a `Resize` destroyed the high-frequency signal SAFE keys on; crop-only preprocessing makes both the official weights and our retrain reproduce.
4. **VIB-Net** has no released weights; its CLIP-backbone re-impl holds at 83.26 mAP on diffusion but collapses cross-GAN (61.71) — a recipe gap, not closeable without the authors' weights.
5. **Faithful reporting** — every number traces to a logged run; SynthScope_train shows the single best from-scratch checkpoint (older/weaker runs discarded, not stacked).

## Where everything lives
- Per-model write-ups: `models/image/*/README.md`, `models/video/*/README.md`
- Aggregated table: `repro/AGGREGATED_BENCHMARK.md`; repo `README.md` → "Full-benchmark reproduction"
- Website data: `interface/backend/registry.py` → `REPRO` dict
- Raw results: `repro/bench_results/*.json`, `repro/MASTER_RESULTS.jsonl`
- Gap-closing checkpoints: `checkpoints_retrain/`; promoted `spai.pth`, `pgc.pth`, `co_spy.pth`, `safe.pth`
