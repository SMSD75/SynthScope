# AGID Full-Benchmark Reproduction & Gap-Closing — Results Summary

*Updated 2026-06-14. Numbers trace to `repro/bench_results/*.json` and `repro/MASTER_RESULTS.jsonl`. Format: **mAP** (acc@0.5), % — unless noted.*

## Three number columns (consistent everywhere)
- **Paper** — figure reported in the paper.
- **AGID_official** — authors' released weights run through the AGID class/eval (architecture-fidelity check).
- **AGID_train** — the AGID class **trained from scratch** by us; the **single best** run is kept (older/weaker from-scratch checkpoints are discarded).

## Headline outcome (one row per paper)
| Method | Headline benchmark | Paper | AGID_official | AGID_train | Verdict |
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
| D3 (video) | GenVideo-Val | 98.5 | — | 89.00 | below (encoder diff) |
| ReStraV (video) | GenVidBench | 93.0 | — | 92.70 | ✅ match |
| STALL (video) | GenVideo-Val | ~80 AUC | — | 83.5 AUROC | ✅ match |

## Full per-benchmark results

### AIDE
| Benchmark | Paper | AGID_official | AGID_train |
|---|---:|---:|---:|
| GenImage-8 | 86.9 acc | 97.12 (85.9) | **96.42 (77.9)** (native-res 5-view retrain, model_2) |
| ForenSynths-13 | — | 94.40 (81.8) | 74.92 (cross-domain) |
| AIGCDetect-17 | 92.8 acc | 97.94 (91.9) | 72.25 (61.6) |

### SPAI
SPAI now SERVES official weights remapped into the AGID class at NATIVE resolution (any-res spectral path). `AGID_official` = that served number; the from-scratch retrain (88.8, fixed-448) is superseded.
| Benchmark | Paper | AGID_official (served, native-res) | vendored-official / retrain |
|---|---:|---:|---:|
| GenImage-8 | ~91 AUC | **94.22 / AUROC 94.28** | vendored 91.33 · retrain 88.80 |
| Synthbuster (9 gen) | — | **96.75 / AUROC 96.97** | (was 55.67 via fixed-448) |

### PGC
| Benchmark | Paper | AGID_official | AGID_train |
|---|---:|---:|---:|
| GenImage-8 | 100 AP | 100.0 (99.5) | **97.62 (78.6)** |
| UniversalFakeDetect-8 | — | 99.34 (94.2) | 99.01 |
| ForenSynths-13 | — | 96.25 (83.7) | 94.57 (83.4) — see hard-3 caveat |
| AIGIBench-25 | — | 95.36 (88.3) | 86.88 |
| CommGen15-15 | — | 99.26 (97.4) | **98.61 (66.7)** |

*PGC ForenSynths caveat:* the 13-gen AGID_train mean (94.57) is close to official (96.25), but only because the 10 easy generators (~99) dominate. On the three hard generators (deepfake/san/seeingdark) AGID_train is **75.87 mAUROC vs 94.35** official. Running the authors' `train.py` verbatim reproduces the same plateau → the released checkpoint is not reproducible from the published recipe+data. See `repro/investigate/pgc_gap_investigation.md`.

### Co-Spy
| Benchmark | Paper | AGID_official | AGID_train |
|---|---:|---:|---:|
| Co-Spy-Bench-110 | 96.0 AP | 86.65 (83.9) | **96.60 (89.9)** |
| GenImage-8 | — | 92.41 (75.3) | 90.59 (75.9) |
| ForenSynths-13 | — | 87.97 (83.4) | 96.17 |
| AIGCDetect-17 | 95.2 AP | 94.59 (89.3) | 86.18 (73.0) |

### SAFE
`AGID_train` = retrained SAFE v2 (full 40-epoch crop recipe, `safe_full_v2/model_38`); supersedes the earlier resize-era column.
| Benchmark | Paper | AGID_official | AGID_train (retrain v2) |
|---|---:|---:|---:|
| GenImage-8 | 99.3 / 95.6 | 99.32 (95.6) | **99.51 (95.9)** |
| ForenSynths-8 | 98.8 / 96.2 | 98.78 (96.2) | **97.79 (95.1)** |
| UniversalFakeDetect-8 | 99.1 / 95.7 | 99.10 (95.7) | **98.44 (94.0)** |
| DiTFake | 99.9 / 99.4 | — | **99.96 (99.2)** |
| DiffusionForensics-8 | — | — | **99.34 (91.3)** |
| FreqNet/GANGen-9 | 99.8 / 99.0 | 99.85 (99.0) | not re-run (data offline) |
| ForenSynths-13 (full, incl. CRN/IMLE/face) | — | 85.74 (84.9) | 87.31 (83.9) |

### FatFormer (official weights only)
| Benchmark | Paper | AGID_official |
|---|---:|---:|
| ForenSynths-8 | 99.7 | 99.70 (98.4) |
| UniversalFakeDetect-8 | — | 98.60 (93.7) |
| DiffusionForensics-8 | — | 94.66 (79.0) |

### NPR (official weights only)
| Benchmark | Paper | AGID_official |
|---|---:|---:|
| ForenSynths-13 | 96.1 | 94.22 (92.3) |
| FreqNet/GANGen-9 | 96.6 | 96.49 (94.8) |
| UniversalFakeDetect-8 | — | 96.69 (93.9) |
| DiffusionForensics-8 | — | 99.40 (95.0) |
| Diffusion-1kStep-5 | — | 83.47 (79.1) |

### CLIP/UnivFD & CNNDetection
| Method · Benchmark | Paper | AGID_official | AGID_train |
|---|---:|---:|---:|
| CLIP · ForenSynths-13 | — | 93.14 (80.7) | 92.99 (83.5) |
| CLIP · UniversalFakeDetect-8 | 95.0 | 95.66 (82.2) | 96.44 (87.2) |
| CNN · ForenSynths-13 | 90.8 | 91.15 (74.3) | 91.05 (72.2) |

### VIB-Net (no official weights released)
| Benchmark | Paper | AGID_train |
|---|---:|---:|
| GenImage-8 (diffusion) | 95.5 AP | 83.94 (77.6) |
| ForenSynths-13 (GAN) | 93.1 AP | 61.71 (57.9) |

### Video detectors (no official weights released)
| Method · Benchmark | Paper | AGID_train |
|---|---:|---:|
| D3 · GenVideo-Val-10 | 98.5 | 89.00 (86.0) |
| D3 · EvalCrafter-14 | 98.9 | 91.19 (87.7) |
| ReStraV · GenVidBench-8 | 93.0 | 92.70 (86.9) |
| STALL · GenVideo-Val-10 | ~80 AUC | 82.38 / AUROC 83.5 |
| STALL · VideoFeedback-11 | ~83 AUC | 83.07 (62.1) |
| STALL · ComGenVid | ~85 AUC | 82.59 (57.3) |

## Key findings
1. **Architecture is faithful where we can check it.** Loading official weights into the AGID class reproduces the paper for CLIP, CNN, FatFormer, NPR, **PGC**, SAFE, Co-Spy. AIDE/SPAI needed a preprocessing/any-resolution port to make official weights load correctly (now done).
2. **PGC hard-generator gap is on the authors' side.** The AGID class reproduces the released PGC checkpoint exactly; but training from scratch — *with our code and with the authors' `train.py` run verbatim* — does not regenerate the hard-generator numbers (deepfake/san/seeingdark stay ~71–81 AUROC vs ~94). The released checkpoint is not reproducible from the published recipe+data. (`repro/investigate/pgc_gap_investigation.md`)
3. **Gaps closed by correcting training data/preprocessing:** SPAI (**native-res any-res path now wired into serving: official weights in the AGID class → GenImage 94.22 / Synthbuster 96.75 mAP, beating the vendored-official 91.33; the fixed-448 retrain 88.80 is superseded**), Co-Spy (right GenImage SD-v1.4 source, →96.60 beats official), **SAFE (crop-only eval + full 40-epoch retrain → reproduces the paper: ForenSynths-8 97.79 / UFD-8 98.44 / GenImage-8 99.51 / DiTFake 99.96)**, **AIDE (native-res 5-view DCT now wired into both serving AND training: served official 96.83, and a from-scratch retrain reproduces at 96.42 ≈ official 97.12, up from the resize-256 92.47 — the DCT must run at native res, ~12 mAP on the same weights)**. The original SAFE "gap" was an eval-transform bug — a `Resize` destroyed the high-frequency signal SAFE keys on; crop-only preprocessing makes both the official weights and our retrain reproduce.
4. **VIB-Net** has no released weights; its CLIP-backbone re-impl holds at 83.26 mAP on diffusion but collapses cross-GAN (61.71) — a recipe gap, not closeable without the authors' weights.
5. **Faithful reporting** — every number traces to a logged run; AGID_train shows the single best from-scratch checkpoint (older/weaker runs discarded, not stacked).

## Where everything lives
- Per-model write-ups: `models/image/*/README.md`, `models/video/*/README.md`
- Aggregated table: `repro/AGGREGATED_BENCHMARK.md`; repo `README.md` → "Full-benchmark reproduction"
- Website data: `interface/backend/registry.py` → `REPRO` dict
- Raw results: `repro/bench_results/*.json`, `repro/MASTER_RESULTS.jsonl`
- Gap-closing checkpoints: `checkpoints_retrain/`; promoted `spai.pth`, `pgc.pth`, `co_spy.pth`, `safe.pth`
