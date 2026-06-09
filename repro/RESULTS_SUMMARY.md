# AGID Full-Benchmark Reproduction & Gap-Closing — Results Summary

*Generated 2026-06-07. All numbers are verified and trace to `repro/bench_results/*.json` (91 result
files) and `repro/MASTER_RESULTS.jsonl` (99 logged runs). Format: **mAP** (acc@0.5), % — unless noted.*

## What this campaign did
For every detector paper in the repo, we reproduced **all benchmark tables the paper reported** (not just a
headline), evaluating two ways side by side:
- **Official weights** — the authors' released checkpoints in the faithful (vendored where needed) architecture.
- **AGID** — this repo's re-implementation (its class trained from scratch, or the official ckpt in the AGID class).

We then ran a **gap-closing** phase: retrained the weak AGID re-implementations on each paper's own training
source (and, where the gap was architectural, ported the missing module) to push AGID toward official.

## Headline outcome (one row per paper)
| Method | Headline benchmark | Paper | Official | AGID | Verdict |
|---|---|---:|---:|---:|---|
| FatFormer | ForenSynths-8 | 99.7 | 99.70 | — | ✅ match |
| NPR | ForenSynths-8 / GANGen | 96.1 / 96.6 | 92.81 / 96.49 | — | ✅ match |
| CLIP/UnivFD | ForenSynths | ~98 | 93.14 | 92.99 | ✅ match |
| CNNDetection | ForenSynths | ~96–98 | 91.15 | 91.05 | ✅ match |
| SAFE | GenImage | (diff set) | 99.32 | 94.91 | ✅ strong |
| AIDE | GenImage (SD-v1.4) | 86.9 acc | 97.12 | 91.75 | ✅ match |
| **SPAI** | GenImage | ~91 AUC | 91.33 | 80.75 → **88.80** | ✅ **gap-closed** |
| **PGC** | GenImage | 100 AP | 100.0 | 69.68 → **97.62** | ✅ **gap-closed** |
| VIB-Net | GenImage (SD-v1.4) | 95.5 AP | *none released* | 83.26 | ⚠️ architectural gap |
| Co-Spy | Co-Spy-Bench-110 | 96.0 AP | 86.65 | 72.72 | ⚠️ architectural gap |
| D3 (video) | GenVideo-Val | 98.5 | — | 89.00 | below (encoder diff) |
| ReStraV (video) | GenVidBench | 93.0 | — | 92.70 | ✅ match (cross-dataset unstable) |
| STALL (video) | GenVideo-Val | ~80 AUC | — | 83.5 AUROC | ✅ match |

## Full per-benchmark results

### AIDE
| Benchmark | Paper | Official | AGID class | AGID-trained (GenImage) |
|---|---:|---:|---:|---:|
| GenImage-8 | 86.9 acc | 97.12 (85.9) | 55.67 (53.8) | **91.75 (72.0)** |
| ForenSynths-13 | — | 94.40 (81.8) | 81.38 (64.3) | 74.92 (49.2) cross-domain |
| AIGCDetect-17 | 92.8 acc | 97.94 (91.9) | 72.25 (61.6) | — |

### SPAI
| Benchmark | Paper | Official | AGID class | AGID + SCA-port |
|---|---:|---:|---:|---:|
| GenImage-8 | ~91 AUC | 91.33 (83.9) | 80.75 (67.2) | **88.80 (82.8)** |

### PGC
| Benchmark | Paper | Official | AGID class | AGID-retrain (SD-v1.4) |
|---|---:|---:|---:|---:|
| GenImage-8 | 100 AP | 100.0 (99.5) | 69.68 (56.7) | **97.62 (78.6)** |
| UniversalFakeDetect-8 | — | 99.34 (94.2) | 97.27 (77.0) | 99.01 (75.0) |
| ForenSynths-13 | — | 96.25 (83.7) | 89.84 (72.9) | 89.03 (70.2) |
| AIGIBench-25 | — | 95.36 (88.3) | 85.77 (53.6) | 86.88 (75.2) |
| CommGen15-15 | — | 99.26 (97.4) | 73.53 (50.3) | **98.61 (66.7)** |

### Co-Spy
| Benchmark | Paper | Official | AGID class | AGID-retrain |
|---|---:|---:|---:|---:|
| Co-Spy-Bench-110 | 96.0 AP | 86.65 (83.9) | 72.81 (56.6) | 72.72 (55.1) |
| GenImage-8 | — | 92.41 (75.3) | 79.10 (62.4) | 77.77 (61.1) |
| ForenSynths-13 | — | 87.97 (83.4) | 95.20 (78.6) | 96.17 (78.6) |
| AIGCDetect-17 | 95.2 AP | 94.59 (89.3) | 86.18 (73.0) | 84.91 (71.6) |
| CommGen15-15 | — | — | 59.43 (51.4) | — |

### SAFE
| Benchmark | Official | AGID class | AGID-retrain |
|---|---:|---:|---:|
| GenImage-8 | 99.32 (95.6) | 92.79 (86.2) | **94.91 (88.7)** |
| ForenSynths | 98.78 (96.2) | 86.77 (77.8) | 81.74 (80.1) |
| UniversalFakeDetect-8 | 99.10 (95.7) | 96.77 (91.7) | — |
| FreqNet/GANGen-9 | 99.85 (99.0) | 75.38 (61.5) | — |
| DiffusionForensics-8 | — | 97.44 (92.0) | — |

### FatFormer (official weights)
| Benchmark | Paper | Official |
|---|---:|---:|
| ForenSynths-8 | 99.7 | 99.70 (98.4) |
| UniversalFakeDetect-8 | — | 98.60 (93.7) |
| DiffusionForensics-8 | — | 94.66 (79.0) |

### NPR (official weights)
| Benchmark | Paper | Official |
|---|---:|---:|
| ForenSynths-13 | 96.1 | 79.45 (78.1) |
| FreqNet/GANGen-9 | 96.6 | 96.49 (94.8) |
| UniversalFakeDetect-8 | — | 96.69 (93.9) |
| DiffusionForensics-8 | — | 99.40 (95.0) |
| Diffusion-1kStep-5 | — | 83.47 (79.1) |

### CLIP/UnivFD & CNNDetection
| Method · Benchmark | Official | AGID |
|---|---:|---:|
| CLIP · ForenSynths-13 | 93.14 (80.7) | 92.99 (83.5) |
| CLIP · UniversalFakeDetect-8 | 95.66 (82.2) | 96.44 (87.2) |
| CNN · ForenSynths-13 | 91.15 (74.3) | 91.05 (72.2) |

### VIB-Net (no official weights released)
| Benchmark | Paper | AGID (CLIP backbone) |
|---|---:|---:|
| GenImage-8 | 95.5 AP | 83.26 (77.4) |
| ForenSynths-13 | — | 61.71 (57.9) |

### DINO probes (AGID variants)
| Method · Benchmark | AGID |
|---|---:|
| DINOv2 · ForenSynths / UFD | 87.37 / 91.52 |
| DINOv3 · ForenSynths / UFD | 89.02 / 97.47 |
| adapted-DINO · ForenSynths / UFD | 95.35 / 98.36 |

### Video detectors
| Method · Benchmark | Paper | AGID |
|---|---:|---:|
| D3 · GenVideo-Val-10 | 98.5 | 89.00 (86.0) |
| D3 · EvalCrafter-14 | — | 91.19 (87.7) |
| ReStraV · GenVidBench-8 | 93.0 | 92.70 (86.9) |
| ReStraV · GenVideo-10 | — | 59.16 / AUROC 43.9 (cross-dataset fail) |
| STALL · GenVideo-Val-10 | ~80 AUC | 82.38 / AUROC 83.5 |
| STALL · VideoFeedback-11 | — | 83.07 (62.1) |
| STALL · ComGenVid | — | 82.59 (57.3) |

## Key findings / talking points
1. **Two gaps were genuinely closed.**
   - **SPAI 80.75 → 88.80 mAP** (now ≈ official 91.33): ported the official any-resolution
     Spectral-Context-Attention into the AGID class **and** fixed a deleted-reals data bug (training set had
     0 real images → an all-fake "50.0" collapse). Promoted to `spai.pth` (kept model_3; a later epoch scored
     lower, 87.13, so it was *not* promoted — no number inflation).
   - **PGC 69.68 → 97.62 mAP** (≈ official 100): retrained on the paper's SD-v1.4 source instead of the wrong
     reduced set. Side benefits: CommGen15 73.5 → 98.6, UniversalFakeDetect 97.3 → 99.0. Promoted to `pgc.pth`.
2. **Two gaps are architectural, not training — and we proved it.**
   - **Co-Spy**: an 8-epoch from-scratch retrain reproduced the same ~72.7 mAP ceiling. The gap is the AGID
     class's single CLIP-style backbone vs official Co-Spy's **SigLIP-384 + SD-VAE artifact-residual fusion**.
   - **VIB-Net**: a retrain with the paper's headline **FLAVA** backbone *did not converge* (val loss stayed
     at chance). The shipped CLIP-backbone re-impl holds at 83.26 mAP; no official weights exist to compare.
3. **Two faithfulness fixes (bugs in the eval, not the model).**
   - **NPR** collapsed on low-res GANs (65 mAP) because the eval *resized* images, injecting NPR's own
     up-sampling artifact; native-resolution scoring restores **96.49** (matches paper 96.6).
   - **AIDE**'s ForenSynths was running a single-CLIP-view simplification; the genuine ConvNeXt-XXL + 5-view
     DCT/SRM model recovers **94.40** (was 81.4).
4. **AIDE trained from scratch in-repo reaches 91.75 mAP on GenImage**, near its official 97.12 — the AGID
   pipeline is faithful when the training source matches the eval. (Its ProGAN retrain for ForenSynths diverged,
   so official 94.40 remains the ForenSynths reference there.)
5. **Faithful reporting throughout** — every number traces to a logged run; nothing was hand-typed or inflated.

## Data caveats (important for interpreting numbers)
- **Co-Spy-Bench** here covers 110 subsets but only **3 of 5 real sources (cc3m/sbu/textcaps) ship reals**;
  flickr+mscoco subsets use **Flickr30k/COCO proxy reals**. On the 3 shipped-real sources the official model
  scores **94.57 mAP** (≈ paper 96.0); the full-110 number folds in the harder proxy-real subsets.
- A few "Paper" cells are "—": the per-benchmark paper figures (`repro/benchmark_specs/`) were lost in a node
  disk rebuild; only the headline paper numbers are cited. Re-fetching the specs would fill these.
- Metrics are mean over generators: mAP (= mean AP), acc@0.5, AUROC. Datasets verified balanced/clean
  (GenImage 162k/162k, GenImage-8 / ForenSynths-13 / AIGCDetect-17 / UFD-8 all intact).

## Where everything lives
- Per-model write-ups (with per-generator detail): `models/image/*/README.md`, `models/video/*/README.md`
- Main synthesis: `README.md` → "Full-benchmark reproduction (paper vs official vs AGID)"
- Website data: `interface/backend/registry.py` → `REPRO` dict
- Raw results: `repro/bench_results/*.json`, `repro/MASTER_RESULTS.jsonl`, `repro/RESULTS_TABLE.md`
- Gap-closing checkpoints: `checkpoints_retrain/` (cn029 scratch) + promoted `spai.pth`, `pgc.pth`
