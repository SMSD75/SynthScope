## Full-benchmark reproduction (all models × all reported benchmarks)

Every benchmark each paper reports, with our verified numbers. Per-model detail in each `models/*/<name>/README.md`; raw results in `repro/bench_results/`, specs in `repro/benchmark_specs/`, matrix in `repro/BENCHMARK_COVERAGE_MATRIX.md`.

**Exactly three number columns:**
- **Paper** — the figure reported in the paper.
- **SynthScope_official** — the authors' released weights run through the SynthScope class / eval (architecture-fidelity check).
- **SynthScope_train** — the SynthScope class **trained from scratch** by us; the **single best** run is kept (older/weaker from-scratch checkpoints are discarded, not shown).

*Snapshot: most rows date from June 2026 and are not regenerated; the current per-benchmark numbers are in `interface/backend/repro_full.json` (the web Leaderboard tab), `repro/RESULTS_SUMMARY.md` and each model README, and detectors added later (DDA, D3QE, OmniAID, Effort, LTF, MIRROR, B-Free, MoE-JEPA, SPLIT) are not listed here. Rows below were corrected on 2026-11-08 where they contradicted those sources.*

*Correction (2026-11-09): the PGC UniversalFakeDetect and ForenSynths-8 SynthScope_official cells below were corrected from 99.17 / 83.57 and 96.41 / 76.34, which were scored while the served PGC dropped its trained LoRA (2026-06-15 to 2026-11-09); the values now shown are the corrected ones (`pgc_official_ufd8_fixed`, FS-8 subset of `pgc_official_forensynths13_fixed`), and step 500 is slightly below the official weights on both (see `repro/RESULTS_SUMMARY.md`, PGC).*

Values are `mAP (acc)` or AUROC as noted. **Status:** reproduced = SynthScope_train matches paper/official · official = SynthScope_official matches (SynthScope_train may still lag) · below = gap remains · reference = no comparable paper figure · missing = not yet evaluated.

| Model | Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|---|
| CLIP / UnivFD | ForenSynths GAN/low-level (11) | — | mAP=92.33 / acc=80.58 | mAP=92.83 / acc=85.39 | reference |
| CLIP / UnivFD | UniversalFakeDetect diffusion/AR (8) | map=95.0 | mAP=95.66 | mAP=96.44 / acc=87.23 | reproduced |
| CNNDetection | ForenSynths test set (11) | map=90.8 | mAP=90.77 / acc=75.82 | mAP=90.44 / acc=74.57 | reproduced |
| FatFormer | GANs — 8 GANs (4-class ProGAN) | map=99.7 / acc=98.4 | mAP=99.7 / acc=98.39 | — | official |
| FatFormer | GANs — 8 GANs (2-class ProGAN) | map=99.2 / acc=95.8 | — | — | missing |
| FatFormer | Diffusion (DMs) — 10 subsets | map=98.8 / acc=95.0 | mAP=98.6 / acc=93.72 | — | official |
| NPR | Table 1 — ForenSynths / Cross-GAN (8) | map=96.1 / acc=92.5 | mAP=94.22 / acc=92.3 | — | official |
| NPR | Table 2 — GANGen-Detection (9) | map=96.6 / acc=93.2 | mAP=96.49 / acc=94.81 | — | official |
| NPR | Table 3 — DiffusionForensics / DIRE | map=99.8 / acc=95.3 | mAP=99.4 / acc=95.03 | — | official |
| NPR | Table 4 — UniversalFakeDetect / Ojha | acc=95.2 | mAP=97.16 / acc=94.57 | — | official |
| NPR | Table 5 — Diffusion1kStep | acc=80.1 | mAP=83.47 / acc=79.13 | — | official |
| AIDE | AIGCDetectBenchmark (B1, ProGAN-trained) | acc=92.77 | mAP=97.94 / acc=91.89 | — | official |
| AIDE | GenImage (B2, SDv1.4-trained) | acc=86.88 | mAP=97.12 / acc=85.91 | mAP=96.42 / acc=77.9 | official |
| AIDE | Chameleon (B3, in-the-wild) | acc=58–66 | — | — | missing |
| SAFE | ForenSynths-8 (Table 1, intra-arch GANs) | map=98.8 / acc=96.2 | mAP=98.78 / acc=96.2 | mAP=97.79 / acc=95.1 | reproduced |
| SAFE | Self-Synthesis / GANGen (Table 2) | map=99.8 / acc=99.0 | mAP=99.85 / acc=99.0 | not re-run (data offline) | official |
| SAFE | Ojha / UniversalFakeDetect-8 (Table 3) | map=99.1 / acc=95.7 | mAP=99.1 / acc=95.71 | mAP=98.44 / acc=94.0 | reproduced |
| SAFE | GenImage-8 (Table 4, DM+GAN) | map=99.3 / acc=95.6 | mAP=99.32 / acc=95.62 | mAP=99.51 / acc=95.9 | reproduced |
| SAFE | DiTFake (Table 6, DiT) | map=99.9 / acc=99.4 | — | mAP=99.96 / acc=99.2 | reproduced |
| SAFE | DiffusionForensics-8 | — | — | mAP=99.34 / acc=91.3 | reproduced |
| SAFE | ForenSynths-13 (full, incl. CRN/IMLE/face) | — | mAP=85.74 / acc=84.9 | mAP=87.31 / acc=83.9 | retrain ≥ official |
| Co-Spy | Co-Spy-Bench | map=96.02 / acc=87.06 | mAP=86.65 / acc=83.88 | mAP=96.70 / acc=90.28 | reproduced |
| Co-Spy | AIGCDetectionBenchmark | map=95.21 / acc=90.58 | mAP=94.59 / acc=89.25 | mAP=86.18 / acc=73.0 | official |
| Co-Spy | GenImage | — | mAP=92.41 | mAP=90.59 / acc=76.23 | reference |
| Co-Spy | ForenSynths (CNNDetection test) | — | mAP=87.97 | mAP=96.17 | reference |
| Co-Spy | In-the-wild (Co-Spy-Bench) | map=84.65 / acc=77.41 | — | — | missing |
| SPAI | SPAI main eval (13 gen × 5 real) | auroc=91.0 | — | — | missing |
| SPAI | GenImage (cross-dataset) | — | mAP=94.22 / AUROC=94.28 | mAP=88.80 / acc=82.81 | reference |
| SPAI | Synthbuster | — | mAP=96.75 / AUROC=96.97 | — | reference |
| VIB-Net | Diffusion-source (SDv1.4), AP — HEADLINE | map=95.53 | *none released* | mAP=83.94 / acc=77.57 | below |
| VIB-Net | Diffusion-source (SDv1.4), ACC | acc=85.83 | *none released* | mAP=83.94 / acc=77.57 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN), AP | map=93.14 | *none released* | mAP=61.71 / acc=57.91 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN), ACC | acc=82.5 | *none released* | mAP=61.71 / acc=57.91 | below |
| PGC | GenImage | map=100.0 / acc=98.3 | mAP=100.0 / acc=99.49 (official code) | mAP=97.62 / acc=78.6 | official |
| PGC | UniversalFakeDetect (UFD) | map=98.0 | mAP=99.35 / acc=94.19 | mAP=98.99 / acc=91.96 | reproduced |
| PGC | ForenSynths-8 (CNNDetection test) [†] | — | mAP=97.83 / acc=89.17 | mAP=97.41 / acc=85.89 | reference |
| PGC | AIGIBench (AIGI) | map=95.0 / acc=86.6 | mAP=95.36 / acc=88.33 (official code) | mAP=86.88 | official |
| PGC | CommGen15 (commercial in-the-wild) | acc=97.9 | mAP=99.26 / acc=97.39 (official code) | mAP=98.61 / acc=66.66 | official |
| CuRe | AIGIBench — paper's 20 full-image-synthesis subsets | acc=92.8 / AUC=97.5 | mAP=97.00 / acc=92.85 / AUROC=97.48 (full test data; every subset within 0.10 pt) | — (training recipe not released) | reproduced |
| CuRe | UniversalFakeDetect (UFD-8, diffusion half) | acc=82.0 (UnivFD; aggregation unspecified) | mAP=99.19 / acc=93.94 | — | reference |
| CuRe | ForenSynths (CNNDetection test, 13) | (GAN half of UnivFD 82.0) | mAP=94.05 / acc=81.95 | — | reference |
| CuRe | GenImage-8 (its own SD-v1.4 training set; sdv1.4 99.73) | — | mAP=97.30 / acc=85.18 (`--limit 1500`/class) | — | reference |
| CuRe | AIGCDetectBenchmark | acc=89.3 | — | — | missing |
| CuRe | EvalGEN / SynthWildX / WildRF / BFree-Online / RRDataset / RealChain / Chameleon | acc=98.1 / 92.7 / 95.8 / 90.9 / 81.5 / 79.7 / 93.8 | — | — | missing |
| D3 | GenVideo (DeMamba GenVideo-Val) | mAP=98.46 | — | AUROC=95.95 / acc=91.71 (XCLIP+edge-crop; mAP harness degenerate) | reproduced (AUROC) |
| D3 | EvalCrafter (14 generators) | mAP=98.87 | — | mAP=91.19 / acc=87.67 / AUROC=95.41 | below |
| D3 | VideoPhy (10 generators) | mAP=99.16 | — | — | missing |
| D3 | VidProM (6 generators) | mAP=88.46 | — | — | missing |
| ReStraV | A) VidProM vs image detectors | acc=97.06 / mAP=98.81 | — | — | missing |
| ReStraV | B) GenVidBench vs SoTA video — Main | Main_M_avg=93.01 (acc) | — | acc=93.33 on the paper's 4 Main-task generators; all 8: mAP=92.7 / acc=86.87 / AUROC=95.9 | reproduced |
| ReStraV | C) DeMamba one-to-many | — | — | — | missing |
| ReStraV | D) Veo3 zero-shot | acc=83.2 / AUROC=86.9 | — | — | missing |
| ReStraV | E) Physics-IQ 2AFC | Pika=97.5 / Sora=99.0 | — | — | missing |
| STALL | GenVideo (DeMamba GenVideo-Val, 10 gen) | avg_AUC≈0.80 | — | mAP=82.38 / acc=56.88 / AUROC=83.51 | reproduced |
| STALL | VideoFeedback (TIGER-Lab, 11 gen) | avg_AUC≈0.83 | — | mAP=83.07 / acc=62.12 / AUROC=85.1 | reproduced |
| STALL | ComGenVid (Sora + VEO3) | avg_AUC≈0.852 | — | mAP=82.59 / acc=57.34 / AUROC=84.0 | below |

**[†] PGC ForenSynths:** SynthScope_train is the official recipe in SynthScope at step 500 (`experiments/image/pgc.sh`, job 266276; eval job 266282), scored with the official checkpoint in the same job and PadCenterCrop eval. The earlier entry (94.57, const-LR ep100, job 185056; hard-3 75.87 mAUROC vs 94.35) overshot the early transfer peak: the released checkpoint is only ~400 steps in, and the gap closes with early stopping (resolved 2026-06-19, `repro/investigate/pgc_gap_investigation.md`).

*Note: `SynthScope_train` shows the single best from-scratch run; earlier/weaker from-scratch checkpoints (e.g. PGC GenImage 69.68, SPAI 80.75) were superseded and are no longer listed.*
