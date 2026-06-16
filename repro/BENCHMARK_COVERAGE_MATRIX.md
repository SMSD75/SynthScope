# Per-paper benchmark coverage matrix (verified from PDFs + official repos, 2026-06-08)

Re-derived specs: `repro/benchmark_specs/*.json`. Status vs `repro/bench_results/`.

| Model | Benchmark | Status | Paper | Ours | Source / cause |
|---|---|---|---|---|---|
| aide | AIGCDetectBenchmark (B1, Setti | reproduced | 92.77 | {'map': | none for official-weights faithful eval. AGID-class trained-from-scrat |
| aide | GenImage (B2, Setting-I, SDv1. | reproduced | 86.88 | {'map': | AGID-class-from-scratch GenImage collapse (55.67) and AGID-trained res |
| aide | Chameleon (B3, Setting-II, in- | missing | — | None | GATED dataset — not on disk and not openly downloadable. README: 'If y |
| clip | ForenSynths GAN/low-level test | reproduced | — | {'map_f | n/a — reproduced. AGID 92.99 (FS-13) ≈ official 93.14 ≈ paper AP_LC 19 |
| clip | UniversalFakeDetect diffusion/ | reproduced | 95.0 | {'map': | n/a — reproduced (AGID 96.44 ≥ official 95.66 ≈ paper ~95). Evaluated  |
| cnn | ForenSynths test set (CVPR pap | reproduced | 90.8 | {'map_f | n/a — reproduced. AGID 91.05 (FS-13) ≈ official 91.15 ≈ paper Blur+JPE |
| co_spy | Co-Spy-Bench | reproduced | 96.02 | {'map': | Reproduced and BEATS official-weights-in-AGID (86.65). Earlier AGID ch |
| co_spy | AIGCDetectionBenchmark (AIGCDe | reproduced | 95.21 | {'map': | Official-weights-in-AGID reproduces (94.59 vs paper 95.21). The AGID * |
| co_spy | GenImage | below | — | {'map_o | AGID retrain (90.59) below official-weights-in-AGID (92.41) by ~1.8 mA |
| co_spy | ForenSynths (CNNDetection test | reproduced | — | {'map_o | AGID retrain (96.17) beats official-weights-in-AGID (87.97) on ForenSy |
| co_spy | In-the-wild (Co-Spy-Bench in-t | missing | 84.65 | None | Dataset NOT on disk and not reproducible end-to-end. /hddstore/.../Co- |
| d3 | GenVideo (DeMamba GenVideo-Val | below | — | {'mAP': | Two compounding causes confirmed against the official repo (d3_repro/R |
| d3 | EvalCrafter (Table 2, 14 gener | below | — | {'mAP': | Same as GenVideo: CLIP-vs-XCLIP encoder substitution + no 10% edge-cro |
| d3 | VideoPhy (Table 3, 10 generato | missing | — | None | Not evaluated in AGID — VideoPhy dataset not downloaded. No bench_resu |
| d3 | VidProM (Table 4, 6 generators | missing | — | None | Not evaluated in AGID — VidProM not downloaded for D3. No bench_result |
| fat_former | GANs dataset — 8 GANs (4-class | reproduced | 99.7 | {'map': | n/a — exact match. The AGID 'number IS the official number': agid.load |
| fat_former | GANs dataset — 8 GANs (2-class | missing | 99.2 | None | Not evaluated: only the 4-class official checkpoint is available; repr |
| fat_former | Diffusion-model (DMs) dataset  | below | 98.8 | {'map_u | PARTIAL coverage, not a model gap. (1) The AGID fatformer_official_ufd |
| npr | Table 1 — ForenSynths / Cross- | below | 96.1 | {'map': | TWO compounding causes. (A) WRONG CHECKPOINT — the primary cause. AGID |
| npr | Table 2 — GANGen-Detection / S | reproduced | 96.6 | {'map': | n/a — 96.49 vs paper 96.6 (≈match). NOTE this benchmark works with the |
| npr | Table 3 — DiffusionForensics / | reproduced | 99.8 | {'map': | n/a — 99.40 vs paper 99.8 (≈match). |
| npr | Table 4 — UniversalFakeDetect  | reproduced | 95.2 | {'map': | n/a — 96.69 mAP / 93.94 acc, close to paper acc 95.2. |
| npr | Table 5 — Diffusion1kStep / Se | reproduced | 80.1 | {'map': | n/a — 83.47 mAP / 79.13 acc ≈ paper acc 80.1. |
| pgc | GenImage | reproduced | 100.0 | {'map_o | Official-weights-in-AGID reproduces exactly (mAP 100.0). AGID from-scr |
| pgc | UniversalFakeDetect (UFD) | reproduced | 98.0 | {'map_o | Reproduced (official-wts 99.34, AGID retrain 99.01) >= paper 98.0; no  |
| pgc | ForenSynths (CNNDetection test | reproduced | — | {'map_o | Official-weights-in-AGID reproduces (96.25 mAP). AGID retrain (89.03)  |
| pgc | AIGIBench (AIGI) | reproduced | 95.0 | {'map_o | Official-weights-in-AGID reproduces (95.36 vs paper 95.0). AGID retrai |
| pgc | CommGen15 (commercial in-the-w | reproduced | 97.9 | {'map_o | Official-weights-in-AGID reproduces the AP (99.26) and acc 97.39 ~= pa |
| restrav | A) VidProM vs image-based dete | missing | 97.06 | None | Not run as the VidProM-vs-image protocol in AGID. AGID only evaluated  |
| restrav | B) VidProM vs VideoSwin — seen | missing | — | None | Not run in AGID (VidProM not downloaded). AGID's restrav README footno |
| restrav | B) GenVidBench vs SoTA video d | reproduced | — | {'acc': | REPRODUCED: AGID mAP 92.70 vs paper Main-task avg-acc 93.01 (the paper |
| restrav | C) DeMamba one-to-many general | missing | — | None | Not run as the one-to-many DeMamba protocol in AGID. AGID DID run rest |
| restrav | D) Veo3 zero-shot generalizati | missing | 83.2 | None | Not run in AGID. Veo3 videos are GATED — Google Veo3 is a closed model |
| restrav | E) Physics-IQ 2AFC vs Gemini 1 | missing | — | None | Not run in AGID. Physics-IQ dataset (Motamed et al., arXiv:2501.09038) |
| safe | ForenSynths-8 (Table 1, intra-ar | reproduced | 98.8 | 97.79 | agid_trained retrain v2 (model_38) = 97.79 mAP on the 8-GAN subset (official 98.78). deepfake AP 58.82@ep10 -> 86.74@ep38. |
| safe | Self-Synthesis / GANGen (Table | reproduced | 99.8 | 99.85(off) | Official faithful matches (99.85). agid_trained retrain not re-run on GANGen (data offline on cn023). |
| safe | Ojha / UniversalFakeDetect-8 (Ta | reproduced | 99.1 | 98.44 | agid_trained retrain v2 (model_38) = 98.44 mAP, reproduces paper 99.1 (official 99.10). |
| safe | GenImage-8 (Table 4, cross-archi | reproduced | 99.3 | 99.51 | agid_trained retrain v2 (model_38) = 99.51 mAP, BEATS official 99.32 ~ paper 99.3. SERVED checkpoint. |
| safe | DiTFake (Table 6, cross-archit | reproduced | 99.9 | 99.96 | agid_trained retrain v2 (model_38) = 99.96 mAP over Flux/PixArt/SD3 = MATCH paper 99.9. |
| spai | SPAI main eval set (13 generat | missing | 91.0 | None | (1) Eval set not on disk: Synthbuster + the GDrive SD3/MJv6.1/GigaGAN/ |
| spai | GenImage (cross-dataset, AGID  | below | — | {'map_o | AGID 88.80 vs official-vendored 91.33 = ~2.5 mAP, on adm/midjourney (o |
| spai | Synthbuster | missing | — | None | Synthbuster archive not on disk (no /hddstore/.../Synthbuster). Must b |
| stall | GenVideo (DeMamba GenVideo-Val | reproduced | — | {'mAP': | REPRODUCED (slightly ABOVE paper): AGID mAUROC 83.51 / mAP 82.38 vs pa |
| stall | VideoFeedback (TIGER-Lab, 11 g | reproduced | — | {'mAP': | REPRODUCED: AGID mAUROC 85.10 / mAP 83.07 vs paper avg AUC 0.83 / AP 0 |
| stall | ComGenVid (paper's new benchma | reproduced | — | {'mAP': | REPRODUCED: AGID mAUROC 84.00 / mAP 82.59 vs paper avg AUC 0.852 / AP  |
| vib_net | Diffusion-source (SDv1.4-train | below | 95.53 | {'map': | No official VIB-Net weights were loaded into the AGID class for a fait |
| vib_net | Diffusion-source (SDv1.4-train | below | 85.83 | {'map': | Same as the AP table: no official weights loaded + GenImage-8 (hard su |
| vib_net | GAN-source (ForenSynths/ProGAN | below | 93.14 | {'map': | AGID never trained a ProGAN/ForenSynths VIB checkpoint, so there is no |
| vib_net | GAN-source (ForenSynths/ProGAN | missing | 82.5 | None | No ProGAN-trained AGID VIB checkpoint and no official ProGAN weights l |

**Totals: {'reproduced': 24, 'missing': 14, 'below': 11}**
