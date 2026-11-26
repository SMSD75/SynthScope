# SynthScope Full-Benchmark Reproduction & Gap-Closing — Results Summary

*Updated 2026-11-21. Numbers trace to `repro/bench_results/*.json` and `repro/MASTER_RESULTS.jsonl`. Format: **mAP** (acc@0.5), % — unless noted.*

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
| AIDE | GenImage (SD-v1.4) | 86.9 acc | 85.91 acc (97.12 mAP) | 80.73 acc (93.90 mAP) | ⚠️ official weights match on acc; **trained in SynthScope with the official recipe: 6.2 pt below the paper on acc, 3.2 mAP below the official weights** (in-domain SD-v1.4/v1.5/Wukong/GLIDE match, ADM / Midjourney / VQDM transfer is weaker) |
| SPAI | GenImage | ~91 AUC | 94.22 | 94.67 | ✅ served native-res (beats vendored 91.33); official-recipe retrain reproduces the official weights (+0.45 mAP; Synthbuster -0.99) |
| PGC | GenImage / ForenSynths-8 / UFD | 100 / — / 98.0 AP | 100.0 / 97.83 / 99.35 | 97.62 / 97.41 / **98.99** | ✅ UFD reproduced by the official recipe in SynthScope (step 500); ForenSynths (not in the paper) slightly below the official checkpoint (corrected 2026-11-09, see the PGC section) |
| VIB-Net | 15 generators (7 GenImage + 8 GANs), SD-v1.4-trained | 95.53 AP / 85.83 acc | *not available* | **94.08 / 85.22** | ✅ reproduced by the reconstructed recipe (native-resolution training) |
| Co-Spy | Co-Spy-Bench-110 | 96.0 AP | 86.65 | 96.70 | ✅ beats official |
| DDA | AIGCDetect / ForenSynths / EvalGEN / SynthWildX / WildRF / GenImage | 87.8 / 81.4 / 97.2 / 90.9 / 90.3 / 91.7 acc | 87.84 / 81.43 / 96.86 / 90.80 / 90.20 / 89.37 acc | 89.58 / 80.79 / 96.79 / 88.00 / 90.56 / **91.35** acc | ✅ official match; trained in SynthScope reproduces 6 of 7 incl. GenImage (SynthWildX 2.9 below) |
| D3QE | ARForensics / ForenSynths-7 / GenImage-6 | 82.11 / 83.73 / 78.61 acc | 81.74 / 83.45 / 79.44 acc | 79.83 / 82.14 / 77.01 acc | ✅ official match; trained in SynthScope: AP matches, acc 1.6-2.3 below |
| OmniAID | GenImage / AIGCDetect / Mirage-Test (GenImage release) | 95.94 / 88.87 / 51.10 acc | 95.82 / 88.87 / 51.10 acc | — | ✅ official match (v1 release too: 97.16 / 92.88 / 88.39 vs 97.24 / 92.88 / 88.39) |
| Effort | GenImage / UniversalFakeDetect (19 subsets, ProGAN-trained) | 91.1 acc / 99.41 mAP, 95.19 acc | 89.83 acc / — (no ProGAN checkpoint released) | — / **99.20 mAP, 94.81 acc** | ✅ official match on GenImage; trained in SynthScope with the official ProGAN recipe reproduces the UFD table (iteration 18000 = the official selection) |
| LTF | GenImage (8 subsets, full val, 100,000 images) | 98.4 acc (Table 4: AUC 99.89 / AP 99.86) | 98.36 acc / 99.88 AUROC / 99.86 AP | — | ✅ official match (every subset within 0.10 pt; the official code equals SynthScope exactly on a 33k subset) |
| MIRROR | AIGCDetect / GenImage / UnivFakeDetect / Synthbuster / EvalGEN / SynthWildX / WildRF / AIGIBench / BFree-Online / RRDataset | 97.3 / 99.8 / 92.4 / 99.2 / 99.8 / 95.1 / 97.8 / 94.9 / 97.6 / 88.3 B.Acc (README, DINOv3-H+) | 97.39 / 99.78 / 92.44 / 99.28 / 99.85 / 95.08 / 97.88 / 94.98 / 97.40 / 86.91 B.Acc | — | ✅ official match on 9 of 10 held (RRDataset 1.4 short); DRCT-2M, Chameleon, CO-SPY in-the-wild, Human-AIGI not run |
| VINA | ForenSynths / UniversalFakeDetect / DiTFake / ARForensics (6 shared subsets) / GenImage (full val) / GenVideo-Val | 86.99 / 94.84 / 97.98 / 96.36 / 94.74 / 96.9 acc (ARForensics: mean of the paper's six per-subset values) | 86.93 / 94.84 / 97.98 / 96.36 / 94.74 / 96.86 acc (pooled accuracy per subset, the paper's metric) | — | ✅ official match on these 6 (every subset; ForenSynths 10 of 11). SynthWildX and WildRF are **below the paper** (91.82 vs 92.8 and 96.12 vs 97.13, the means of the paper's printed per-subset values) and AIGIBench differs (SocialRF -1.18, CommunityAI +1.83), but the unmodified official `eval.py` gives the same per-file scores on our files (max diff <= 5.8e-05, 0 flips), so the gap is the data, not the code; RRDataset 82.89 vs 82.7; Chameleon, Magic Videos, GenBuster++, DeepTraceReward not run |
| PROBE (DINOv2 `probe` / ResNet-50 `probe_resnet`) | GenImage (val100k, fakes JPEG q96) / SynthWildX / WildRF / AIGI-Bench (CommunityAI + SocialRF); Synthbuster with different reals | DINOv2 96.8 / 96.4 / 96.9 / 89.9; ResNet-50 78.4 / 75.6 / 81.3 / 70.4 bAcc (Synthbuster 97.5 / 79.1) | DINOv2 97.47 / 97.30 / 96.92 / 89.88; ResNet-50 78.38 / 75.20 / 81.63 / 70.44 bAcc (Synthbuster, 5000 MSCOCO reals not RAISE-1k: 98.20 / 83.75) | — (training code not released) | ✅ official match on the 4 like-for-like benchmarks (DINOv2 within 0.9, ResNet-50 within 0.4 per benchmark; DINOv2 ADM / GLIDE / BigGAN about 2 points above the paper). Synthbuster is a different-reals comparison, not a reproduction claim. Chameleon and AIGI-Quality-Paradox not run (no loader), so the paper's 7-benchmark averages (93.9 / 78.1) are not comparable with our 5-benchmark means (95.95 / 77.88; paper's same 5: 95.50 / 76.96). Licence unclear: not redistributed |
| Veritas++ (VLM, `veritaspp`) | WildRF (full data) / RealChain (1,008-image subset) / EvalGEN (500-fake subset); plus 15 held benchmarks subsampled against 27 other detectors | 96.4 / 82.3 / 99.5 acc (Tables III, III, II) | 95.65 / 82.74 / 98.20 acc (official weights, greedy decoding; EvalGEN balanced 99.10) | — (inference only, no training code in SynthScope) | ⚠️ **partial**: WildRF -0.75 on full data (sampled setting, subset: +0.06); RealChain +0.44 on a subset; EvalGEN is **not reproduced** on full data (500-fake subset, -1.30 on fakes only, which is our inference of the paper's metric; -0.40 balanced). Fidelity vs a direct transformers call: 15 / 15 identical greedy tokens and verdicts, max logit-gap diff 0.0. Apache-2.0; 17.5 GB folder, not in the zoo or on the Space |
| Align, Don't Blend (full method, `adb_full` = VLM OR DDA) | WildRF (full data) / AIGI-Bench CommunityAI + SocialRF (full data, 18,000) / BFree-Online (807 of 1,111 held) / GenImage (10,000: 625 + 625 per generator) and AIGCDetect (3,400: 100 + 100 per generator), fakes JPEG q96 | 92.9 / 88.8 / 97.5 / 91.7 / 87.7 bAcc | 93.01 / 89.22 / 96.84 / 88.86 / 84.76 bAcc (official adapter + official DDA checkpoint, the official rule and request of `predict.py` @ cf168db; VLM bf16 greedy, RTX A5000) | — (inference only) | ✅ official on full WildRF (+0.11; Twitter -2.10) and full AIGI-Bench (+0.42); BFree-Online -0.66 on partial data; GenImage (95% CI 88.32-89.38) and AIGCDetect (83.82-85.74) samples below (-2.84 / -2.94, reference; DDA alone is above the fused number there). Synced to upstream cf168db (prompt, decoding, pixel cap, unparsed answers Real, JPEG96): paired change on the same images, WildRF +1.79 (2,503), AIGI-Bench +1.97 (the earlier 9,000), BFree-Online -0.09 (807), GenImage +2.16 (10,000), AIGCDetect +0.88 (the earlier 1,020). Fidelity vs the unmodified official `predict.py` @ cf168db (VLM + DDA): identical final, DDA and VLM decisions and VLM text 24 / 24, bf16 and float32. CO-SPY in-the-wild not held. Apache-2.0; not on the Space |
| Align, Don't Blend (VLM branch, `adb_vlm`) | WildRF (full data) / BFree-Online (269 real + 500 fake subset) / AIGI-Bench CommunityAI + SocialRF (40-image slices) / GenImage and AIGCDetect (subsets, fakes JPEG q96); plus 15 held benchmarks subsampled against Veritas++ and 27 other detectors | **no VLM-only number in the paper**; the printed numbers are the full method (row above) | 93.98 / 87.96 / 82.50 / 65.97 / 63.92 bAcc (official adapter, VLM branch alone, greedy) | — (inference only, no training code in SynthScope) | reference, measured with the upstream f4b4794 request (the cf168db request: full WildRF 95.96, see the README) (no paper number for the branch alone; the full method is `adb_full`). Fidelity vs the official ms-swift 4.4.2 (pt backend; vLLM backend not compared): 16 / 16 identical text, tokens and verdicts. Apache-2.0; not mirrored to the zoo, not on the Space |
| FakeVLM (VLM, `fakevlm`) | none of the paper's benchmarks is held (FakeClue, LOKI, DMimage, DD-VQA, FF++); 15 held benchmarks subsampled against Veritas++, Align, Don't Blend and 27 other detectors | — (not held) | held-out mean of 14 (without the in-distribution GenImage) 84.15 acc / 91.16 AUROC; all 15: 85.19 / 91.74 (official weights, greedy) | — (inference only, no training code in SynthScope) | ⚠️ **not compared**: no paper benchmark is held, so no reproduction claim. GenImage is IN-DISTRIBUTION (FakeClue contains GenImage; 99.70, 1st of 28). Fidelity vs the official `eval.py`: first sentence and parsed verdict 16 / 16 against all five official runs; text 16 / 16 against the official code run one image at a time under transformers 4.45.2. JPEG95 drops UFD-8 by 31.35 acc (reals 98.61 unchanged, fakes 95.44 -> 32.74). No licence: not in the zoo or on the Space |
| D3 (video) | GenVideo-Val | 98.5 AP | 97.93 (official code) | 98.95 (training-free; AP with real as positive) | ✅ reproduced (D3 protocol; preprocessing matched to the official code) |
| ReStraV (video) | GenVidBench (Main task: 4 test generators) | 93.01 acc | — | 93.33 acc on those 4; 92.70 mAP / 86.87 acc on all 8 | ✅ close on acc on the paper's 4 generators (not the paper's train/test split) |
| STALL (video) | GenVideo-Val | ~80 AUC | — | 83.5 AUROC (training-free) | ✅ match |
| SPLIT (video) | GenVideo-Val, fake recall at FPR 0.1 / 1 / 5% | 82.1 / 91.7 / 97.2 | 82.22 / 90.17 / 96.89 (official code) | 98.87 / 99.86 / 99.98 (training-free) | ✅ the official code reproduces the paper; SynthScope's SPLIT skips the official JPEG frame round trip, so it scores higher (see `models/video/split/README.md`) |
| WaveRep (video; `waverep` = G1, the paper's model) | The 8 GenVideo generators of the paper's Tables 1 and 7, on GenVideo-Val (PARTIAL: the paper's own 7-generator test set is not held) | 94.83 bAcc / 98.03 AUC (means of the printed per-generator values) | 90.30 bAcc / 95.73 AUC (official G1 weights; G4, trained on four generators, 96.30 / 99.05, reference only) | — (no training code) | ⚠️ **partial**: faithful port (the unmodified official demo agrees on 180 of 180 benchmark files, max logit diff 1.05e-5, and on 15 videos / 6 codecs with bit-exact inputs); 5 of the 8 generators within about 2 points, but HotShot (60.10 vs 81.4 bAcc), Show-1 (79.67 vs 92.1) and LaVie (92.06 vs 95.5) are **not reproduced** (HotShot fake recall 21%, AUC 75.6 vs 91.0, so not a threshold offset; cause untested; the paper does not say which reals it paired, we use GenVideo-Val's 10,000). Zero-shot reference rows only for VideoPhy / VidProM / EvalCrafter. GRIP non-profit licence: not redistributed |
| NSG-VD (video; `nsg_vd` = standard-SEINE-mp, also `nsg_vd_pika`, `nsg_vd_unbalance`) | GenVideo-Val, the authors' released-CSV protocol (500 per generator, Sora 51 + 56). The paper's own protocol (120 + 120 per generator, confirmed by the authors 2026-11-13) is run as L120 (Table 2 reproduces, Tables 1 and 3 do not; below) | released results CSV (not the paper): 88.97 acc / 97.68 AUROC; paper Table 2 (120 per generator): 86.05 / 95.46 | 89.22 acc / 97.51 AUROC (official code on our data) | 88.96 acc / 97.50 AUROC (served path; official pooled accuracy) | ⚠️ **partial**: the released-CSV protocol **reproduces** for all 3 keys on both paths (per-generator accuracy, recall, F1 and precision within 0.43 of the CSVs, Avg accuracy within 0.25); under the paper's own confirmed protocol (L120) the default key **reproduces paper Table 2** on all four metrics (within 2 points; Sora 51 fakes here, 56 in the paper), while Table 1 (Pika-mp) and Table 3 (unbalance-SEINE-mp) are **not reproduced** (recall +8.65 and +5.45 above the paper); the released checkpoints are separate training runs, not those behind the tables, per the authors (2026-11-15), and the released CSVs differ from the tables by up to 7.8; Sora AUROC is below the CSV for every checkpoint (unexplained). Exact parity with the official code at its own batching. Details below and in `models/video/nsg_vd/README.md` |
| WeVisForensics (video agent, `wevis`) | GenBuster-Bench / GenBuster-Bench++ (the paper's GB / GB++; 40% subsample of every folder, 1,260 / 800 clips; IN-DISTRIBUTION); plus 4 held subsets against 8 video detectors | 97.7 / 97.5 acc (Table 4) | 97.30 [96.25, 98.06] / 97.88 [96.62, 98.67] acc (official adapters + code, greedy, audio off; one row per clip, 95% Wilson) | — (inference only, no training code in SynthScope) | ⚠️ **partial**: GB / GB++ **reproduce the paper on a 40% subsample** (-0.40 / +0.38, both paper values inside our intervals), but GenBuster-200K is in its training data, so they are **in-distribution, not held out**. **Held-out headline:** 4 held subsets (GenVideo-Val, VideoPhy, VidProM, EvalCrafter), 85.59 acc / 90.23 AUROC, **7th of 9** (SPLIT 99.41): real accuracy 72-75 on the low-resolution MSR-VTT reals vs fake accuracy 97.7-98.4. Fidelity vs the official transformers backend: 17 / 17 identical traces, texts and verdicts, logit-gap diff 0.0 (vLLM engine not reproduced). Apache-2.0; needs a ~22 GB GPU, not in the zoo or on the Space |
| B-Free | SynthWildX / WildRF | 95.6 / 95.6 · 86.2 · 97.3 bAcc | 95.64 / 95.31 · 86.47 · 97.66 bAcc | — | ✅ official match (details in `models/image/bfree/README.md`) |
| MoE-JEPA | SID-Set (3-way) | 95.54 acc | 96.37 acc | — | ✅ official match (details in `models/image/moejepa/README.md`) |

## Full per-benchmark results

### AIDE
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| GenImage-8 | 86.9 acc | 97.12 (85.9) | **93.90 (80.73)** (official recipe, `experiments/image/aide.sh`, model_best = epoch 4; full test set) |
| ForenSynths-13 | — | 94.40 (81.8) | 80.42 (62.5) (same checkpoint, cross-domain, 300 images per class) |
| AIGCDetect-17 | 92.8 acc | 97.94 (91.9) | — (not trained in this setting) |

SynthScope_train (GenImage-8, ForenSynths-13) = the official AIDE recipe in SynthScope (`bash experiments/image/aide.sh`, train job 281041: 4 x A6000, effective batch 256 = 4 GPUs x micro-batch 8 x accumulation 8 with SyncBatchNorm over 32 images as on each official GPU, AdamW lr 5e-4, weight decay 0, 5 epochs, per-iteration cosine to 1e-6, label smoothing 0.1, fp32, no clipping; training images get the official Gaussian blur and JPEG perturbations (p 0.1 each) at native resolution before the 5-view DCT selection, no flip; the long side is capped at 1024 before the DCT, the only deviation). `model_best` = epoch 4 (validation accuracy 0.9984; every epoch was still improving). Evaluated by DAS6 job 66956 through `bench_eval --model aide --ckpt`: tags `aide_trained_genimage8` (all 1500 + 1500 images per generator, 6000 + 6000 for sdv1.4: 93.90 mAP / 91.66 mAUROC / 80.73 acc), `aide_trained_genimage8_l300` (first 300 per class, the protocol of the previous retrain and the served row: 92.57 / 89.63 / 80.69), `aide_trained_forensynths` (300 per class: 80.42 / 80.41 / 62.5). Accuracy per generator (full): ADM 58.63, BigGAN 50.87, GLIDE 92.00, Midjourney 74.60, SD v1.4 99.97, SD v1.5 100.0, VQDM 69.83, Wukong 99.93 (official weights: 76.60, 60.63, 90.83, 83.70, 99.76, 99.80, 76.80, 99.13). Real accuracy is 99.93-100 everywhere: the loss is fake recall at threshold 0.5 on generators other than SD v1.4. The previous retrain (`aide_native_v2/model_2`; checkpoint deleted 2026-11-10 to free disk space, its result JSONs remain: native-resolution 5-view, but the earlier recipe with lr 1e-4, batch 32, horizontal flip, no perturbations, no label smoothing, DeepSpeed gradient clipping 1.0 and the epoch picked by validation loss; 300 per class) scored 96.42 mAP / 77.87 acc, so the faithful recipe gains 2.8 pt accuracy (80.69 on the same 300 images per class) and loses 3.9 mAP (92.57); neither reaches the paper's 86.88 accuracy within the 2-point tolerance.

SynthScope_official is `sd14_train.pth` (the served `aide.pth`) on GenImage and the ProGAN-trained `progan_train.pth` on ForenSynths and AIGCDetect. The earlier 72.25 (61.6) AIGCDetect entry was `progan_train.pth` under the old resize-before-DCT eval, not a SynthScope-trained model.

**Per-epoch diagnostic (not a reproduction claim; the headline stays `model_best` = epoch 4, chosen by validation accuracy as in the official code).** Each epoch checkpoint `model_0..model_4` of train job 281041 was scored on GenImage-8 (first 300 images per class and generator; epoch 4 is `aide_trained_genimage8_l300`) and, as a transfer probe, on Synthbuster (not an AIDE paper benchmark: the 9 generators' fakes against the 5,000 shared MSCOCO reals, first 300 per class and generator, 5,400 images), DAS6 job 66991, tags `aide_trained_e<k>_genimage8_l300` / `aide_trained_e<k>_synthbuster`:

| Epoch | val acc | GenImage-8 mAP / acc | GenImage-8 ADM acc | Synthbuster mAP / acc | Synthbuster DALL-E 2 AP |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9913 | 93.20 / 77.83 | 52.67 | **96.16** / 72.20 | 98.90 |
| 1 | 0.9956 | 91.99 / 77.88 | 54.17 | 94.20 / 69.63 | 93.79 |
| 2 | 0.9973 | 92.96 / 79.73 | 54.67 | 93.13 / 70.37 | 86.57 |
| 3 | 0.9979 | 92.50 / 79.73 | 54.33 | 92.56 / 70.24 | 84.96 |
| 4 (best) | 0.9984 | 92.57 / 80.69 | 56.83 | 92.20 / 70.67 | 81.79 |

Reading: Synthbuster mAP falls monotonically with training (96.16 to 92.20; DALL-E 2 98.9 to 81.8, Firefly 82.8 to 78.6), the signature of over-adapting to SD v1.4 artifacts, while GenImage-8 mAP is flat (92.0-93.2, within single-run noise) and its accuracy creeps up (77.8 to 80.7) as the 0.5 threshold calibrates. No early epoch comes near the official weights' 97.12 mAP on GenImage-8, so an early checkpoint does not explain the gap on the paper's benchmark. One run, one seed, 300 images per class.

### SPAI
SPAI SERVES official weights remapped into the SynthScope class at NATIVE resolution (any-res spectral path). `SynthScope_official` = that served number; `SynthScope_train` = the official-recipe retrain in SynthScope (35 epochs, job 267192), epoch 8 chosen by validation loss (epoch 17 as a note). It reproduces the official weights within about 1 point; the earlier from-scratch fixed-448 retrain (88.80) is superseded. Protocol: native resolution, GenImage-8 300 per class per generator, Synthbuster 200 with GenImage-nature proxy reals; eval jobs 273444-273446, 273597.
| Benchmark | Paper | SynthScope_official (served, native-res) | SynthScope_train (epoch 8) | Epoch 17 |
|---|---:|---:|---:|---:|
| GenImage-8 | ~91 AUC | **94.22 / AUROC 94.28** (vendored 91.33) | **94.67 / AUROC 94.77** (acc 87.25) | 93.52 / 93.48 |
| Synthbuster (9 gen) | — | **96.75 / AUROC 96.97** (was 55.67 via fixed-448) | **95.76 / AUROC 96.37** (acc 88.14) | 94.57 / 95.85 |

### PGC
> **Correction (2026-11-09).** From 2026-06-15 to 2026-11-09 the served PGC (official `PGC_train_progan_ckpt.pth`) loaded only its 28 head tensors: the checkpoint's `rgb_stream.` prefix was not remapped and the load did not fail, so every `rgb_stream.*` tensor, including the trained LoRA, was silently dropped. Accuracy collapsed and mAP stayed near-saturated (UFD 99.17 / 83.57 instead of 99.35 / 94.19; ForenSynths-13 93.80 / 70.58 instead of 96.25 / 83.76). On the generator benchmarks the broken model had a higher AUROC (FLUX.2 klein 4B 97.33, rank 4 of 23, against 90.70, rank 10, now). All 13 affected tags were rescored as `<tag>_fixed` (`repro/bench_results/`, summary of the generator rows in `repro/pgc_load_fix_genbench_summary.json`); the old JSONs are in `repro/bench_results/_superseded/`, and their lines stay in `repro/MASTER_RESULTS.jsonl`. The fixed class matches the official code to 0.0 probability difference on 24 images (`tests/test_pgc_load.py`). SynthScope-trained PGC checkpoints and the GenImage / AIGIBench / CommGen15 official-code numbers were never affected.

| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| GenImage-8 | 100 AP | 100.0 (99.5) | **97.62 (78.6)** |
| UniversalFakeDetect-8 | 98.0 AP | 99.35 (94.2) | **98.99 (92.0)** |
| ForenSynths-13 | — | 96.25 (83.8) (FS-8 97.83 / 89.2) | 95.78 (FS-8 97.41 / 85.9) |
| AIGIBench-25 | — | 95.36 (88.3) | 86.88 |
| CommGen15-15 | — | 99.26 (97.4) | **98.61 (66.7)** |

SynthScope_official uses the release matching each benchmark's training setting: `PGC_train_sdv1_4_ckpt.pth` for GenImage and CommGen15, `PGC_train_progan_ckpt.pth` (the served `pgc.pth`) for UniversalFakeDetect and ForenSynths, `PGC_train_progan_sdv1_4_ckpt.pth` for AIGIBench. The UniversalFakeDetect and ForenSynths SynthScope_official numbers are the corrected rescore of the served weights (`pgc_official_ufd8_fixed`, `pgc_official_forensynths13_fixed`, DAS6 job 66717, PadCenterCrop eval); the step-500 numbers are from jobs 266660 and 266282. The GenImage, AIGIBench and CommGen15 entries are official-code numbers: 2026-06-04 runs of the official PGC code (`pgc_official_eval.py`, PadCenterCrop). Those runs gave UFD 99.34 (94.2) and FS-13 96.25 (83.7); the SynthScope class on the same weights now gives 99.35 / 94.19 and 96.25 / 83.76, within 0.03 of the official code. The 'known model-forward difference' quoted before 2026-11-09 (99.17 / 83.57 and 93.80 / 70.58) was the dropped-LoRA load.

*PGC training:* the released ProGAN checkpoint is only ~400 steps in (the authors, on our issue), and DINOv2's prior overfits ProGAN within the first epoch, so the earlier epoch-level retrains (GenImage, AIGIBench, CommGen15 rows above, and the old 94.57 / hard-3 75.87 ForenSynths entry) overshot the transfer peak. The official recipe run in SynthScope with a 2000-micro-batch budget (`experiments/image/pgc.sh`, checkpoint every 100) lands at or just below the official checkpoint (corrected 2026-11-09, see below): step 500 (the saved step nearest ~400) gives FS-8 97.41 mAP / 85.89 acc and hard-3 AUROC 93.93 against 97.83 / 89.17 / 94.34 for the official weights on the same evaluation, and UFD 98.99 / 91.96 (paper 98.0; official weights 99.35 / 94.19); transfer peaks around steps 500-900 and erodes after. GenImage, AIGIBench and CommGen15 have not been rerun with this recipe. See `repro/investigate/pgc_gap_investigation.md` and `models/image/pgc/README.md`.

### Co-Spy
| Benchmark | Paper | SynthScope_official | SynthScope_train |
|---|---:|---:|---:|
| Co-Spy-Bench-110 | 96.0 AP | 86.65 (83.9) | **96.70 (90.3)** |
| GenImage-8 | — | 92.41 (75.3) | 90.59 (76.2) |
| ForenSynths-13 | — | 87.97 (83.4) | 96.17 |
| AIGCDetect-17 | 95.2 AP | 94.59 (89.3) | 86.18 (73.0) |

SynthScope_train on Co-Spy-Bench and GenImage is the served `co_spy.pth` (`checkpoints_retrain/cospy_full/model_1.pth`, GenImage SD v1.4; `cospy_full_cospybench` also has rows for model_0 96.12 and model_2 96.60). The ForenSynths and AIGCDetect SynthScope_train entries are from earlier SynthScope checkpoints and have not been re-run with it. SynthScope_official is the official SD v1.4 checkpoint on Co-Spy-Bench and GenImage and the official ProGAN one on ForenSynths and AIGCDetect.

The ForenSynths-13 SynthScope_official entry (`cospy_official_forensynths`, 87.97 mAP / 83.42 acc, 2026-06-04) is a **1000-per-class** run (`--limit 1000` on the sorted path list: every per-class rate is a multiple of 0.1, and the June job log was not kept), not the full test sets. The full ForenSynths-8 run with the same checkpoint and evaluator (`cospy_official_fs8_nov`, DAS6 job 66671, `repro/logs/lbfu_cospyoff_das6_66671.out`) gives 98.85 mAP / 95.12 acc / 99.03 mAUROC; that is the leaderboard's CO-SPY (official weights) row (it previously used the 8 ForenSynths-8 sets of the 1000-per-class file: 98.50 / 94.64).

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

**SynthScope_train (the paper's 3-stage recipe, `experiments/image/cure.sh`; investigation scripts and results in `repro/cure_stage1/`; all scored with `bench_eval --ckpt` and the official rows' flags; `models/image/cure/README.md` "Trained checkpoint")**

Columns:
- **Original:** the recipe as written, last epoch (job 66658).
- **R1 e0:** the recipe, every epoch saved, epoch chosen by a pre-registered non-benchmark rule (Synthbuster fakes vs SD1.4 test reals).
- **R2 final:** global batch 512.
- **R2 e1:** chosen after the fact by its weight-norm match to the release; diagnostic only.
- **R3 e0:** R1 + DDA-style training augmentation (an assumption, not in the paper), same rule. **This is the main trained checkpoint.**

Each cell is acc / AUROC; R means reproduced.

| Benchmark | Paper | Official | Original | R1 e0 | R2 final | R2 e1 (diag) | **R3 e0** |
|---|---|---|---|---|---|---|---|
| AIGIBench-20 | 92.8/97.5 | 92.85/97.48 | 85.05/91.84 | 88.85/95.86 | 87.10/93.73 | 88.81/95.78 | **91.12/97.10 R** |
| AIGCDetect | 89.3 | 89.30/98.02 | 81.55/92.22 | 92.25/98.47 R | 82.29/94.40 | 88.20/97.84 R | **92.01/98.38 R** |
| EvalGEN | 98.1 | 98.14/99.66 | 90.65/95.96 | 91.57/98.63 | 92.45/97.90 | 93.98/99.17 | 93.32/98.85 (-4.8) |
| SynthWildX | 92.7/96.6 | 92.69/96.50 | 76.11/86.84 | 90.99/96.85 R | 81.56/91.85 | 88.65/95.90 | **91.76/97.09 R** |
| WildRF | 95.8/99.1 | 95.89/99.14 | 85.80/94.59 | 95.23/99.06 R | 90.24/96.19 | 93.91/98.43 R | **95.58/99.25 R** |
| BFree-Online (807 imgs) | 90.9 | 91.64/98.08 | 66.64/85.77 | 72.30/81.80 | 69.52/81.51 | 74.54/89.90 | 71.19/79.51 (-19.7) |
| RRDataset | 81.5 | 82.04/88.88 | 66.36/76.87 | 72.62/85.25 | 68.01/78.57 | 72.65/84.82 | 73.46/85.32 (-8.0) |
| RealChain | 79.7 | 79.69/85.47 | 63.33/76.50 | 70.34/84.87 | 67.49/82.20 | 72.28/85.02 | 72.75/84.70 (-7.0) |
| UFD-8 / ForenSynths-13 / GenImage-8 (ref) | - | 93.94 / 81.95 / 85.18 | 82.69 / 75.83 / 77.70 | 93.99 / 89.79 / 86.05 | 83.81 / 74.35 / 79.89 | 92.14 / 84.26 / 83.48 | 93.89 / 89.36 / 86.22 |
| Mean acc (8 comparable) | 90.10 | 90.28 | 76.94 | 84.27 | 79.83 | 84.13 | **85.15** |
| Reproduced | | | 0/8 | 3/8 | 0/8 | 2/8 | **4/8** |

**Verdict: the recipe is provided and partially reproduces the paper.** The repo tolerance is acc >= paper - 2.0 and AUROC >= paper - 2.5.
- **Last epoch, as written:** 13 points short. Stage 1 over-adapts to SD v1.4:
  - LoRA update 1.71x the release's;
  - a 3x tighter fit of the SD1.4 mixing ratio;
  - a SD1.4-fitted probe transfers to Synthbuster worse than the frozen base (81.0 vs 92.5; release 95.0; `repro/cure_stage1/results/epoch_select_r1.json`).
- **Early stopping by the pre-registered rule (epoch 0):** recovers more than half of the gap.
- **Augmentation:** adds AIGIBench.
- **What remains with R3 e0:**
  - mean -5;
  - BFree-Online -19.7 with AUROC 79.5 against the official adapter's 98.1, a ranking failure;
  - RRDataset -8 (AUROC 85.3 vs 88.9);
  - RealChain -7, which is calibration (AUROC 84.7 vs the official 85.5);
  - EvalGEN -5, mostly the threshold (AUROC 98.85 vs 99.66).
- **Disclosure:** the pre-registered rule selects an epoch within a run.
  - Choosing r3e0 over r1e0 as the main checkpoint was decided after R1's benchmark results were known.
  - Cross-run selection scores come from different GPUs: R1 RTX 6000 Ada, R2 A5000, R3 A6000. The released reference scores 95.2 on the A5000 and 95.16 on the A6000.
  - Only the AIGIBench "reproduced" depends on the augmentation; r1e0 alone reproduces AIGCDetect, SynthWildX and WildRF.
- **Seed and global batch:** single seed (paper std 0.3-0.7). Global batch 512 alone (R2) does not reproduce the paper.
- **Stages 2-3:** verified. The released encoder through our stage 2 + 3 reproduces the paper (jobs 66720 / 66726).
- **Engineering notes:**
  - On RTX 6000 Ada GPUs, PE-Core fp16 features depend on the batch size (logits differ by about 2e-2 between batches of 64 and 128; Ampere 1e-5). The stage-3 export check now compares at the encoding batch size.
  - The multi-encoder stage-2/3 driver is bit-identical to the single path.
- **Serving:** the served `cure` model stays the official adapter.

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

### PROBE (official checkpoints, DINOv2 and ResNet-50)
Fidelity: SynthScope's classes equal the official `dino_classifier` / `ResNet` (442 and 320 state-dict tensors identical, strict load; patch tensors identical; P(fake) and mean logit equal the official batch-1 run with max |diff| 0.0 and 0 flips on 34 images of varied sizes, CPU). The released checkpoints are the paper's PROBE-DINOv2 and PROBE-ResNet50 rows; the generator-probing and fine-tuning code is not released, so there is no SynthScope-trained number. Protocol: balanced accuracy at 0.5 (positive = fake), mean over the paper's subsets, the paper's metric (`means.acc`); official inference (pad to the crop size, mean logit of non-overlapping 336 / 224 patches), batch size 1, GPU. GenImage = the full val split (100,000 images) with all fakes JPEG q96 as in the paper (sdv1.5 has 8000 per class here, 6000 in the paper's 48k).

| Benchmark | `probe` bAcc | `probe_resnet` bAcc | Paper DINOv2 / ResNet-50 (Table 1) | Note |
|---|---:|---:|---|---|
| GenImage (8 subsets, val100k, fakes JPEG q96) | **97.47** | **78.38** | 96.8 / 78.4 | ResNet-50 equal (7 of 8 subsets within 0.1, ADM 58.94 vs 60.0). DINOv2 +0.67: ADM +2.2, GLIDE +1.8, BigGAN +2.1, the other five within 0.2; the official code equals SynthScope wherever compared (34 parity images, WildRF, SynthWildX; not run on GenImage), so the difference is on the paper's data / preprocessing side, undetermined |
| Synthbuster (9 subsets) vs **5000 MSCOCO reals** | 98.20 | 83.75 | 97.5 / 79.1 | **different reals** from the paper's RAISE-1k (registration-gated, not in SynthScope): not like-for-like; real accuracy 99.54 / 99.14. ResNet-50: the paper's SD 1.3 / 1.4, DALL-E 2 and DALL-E 3 values follow from our fake accuracies if the RAISE real accuracy is about 89.8 (inferred, not measured) |
| SynthWildX | **97.30** | **75.20** | 96.4 / 75.6 | +0.90 / -0.40; the unmodified official code gives the same bAcc on every subset (413 reals, 392 / 473 / 435 fakes; paper 0.5k + 1.5k) |
| WildRF | **96.92** | **81.63** | 96.9 / 81.3 | +0.02 / +0.33; 2503 files. The official code drops the 262 `.jfif` files of twitter (421 of 683 files read); on the common files it equals SynthScope |
| AIGI-Bench (CommunityAI + SocialRF only, 9k + 9k) | **89.88** | **70.44** | 89.9 / 70.4 | equal; the other AIGIBench subsets were not scored |
| Chameleon · AIGI-Quality-Paradox | — | — | 86.6 / 74.0 · 92.9 / 87.7 | not run (no loader; Chameleon by request only) |
| Mean of the 5 run (4 like-for-like) | 95.95 (95.39) | 77.88 (76.41) | 95.50 (95.00) / 76.96 (76.43) | the paper's Avg over all 7 (93.9 / 78.1) is not comparable |

**Official code on our data** (unmodified `evaluate_dino.py` / `evaluate_resnet.py`, job 66753, WildRF and SynthWildX): **0 flips at 0.5 over every common file in all four runs; max |dP| 4.8e-6 (DINOv2), 1.3e-3 (ResNet-50, GPU convolution algorithms / TF32 at batch 4 vs 1)**. Its own "Overall Accuracy" is pooled accuracy on a seed-42 class-balanced subsample, which equals bAcc up to sampling noise (at most 2.0 per subset, ResNet-50 on WildRF twitter: 82.29 vs 84.27; at most 0.66 per benchmark, ResNet-50 WildRF: 80.99 vs 81.65). **Verdict: faithful port; the released checkpoints reproduce the paper on GenImage (JPEG q96), SynthWildX, WildRF and AIGI-Bench; Synthbuster is not a like-for-like comparison and Chameleon / AIGI-Quality-Paradox were not run.** Jobs (DAS6, A6000): 66753 (SynthWildX, WildRF, official runs, GenImage), 66755 (Synthbuster), 66790 (AIGI-Bench); tags `probe_<x>`, `probe_resnet_<x>` in `repro/bench_results/`; per-subset tables, protocol and scripts: `models/image/probe/README.md`, `repro/probe_task2/`. Licence unclear (no LICENSE file): not redistributed, not on the Space.

### Veritas++ (official weights, VLM kind; inference only; subsampled protocol)
Qwen3-VL-8B fine-tuned to write a rationale and answer `<answer> real/fake </answer>` (EricTan7/VeritasPP @ 22ba6c6b, Apache-2.0). **Fidelity:** SynthScope's path (transformers, the official prompts and processor, vLLM's image loading ported and checked against vLLM 0.31.0 on 16 synthetic files) equals a direct transformers `generate` with the official request on 15 images: 15 / 15 identical greedy tokens and parsed verdicts, max |logit-gap diff| 0.0, all 750 tensors equal to the shards (job 292005, `repro/veritaspp_fidelity.json`). 17.35 s per image on an RTX A5000, peak 18.9 GB up to 2048x2048 and 22.96 GB at the 16.8 MP processor cap (no `max_pixels` cap needed).

**Protocol.** 15 held benchmarks, each subsampled with `L = ceil(500 / generators)` reals and fakes per generator (the first L sorted; manifests `repro/vlm_bench/subsets/`), the same images for Veritas++ and the 27 field-report detectors; 14,937 images each. Greedy decoding; an unparsed answer is counted wrong (10 of 17,935, all truncations at 2,048 tokens); AUROC from the verdict-token logit gap. "Best publicly licensed" excludes LTF, MIRROR, VINA, PROBE and PROBE-ResNet (no clear licence; `internal_only` in `repro/vlm_bench/spec.json`). Ties for best are broken by the alphabetically first key. Source: `repro/vlm_bench/comparison.json`.

| benchmark (images per class and generator L) | Veritas++ acc | best overall | best publicly licensed | median of 27 | rank of 28 | unparsed | mean fake / real acc |
|---|---:|---|---|---:|---:|---:|---:|
| GenImage val (100k) (L=63) | **76.88** | 99.50 (`mirror`) | 96.03 (`omniaid_v1`) | 82.14 | 17 | 1 | 54.2 / 99.6 |
| ForenSynths (13 generators) (L=39) | **63.51** | 90.34 (`fat_former`) | 90.34 (`fat_former`) | 79.29 | 25 | 0 | 37.7 / 89.3 |
| UniversalFakeDetect (8 diffusion sets) (L=63) | **73.12** | 97.92 (`mirror`) | 95.83 (`npr`) | 87.40 | 25 | 3 | 49.2 / 97.0 |
| AIGCDetectBenchmark (17) (L=30) | **78.04** | 96.86 (`mirror`) | 93.73 (`omniaid_v1`) | 83.63 | 19 | 0 | 61.0 / 95.1 |
| Synthbuster (MSCOCO reals) (L=56) | **89.98** | 99.31 (`probe`) | 96.63 (`omniaid_v1`) | 87.80 | 13 | 0 | 80.0 / 100.0 |
| EvalGEN (L=100) | **99.10** | 99.90 (`mirror`) | 98.40 (`cure`) | 83.40 | 3 | 0 | 98.2 / 100.0 |
| SynthWildX (L=167) | **92.11** | 97.70 (`probe`) | 96.31 (`bfree`) | 61.28 | 6 | 1 | 93.8 / 90.4 |
| WildRF (L=167) | **95.64** | 97.47 (`mirror`) | 96.24 (`cure`) | 67.16 | 4 | 0 | 96.8 / 94.5 |
| AIGIBench (25 subsets, full) (L=20) | **82.40** | 89.00 (`mirror`) | 87.80 (`cure` (+1 tied)) | 79.20 | 11 | 0 | 69.6 / 95.2 |
| RRDataset (L=167) | **89.92** | 88.72 (`vina`) | 76.84 (`cure`) | 60.28 | 1 | 2 | 82.0 / 97.8 |
| BFree-Online (L=500) | **79.35** | 97.67 (`mirror`) | 94.93 (`dda`) | 58.03 | 8 | 0 | 70.6 / 88.1 |
| Co-Spy-Bench (110 subsets) (L=5) | **93.82** | 99.73 (`mirror`) | 98.18 (`safe`) | 89.27 | 7 | 0 | 91.6 / 96.0 |
| DiTFake (L=167) | **99.50** | 100.00 (`mirror`) | 99.10 (`safe`) | 94.31 | 2 | 0 | 99.2 / 99.8 |
| ARForensics (L=72) | **74.40** | 99.50 (`mirror`) | 97.72 (`omniaid_v1`) | 83.04 | 19 | 0 | 49.4 / 99.4 |
| RealChain (L=72) | **82.74** | 80.75 (`mirror` (+1 tied)) | 78.17 (`cure`) | 61.71 | 1 | 0 | 69.4 / 96.0 |

**Reading** (all from `comparison.json`): strong on in-the-wild and recent generators (1st of 28 on RRDataset and RealChain, 2nd DiTFake, 3rd EvalGEN, 4th WildRF), ahead of the best publicly licensed detector on RRDataset (+13.1), RealChain (+4.6), EvalGEN (+0.7) and DiTFake (+0.4); weak on GANs, early diffusion and AR generators (ForenSynths 63.51, 25th; UFD 25th; ARForensics 19th; GenImage 17th), where real accuracy stays at 89-100% while fake accuracy is 38-54% on average (below 50% on 9 / 13 ForenSynths, 4 / 8 GenImage, 4 / 7 ARForensics generators and 5 / 8 UFD sets). Its AUROC is higher than its accuracy on every one of those sets.

**JPEG quality 95 on every image** (same subsets): DiTFake 99.50 -> 99.60 (+0.10, AUROC +0.06; median detector -2.00 / -1.14); UFD-8 73.12 -> 67.86 (-5.26, AUROC -5.68; median -2.08 / -1.59), mostly on fakes (fake accuracy 49.2 -> 38.5). **Greedy vs the authors' sampled setting** (WildRF subset, 988 images): verdict agreement 96.96%, accuracy 95.64 -> 96.45, AUROC 99.25 -> 99.49. T 0.7 comes from the authors' `infer_vllm_single.py` (the paper states no temperature); top-p 0.8 / top-k 20 from the released `generation_config.json`.

**Against the paper (Acc., VERITAS++ row):** WildRF 96.4 vs **95.65** on the full data (2,503 images; -0.75; sampled subset 96.46, +0.06); RealChain 82.3 vs **82.74** on a 1,008-image subset (+0.44, 7% of the data); EvalGEN 99.5 vs **98.20** on a 500-fake subset of 55,298 (fakes only, -1.30) or **99.10** balanced with MSCOCO reals (-0.40). The paper does not say how EvalGen is scored: fakes only is our inference from DDA's row (97.2 = DDA's fakes-only EvalGEN accuracy). EvalGEN is **not reproduced** on full data (about 55k images at 17 s each is not affordable). The WildRF full run cost about 13.8 GPU-h (8.7 h beyond the subset), over the 12 h guideline: a judgement call to get one full-data paper number. `acc_pooled` in the result JSONs counts a shared real pool once per generator (UFD, Synthbuster); the headline `acc` is unaffected.

### Align, Don't Blend, VLM branch (official adapter, VLM kind; inference only; subsampled protocol)
The released VLM branch of Align, Don't Blend (arXiv 2512.06746 v3): the authors' LoRA (Gaffeyzz/Align-Dont-Blend-VLM @ 49975ccd,
Apache-2.0) on Qwen2.5-VL-7B-Instruct @ cc594898, which writes a short rationale and answers "This is an authentic /
AI-generated image.". The paper's detector ORs this branch with a pixel-artifact expert, the released DDA checkpoint
(confirmed by the authors, 2026-11-20): that full method is `adb_full` (next section), +0.11 above the paper on full
WildRF after the upstream sync. **The paper prints no VLM-only number for the released adapter.** **This section is the
upstream f4b4794 request** (not rerun on the subsets after the cf168db sync; the VLM-alone numbers with the cf168db
request on the paper benchmarks are in `models/image/adb_vlm/README.md`).

**Fidelity** (job 293634, `repro/adb_vlm_fidelity.json`), on 16 images:
- vs the unmodified official `predict.py` (ms-swift 4.4.2, pt backend): identical text, tokens (1,570) and parsed
  verdicts, 16 / 16;
- vs swift with `--logprobs`: chosen-token log-probability max diff 0.0;
- vs a direct transformers + peft call: verdict-token logit gap max diff 0.0 (this gap is not exposed by swift);
- the vLLM backend (the official default) was not compared;
- 0.204 images/s and a 17.26 GB peak on an RTX A5000.

**Protocol.** The Veritas++ subsets (identical images, 14,937 per detector), the official greedy decoding, and 0 unparsed
answers in every run. AUROC is read at the verdict word the official parser read: after `<answer>`, or, for the 446 of
23,981 per-image records (386 of 19,286 scored images; at most 4.2% per result JSON) whose answer is a bare final line
"This is an AI-generated / authentic image." without tags, at that line's word; 0 unscored answers (`n_unscored`) in
every run. Source: `repro/vlm_bench/comparison.json` (`vlms.adb_vlm`, `subset_means.adb_vlm`,
`paper_comparison.adb_vlm`); logs `repro/logs/adb_ivi_*.out` (the tag-less answers rescored in jobs 293800 / 293801 with
identical text, tokens and verdicts, so no accuracy changed).

| benchmark (L) | ADB acc | ADB AUROC | Veritas++ acc | best publicly licensed | median of 27 | rank of 28 (acc) | mean fake / real acc |
|---|---:|---:|---:|---|---:|---:|---:|
| GenImage val (100k) (L=63) | **68.05** | 82.70 | 76.88 | 96.03 (`omniaid_v1`) | 82.14 | 24 | 44.2 / 91.9 |
| ForenSynths (13) (L=39) | **55.72** | 64.55 | 63.51 | 90.34 (`fat_former`) | 79.29 | 26 | 39.2 / 72.2 |
| UniversalFakeDetect (8) (L=63) | **52.88** | 54.65 | 73.12 | 95.83 (`npr`) | 87.40 | 28 | 41.5 / 64.3 |
| AIGCDetectBenchmark (17) (L=30) | **65.20** | 74.24 | 78.04 | 93.73 (`omniaid_v1`) | 83.63 | 26 | 41.0 / 89.4 |
| Synthbuster (MSCOCO reals) (L=56) | **80.95** | 93.22 | 89.98 | 96.63 (`omniaid_v1`) | 87.80 | 21 | 63.7 / 98.2 |
| EvalGEN (L=100) | **97.30** | 99.79 | 99.10 | 98.40 (`cure`) | 83.40 | 5 | 99.6 / 95.0 |
| SynthWildX (L=167) | **89.22** | 97.74 | 92.11 | 96.31 (`bfree`) | 61.28 | 7 | 95.2 / 83.2 |
| WildRF (L=167) | **93.92** | 98.81 | 95.64 | 96.24 (`cure`) | 67.16 | 5 | 95.1 / 92.7 |
| AIGIBench (25) (L=20) | **76.70** | 87.68 | 82.40 | 87.80 (`cure` (+1 tied)) | 79.20 | 16 | 62.2 / 91.2 |
| RRDataset (L=167) | **69.86** | 77.42 | 89.92 | 76.84 (`cure`) | 60.28 | 7 | 44.3 / 95.4 |
| BFree-Online (269 real + 500 fake) | **87.96** | 98.36 | 79.35 | 94.93 (`dda`) | 58.03 | 6 | 77.4 / 98.5 |
| Co-Spy-Bench (110) (L=5) | **78.36** | 84.84 | 93.82 | 98.18 (`safe`) | 89.27 | 19 | 72.7 / 84.0 |
| DiTFake (L=167) | **95.11** | 98.86 | 99.50 | 99.10 (`safe`) | 94.31 | 10 | 93.0 / 97.2 |
| ARForensics (L=72) | **64.78** | 68.27 | 74.40 | 97.72 (`omniaid_v1`) | 83.04 | 23 | 36.9 / 92.7 |
| RealChain (L=72) | **72.72** | 80.57 | 82.74 | 78.17 (`cure`) | 61.71 | 8 | 63.7 / 81.8 |
| **mean of 15** | **76.58** | **84.11** | 84.70 (AUROC 91.33) | | | | |

**Reading.**
- **In the wild it does well:** 5th to 8th of 28 on WildRF, EvalGEN, BFree-Online, SynthWildX, RRDataset and RealChain.
  It is behind the best publicly licensed detector on all 15.
- **Against Veritas++:** ahead in accuracy only on BFree-Online (+8.61), and in AUROC on BFree-Online and SynthWildX.
- **Near chance on GANs, early diffusion and AR generators:** UFD-8 52.88 (last), ForenSynths 55.72, ARForensics 64.78,
  AIGCDetect 65.20, GenImage 68.05. Fake accuracy is 37-44% on average there. This fits a VLM trained on semantic
  implausibility that leaves pixel artifacts to its DDA partner (`adb_full`).
- **JPEG quality 95 on every image:** DiTFake +0.30 acc, UFD-8 -0.70 (median detector -2.00 / -2.08; Veritas++ +0.10 / -5.26).

**Against the paper (balanced accuracy).** The printed AlignGemini numbers are the full method (`adb_full`, next
section), so for the VLM branch alone they are context, not targets:

| benchmark | paper, full method (context) | SynthScope, VLM branch only |
|---|---:|---|
| WildRF (full, 2,503 images) | 92.9 | 93.98 |
| BFree-Online (269 real + 500 fake subset) | 97.5 | 87.96 |
| AIGI-Bench CommunityAI + SocialRF (20 + 20 slices) | 88.8 | 82.50 |
| GenImage (subset, fakes q96) | 91.7 | 65.97 |
| AIGCDetect (subset, fakes q96, the official protocol) | 87.7 | 63.92 |
| CO-SPY in-the-wild | 94.3 | not held |

Notes:
- **GenImage and AIGCDetect:** fakes JPEG q96 in memory, reals untouched: the official protocol since upstream cf168db (`tools/jpeg96.py`), which names both benchmarks.
  SynthScope had applied it before (q96 following DDA; on AIGCDetect as its own inference), so these rows did not change.
- **Tables 4 / 5:** their Qwen2.5-VL-7B rows are the ambiguous / untrained base model, not the released adapter.
- **Cache identity:** editing any `models/vlm/*.py` re-keys the per-image caches of every VLM detector, Veritas++
  included; the identity also holds the base model's revision and the resize parameters. The committed results are
  unaffected. The ADB caches were carried over the final-review re-key by `repro/vlm_bench/migrate_vlm_cache.py`
  (18,900 records kept, the 386 tag-less ones rescored). Details in `models/image/adb_vlm/README.md`.

### Align, Don't Blend, full method (`adb_full` = VLM OR DDA; official weights, the official decision rule)
The paper's detector, as released: the `adb_vlm` adapter OR the DDA checkpoint (`dda.pth` = HF `DDA_ckpt.pth`, sha256
identical), fused per image by the rule of the official `predict.py` (GzZning/Align-Dont-Blend @ cf168db): fake if the
VLM's verdict is "AI-generated" (an answer the official parser rejects is Real) or DDA's P(fake) >= 0.5. AUROC / AP use
SynthScope's fused score, `max(P_vlm clamped to its verdict's side of 0.5, P_dda)`, which is >= 0.5 exactly when the
official decision is fake.

**Upstream sync (cf168db).** Upstream commits 2f0adbc (the task prompt as the user message, system "You are a helpful
assistant.", repetition penalty 1.0, max 1,210,000 pixels), 898b607 (unparsed answers Real) and cf168db (fakes of GenImage
and AIGCDetectBenchmark JPEG q96 in memory, `tools/jpeg96.py`, = `bench_eval --jpeg-fakes 96` pixel for pixel) are
followed. Every number below is the cf168db request; the earlier rows (upstream f4b4794 request) stay in
`paper_comparison.json` for the paired comparison.

**Fidelity** (job 301130, RTX A6000, `repro/adb_full_fidelity.json`): the unmodified official `predict.py` @ cf168db with
the README's expert command, on 24 images (the 16 `adb_vlm` fidelity images + 8 GenImage images where the components
disagreed). Final, DDA and VLM decisions, `vlm_answer_parsed` and the VLM text are identical 24 / 24 in bf16 (pt backend)
and in the README's float32 / eager command. DDA P(fake) is within 2.1e-6 of the official expert's. A CPU test runs the
official `predict.py` on mocked component outputs, including the 0.5 threshold edge and unparsed answers: 49 / 49 identical.

**Paper benchmarks** (`repro/adb_full/paper_comparison.json`, the `*_cf168db` rows; VLM bf16 greedy on RTX A5000, job 301129;
DDA per image on the same files, sha1-checked; bootstrap 95% CIs):

| benchmark | data (n) | paper | `adb_full` [95% CI] | delta | AUROC | VLM alone | DDA alone | before the sync (paired change) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WildRF | full (2,503) | 92.9 | **93.01** [91.87, 94.17] | +0.11 | 99.47 | 95.96 | 90.20 | 91.22 (+1.79 [+0.98, +2.65]) |
| AIGI-Bench CommunityAI + SocialRF | full (18,000) | 88.8 | **89.22** [88.75, 89.69] | +0.42 | 98.03 | 88.81 | 83.18 | 9,000-image selection: 87.52 -> 89.49 (+1.97 [+1.50, +2.47]) |
| BFree-Online | 807 of 1,111 held | 97.5 | **96.84** [95.44, 98.05] | -0.66 | 99.41 | 81.69 | 94.98 | 96.93 (-0.09 [-1.02, +0.84]) |
| GenImage (fakes q96) | first 625 + 625 per generator (10,000) | 91.7 | **88.86** [88.32, 89.38] | -2.84 | 96.47 | 59.50 | 89.35 | 86.70 (+2.16 [+1.77, +2.53]) |
| AIGCDetect (fakes q96) | first 100 + 100 per generator (3,400) | 87.7 | **84.76** [83.82, 85.74] | -2.94 | 92.93 | 59.06 | 85.94 | 1,020 subset: 83.82 -> 84.71 (+0.88 [-0.29, +2.06]) |
| CO-SPY in-the-wild | not held | 94.3 | — | | | | | |

- **WildRF and AIGI-Bench reproduce** on the paper's full data (+0.11, +0.42 on the mean). WildRF Twitter is still 2.10
  below the paper (93.40 vs 95.5); Facebook (+1.24) and Reddit (+1.30) are above it.
- **BFree-Online** is close (-0.66) on partial data (807 of 1,111 images still online).
- **GenImage and AIGCDetect** are samples, about 3 points below the paper. There, DDA alone is still above the fused
  number (paired: -0.49 [-0.73, -0.27] and -1.18 [-1.76, -0.59]).
- **The sync** mainly cut the VLM's false positives on reals, which is what limits the OR: the fused number rose on four
  of five benchmarks although the VLM-alone number fell on BFree-Online (-5.39), GenImage (-1.55) and AIGCDetect (-4.31).
- **Unparsed VLM answers** (Real, as `predict.py`): 287 of the 34,710 answers (0.8%; `repro/adb_full/vlm_cf168db/unparsed_answers.json`).
- **The bf16 text depends on precision and GPU.** On the 24 fidelity images, official bf16 and float32 give the same VLM
  text on 2 and the same final decision on 24. Every row above is bf16 on A5000; with the earlier request a float32 run
  on an A6000 moved full WildRF by +0.12.
- **Logs** (per row in `paper_comparison.json`: `vlm_logs`, `dda_logs`): fusion `repro/logs/adb_full_fuse.out`; VLM
  units and the new DDA runs `repro/logs/adbsync_301129*.out`; the reused DDA records `adbfull_bench_297122.out` and
  `adbfull_gi10k_297930.out`. `trace_check.py` traces every JSON to them.

### FakeVLM (official weights, VLM kind; inference only; subsampled protocol; GenImage in-distribution)
FakeVLM (arXiv 2503.14905, NeurIPS 2025): LLaVA-1.5-7B fine-tuned on FakeClue (HF lingcco/fakeVLM @ dce9e88d, no licence;
Llama 2-licensed base), with the llava-hf/llava-1.5-7b-hf @ a272c74 processor as the official `eval.py` loads it. It
answers "This is a real / fake image." and then explains the artifacts. **None of the paper's benchmarks (FakeClue, LOKI,
DMimage, DD-VQA, FF++) is held, so there is no reproduction claim.** **GenImage is in-distribution**: FakeClue contains
GenImage images. FakeClue also draws on other sources (Chameleon, FF++, documents, satellite images), and its overlap
with the other 14 subsets has not been checked.

**Fidelity** (job 293873, `repro/fakevlm_fidelity.json`), on 16 images:
- first sentence and parsed verdict equal all five official runs (unmodified `eval.py` in batches of 4 and 2 under
  transformers 4.45.2 and of 4 under 5.5.4; the official code one image at a time under 4.45.2 and 5.5.4), 16 / 16 each;
- the greedy text equals the single-image run under the authors' 4.45.2 on 16 / 16. The batched official texts differ
  among themselves with the batch size (4.45.2 bs4 vs bs2: 14 / 16), so the batch-independent single-image run is the
  token-level reference; `eval.sh`'s batch of 16 and flash-attention 2 were not run;
- engine vs a direct transformers call on the same runtime: logit-gap max diff 0.0;
- three transformers-version steps make the request equal to 4.45.2's: the processor's image-expansion attributes, the
  4.45.2 PIL resize + crop (pixels to 2.4e-7), and padding to 1,599 positions;
- 0.294 images/s and a 15.1 GB peak on an RTX A5000.

**Protocol.** The Veritas++ subsets (identical images, 14,937 per detector), the official greedy decoding with 256 new
tokens, the official parse; 1 unparsed answer (an ARForensics real) and 0 unscored over the 16,947 scored images (15
subsets + 2 JPEG95 probes). Source: `repro/vlm_bench/comparison.json` (`vlms.fakevlm`, `subset_means.fakevlm`,
`paper_comparison.fakevlm`); logs `repro/logs/fakevlm_bench_293884.out` / `293885.out`.

| benchmark (L) | FakeVLM acc | FakeVLM AUROC | Veritas++ acc | ADB acc | best publicly licensed | median of 27 | rank of 28 (acc) | mean fake / real acc |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| GenImage val (100k) (L=63), **in-distribution** | **99.70** | 99.99 | 76.88 | 68.05 | 96.03 (`omniaid_v1`) | 82.14 | 1 | 99.8 / 99.6 |
| ForenSynths (13) (L=39) | **76.43** | 88.49 | 63.51 | 55.72 | 90.34 (`fat_former`) | 79.29 | 17 | 96.06 / 56.8 |
| UniversalFakeDetect (8) (L=63) | **97.02** | 99.90 | 73.12 | 52.88 | 95.83 (`npr`) | 87.40 | 2 | 95.44 / 98.61 |
| AIGCDetectBenchmark (17) (L=30) | **92.45** | 97.55 | 78.04 | 65.20 | 93.73 (`omniaid_v1`) | 83.63 | 5 | 97.45 / 87.45 |
| Synthbuster (MSCOCO reals) (L=56) | **96.52** | 99.85 | 89.98 | 80.95 | 96.63 (`omniaid_v1`) | 87.80 | 6 | 98.41 / 94.64 |
| EvalGEN (L=100) | **98.10** | 99.94 | 99.10 | 97.30 | 98.40 (`cure`) | 83.40 | 4 | 99.2 / 97.0 |
| SynthWildX (L=167) | **65.46** | 87.78 | 92.11 | 89.22 | 96.31 (`bfree`) | 61.28 | 13 | 98.0 / 32.93 |
| WildRF (L=167) | **64.11** | 80.86 | 95.64 | 93.92 | 96.24 (`cure`) | 67.16 | 17 | 94.81 / 33.41 |
| AIGIBench (25) (L=20) | **83.90** | 95.38 | 82.40 | 76.70 | 87.80 (`cure` (+1 tied)) | 79.20 | 9 | 97.4 / 70.4 |
| RRDataset (L=167) | **62.67** | 74.64 | 89.92 | 69.86 | 76.84 (`cure`) | 60.28 | 14 | 75.25 / 50.1 |
| BFree-Online (269 real + 500 fake) | **70.47** | 70.17 | 79.35 | 87.96 | 94.93 (`dda`) | 58.03 | 9 | 95.6 / 45.35 |
| Co-Spy-Bench (110) (L=5) | **96.00** | 99.02 | 93.82 | 78.36 | 98.18 (`safe`) | 89.27 | 6 | 96.0 / 96.0 |
| DiTFake (L=167) | **99.30** | 99.99 | 99.50 | 95.11 | 99.10 (`safe`) | 94.31 | 2 | 99.8 / 98.8 |
| ARForensics (L=72) | **97.92** | 99.84 | 74.40 | 64.78 | 97.72 (`omniaid_v1`) | 83.04 | 2 | 96.43 / 99.4 |
| RealChain (L=72) | **77.78** | 82.77 | 82.74 | 72.72 | 78.17 (`cure`) | 61.71 | 5 | 84.13 / 71.43 |
| **mean of the 14 held-out (without GenImage)** | **84.15** | **91.16** | | | | | | |
| mean of 15 (with GenImage) | 85.19 | 91.74 | 84.70 (AUROC 91.33) | 76.58 (AUROC 84.11) | | | | |

**Reading.**
- **The headline is the held-out mean, 84.15 / 91.16.** GenImage (99.70, 1st of 28) is in the training data.
- **Strong on GAN, diffusion and autoregressive sets:** 2nd of 28 on UFD-8, DiTFake and ARForensics, each above the best
  publicly licensed detector (+1.19, +0.20, +0.20); 4th-6th on EvalGEN, AIGCDetect, Synthbuster and Co-Spy-Bench.
- **Weak in the wild, because it calls real photos fake:** real accuracy 32.93 on SynthWildX, 33.41 on WildRF, 45.35 on
  BFree-Online and 50.1 on RRDataset, with fake accuracy 75.25-98.0. This is the opposite of ADB.
- **JPEG quality 95 on every image:** DiTFake -0.40 acc / -0.04 AUROC; UFD-8 **-31.35 / -12.05** (median detector
  -2.08 / -1.59; Veritas++ -5.26 / -5.68; ADB -0.70 / -0.84). On UFD-8 the real accuracy stays at 98.61 and the mean fake
  accuracy falls from 95.44 to 32.74, so the re-encoded fakes are called real. That the model relies on a format or
  compression cue is a hypothesis, not tested.
- **Cache identity:** the benchmark caches were written with VLM code version 02852acf62. The later
  `num_additional_image_tokens` load check (4a950958ed) does not change any score, and
  `repro/vlm_bench/migrate_vlm_cache.py` carries the FakeVLM caches, and the ADB / Veritas++ caches of main e5749ae,
  over byte for byte. Details in `models/image/fakevlm/README.md`.

### WaveRep (official checkpoints G1 and G4; PARTIAL paper comparison)
Fidelity: SynthScope's class == the official timm `vit_base_patch14_reg4_dinov2` model (177/177 state-dict tensors identical, strict load, both checkpoints). The unmodified official demo (`main_avideo.py` / `main_directory.py` @ `0fd6010`) and SynthScope agree: 15 videos in 6 codecs (model inputs bit-exact; video logit max diff 6.0e-7; CPU) and 180 GenVideo-Val files per checkpoint (max |logit diff| 1.02e-5 G1 / 1.05e-5 G4, 0 sign flips, 0 NaN; GPU-vs-GPU; `repro/waverep_official_parity_genvideo180.json`, DAS6 67291). Protocol: official OpenCV reader, the first 64 frames (paper App. D; the demo's default is every frame, `num_frames=0`), 504 centre crop, mean frame logit, balanced accuracy at 0.5 and AUC (the paper's metrics). Two released checkpoints: `waverep` = G1 ("a single generator" per the official demo README; we infer Pyramid Flow, the paper's detector), `waverep_g4` = G4 ("four generators", the demo's default; we infer the Table 5 set). No training code is released, so there is no SynthScope-trained number.

**What is held.** The paper's own test set (7 generators vs 300 Panda-70M YouTube reals) is **NOT held**: the reals need a YouTube downloader and were not scraped. The comparison below uses the 8 GenVideo generators that are among the paper's columns, scored on full GenVideo-Val with its own 10,000 reals; the paper does not say which reals it paired, so this is a **PARTIAL** comparison.

| Generator (n fake) | Paper bAcc (Table 1) | G1 `waverep` bAcc | G4 `waverep_g4` bAcc (reference only) | Paper AUC (Table 7) | G1 AUC | G4 AUC |
|---|---:|---:|---:|---:|---:|---:|
| Crafter (1398) | 98.3 | 98.99 | 98.90 | 99.8 | 99.90 | 99.98 |
| Gen2 (1380) | 98.8 | 99.49 | 98.90 | 100.0 | 99.99 | 99.99 |
| HotShot (700) | 81.4 | **60.10** | 83.40 | 91.0 | **75.60** | 94.15 |
| LaVie (1400) | 95.5 | **92.06** | 97.33 | 97.7 | 97.03 | 99.26 |
| ModelScope (700) | 97.1 | 95.24 | 97.54 | 99.4 | 99.31 | 99.69 |
| MorphStudio (700) | 97.0 | 97.74 | 98.61 | 99.3 | 99.74 | 99.94 |
| Show-1 (700) | 92.1 | **79.67** | 96.83 | 97.1 | 94.27 | 99.43 |
| MoonValley (626) | 98.4 | 99.12 | 98.89 | 99.9 | 99.97 | 99.98 |
| 8-generator mean | 94.83 | 90.30 | 96.30 | 98.03 | 95.73 | 99.05 |

Reading: 5 of 8 G1 cells are within about 2 points (Crafter, Gen2, MorphStudio, MoonValley, ModelScope -1.86); **HotShot (-21.3 bAcc / -15.4 AUC), Show-1 (-12.4 / -2.8) and LaVie (-3.4 / -0.7) are NOT reproduced.** HotShot: G1 real accuracy 99.2 but fake recall 21.0% at P >= 0.5 (Show-1 60.1%, LaVie 84.9%); AUC falls too, so it is not a calibration or threshold offset. The cause is untested (a different real pool or fake subset in the paper is a possibility; we cannot check it). G4 is trained on four generators, so its higher numbers are not single-generator generalization. Source: `repro/bench_results/waverep_genvideo_full.json`, `waverep_g4_genvideo_full.json` (`per_generator` `acc`, `auroc`); paper values from arXiv 2506.16802 Tables 1 and 7. Sora (56 fakes: G1 67.46 / 68.54, G4 76.65 / 83.61) and WildScrape (924: 78.87 / 86.92, 85.77 / 90.83) are GenVideo-Val extras with no paper cell.

**Zero-shot reference rows (NOT paper reproductions; the paper reports none of these sets).** Means over the generators; bAcc at 0.5 / mAP (fake positive) / mAUROC.

| Set | gens | G1 `waverep` | G4 `waverep_g4` |
|---|---:|---|---|
| GenVideo-Val full (10,000 reals + all fakes) | 10 | 86.87 / 79.85 / 92.13 | 93.28 / 88.88 / 96.69 |
| GenVideo-Val `--limit 1000` (D3 protocol) | 10 | 86.62 / 89.81 / 92.09 | 92.65 / 94.47 / 96.53 |
| VideoPhy `--limit 1000` | 8 | 90.00 / 93.21 / 96.54 | 95.73 / 98.20 / 99.29 |
| VidProM `--limit 1000` | 4 | 89.85 / 98.11 / 98.06 | 95.80 / 99.29 / 99.25 |
| EvalCrafter (fakes vs the **first 1000 GenVideo-Val MSR-VTT reals**; EvalCrafter has none of its own) | 14 | 92.84 / 96.93 / 97.17 | 96.55 / 99.19 / 99.29 |

`--limit 1000` = the first 1000 clips per class per generator. Two VidProM Pika fakes (`00642_pika-9e93dcfc...`, `00891_pika-e025f9ee...`) are empty (0-byte) files in our copy, so OpenCV decodes no frame; they are excluded (998 scored); every other set decodes fully. Weakest G1 cells: VideoPhy OpenSora 62.15 / ZeroScope 63.75, VidProM ModelScope 71.55, GenVideo Sora 67.46; G4 is better on all of them. Full GenVideo-Val mAP (fake positive) is low only through the 10,000 : 56-1400 class imbalance. Jobs: IVI 285376_0, 285390_3-4, 285389_2 / 285886_2, DAS6 67328 / 67404 (scoring), IVI 286053 / 286059 (aggregation), DAS6 67291 (official demo); logs `repro/logs/waverep_*`. Licence: GRIP-UNINA, free for informational / non-profit use only, so the weights are not redistributed, not in the model zoo, and not on the Space (pending user approval: needs uploading the checkpoints to the hub; about 30 s per clip on CPU). Details: `models/video/waverep/README.md`.

### NSG-VD (official MMD-MP checkpoints; the released-CSV protocol reproduces, the paper's Tables 1-3 do not)
Fidelity: SynthScope's pipeline (ffmpeg frame selection, the guided-diffusion normalized spatiotemporal gradient, the TALL discriminator, MMD against the locally built 100-video MSR-VTT reference) equals the unmodified official code: all 47 tensors of each of the 6 released checkpoints load strictly; the reference file is bit-identical to the official `ref_data`; frames identical on 75 of 75 videos; NSG features and scores **bit-identical (max abs diff 0.0)** at the official grouping (3 videos per diffusion pass, jobs 286253 / 286389, `repro/nsg_vd_official_parity_genvideo75.json`). The served path scores one video per pass and differs from the official batching, below. Metrics are the official ones (`test_dMMD.py`: pooled accuracy over real + fake, recall / precision / F1 on the fake class, AUROC of the raw statistic, fake iff the statistic is above 1; `repro/nsg_vd_benchmark_genvideo_val.json`), not `bench_eval_video`'s balanced accuracy.

| `nsg_vd` = standard-SEINE-mp, 10-generator average | accuracy | recall | F1 | AUROC | precision |
|---|---:|---:|---:|---:|---:|
| released results CSV (500 per generator, Sora 51 + 56) | 88.97 | 98.54 | 89.88 | 97.68 | 82.63 |
| official code on our data (same protocol) | 89.22 | 98.50 | 90.08 | 97.51 | 83.00 |
| SynthScope (served path, same protocol) | 88.96 | 98.52 | 89.87 | 97.50 | 82.63 |
| paper Table 2 (120 per generator) | 86.05 | 97.13 | 87.45 | 95.46 | not printed |
| L120 = the paper protocol (official code / SynthScope; Sora 56 + 51) | 86.90 / 86.52 | 98.72 / 98.72 | 88.23 / 87.93 | 97.16 / 97.18 | 79.76 / 79.28 |

All three registered keys (Avg. accuracy / AUROC; released CSV, official code, SynthScope): `nsg_vd` 88.97 / 97.68, 89.22 / 97.51, 88.96 / 97.50; `nsg_vd_pika` (standard-Pika-mp, paper Table 1) 89.34 / 96.80, 89.42 / 96.69, 89.15 / 96.75; `nsg_vd_unbalance` (unbalance-SEINE-mp, paper Table 3) 90.37 / 96.95, 90.37 / 96.83, 90.47 / 96.90. Per-generator tables for all three: `models/video/nsg_vd/README.md`.

* **Paper Tables 1-3, under the paper's own protocol: Table 2 (default key) reproduces; Tables 1 and 3 do not.** The authors confirmed (2026-11-13) that Tables 1-3 used 120 reals + 120 fakes per generator (56 each for Sora), selected from the released test splits via `test_load_len` in `test.yaml`, with ffmpeg 4.2.2. That is our L120 run (the official `test_dMMD` with `test_load_len` = 120 on our data); the one difference is Sora, where the released test split has only 51 fakes (we have 56 reals + 51 fakes; the paper 56 + 56). L120 (official code) against the paper's Avg., accuracy / recall / F1 / AUROC gaps: `nsg_vd` (Table 2) +0.85 / +1.59 / +0.78 / +1.69, **reproduced** (all within 2 points); `nsg_vd_pika` (Table 1) -1.75 / +8.65 / -0.56 / +0.73, **not reproduced** (recall); `nsg_vd_unbalance` (Table 3) +2.11 / +5.45 / +2.35 / +1.35, **not reproduced** (recall, accuracy). Raw numbers: acc / recall / AUROC 86.90 / 98.72 / 97.16 vs 86.05 / 97.13 / 95.46; 89.71 / 96.67 / 96.88 vs 91.46 / 88.02 / 96.14; 91.27 / 98.67 / 96.26 vs 89.16 / 93.21 / 94.91. The released CSVs differ from the tables too (CSV minus paper, Avg., largest metric per key: SEINE-mp accuracy +2.92, Pika-mp recall +7.84, unbalance-SEINE-mp recall +5.23). For Tables 1 and 3 the cause is the checkpoints. The authors confirmed (2026-11-15) that the released checkpoints come from separate training runs made after they moved to a different machine, so they are not the exact checkpoints behind Tables 1-3; they verified that the released checkpoints and evaluation code reproduce the CSVs in their repository, which matches our reproduction. Their stated paper-vs-CSV gaps: average recall in the CSVs is higher by 1.41-7.84 points and average AUROC differs by 0.66-2.22; recall is measured at a fixed threshold and is more sensitive to run-to-run variation. Worst cells: Table 1 ModelScope recall 68.33 vs 90.00, Table 3 Sora recall 82.14 vs 100.00.
* **Sora AUROC: unexplained.** Sora's accuracy and recall equal the CSV, but its AUROC is below it for every checkpoint: official code 0.88 / 0.46 / 0.84 below for SEINE-mp / Pika-mp / unbalance-SEINE-mp (0.87 / 0.84 / 0.84 for the unregistered MMD-D ones), SynthScope 0.84 / 0.32 / 0.37. Candidates, none verified: a different copy or encode of the Sora files, a different reference or cache state. Against a global build effect: with 4.2.2 the other nine generators match the CSV AUROC within 0.17, and with 9.0.2 ModelScope moves away from it.
* **Served path vs official batching** (9,107 pairs per checkpoint): per-video max |diff| 0.096 / 0.267 / 0.069 (the 0.267 is WildScrape D87 at the kernel ceiling, 1.146 vs 1.413, fake on both paths; within 0.1 of s = 1 the gap is at most 0.03); 28 / 33 / 32 verdicts differ; Avg accuracy differs by -0.26 / -0.27 / +0.10. Cause: batching (MMD batch: official shuffled batches of 24 vs 1; diffusion pass: official 3 videos vs 1). On the 75-video parity run the MMD batch alone reproduces the gap for the three MMD-MP checkpoints (0.037 / 0.035 / 0.022 with only the MMD batch changed vs 0.038 / 0.024 / 0.022 for the served path) and the diffusion grouping adds to it for some -d checkpoints; the split is not measured on the 9,107 pairs. The official code against itself (other seeds) moves 0.03-0.06 per video with 0-3 flips, so the served path is outside its shuffle noise.
* **ffmpeg is pinned to 4.2** (`$SYNTHSCOPE_FFMPEG_NSG_VD`); the authors used 4.2.2 for the paper (confirmed 2026-11-13), which is our pin. 7.0.2 is unusable (it drops frames and selects different ones, `repro/nsg_vd_ffmpeg_builds_genvideo75.json`). 9.0.2 (SynthScope, job 286453): ModelScope accuracy 88.50 vs 88.40, AUROC 96.39 vs 95.99, 51 of 1,000 verdicts flip; Sora accuracy 87.85 vs 86.92, AUROC 97.72 vs 96.08.
* **Zero-shot reference rows** (`nsg_vd` only; the reference set stays the 100 MSR-VTT val videos; NOT paper results; `--limit 1000`; jobs 286717-286719): VidProM (4 generators) accuracy 86.80 / recall 95.80 / precision 81.16 / AUROC 95.49; VideoPhy (8) 83.17 / 99.23 / 59.87 / 97.88; EvalCrafter (14, reals = GenVideo-Val's MSR-VTT) 86.47 / 99.12 / 75.35 / 97.79. The false-positive rate at s > 1 is 22.2% on every real pool, so accuracy and precision look low where fakes are few. 12 of the 100 reference reals are in each real pool; removing them changes accuracy, recall and AUROC by at most 0.03 and precision / F1 by up to 0.22 / 0.17 (averages).
* **Deviations.** Frames for the official run were extracted beforehand by the official `process_video` (job 286532) because the official moviepy readers were OOM-killed (jobs 286429 / 286438 / 286449); every official number is a warm-cache run (a cold run, job 286533, was OOM-killed after the extraction); seed 42 (seeds 1 and 2 are in the JSON); a pandas `applymap` alias was needed (applied after the metrics; the logged metrics equal the recomputed ones); the official `.py` sources were identical to the repository's in every job (only `__pycache__` differed; the job check did not list files present only in the staged copy). Drivers are private and named by job id; the `_tmp` paths in `repro/bench_results/nsg_vd*.json` are node-local.
* **Availability and licence.** Apache-2.0 code and checkpoints, MIT guided-diffusion; the reference file is derived from MSR-VTT (research use), so it is not redistributed. Not on the Space or in the model zoo: that needs approval, the MSR-VTT terms check, the 2.2 GB diffusion model plus the 482 MB reference file, ffmpeg 4.2 in the image, and CPU time (logged job 286994, `repro/nsg_vd_cpu_timing.json`: about 76 s per clip with 2 threads, 32-35 s with 8, 25 s with 16; one clip, shared node; 0.45-0.73 s per video on an RTX A5000).

### WeVisForensics (official adapters + code, agent kind; inference only; subsampled protocol; GenBuster in-distribution)
WeVisForensics (arXiv 2611.21373; Tencent/WeVisForensics @ 43ae1c0, HF tencent/WeVisForensics @ 9fa025a0, Apache-2.0) is an
agentic detector. A Qwen3.5-4B investigator (`main` LoRA) looks at a 1 fps overview, calls up to four inspector LoRAs on
the same base (visual appearance, physical consistency, common sense, OCR) and answers real or fake.

How SynthScope runs it:
- the official transformers backend (`demo/wvf_local.py`) runs unmodified in a separate process (`models/video/wevis/runner.py`),
  with transformers 5.23.0, flash-linear-attention 0.6.0 and PaddleOCR 3.7 / paddlepaddle 3.2.2 pinned and checked;
- greedy decoding;
- audio off (`has_audio=False`): the audio tools need gated / non-commercial models, and in our benchmarks only some fakes
  carry audio, so the flag alone would give the label away;
- the verdict is the agent's answer, and AUROC ranks the fake-minus-real logit at the final verdict token, read during
  generation.

**Fidelity** (job 295351, RTX A5000, `repro/wevis_fidelity.json`):
- 17 / 17 clips identical to the official backend in answer, tool calls, every turn's raw text and the final text;
- the logit gap equals a direct `generate(output_logits=True)` call exactly;
- 7 / 7 demo verdicts equal the recorded vLLM trajectories, but some tool traces differ there; the authors' multi-GPU
  vLLM evaluator is not reproduced;
- one 74 s 1080p clip ran out of memory on 24 GB in both backends.

A spot check after the fix round (job 295556, RTX A6000) gave 3 / 3 identical. The A6000 logits differ from the A5000
ones by up to 2.75, with the same traces, so every benchmark row ran on RTX A6000s (ivi-cn023).

**Protocol** (`repro/video_agent_bench/`: `spec.json`, `comparison.json`, `results/<subset>/*.json`; logs
`repro/logs/wevis_wevis_w0_295637.out`, `wevis_wevis_w1_295640.out`, `wevis_wevis_w0_295611.out`):
- GenVideo-Val, VideoPhy, VidProM and EvalCrafter at L = ceil(250 / generators) per class per generator (EvalCrafter vs
  the first L GenVideo-Val MSR-VTT reals);
- GenBuster-Bench / ++ at the first ceil(0.4 n) sorted clips of every folder (the full sets would cost about 60 GPU-h);
- the agent and 8 video detectors score byte-identical files;
- 3,208 agent rows: 0 tool errors, 0 OOM, 0 unparsed; 41.1 s and 5.2 model calls per video; peak 22.69 GiB; 36.6 GPU-h;
- `trace_check.py`: 54 / 54 JSONs and 3,208 / 3,208 per-video records trace to the logs.

| subset | wevis acc | AUROC | real / fake acc | rank acc (of 9) | best comparator acc | median acc |
|---|---:|---:|---:|---:|---|---:|
| GenVideo-Val (L=25) | 85.20 | 92.20 | 72.00 / 98.40 | 7 | SPLIT 100.0 | 88.60 |
| VideoPhy (L=32) | 86.33 | 90.05 | 75.00 / 97.66 | 7 | VINA 99.41 (public: SPLIT 98.44) | 92.19 |
| VidProM (L=63) | 85.52 | 89.11 | 73.02 / 98.02 | 7 | SPLIT 99.21 | 89.68 |
| EvalCrafter + MSR-VTT reals (L=18) | 85.32 | 89.57 | 72.22 / 98.41 | 6 | SPLIT 100.0 | 92.85 |
| **mean of the 4 held (headline)** | **85.59** | **90.23** | | **7** | SPLIT 99.41 | 91.40 |
| GenBuster-Bench, **in-distribution**, pooled per clip (n = 1,260) | 97.30 | 96.11 | 93.25 / 99.19 | 1 | VINA 89.37 (public: WaveRep-G4 85.63) | 69.80 |
| GenBuster-Bench++, **in-distribution**, pooled per clip (n = 800) | 97.88 | 98.65 | 98.25 / 97.50 | 1 | VINA 89.00 (public: WaveRep-G4 80.00) | 52.56 |

The held-subset accuracy is balanced: real and fake per generator, then the mean over generators. The GenBuster rows
are the paper's metric, one row per clip, and their AUROC is the per-generator mean.

**Reading.**
- **The held-out headline is 7th of 9.** The real videos drive it: the agent calls about 1 in 4 of the low-resolution
  GenVideo-Val MSR-VTT reals fake (37 of 138 real rows), while fake accuracy is 97.7-98.4. All 63 distinct reals are
  298x224 at 3 fps; the fakes' median frame height is 512 (`repro/video_agent_bench/msrvtt_probe.json`). The real pools are small
  (18-63 per subset).
- **GenBuster-Bench / ++ reproduce the paper** on the 40% subsample: 97.30 vs 97.7 and 97.88 vs 97.5, both paper values
  inside our 95% intervals. Also fake recall 99.19 / 97.50 vs 99.3 / 97.2, and in-domain / OOD / wild fakes 99.5 /
  99.25 / 96.67 vs 99.7 / 99.6 / 95.3.
- **GenBuster is not a held-out result.** GenBuster-200K is in its training data, and the comparators were trained
  elsewhere (SPLIT and D3 call nearly every GenBuster real fake). The 6-subset mean (89.29, 2nd of 9) includes these
  sets, so it is not a held-out number either.
- **Near-binary scores:** |logit| 9.1-24.75, so AUROC adds little to accuracy.
- **Determinism:** 136 / 136 repeated-file pairs are identical.
- **Not held:** the paper's MintVid (mean 88.0) and AgenticCue benchmarks.

Details: `models/video/wevis/README.md`.

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
1. **Architecture is faithful where we can check it.** Loading official weights into the SynthScope class reproduces the paper for CLIP, CNN, FatFormer, NPR, **PGC** (since the 2026-11-09 load fix; before it the served PGC dropped its trained LoRA, see the PGC section), SAFE, Co-Spy. AIDE/SPAI needed a preprocessing/any-resolution port to make official weights load correctly (now done).
2. **PGC reproduces from the published recipe with early stopping (resolved 2026-06-19).** The released ProGAN checkpoint is only ~400 steps in (the authors, on our issue); the earlier epoch-level retrains overshot the transfer peak, which is why the hard generators (deepfake/san/seeingdark) stayed at ~71–81 AUROC. With the official code and evaluator, the official `train.py` evaluated every 100 steps reaches hard-3 AUROC 94.56 at step 900 (the released checkpoint: 94.35). In the SynthScope class, the official recipe at step 500 gives FS-8 97.41 / UFD 98.99 mAP, just below the released checkpoint in the same class (97.83 / 99.35, corrected 2026-11-09; the earlier comparison against 96.41 / 99.17 used the dropped-LoRA load and is withdrawn). The official `train.py` retrain stopped at step 900 scores FS-13 97.34 mAP / 80.68 acc and hard-3 AUROC 94.56 through the same class (released checkpoint: FS-13 96.25 / 83.76, hard-3 94.34). (`repro/investigate/pgc_gap_investigation.md`, `models/image/pgc/README.md`)
3. **Gaps closed by correcting training data/preprocessing:** SPAI (**native-res any-res path now wired into serving: official weights in the SynthScope class → GenImage 94.22 / Synthbuster 96.75 mAP, beating the vendored-official 91.33; the fixed-448 retrain 88.80 is superseded by the official-recipe retrain, 94.67 GenImage / 95.76 Synthbuster, within ~1 point of the official weights**), Co-Spy (right GenImage SD-v1.4 source, →96.70 beats official), **SAFE (crop-only eval + full 40-epoch retrain → reproduces the paper: ForenSynths-8 97.79 / UFD-8 98.44 / GenImage-8 99.51 / DiTFake 99.96)**, **AIDE (native-res 5-view DCT now wired into both serving AND training: served official 96.83, and a from-scratch retrain reproduces the mAP at 96.42 ≈ official 97.12 (acc 77.9 vs 85.9), up from the resize-256 92.47 — the DCT must run at native res, ~12 mAP on the same weights)**. The original SAFE "gap" was an eval-transform bug — a `Resize` destroyed the high-frequency signal SAFE keys on; crop-only preprocessing makes both the official weights and our retrain reproduce.
4. **VIB-Net** (weights on a restricted Drive): reproduced by training in SynthScope with the reconstructed recipe at native resolution (94.08 AP / 85.22 acc vs 95.53 / 85.83 over the paper's 15 generators); the resize-256 training pipeline alone costs 11 AP points (82.98 with it); the earlier model also had an fp16 CLIP and an added BatchNorm.
5. **Faithful reporting** — every number traces to a logged run; SynthScope_train shows the single best from-scratch checkpoint (older/weaker runs discarded, not stacked).

## Where everything lives
- Per-model write-ups: `models/image/*/README.md`, `models/video/*/README.md`
- Aggregated table: `repro/AGGREGATED_BENCHMARK.md`; repo `README.md` → "Full-benchmark reproduction"
- Website data: `interface/backend/registry.py` → `REPRO` dict
- Raw results: `repro/bench_results/*.json`, `repro/MASTER_RESULTS.jsonl`
- Gap-closing checkpoints: `checkpoints_retrain/`; promoted `co_spy.pth` (`cospy_full/model_1`) and `safe.pth` (`safe_full_v2/model_38`); `spai.pth` and `pgc.pth` are now the official weights
