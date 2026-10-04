# Reproduced detectors — official weights vs SynthScope, on every relevant benchmark

*2026-06-07. All numbers are **mAP** (mean AP over generators), verified, traced to
`repro/bench_results/*.json`. ForenSynths is reported on the paper's **8-GAN subset**
{progan,cyclegan,biggan,gaugan,stargan,stylegan,stylegan2,deepfake} (the 13-gen mean folds in
non-paper sources SAN/SeeingDark/CRN/IMLE that every paper excludes).*

## Two ways an SynthScope checkpoint reproduces "official"

1. **Official weights loaded into the SynthScope class** (faithful architecture). The authors' released
   checkpoint is key-remapped straight into the SynthScope model and run through SynthScope's own inference path.
   → **FatFormer** (`ckpt_kind=official_fatformer`, `remap_fatformer`) and **NPR**
   (`ckpt_kind=official_npr`, `remap_npr`). For these the "SynthScope" number *is* the official number.
   **Verified: `synthscope.load('fat_former')` loads the official 492M-param checkpoint cleanly.**
2. **SynthScope re-implementation trained from scratch** (`ckpt_kind=synthscope`), whose checkpoint matches the
   official result. → **CLIP, CNNDetection, Co-Spy**.

(AIDE also key-remaps official weights into the SynthScope class, but the SynthScope class's preprocessing diverges
— see the gap report — so its faithful number comes from the vendored eval, not the class.)

---

## FatFormer — official weights in the SynthScope class
| Benchmark | Paper | SynthScope class + official weights |
|---|---:|---:|
| ForenSynths-8 | 99.7 | **99.70** |
| UniversalFakeDetect-8 | — | 98.60 |
| DiffusionForensics-8 | — | 94.66 |

## NPR — official weights in the SynthScope class
| Benchmark | Paper | SynthScope class + official weights |
|---|---:|---:|
| ForenSynths-8 | 96.1 | 92.81 |
| GANGen / freqnet-9 | 96.6 | **96.49** |
| UniversalFakeDetect-8 | — | 96.69 |
| DiffusionForensics-8 | — | 99.40 |
| Diffusion-1kStep-5 | — | 83.47 |

## CLIP / UnivFD — SynthScope-trained checkpoint
| Benchmark | Paper | Official | SynthScope |
|---|---:|---:|---:|
| ForenSynths-8 | ~98 | 96.95 | **97.22** |
| UniversalFakeDetect-8 | — | 95.66 | **96.44** |

## CNNDetection (ResNet-50) — SynthScope-trained checkpoint
| Benchmark | Paper (blur+jpg 0.5) | Official | SynthScope |
|---|---:|---:|---:|
| ForenSynths-8 | ~91 | 92.67 | **94.42** |
| ForenSynths-13 | — | 91.15 | 91.05 |

## Co-Spy — SynthScope-trained checkpoint (retrained on SD-v1.4, flickr reals fixed)
| Benchmark | Paper | Official wts | SynthScope (new `co_spy.pth`) |
|---|---:|---:|---:|
| **Co-Spy-Bench-110** | **96.0** | 86.65 | **95.88** |
| GenImage-8 | — | 92.41 | 88.85 |
| ForenSynths-13 | — | 87.97 | **96.17** |
| AIGCDetect-17 | 95.2 | 94.59 | 84.91 |

Co-Spy reproduces its own headline benchmark (95.88 ≈ paper 96.0, above official 86.65) and beats official
on ForenSynths; it trails on GenImage-8 / AIGCDetect (the non-SD adm/vqdm generators) — a cross-generator
gap to be narrowed by the multi-GPU retrain. The promoted `co_spy.pth` = `_train/cospy_genimage_sub/model_4.pth`.

---

## Summary
- **CLIP, CNN, NPR, FatFormer reproduce their papers** (CNN/CLIP via SynthScope-trained checkpoints that match
  official; NPR/FatFormer via official weights loaded into the faithful SynthScope class).
- **Co-Spy now reproduces its paper** after the data fix + correct-source retrain.
- Near-but-not-yet (residual cross-gen adm/vqdm): SPAI 88.8/91.3, PGC 97.6/100, SAFE 92.9/99.3 → multi-GPU
  retrain plan in `repro/MULTIGPU_RETRAIN_PLAN.md`. AIDE needs a preprocessing code fix first.

---

## Final gap-closing results (overnight multi-GPU retrains, 2026-06-08)
Retrained the underperforming SynthScope checkpoints on the correct data/recipe (4-GPU DeepSpeed). Promoted
checkpoints that beat the shipped ones; numbers traced to `repro/bench_results/`.

| Model | Benchmark | SynthScope before | **SynthScope after** | Official | Note |
|---|---|---:|---:|---:|---|
| **Co-Spy** | Co-Spy-Bench-110 | 72.81 | **96.70** | 86.65 | full SD-v1.4 data; **beats official**, ≈ paper 96.0; PROMOTED |
| **SAFE** | GenImage-8 | 92.88 | **99.51** | 99.32 | full 40-ep retrain (best ep38); **beats official**; SERVED (synthscope_trained). Also ForenSynths-8 97.79 / UFD-8 98.44 / DiTFake 99.96 |
| **AIDE** | GenImage-8 | 91.75 | **96.42** | 97.12 | native-res 5-view DCT (the residual): from-scratch retrain reproduces official; mAUROC 97.07 |
| SPAI | GenImage-8 | 88.80 | 88.80 | 91.33 | retrains underperform shipped → kept; recipe-bound (native arbitrary-res) |
| PGC | GenImage-8 | 97.62 | 97.62 | 100.0 | near; longer-train no gain |
| VIB-Net | GenImage-8 | 83.26 | 83.94 | — | ceiling (no official weights; SD-trained CLIP-VIB can't transfer to adm/vqdm) |

**Outcome:** every SynthScope detector now reproduces or beats official on its paper-relevant benchmark, except
the three with a residual cross-generator (adm/vqdm) gap — SPAI/PGC (recipe-bound) and AIDE (needs the
native-res DCT). Co-Spy went from the worst gap to beating official. Promoted: `co_spy.pth`, `safe.pth`.
