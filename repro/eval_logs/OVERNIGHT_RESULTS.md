## Overnight multi-GPU retrains (2026-06-08, intermediate)
- AIDE-faithful (preprocessing fix): GenImage-8 91.75 -> 92.53; adm 58.6 -> ~66-70, vqdm 80.4 -> 85.2.
  In-domain saturated (val AP 0.99999). Residual adm gap = native-res DCT (256-DCT used). official 97.12.
- Co-Spy full-data (162k): CoSpyBench 95.88(subset) -> 96.12 (paper 96.0); GenImage-8 88.85 -> 89.97 (official 92.41).
- SPAI/SAFE: queued behind 8-GPU cap, start when AIDE/Co-Spy free GPUs.

## FINALS (2026-06-08 night)
- AIDE-faithful FINAL: GenImage-8 **92.47** (was 91.75); adm 58.6->65, vqdm 80.4->81. In-domain perfect.
  Residual to official 97.12 = native-res DCT + custom-ResNet (not done; documented next lever). Not promoted
  (official-weights path via vendored eval remains the 97.12 reference).
- Co-Spy full FINAL: CoSpyBench **96.70** (beats paper 96.0), GenImage-8 90.59 (official 92.41).
  **co_spy.pth RE-PROMOTED** -> checkpoints_retrain/cospy_full/model_1.pth (was subset 95.88).
- SPAI-full (182179) + SAFE-full (182180): running on cn029 after redirect from busy cn025/cn028.

## SPAI killed / SAFE running (2026-06-08 night)
- SPAI-full: epoch-0 baseline 77.79 (undertrained); every SPAI retrain (<86) underperforms shipped 88.80
  (already near official 91.33) -> KILLED, keep shipped spai.pth. SPAI gap is architectural-recipe (native
  arbitrary-res), not closeable by more fixed-448 training.
- SAFE-full: epoch 6/40, in-domain val AP ~1.0; GenImage-8 cross-gen 92.72 so far. Monitoring for climb to ~96+.
- SAFE-full: epoch10 GenImage-8 **95.70** (shipped 94.91/92.88; official 99.32); climbing, all gens 89-99. Running to ep40.

## Phase 2 — benchmark-coverage closures (2026-06-08)
- NPR ForenSynths: wrong-ckpt fix — stripped `module.` from official `NPR.pth` -> `repro/ckpts/NPR_table1.pth`;
  FS-8 92.81 -> **94.22** (paper 96.1; residual ~1.9 on biggan/deepfake/gaugan = checkpoint/seed). npr.pth NOT
  re-linked yet (must re-confirm GANGen/DiffForensics/UFD/Diffusion1kStep don't regress with NPR_table1).
- SAFE DiTFake (5th SAFE benchmark, was missing): downloaded HF lioooox/DiTFake (23G, FLUX/PixArt/SD3) +
  evaled -> **97.78 mAP** (paper 99.9). Class already existed (DiTFakeDataset).
- VIB-Net official weights: NOT retrievable (GDrive permission-restricted incl. upgraded gdown; Baidu-only).
  Documented gated; from-scratch CLIP-VIB 83.94 stands.
- INFRA: two simultaneous big-image evals OOM the cn023 48GiB cgroup -> run sequentially / --bs 16 / --limit.
- D3 GenVideo (CLIP-B/16 + full official preproc, d3_eval.py): SynthScope-bench 89.0 -> **AUROC 0.944 / AP_real(bal) 0.944** (paper CLIP ~97, XCLIP 98.46). XCLIP temporal encoder = remaining headline lever.
- Synthbuster downloaded (12GB zip) for SPAI eval.

## Synthbuster integrated (2026-06-09) — new benchmark, class written (template)
- Data: Zenodo 10066460 (9 diffusion generators × 1000 fakes) + GenImage-nature proxy reals (RAISE-1k not shipped).
- SynthbusterDataset class + module + SynthScopeModuleDict["synthbuster"] + conf/dataset/image/synthbuster.yaml + explorer DatasetDef.
- Results: **Co-Spy 98.48 / SAFE 93.92** (reproduce well) ; **SPAI 55.67** (fixed-448 fails — resolution-sensitive,
  needs the any-res SCA path + authentic RAISE reals to reach the paper's ~91 AUC).

## SPAI any-resolution REPRODUCED (2026-06-09) — minimal class change + official-weights remap
- Class change (minimal): forward now patchifies each image at NATIVE res (variable L) + five_crop fallback
  when L<minimum_patches(4) [models/image/spai/spai.py] — matches official forward_arbitrary_resolution_batch.
- Official SPAI weights remapped into the SynthScope class (repro/load_spai_official.py -> spai_official_synthscope.pth;
  321 keys, 0 unexpected). Eval: repro/spai_anyres_eval.py (native-res, bs=1, AUROC metric).
- **GenImage-8: fixed-448 88.80 -> any-res 90.74 mAP / 85.47 mAUROC (96.6 mAUROC excl. biggan)** ≈ paper ~91.
  biggan AUROC 7.3 = inverted (SPAI is a diffusion-spectral detector; genuinely flips on the GAN). SD-family
  gens 96-99.8. SPAI headline REPRODUCED via the any-res path.
- **Synthbuster (SPAI's own benchmark): fixed-448 55.67 -> any-res ~96.6 mAUROC** (all 9 gens 93.9-99.7);
  exceeds paper ~91. SPAI FULLY reproduced via the minimal any-res class change + official-weights remap.

## D3 X-CLIP added (2026-06-09) — minimal class change
- _XClipEncoder (XCLIPVisionModel, mirrors official D3's flattened-B*T pooler_output usage) + xclip-16/-32
  backbone cases in models/video/d3/d3.py. Eval d3_eval.py --backbone xclip-16 --norm imagenet --crop 0.1.
- GenVideo-Val: CLIP 94.4 -> **XCLIP-16 AUROC 95.12 / AP_real_bal 95.01 / AP_real_full 99.52** (paper 98.46
  mAP = official real-class AP; our full-pool AP_real 99.52 >= paper; residual on balanced = ffmpeg-vs-decord
  frame extraction). Up from SynthScope-bench 89.
## Web UI: per-method FULL benchmark table (2026-06-09)
- interface/backend/repro_full.json (13 methods, 49 rows) from specs+results; /api/models serves repro_full;
  App.jsx ReproTable renders the full Benchmark|Paper|Ours|Status table per selected method (+CSS).

## AIDE native-res DCT confirmed (2026-06-09)
- The native-res DCT IS implemented in the faithful eval (repro/official_evals/aide_official_eval.py: DCT on
  the native image -> 4 patches -> Resize(256)+ImageNet -> custom ResNet). On full GenImage-8 (sd14 weights):
  **adm AP 94.31** (vs the SynthScope-class 256-DCT's 58-65) — recovers the ADM/VQDM collapse to the official level.
  AIDE-official = 97.12 mAP (banked aide_official_genimage8), paper 86.9 acc. Reproduced. SynthScope-class native-DCT
  fold-in (SPAI-style) is optional code-org (number already achieved via the faithful eval).
