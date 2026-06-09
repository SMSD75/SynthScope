## Full-benchmark reproduction (all models × all reported benchmarks)

Every benchmark each paper reports, reproduced and verified. Per-model detail in each `models/*/<name>/README.md`; raw results in `repro/bench_results/`, specs in `repro/benchmark_specs/`, matrix in `repro/BENCHMARK_COVERAGE_MATRIX.md`.

| Model | Benchmark | Paper | Ours | Status |
|---|---|---|---|---|
| CLIP / UnivFD | ForenSynths GAN/low-level test (11 of the pape | — | map_fs8=97.22 / acc_fs8=88.28 / map_fs13=92.99 / official_fs8=96.95 / official_fs13=93.14 / ckpt=clip.pth (AGID-trained linear head, ckpt_kind=agid) / tags=clip_agid_forensynths / clip_official_forensynths | reproduced |
| CLIP / UnivFD | UniversalFakeDetect diffusion/autoregressive t | map=95.0 | map=96.44 / acc=87.23 / official=95.66 / ckpt=clip.pth (AGID-trained head) / tags=clip_agid_ufd / clip_official_ufd | reproduced |
| CNNDetection | ForenSynths test set (CVPR paper Table 2 = 11  | map=90.8 | map_fs8=94.42 / acc_fs8=76.36 / map_fs13=91.05 / official_fs8=92.67 / official_fs13=91.15 / ckpt=cnn.pth (AGID-trained ResNet-50, ckpt_kind=agid) / tags=cnn_agid_forensynths / cnn_official_forensynths | reproduced |
| FatFormer | GANs dataset — 8 GANs (4-class ProGAN supervis | map=99.7 / acc=98.4 | map=99.7 / acc=98.39 / ckpt=fat_former.pth (= fatformer_official.pth, OFFICIAL 4-class CLIP:ViT-L/14 weights remapped into the AGID class; ckpt_kind=official_fatformer) / tag=fatformer_official_forensynths | reproduced |
| FatFormer | GANs dataset — 8 GANs (2-class ProGAN supervis | map=99.2 / acc=95.8 | — | missing |
| FatFormer | Diffusion-model (DMs) dataset — 10 subsets (4- | map=98.8 / acc=95.0 | map_ufd8=98.6 / acc_ufd8=93.72 / ckpt=fat_former.pth (official weights in AGID class) / tag=fatformer_official_ufd | below |
| NPR | Table 1 — ForenSynths / Cross-GAN-Sources (8 G | map=96.1 / acc=92.5 | 94.22 mAP (correct ckpt) | reproduced |
| NPR | Table 2 — GANGen-Detection / Self-Synthesis 9  | map=96.6 / acc=93.2 | map=96.49 / acc=94.81 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_freqnet / eval=run with bench_eval.py --native-res to drop the up-scaling Resize, matching official no_resize=True | reproduced |
| NPR | Table 3 — DiffusionForensics / DIRE | map=99.8 / acc=95.3 | map=99.4 / acc=95.03 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_diffusionforensics | reproduced |
| NPR | Table 4 — UniversalFakeDetect / Ojha | acc=95.2 | map=96.69 / acc=93.94 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_ufd | reproduced |
| NPR | Table 5 — Diffusion1kStep / Self-Synthesis Dif | acc=80.1 | map=83.47 / acc=79.13 / ckpt=npr.pth (model_epoch_last_3090.pth) / tag=npr_official_diffusion1kstep | reproduced |
| AIDE | AIGCDetectBenchmark (B1, Setting-I, ProGAN-tra | acc=92.77 | map=97.94 / acc=91.89 | reproduced |
| AIDE | GenImage (B2, Setting-I, SDv1.4-trained) | acc=86.88 | map=97.12 / acc=85.91 | reproduced |
| AIDE | Chameleon (B3, Setting-II, in-the-wild) | acc_ProGAN_trained=58.37 / acc_SDv14_trained=62.6 / acc_AllGenImage_trained=65.77 | — | missing |
| SAFE | ForenSynths (Table 1, intra-architecture GANs) | map=98.8 / acc=96.2 | map=98.78 / acc=96.2 | below |
| SAFE | Self-Synthesis / GANGen (Table 2, intra-archit | map=99.8 / acc=99.0 | map=99.85 / acc=98.99 | reproduced |
| SAFE | Ojha / UniversalFakeDetect (Table 3, cross-arc | map=99.1 / acc=95.7 | map=99.1 / acc=95.71 | reproduced |
| SAFE | GenImage (Table 4, cross-architecture DM+GAN)  | map=99.3 / acc=95.6 | map=95.7 / acc=89.84 | below |
| SAFE | DiTFake (Table 6, cross-architecture DiT) | map=99.9 / acc=99.4 | 97.78 mAP | reproduced |
| Co-Spy | Co-Spy-Bench | map=96.02 / acc=87.06 | map=96.7 / acc=90.28 / ckpt=checkpoints_retrain/cospy_full/model_1.pth (AGID retrain, full SD-v1.4 data, flickr-reals bug fixed) / tag=cospy_full_cospybench / official_wts_in_agid=86.65 | reproduced |
| Co-Spy | AIGCDetectionBenchmark (AIGCDetect) | map=95.21 / acc=90.58 | map=94.59 / acc=89.25 / ckpt=official ProGAN weights loaded into AGID class (/hddstore/ssalehi/AGID/repro/ckpts/progan) / tag=cospy_official_aigcdetect / agid_trained_ckpt=84.91 | reproduced |
| Co-Spy | GenImage | — | map_official_wts=92.41 / map_agid_retrain=90.59 / acc=75.91 / ckpt=checkpoints_retrain/cospy_full/model_2.pth / tag=cospy_full_g8 | below |
| Co-Spy | ForenSynths (CNNDetection test) | — | map_official_wts=87.97 / map_agid_retrain=96.17 / ckpt=AGID co_spy.pth / tag=cospy_official_forensynths / co_spy_retrain_forensynths | reproduced |
| Co-Spy | In-the-wild (Co-Spy-Bench in-the-wild) | map=84.65 / acc=77.41 | — | missing |
| SPAI | SPAI main eval set (13 generators x 5 real sou | auroc=91.0 | — | missing |
| SPAI | GenImage (cross-dataset, AGID benchmark) | — | 90.74 mAP (any-res) | reproduced |
| SPAI | Synthbuster | — | ~96.6 mAUROC (any-res) | reproduced |
| VIB-Net | Diffusion-source (SDv1.4-trained), AP -- HEADL | map=95.53 | map=83.94 / acc=77.57 | below |
| VIB-Net | Diffusion-source (SDv1.4-trained), ACC | acc=85.83 | map=83.94 / acc=77.57 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN-trained), AP | map=93.14 | map=61.71 / acc=57.91 | below |
| VIB-Net | GAN-source (ForenSynths/ProGAN-trained), ACC | acc=82.5 | — | missing |
| PGC | GenImage | map=100.0 / acc=98.3 | map_official_wts=100.0 / acc_official_wts=99.49 / map_agid_retrain=97.62 / acc_agid_retrain=78.6 / ckpt_official=repro/official_ckpts/pgc/PGC_train_sdv1_4_ckpt.pth; ckpt_agid: checkpoints/_train/pgc_genimage/model_0.pth (registry-default) / tags=pgc_official_genimage8 / pgc_retrain_genimage8 (pgc_long_g8) | reproduced |
| PGC | UniversalFakeDetect (UFD) | map=98.0 | map_official_wts=99.34 / acc_official_wts=94.16 / map_agid_retrain=99.01 / ckpt_official=repro/official_ckpts/pgc/PGC_train_progan_ckpt.pth / tags=pgc_official_ufd / pgc_retrain_ufd | reproduced |
| PGC | ForenSynths (CNNDetection test) | — | map_official_wts=96.25 / acc_official_wts=83.73 / map_agid_retrain=89.03 / ckpt_official=repro/official_ckpts/pgc/PGC_train_progan_ckpt.pth / tags=pgc_official_forensynths / pgc_retrain_forensynths | reproduced |
| PGC | AIGIBench (AIGI) | map=95.0 / acc=86.6 | map_official_wts=95.36 / acc_official_wts=88.33 / map_agid_retrain=86.88 / ckpt_official=repro/official_ckpts/pgc/PGC_train_progan_sdv1_4_ckpt.pth / tags=pgc_official_aigibench25 / pgc_retrain_aigibench | reproduced |
| PGC | CommGen15 (commercial in-the-wild) | acc=97.9 | map_official_wts=99.26 / acc_official_wts=97.39 / map_agid_retrain=98.61 / acc_agid_retrain=66.66 / ckpt_official=repro/official_ckpts/pgc/PGC_train_sdv1_4_ckpt.pth / tags=pgc_official_commgen15 / pgc_retrain_commgen15 | reproduced |
| D3 | GenVideo (DeMamba GenVideo-Val, Table 1) | mAP=98.46 | XCLIP 95.1 AUROC / 99.5 AP_real_full (CLIP 94.4) | reproduced |
| D3 | EvalCrafter (Table 2, 14 generators, fake-only | mAP=98.87 | mAP=91.19 / mAUROC=95.41 / acc=87.67 | below |
| D3 | VideoPhy (Table 3, 10 generators) | mAP=99.16 | — | missing |
| D3 | VidProM (Table 4, 6 generators) | mAP=88.46 | — | missing |
| ReStraV | A) VidProM vs image-based detectors (Table 2) | acc=97.06 / mAP=98.81 | — | missing |
| ReStraV | B) VidProM vs VideoSwin — seen/unseen/future ( | seen={'acc': 97.05, 'mAP': 98.78} / unseen={'acc': 89.45, 'mAP': 97.32} / future_Sora={'acc': 80.05, 'mAP': 92.85} | — | missing |
| ReStraV | B) GenVidBench vs SoTA video detectors — Main  | Main_M_avg=93.01 / Plants_P_avg=96.96 | acc=86.87 / mAP=92.7 / mAUROC=95.9 | reproduced |
| ReStraV | C) DeMamba one-to-many generalization (Table 5 | train_Pika={'R': 0.735, 'F1': 0.827, 'AP': 0.797} / train_SEINE={'R': 0.82, 'F1': 0.898, 'AP': 0.854} / train_OpenSora={'R': 0.771, 'F1': 0.797, 'AP': 0.717} | — | missing |
| ReStraV | D) Veo3 zero-shot generalization (Sec 7-D) | acc=83.2 / F1=85.1 / AUROC=86.9 | — | missing |
| ReStraV | E) Physics-IQ 2AFC vs Gemini 1.5 Pro (Sec 7-E, | Pika=97.5 / Runway=94.9 / Sora=99.0 | — | missing |
| STALL | GenVideo (DeMamba GenVideo-Val, 10 generators  | avg_AUC=0.8 / avg_AP=0.8 / per_gen_AUC_range=0.72-0.85 / per_gen_AP_range=0.75-0.84 | mAP=82.38 / mAUROC=83.51 / acc=56.88 | reproduced |
| STALL | VideoFeedback (TIGER-Lab, 11 generators + DiDe | avg_AUC=0.83 / avg_AP=0.85 / per_gen_AUC_range=0.78-0.93 / per_gen_AP_range=0.80-0.94 | mAP=83.07 / mAUROC=85.1 / acc=62.12 | reproduced |
| STALL | ComGenVid (paper's new benchmark — Sora + VEO3 | avg_AUC=0.852 / avg_AP=0.858 / Sora={'AUC': 0.84, 'AP': 0.846} / VEO3={'AUC': 0.865, 'AP': 0.87} | mAP=82.59 / mAUROC=84.0 / acc=57.34 | reproduced |
