# 📈 What's Trending — AI-generated content detection

*High-quality recent papers for image & video AIGC detection. High-quality = (1) a solid public GitHub repository, and (2) strong results versus previous methods. Auto-refreshed daily. Last updated: 2026-10-18.*

Entries marked _in SynthScope_ are integrated in this repo; the rest are tracked, with the reason an integration is pending noted in their _Results_ line. See [docs/ADD_DETECTOR.md](docs/ADD_DETECTOR.md) for how a paper is integrated and reproduced.

## Image detectors
- **[CuRe: Learning Continuous Source Responses for Generalizable AI-Generated Image Detection](https://github.com/manic-cui/CuRe)** (manic-cui · arXiv · 2026) — [code](https://github.com/manic-cui/CuRe) · _in SynthScope: `cure`_  
  Reframes backbone adaptation as regressing real-generated MIXING RATIOS instead of binary classification (finer supervision), with a compact source-response subspace that suppresses nuisance variation/shortcut cues. Frozen Perception-Encoder (PE-Core L/14-336) backbone + LoRA + subspace + linear head; the 25 MB adapter ships in the repo and infer.py scores a single image.  
  _Results: 89.7% average balanced accuracy over ten public benchmarks, +5.2 pts over the second-best method, robust to common degradations (paper). [Integrated in SynthScope as `cure` (merged to main 2026-10-03): SynthScope's class is identical to the official model (395/395 state-dict tensors, logit diff 4.8e-6, 0 verdict flips). SynthScope-protocol results with the official adapter: UFD-8 93.9 / ForenSynths-13 82.0 / GenImage-8 85.2 balanced acc (SD-v1.4, its training set, 99.7). AIGIBench REPRODUCED: on the paper's own 20 full-image-synthesis subsets (full test data) the port scores 92.85 balanced acc / 97.48 AUROC vs the paper's 92.8 / 97.5, every subset within 0.10 pt (per-subset mean abs diff 0.03; the paper excludes the 5 face-swap subsets, which is why all 25 gives 85.95). Still unconfirmed: UnivFD (aggregation unspecified), AIGCDetect (data only on cn023, not run), and 7 of the paper's 10 benchmarks (not in SynthScope).]_
- **[PGC: Peak-Guided Calibration for AI-Generated Image Detection](https://github.com/xiaoyu6868/PGC)** (Zhou et al. · 2026) — [code](https://github.com/xiaoyu6868/PGC) · _in SynthScope: `pgc`_  
  LoRA-tuned DINOv2 RGB stream + a YCbCr-quantization-residual CNN, fused by a peak-guided (log-sum-exp) calibration. Strong cross-generator transfer.  
  _Results: ForenSynths 96.25 mAP / GenImage 100 mAP (official weights)._
- **[PROBE: Probing Generative Space for Generalizable AIGI Detection](https://github.com/Amamiya-C/PROBE-AIGI-Detection)** (arXiv · 2026) — [code](https://github.com/Amamiya-C/PROBE-AIGI-Detection) · _in SynthScope: `probe`_  
  Boosts cross-generator generalization by augmenting training with images sampled from probed generative-space variations; the detector is a fine-tuned DINOv2 (sliding-patch) classifier. Released weights on ModelScope.  
  _Results: PROBE-DINOv2 93.9% avg balanced-acc over 7 benchmarks (paper). SynthScope reproduces the official weights: GenImage 98.54 / Synthbuster 99.27 / UFD 95.95 mAP (mAUROC 96-99). [Integration on branch trending/probe, under review.]_
- **[DiCoME: Divide-and-Conquer Multi-View Evidential Learning for Deepfake Detection](https://github.com/kxl0825/DiCoME)** (arXiv · 2026) — [code](https://github.com/kxl0825/DiCoME)  
  Disentangles semantic vs structural-artifact views and fuses them with evidential learning to counter the 'semantic masking effect'. Released checkpoint (HF).  
  _Results: SOTA cross-dataset deepfake generalization (DeepfakeBench protocol). [face-deepfake domain; SynthScope lacks those datasets locally — integration pending data.]_
- **[Images as Tables: In-Context AIGI Detection with TabPFN](https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images)** (arXiv · 2026) — [code](https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images)  
  Frozen DINOv3 CLS feature -> PCA(500) -> TabPFN in-context tabular classification; strong few-shot/low-data generalization with no task-specific training.  
  _Results: Generalizable detection in the low-data regime (few labeled examples per new generator). [integration pending: in-context TabPFN paradigm — a clean SynthScope forward->logit needs the released fitted TabPFN+PCA+labeled-support-set checkpoint baked in (GDrive), plus an external version-sensitive tabpfn dependency; a different paradigm from the trained-detector template, and the headline is a low-data/few-shot niche (multi-multi acc 0.8765 / AUROC 0.9501 @625 samples) rather than full-data SOTA. Verified repo 2026-06-20: weights on GDrive, frozen DINOv3->PCA(500)->TabPFN.]_
- **[Co-Spy: Fusing Semantic and Artifact Cues for Synthetic Image Detection](https://github.com/Megum1/Co-Spy)** (Cheng et al. · CVPR · 2025) — [code](https://github.com/Megum1/Co-Spy) · _in SynthScope: `co_spy`_  
  Adaptive fusion of a SigLIP semantic stream and an SD-VAE artifact-residual stream.  
  _Results: Co-Spy-Bench ~96 mAP; SynthScope retrain reaches 96.6 (beats released weights on that bench)._
- **[SAFE: High-Frequency Artifact Detector with Targeted Augmentation](https://github.com/Ouxiang-Li/SAFE)** (Ouxiang Li et al. · KDD · 2025) — [code](https://github.com/Ouxiang-Li/SAFE) · _in SynthScope: `safe`_  
  Lightweight DWT high-frequency artifact detector with a carefully designed augmentation recipe.  
  _Results: SynthScope-trained reproduction (served): ForenSynths-8 97.79 / UFD-8 98.44 / GenImage-8 99.51 / DiTFake 99.96 mAP — beats the official checkpoint on GenImage._
- **[AIDE: ConvNeXt Semantic + Low-Level DCT/SRM Cues](https://arxiv.org/abs/2406.19435)** (Yan et al. · ICLR · 2025) — [arXiv](https://arxiv.org/abs/2406.19435) · _in SynthScope: `aide`_  
  Fuses ConvNeXt semantic features with low-level DCT/SRM frequency cues; native-resolution preprocessing.  
  _Results: GenImage 97.12 mAP / AIGCDetect 97.94 mAP (official weights)._
- **[SPAI: Spectral AI-Image Detection (any-resolution)](https://github.com/mever-team/spai)** (Karageorgiou et al. · CVPR · 2025) — [code](https://github.com/mever-team/spai) · _in SynthScope: `spai`_  
  Spectral Reconstruction Similarity + Spectral Context Attention; detects at native/arbitrary resolution.  
  _Results: SynthScope serves the official weights at native resolution (the faithful any-res path): GenImage 94.22 / Synthbuster 96.75 mAP — reproduces & beats the vendored-official 91.33._
- **[FatFormer: Forgery-Aware Adaptive Transformer](https://github.com/Michel-liu/FatFormer)** (Liu et al. · CVPR · 2024) — [code](https://github.com/Michel-liu/FatFormer) · _in SynthScope: `fat_former`_  
  Frozen CLIP + forgery-aware adapters + learnable text-prompt context.  
  _Results: ForenSynths-8 99.7 mAP (official weights)._
- **[NPR: Neighboring-Pixel Relationships](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)** (Tan et al. · CVPR · 2024) — [code](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) · _in SynthScope: `npr`_  
  Small ResNet on neighboring-pixel residuals that expose generator up-sampling artifacts.  
  _Results: GANGen 96.49 mAP / DiffusionForensics 99.4 mAP (official weights)._
- **[UnivFD / CLIP: Towards Universal Fake Image Detectors](https://github.com/WisconsinAIVision/UniversalFakeDetect)** (Ojha et al. · CVPR · 2023) — [code](https://github.com/WisconsinAIVision/UniversalFakeDetect) · _in SynthScope: `clip`_  
  Linear probe on frozen CLIP ViT-L/14 features — a simple, strong cross-generator baseline.  
  _Results: UniversalFakeDetect 95.66 mAP; SynthScope-trained 96.44._
- **[MoE-JEPA: Unifying Semantic Priors and High-Frequency Traces (V-JEPA for synthetic image forensics)](https://github.com/ALCOR-Lab-DIAG/MoE-JEPA)** (ALCOR-Lab-DIAG · ECCV'26 Workshop (AI for Multimedia Forensics) · 2026) — [code](https://github.com/ALCOR-Lab-DIAG/MoE-JEPA)  
  Frozen V-JEPA 2 ViT-L backbone with a Residual Mixture-of-Experts injected into the last blocks, a Bayar-conv noise branch and Gated-Attention MIL pooling; uses JEPA 'world understanding' as a semantic prior.  
  _Results: 95.54% acc / 94.21 F1 on SID-Set (300K images), outperforming much larger models (paper). [integration pending (assessed 2026-10-02): the authors' trained weights are NOT linked (only the frozen V-JEPA 2 backbone is downloaded separately) and there is no single-image inference entrypoint; can only be retrained. Re-checked 2026-10-05: unchanged.]_
- **[RAID: Towards Robust AI-Generated Image Detection with Bit-Reversed Images](https://github.com/renxi-seu/RAID)** (renxi-seu · arXiv · 2026) — [code](https://github.com/renxi-seu/RAID)  
  Detects via bit-planes: a bit-reversed image, gradient-based patch selection and a small convolutional classifier, with a theoretical analysis; ~100x faster than counterparts.  
  _Results: Claims to outperform existing methods on 40+ benchmarks across cross-generator, cross-dataset and zero-shot settings (paper; README gives no numbers). [integration pending (assessed 2026-10-02): the SD-v1.4-trained weights are only shared via Baidu Cloud (not fetchable programmatically), and the README leaves the architecture and patch-score aggregation unspecified; revisit if a direct weight link appears. Re-checked 2026-10-05: unchanged.]_
- **[Fleet: Few Shots Lead Effective AI-generated Image Detection](https://github.com/ICTMCG/Fleet)** (ICTMCG · arXiv · 2026) — [code](https://github.com/ICTMCG/Fleet)  
  Dynamic few-shot adaptation: 'constrained routing correction' redirects novel AI samples away from non-AI-dominated routes in decoupled subspaces (DINOv3 backbone), so the detector adapts to rapidly-evolving/commercial generators from a handful of labeled examples. Introduces the Treasure benchmark (64 generators, 360k images, 20 closed-source commercial engines).  
  _Results: On a modern generator (Doubao Seedream 4.0), 20.4% -> 73.1% with only 10-shot adaptation, where static SOTA collapses (paper). Repo now live with code (re-listed 2026-10-03; it had been pruned 2026-07-04 while an empty placeholder). [integration pending (assessed 2026-10-03): not a single-checkpoint detector — the Hugging Face repo wangjiaan/Fleet holds NINE per-target-generator few-shot adapted models (ADM, FLUX.2, GPT4O, HunyuanImage-3.0, NextStep, Qwen-Image, StarGAN, wan2.5, ...; 1.3 GB each), the generic stage-1 base is only on Baidu Pan, inference requires few-shot support data per target generator, and the Treasure benchmark is gated. No clean forward->logit that represents the method; revisit if a generic checkpoint appears on HF.]_

## Video detectors
- **[D3: Disentangled Detail Dynamics (training-free)](https://github.com/Zig-HS/D3)** (Zheng et al. · ICCV · 2025) — [code](https://github.com/Zig-HS/D3) · _in SynthScope: `d3`_  
  Training-free: 2nd-order std of frozen per-frame features as the score.  
  _Results: GenVideo-Val ~98 mAP (paper). SynthScope faithful protocol (X-CLIP-B/16 + 10% edge-crop): AUROC 95.95 / acc 91.7 (the paper's mAP isn't comparable on SynthScope's 10000:56 real:fake pool — AUROC is the robust metric)._
- **[ReStraV: Representation-Trajectory Geometry](https://github.com/ChristianInterno/ReStraV)** (Internò et al. · 2025) — [code](https://github.com/ChristianInterno/ReStraV) · _in SynthScope: `restrav`_  
  Frozen DINOv2 trajectory-geometry features + a standardized MLP head.  
  _Results: GenVidBench Main ~93; SynthScope 92.7 mAP._
- **[STALL: Spatio-Temporal whitened Log-Likelihoods (training-free)](https://github.com/OmerBenHayun/STALL)** (Ben Hayun et al. · CVPR · 2026) — [code](https://github.com/OmerBenHayun/STALL) · _in SynthScope: `stall`_  
  Training-free spatial + temporal whitened-Gaussian log-likelihoods vs a real-video calibration.  
  _Results: GenVideo-Val ~0.80 AUC (paper); SynthScope AUROC 83.5._
- **[ATSS: Anomalous Temporal Self-Similarity for AI-Generated Video Detection](https://github.com/hwang-cs-ime/ATSS)** (Wang et al. (hwang-cs-ime) · arXiv · 2026) — [code](https://github.com/hwang-cs-ime/ATSS)  
  Multimodal detector that exploits anomalous temporal self-similarity fingerprints across frames of AI-generated video. Inference code + a pretrained checkpoint (best.pt) are released; the data sampling/extraction pipeline is still being cleaned for release.  
  _Results: Claims SOTA in AP/AUC/ACC over GenVideo, EvalCrafter, VideoPhy, and VidProM vs prior methods (e.g. DeMamba). Repo live 2026-06-21 with code+weights; per-generator numbers not yet in the README. [integration pending: the released checkpoint's forward() consumes PRECOMPUTED multimodal temporal-self-similarity tensors (sim_img/sim_txt/sim_cross, BLIP2-based), not raw video — and the feature-extraction/data-sampling front-end that produces them is 'still being cleaned for release (coming soon)'. SynthScope feeds raw video clips, so there is no faithful way to run ATSS until that pipeline is published; will revisit when released. Re-checked 2026-10-05: unchanged.]_
- **[G2VD: Generalizable AI-Generated Video Detection via Counterfactual Intervention and Causal Disentanglement](https://github.com/DMOSCAR-98/G2VD)** (Du et al. (dumeng98) · arXiv · 2026) — [code](https://github.com/DMOSCAR-98/G2VD)  
  Counterfactual-intervention + causal-disentanglement detector: a training-time CFIPipeline synthesizes controlled counterfactual samples (VAEs + frequency/pixel alignment), and a causal-disentanglement classifier with two domain-anchored branches and an HSIC independence constraint isolates the generalizable artifact signal from domain-specific content.  
  _Results: GenVidBench cross-domain: >90% accuracy / AUC ~0.95 using only 10% of the training data (paper). Repo live 2026-07-07 (CLIP/XCLIP/DeMamba backbones — the same family as SynthScope's D3). [integration pending (re-checked 2026-10-05): repo moved from dumeng98/G2VD (now 404) to DMOSCAR-98/G2VD; checkpoints are still NOT released ('not stored in this repository', MODEL_ZOO.md lists no links) and the CFIPipeline VAE-pool is part of the staged pipeline. Revisit when checkpoints appear.]_
- **[RIFT: Mind the Rift - Cross-Scale Coupling Mismatch for AI-Generated Video Detection](https://github.com/Litsay/RIFT)** (Litsay · arXiv (submitted ACM MM'26) · 2026) — [code](https://github.com/Litsay/RIFT)  
  Treats a cross-scale coupling mismatch between macro temporal dynamics (differential geometry / persistent homology on frozen DINOv2 features) and micro residuals (steganalytic filtering + RAFT motion) as the forensic signal, scored by a coupling-divergence module.  
  _Results: Own benchmark: 99.22% acc / 99.33 F1 / 99.96 AUC, 97.87% leave-one-generator-out (paper; no head-to-head against named prior detectors in the abstract, so not directly comparable). [integration pending (assessed 2026-10-02): trained weights are 'released upon acceptance' (not available), and it needs DINOv2 + RAFT features over ~120K training videos. Re-checked 2026-10-05: unchanged.]_

## Benchmarks & curated lists
- [GenImage: million-scale real-vs-fake benchmark](https://github.com/GenImage-Dataset/GenImage) — ImageNet-1k reals + ~equal synthetic from diverse generators.
- [AIGIBench: robustness/generalization benchmark (23 fake subsets)](https://github.com/HorizonTEL/AIGIBench) — Multi-source generalization, degradation robustness, augmentation sensitivity, preprocessing impact.
- [AI-GenBench: temporal (GAN→diffusion) generalization benchmark](https://github.com/MI-BioLab/AI-GenBench) — Incremental, historically-ordered training to test generalization to new generators (Verimedia @ IJCNN 2025).
- [GenVidBench: 6M AI-generated video detection benchmark](https://arxiv.org/abs/2501.11340) — Large-scale benchmark for AI-generated video detection.
- [DeepfakeBench: comprehensive deepfake-detection benchmark](https://github.com/SCLBD/DeepfakeBench) — Unified benchmark across many deepfake datasets/detectors.
- [Awesome-AIGC-Image-Video-Detection (curated list)](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection) — Continuously-updated collection of image + video AIGC-detection papers and code.
- [Awesome-Comprehensive-Deepfake-Detection (curated list)](https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection) — Broad survey-style index of deepfake-detection research and repos.

---
*Auto-generated from `interface/backend/trending.json` by `repro/generate_trending_readme.py` — edit the JSON, not this file.*
