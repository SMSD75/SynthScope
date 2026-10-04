# SOTA AI-Generated Media Detection — papers worth looking into

Curated for the SynthScope project (AI-generated **image** and **video** detection). ✓ = already
implemented in this repo. Venues are best-effort; verify before citing.

## Already in SynthScope
- **Image:** CNNDetection ✓, UnivFD/CLIP ✓, NPR ✓, FatFormer ✓, AIDE ✓, CO-SPY ✓, SAFE ✓,
  SPAI ✓, VIB-Net ✓ (+ DINOv2 / CLIP / ResNet linear-probe baselines).
- **Video:** D3 ✓, ReStraV ✓.

## AI-generated IMAGE detection
| Method | Paper | Venue | Idea |
|---|---|---|---|
| CNNDetection ✓ | CNN-generated images are surprisingly easy to spot… for now (Wang et al.) | CVPR 2020 | ResNet-50 on ProGAN + blur/JPEG aug; strong cross-GAN transfer. |
| UnivFD ✓ | Towards Universal Fake Image Detectors that Generalize Across Generative Models (Ojha et al.) | CVPR 2023 | Frozen CLIP features + linear/NN probe. |
| DIRE | DIRE for Diffusion-Generated Image Detection (Wang et al.) | ICCV 2023 | Reconstruction error under a diffusion model. |
| LGrad | Learning on Gradients (Tan et al.) | CVPR 2023 | Gradient-based generalized artifact representation. |
| NPR ✓ | Rethinking the Up-Sampling Operations… (Tan et al.) | CVPR 2024 | Neighboring-pixel residuals from up-sampling. |
| FatFormer ✓ | Forgery-Aware Adaptive Transformer (Liu et al.) | CVPR 2024 | Frozen CLIP + forgery-aware adapters + language-guided alignment. |
| FreqNet | Frequency-Aware Deepfake Detection (Tan et al.) | AAAI 2024 | High-frequency representation learning. |
| DRCT | Diffusion Reconstruction Contrastive Training (Chen et al.) | ICML 2024 | Reconstruct-then-contrast for diffusion images. |
| RINE | Leveraging Representations from Intermediate Encoder Blocks (Koutlis & Papadopoulos) | ECCV 2024 | Uses intermediate CLIP features, not just CLS. |
| Cozzolino et al. | Raising the Bar of AI-generated Image Detection with CLIP | CVPRW 2024 | Strong, simple CLIP-feature baseline + protocol. |
| FID / "Breaking Semantic Artifacts" | Breaking Semantic Artifacts for Generalized AIGI Detection (Zheng et al.) | NeurIPS 2024 | Local-feature method; the strongest image baseline in the D3 paper. |
| AIDE ✓ | AIDE: AI-generated Image Detection with Hybrid Features (Yan et al.) | ICLR 2025 | ConvNeXt semantics + DCT/SRM low-level patches. |
| CO-SPY ✓ | Combining Semantic and Pixel Features (Cheng et al.) | CVPR 2025 | SigLIP semantics + VAE-residual artifacts, adaptive fusion. |
| SAFE ✓ | Improving SID Towards Generalization: An Image Transformation Perspective (Li et al.) | KDD 2025 | Lightweight DWT high-freq detector + augmentation recipe. |
| VIB-Net ✓ | Universal AIGI Detection by Variational Information Bottleneck (Zhang et al.) | CVPR 2025 | VIB on CLIP features to suppress shortcut cues. |

## AI-generated VIDEO detection
| Method | Paper | Venue | Idea |
|---|---|---|---|
| DeMamba | AI-Generated Video Detection on Million-Scale GenVideo (Chen et al.) | 2024 (arXiv 2405.19707) | GenVideo benchmark + Mamba temporal module; strong trained baseline. |
| DeCoF | Generated Video Detection via Frame Consistency (Ma et al.) | 2024 (arXiv 2402.02085) | Temporal frame-consistency features. |
| D3 ✓ | Training-Free AIGV Detection Using Second-Order Features (Zheng et al.) | ICCV 2025 | Training-free; std of 2nd-order inter-frame feature differences. |
| ReStraV ✓ | AI-Generated Video Detection via Perceptual Straightening (Internò et al.) | 2025 (arXiv 2507.00583) | DINOv2 trajectory curvature/distance + small classifier. |
| AIGVDet | Spatial-Temporal Anomaly Learning (Bai et al.) | PRCV 2024 | Optical-flow + spatial anomaly streams. |
| DuB3D | Distinguish Any Fake Videos (Ji et al.) | 2024 (arXiv 2405.15343) | Large-scale data + motion features. |
| "Turns out I'm not real" | Towards Robust Detection of AI-Generated Videos (Liu et al.) | 2024 (arXiv 2406.09601) | DIRE residuals + CNN/LSTM, robustness focus. |

Face-deepfake-video methods often cited as baselines (less relevant to *generative* video, but useful): FTCN (ICCV 2021), TALL (ICCV 2023), MINTIME (TIFS 2024), STIL (ACM MM 2021).

## Benchmarks / datasets
- **Image:** GenImage, ForenSynths (CNNDetection), UniversalFakeDetect, DiffusionForensics, Co-Spy-Bench, AIGIBench, DiTFake.
- **Video:** GenVideo (DeMamba), GenVidBench, VidProM, EvalCrafter, VideoPhy / VideoPhy-2, Physics-IQ.

## Suggested starting points
1. **Cross-modal trend:** frozen foundation features (CLIP/DINOv2/SigLIP) + light heads now dominate both image and video — UnivFD → FatFormer → AIDE/CO-SPY (image), ReStraV/D3 (video).
2. **Training-free direction:** D3 (video) and the local-feature image methods suggest detection signals that need no generated training data — worth exploring for generalization.
3. **For video specifically:** DeMamba (benchmark + trained SOTA) vs. D3/ReStraV (cheap, training-free/light) is the key comparison; GenVideo is the common benchmark.
