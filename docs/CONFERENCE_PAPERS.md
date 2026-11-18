# 📚 Awesome AI-Generated Image and Video Detection

*Top-tier conference and journal papers on detecting AI-generated images and videos, verified and grouped by venue.*

> **Last updated: 2026-11-18** · 264 papers · updated monthly · part of [SynthScope](../README.md)

A curated list of peer-reviewed papers on detecting **AI-generated images and videos**, grouped by venue and year (in the style of [Awesome Industrial Anomaly Detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)). It covers generalizable detectors, localization, explainable and MLLM-based detection, and benchmarks/datasets, from 2020 on. Face-swap-only deepfake detection, generation, watermarking-only and attribution-only papers are out of scope.

**How entries are verified.** Every paper links to its **official** acceptance record: CVF Open Access (CVPR, ICCV, WACV), ecva.net (ECCV), proceedings.neurips.cc or the accepted OpenReview decision (NeurIPS, ICLR), PMLR (ICML), ojs.aaai.org (AAAI), ijcai.org (IJCAI), the ACM Digital Library (ACM MM, KDD), IEEE Xplore (TPAMI) or Springer (IJCV). Papers accepted but not yet in the proceedings link to the venue's official accepted-papers page. A code link is listed only if the GitHub repository exists. Venue tags from other lists or arXiv comments alone are not enough. [`tools/check_conference_list.py`](../tools/check_conference_list.py) checks the format, the link domains, duplicates and the counts below.

**Legend.** `Image` / `Video`: modality · `Benchmark`: a dataset or benchmark · `Localization`: localizes generated or edited regions or segments · `MLLM`: multimodal-LLM-based or explainable detection · ✅: the detector is implemented in SynthScope (see the [detector tables](../README.md#-detectors)) · first author shown, "et al." when there are more.

Missing a paper? Open a pull request that adds it in this format, with its official link, and run `python tools/check_conference_list.py`.

## Overview

| Venue | 2026 | 2025 | 2024 | 2023 | 2022 | 2021 | 2020 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [CVPR](#cvpr) | 28 | 16 | 5 | 3 | 1 | 1 | 2 | 56 |
| [ICCV](#iccv) | · | 13 | · | 1 | · | · | · | 14 |
| [ECCV](#eccv) | 10 | · | 3 | · | 3 | · | 1 | 17 |
| [NeurIPS](#neurips) | 23 | 17 | 3 | 2 | · | · | 1 | 46 |
| [ICLR](#iclr) | 11 | 5 | · | · | · | · | · | 16 |
| [ICML](#icml) | 24 | 6 | 1 | · | · | · | 2 | 33 |
| [AAAI](#aaai) | 16 | 6 | 1 | · | 1 | · | · | 24 |
| [IJCAI](#ijcai) | 2 | 2 | · | 1 | · | 1 | 1 | 7 |
| [ACM MM](#acm-mm) | 17 | 13 | 2 | · | · | · | · | 32 |
| [KDD](#kdd) | 1 | 1 | · | · | · | · | · | 2 |
| [WACV](#wacv) | 4 | 2 | 3 | 2 | 1 | · | · | 12 |
| [TPAMI](#journals) | 3 | · | · | · | · | · | · | 3 |
| [IJCV](#journals) | 1 | 1 | · | · | · | · | · | 2 |
| **Total** | **140** | **82** | **18** | **9** | **6** | **2** | **7** | **264** |

## CVPR

### CVPR 2026

- **A Debiased Reconstruction-based Framework for Training-Free Detection of AI-Generated Images** · Sungik Choi et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Choi_A_Debiased_Reconstruction-based_Framework_for_Training-Free_Detection_of_AI-Generated_Images_CVPR_2026_paper.html) · `Image`
- **A Difference-in-Difference Approach to Detecting AI-Generated Images** · Xinyi Qi et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Qi_A_Difference-in-Difference_Approach_to_Detecting_AI-Generated_Images_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2602.23732) · `Image`
- **ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos** · Peijun Bao et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Bao_ActivityForensics_A_Comprehensive_Benchmark_for_Localizing_Manipulated_Activity_in_Videos_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2604.03819) · `Video` `Benchmark` `Localization`
- **AVFakeBench: A Comprehensive Audio-Video Forgery Detection Benchmark for AV-LMMs** · Shuhan Xia et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_AVFakeBench_A_Comprehensive_Audio-Video_Forgery_Detection_Benchmark_for_AV-LMMs_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2511.21251) · `Video` `Benchmark` `MLLM`
- **CoCoVideo: The High-Quality Commercial-Model-Based Contrastive Benchmark for AI-Generated Video Detection** · Huidong Feng et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_CoCoVideo_The_High-Quality_Commercial-Model-Based_Contrastive_Benchmark_for_AI-Generated_Video_Detection_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2606.00101) · [[code]](https://github.com/DonoToT/CoCoVideo) · `Video` `Benchmark`
- **Cross-modal Representation Learning for Diffusion-generated Image Detection** · Tao Gong et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026_paper.html) · `Image`
- **Detect Any AI-Counterfeited Text Image** · Chenfan Qu et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Qu_Detect_Any_AI-Counterfeited_Text_Image_CVPR_2026_paper.html) · `Image`
- **Detecting AI-Generated Forgeries via Iterative Manifold Deviation Amplification** · Jiangling Zhang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Detecting_AI-Generated_Forgeries_via_Iterative_Manifold_Deviation_Amplification_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2602.18842) · `Image` `Localization`
- **Detecting Compressed AI-Generated Images via Phase Spectrum Robustness** · Kai Li et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Detecting_Compressed_AI-Generated_Images_via_Phase_Spectrum_Robustness_CVPR_2026_paper.html) · `Image`
- **DFD-HR: Generalizable Deepfake Detection via Hierarchical Routing Learning** · Jiamu Sun et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_DFD-HR_Generalizable_Deepfake_Detection_via_Hierarchical_Routing_Learning_CVPR_2026_paper.html) · `Image`
- **Diversity over Uniformity: Rethinking Representation in Generated Image Detection** · Qinghui He et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/He_Diversity_over_Uniformity_Rethinking_Representation_in_Generated_Image_Detection_CVPR_2026_paper.html) · `Image`
- **Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection** · Jianwei Fei et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_Enabling_Supervised_Learning_of_Generative_Signatures_for_Generalized_AI-Generated_Images_CVPR_2026_paper.html) · `Image`
- **Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection** · Yawen Yang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Layer_Consistency_Matters_Elegant_Latent_Transition_Discrepancy_for_Generalizable_Synthetic_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2603.10598) · [[code]](https://github.com/yywencs/LTD) · `Image`
- **Locate-Then-Examine: Grounded Region Reasoning Improves Detection of AI-Generated Images** · Yikun Ji et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_Locate-Then-Examine_Grounded_Region_Reasoning_Improves_Detection_of_AI-Generated_Images_CVPR_2026_paper.html) · `Image` `MLLM`
- **Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection** · Tianxiao Li et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Omni-Fake_Benchmarking_Unified_Multimodal_Social_Media_Deepfake_Detection_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2605.01638) · `Image` `Video` `Benchmark` `Localization`
- **Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision** · Kartik Kuckreja et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Kuckreja_Pixels_Dont_Lie_But_Your_Detector_Might_Bootstrapping_MLLM-as-a-Judge_for_CVPR_2026_paper.html) · `Image` `Benchmark` `MLLM`
- **PPM-CLIP: Probabilistic Prompt Modeling for Generalizable AI-Generated Image Detection** · Xinyuan Wang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_PPM-CLIP_Probabilistic_Prompt_Modeling_for_Generalizable_AI-Generated_Image_Detection_CVPR_2026_paper.html) · `Image`
- **ReAlign: Generalizable Image Forgery Detection via Reasoning-Aligned Representation** · Qing Huang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_ReAlign_Generalizable_Image_Forgery_Detection_via_Reasoning-Aligned_Representation_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2605.16080) · `Image` `MLLM`
- **SAIDO: Generalizable Detection of AI-Generated Images via Scene-Aware and Importance-Guided Dynamic Optimization in Continual Learning** · Yongkang Hu et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_SAIDO_Generalizable_Detection_of_AI-Generated_Images_via_Scene-Aware_and_Importance-Guided_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2512.00539) · [[code]](https://github.com/edu-yinzhaoxia/SAIDO-CVPR-2026) · `Image`
- **Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes** · Ziheng Qin et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_Scaling_Up_AI-Generated_Image_Detection_with_Generator-Aware_Prototypes_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2512.12982) · [[code]](https://github.com/UltraCapture/GAPL) · `Image`
- **SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images** · Aayush Dhakal et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Dhakal_SimLBR_Learning_to_Detect_Fake_Images_by_Learning_to_Detect_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2602.20412) · `Image`
- **Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning** · Yifei Li et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Skyra_AI-Generated_Video_Detection_via_Grounded_Artifact_Reasoning_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2512.15693) · [[code]](https://github.com/JoeLeelyf/Skyra) · `Video` `MLLM`
- **Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning** · Yiheng Li et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Towards_Generalizable_AI-Generated_Image_Detection_via_Image-Adaptive_Prompt_Learning_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2508.01603) · [[code]](https://github.com/liyih/IAPL) · `Image`
- **Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods** · Omer Ben Hayun et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Hayun_Training-free_Detection_of_Generated_Videos_via_Spatial-Temporal_Likelihoods_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2603.15026) · [[code]](https://github.com/OmerBenHayun/STALL) · `Video` · ✅
- **UniGenDet: A Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection** · Yanran Zhang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_UniGenDet_A_Unified_Generative-Discriminative_Framework_for_Co-Evolutionary_Image_Generation_and_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2604.21904) · [[code]](https://github.com/Zhangyr2022/UniGenDet) · `Image`
- **VMD-FACT: A New Video Dataset and MLLM-based method for Detecting Realistic AI-Generated Video Misinformation** · Yongkang Zhang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_VMD-FACT_A_New_Video_Dataset_and_MLLM-based_method_for_Detecting_CVPR_2026_paper.html) · `Video` `Benchmark` `MLLM`
- **Your One-Stop Solution for AI-Generated Video Detection** · Long Ma et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Your_One-Stop_Solution_for_AI-Generated_Video_Detection_CVPR_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2601.11035) · [[code]](https://github.com/LongMa-2025/AIGVDBench) · `Video` `Benchmark`
- **Zero-shot Detection of AI-Generated Image via RAW-RGB Alignment** · Haiwei Wu et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Zero-shot_Detection_of_AI-Generated_Image_via_RAW-RGB_Alignment_CVPR_2026_paper.html) · `Image`

### CVPR 2025

- **A Bias-Free Training Paradigm for More General AI-generated Image Detection** · Fabrizio Guillaro et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2412.17671) · [[code]](https://github.com/grip-unina/B-Free) · `Image` · ✅
- **AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark** · Li Lin et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_AI-Face_A_Million-Scale_Demographically_Annotated_AI-Generated_Face_Dataset_and_Fairness_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2406.00783) · [[code]](https://github.com/Purdue-M2/AI-Face-FairnessBench) · `Image` `Benchmark`
- **Any-Resolution AI-Generated Image Detection by Spectral Learning** · Dimitrios Karageorgiou et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2411.19417) · [[code]](https://github.com/mever-team/spai) · `Image` · ✅
- **Beyond Generation: A Diffusion-based Low-level Feature Extractor for Detecting AI-generated Images** · Nan Zhong et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Zhong_Beyond_Generation_A_Diffusion-based_Low-level_Feature_Extractor_for_Detecting_AI-generated_CVPR_2025_paper.html) · `Image`
- **CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI** · Siyuan Cheng et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_CO-SPY_Combining_Semantic_and_Pixel_Features_to_Detect_Synthetic_Images_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2503.18286) · [[code]](https://github.com/Megum1/CO-SPY) · `Image` · ✅
- **Community Forensics: Using Thousands of Generators to Train Fake Image Detectors** · Jeongsoo Park et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2411.04125) · `Image` `Benchmark`
- **D^3: Scaling Up Deepfake Detection by Learning from Discrepancy** · Yongqi Yang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2404.04584) · [[code]](https://github.com/BigAandSmallq/D3) · `Image`
- **FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error** · Beilin Chu et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Chu_FIRE_Robust_Detection_of_Diffusion-Generated_Images_via_Frequency-Guided_Reconstruction_Error_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2412.07140) · `Image`
- **Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images** · Tai D. Nguyen et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Nguyen_Forensic_Self-Descriptions_Are_All_You_Need_for_Zero-Shot_Detection_Open-Set_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2503.21003) · `Image`
- **Forensics-Bench: A Comprehensive Forgery Detection Benchmark Suite for Large Vision Language Models** · Jin Wang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Forensics-Bench_A_Comprehensive_Forgery_Detection_Benchmark_Suite_for_Large_Vision_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2503.15024) · `Image` `Benchmark` `MLLM`
- **OpenSDI: Spotting Diffusion-Generated Images in the Open World** · Yabin Wang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_OpenSDI_Spotting_Diffusion-Generated_Images_in_the_Open_World_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2503.19653) · [[code]](https://github.com/iamwangyabin/OpenSDI) · `Image` `Benchmark` `Localization`
- **Secret Lies in Color: Enhancing AI-Generated Images Detection with Color Distribution Analysis** · Zexi Jia et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Secret_Lies_in_Color_Enhancing_AI-Generated_Images_Detection_with_Color_CVPR_2025_paper.html) · `Image`
- **SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model** · Zhenglin Huang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_SIDA_Social_Media_Image_Deepfake_Detection_Localization_and_Explanation_with_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2412.04292) · `Image` `Localization` `MLLM`
- **Towards a Universal Synthetic Video Detector: From Face or Background Manipulations to Fully AI-Generated Content** · Rohit Kundu et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Kundu_Towards_a_Universal_Synthetic_Video_Detector_From_Face_or_Background_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2412.12278) · `Video`
- **Towards Universal AI-Generated Image Detection by Variational Information Bottleneck Network** · Haifeng Zhang et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Towards_Universal_AI-Generated_Image_Detection_by_Variational_Information_Bottleneck_Network_CVPR_2025_paper.html) · `Image` · ✅
- **Where's the Liability in the Generative Era? Recovery-based Black-Box Detection of AI-Generated Content** · Haoyue Bai et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Bai_Wheres_the_Liability_in_the_Generative_Era_Recovery-based_Black-Box_Detection_CVPR_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2505.01008) · `Image`

### CVPR 2024

- **AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error** · Jonas Ricker et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2401.17879) · [[code]](https://github.com/jonasricker/aeroblade) · `Image`
- **FakeInversion: Learning to Detect Images from Unseen Text-to-Image Models by Inverting Stable Diffusion** · George Cazenavette et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Cazenavette_FakeInversion_Learning_to_Detect_Images_from_Unseen_Text-to-Image_Models_by_CVPR_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2406.08603) · `Image`
- **Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection** · Huan Liu et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2312.16649) · [[code]](https://github.com/Michel-liu/FatFormer) · `Image` · ✅
- **LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection** · Yunpeng Luo et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_LaRE2_Latent_Reconstruction_Error_Based_Method_for_Diffusion-Generated_Image_Detection_CVPR_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2403.17465) · [[code]](https://github.com/luo3300612/LaRE) · `Image`
- **Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection** · Chuangchuang Tan et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2312.10461) · [[code]](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) · `Image` · ✅

### CVPR 2023

- **Hierarchical Fine-Grained Image Forgery Detection and Localization** · Xiao Guo et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Hierarchical_Fine-Grained_Image_Forgery_Detection_and_Localization_CVPR_2023_paper.html) · [[arXiv]](https://arxiv.org/abs/2303.17111) · [[code]](https://github.com/CHELSEA234/HiFi_IFDL) · `Image` `Localization`
- **Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection** · Chuangchuang Tan et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.html) · `Image`
- **Towards Universal Fake Image Detectors That Generalize Across Generative Models** · Utkarsh Ojha et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html) · [[arXiv]](https://arxiv.org/abs/2302.10174) · [[code]](https://github.com/WisconsinAIVision/UniversalFakeDetect) · `Image` · ✅

### CVPR 2022

- **Think Twice Before Detecting GAN-Generated Fake Images From Their Spectral Domain Imprints** · Chengdong Dong et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Think_Twice_Before_Detecting_GAN-Generated_Fake_Images_From_Their_Spectral_CVPR_2022_paper.html) · `Image`

### CVPR 2021

- **A Closer Look at Fourier Spectrum Discrepancies for CNN-Generated Images Detection** · Keshigeyan Chandrasegaran et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Chandrasegaran_A_Closer_Look_at_Fourier_Spectrum_Discrepancies_for_CNN-Generated_Images_CVPR_2021_paper.html) · [[arXiv]](https://arxiv.org/abs/2103.17195) · `Image`

### CVPR 2020

- **CNN-Generated Images Are Surprisingly Easy to Spot... for Now** · Sheng-Yu Wang et al. · [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html) · [[arXiv]](https://arxiv.org/abs/1912.11035) · [[code]](https://github.com/PeterWang512/CNNDetection) · `Image` · ✅
- **Global Texture Enhancement for Fake Face Detection in the Wild** · Zhengzhe Liu et al. · [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Global_Texture_Enhancement_for_Fake_Face_Detection_in_the_Wild_CVPR_2020_paper.html) · [[arXiv]](https://arxiv.org/abs/2002.00133) · `Image`

## ICCV

### ICCV 2025

- **AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models** · Ziyin Zhou et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2507.02664) · [[code]](https://github.com/wyczzy/AIGI-Holmes) · `Image` `MLLM`
- **Bi-Level Optimization for Self-Supervised AI-Generated Face Detection** · Mian Zou et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Bi-Level_Optimization_for_Self-Supervised_AI-Generated_Face_Detection_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2507.22824) · `Image`
- **Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios** · Chunxiao Li et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2509.09172) · `Image` `Benchmark`
- **D3: Training-Free AI-Generated Video Detection Using Second-Order Features** · Chende Zheng et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_D3_Training-Free_AI-Generated_Video_Detection_Using_Second-Order_Features_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2508.00701) · [[code]](https://github.com/Zig-HS/D3) · `Video` · ✅
- **D3QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image Detection** · Yanran Zhang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_D3QE_Learning_Discrete_Distribution_Discrepancy-aware_Quantization_Error_for_Autoregressive-Generated_Image_ICCV_2025_paper.html) · [[code]](https://github.com/Zhangyr2022/D3QE) · `Image` · ✅
- **Diffusion Epistemic Uncertainty with Asymmetric Learning for Diffusion-Generated Image Detection** · Yingsong Huang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Diffusion_Epistemic_Uncertainty_with_Asymmetric_Learning_for_Diffusion-Generated_Image_Detection_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2601.14625) · `Image`
- **Forensic-MoE: Exploring Comprehensive Synthetic Image Detection Traces with Mixture of Experts** · Mingqi Fang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Fang_Forensic-MoE_Exploring_Comprehensive_Synthetic_Image_Detection_Traces_with_Mixture_of_ICCV_2025_paper.html) · `Image`
- **ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection** · Yingjian Chen et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_ForgeLens_Data-Efficient_Forgery_Focus_for_Generalizable_Forgery_Image_Detection_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2408.13697) · [[code]](https://github.com/Yingjian-Chen/ForgeLens) · `Image`
- **HumanSAM: Classifying Human-centric Forgery Videos in Human Spatial, Appearance, and Motion Anomaly** · Chang Liu et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_HumanSAM_Classifying_Human-centric_Forgery_Videos_in_Human_Spatial_Appearance_and_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2507.19924) · `Video`
- **LEGION: Learning to Ground and Explain for Synthetic Image Detection** · Hengrui Kang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Kang_LEGION_Learning_to_Ground_and_Explain_for_Synthetic_Image_Detection_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2503.15264) · [[code]](https://github.com/opendatalab/LEGION) · `Image` `Localization` `MLLM`
- **LOTA: Bit-Planes Guided AI-Generated Image Detection** · Hongsong Wang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LOTA_Bit-Planes_Guided_AI-Generated_Image_Detection_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2510.14230) · [[code]](https://github.com/hongsong-wang/LOTA) · `Image`
- **Semantic Discrepancy-aware Detector for Image Forgery Identification** · Ziye Wang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Semantic_Discrepancy-aware_Detector_for_Image_Forgery_Identification_ICCV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2508.12341) · [[code]](https://github.com/wzy1111111/SSD) · `Image`
- **Spatial-Temporal Forgery Trace based Forgery Image Identification** · Yilin Wang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Spatial-Temporal_Forgery_Trace_based_Forgery_Image_Identification_ICCV_2025_paper.html) · `Image`

### ICCV 2023

- **DIRE for Diffusion-Generated Image Detection** · Zhendong Wang et al. · [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DIRE_for_Diffusion-Generated_Image_Detection_ICCV_2023_paper.html) · [[arXiv]](https://arxiv.org/abs/2303.09295) · [[code]](https://github.com/ZhendongWang6/DIRE) · `Image`

## ECCV

### ECCV 2026

- **Beyond Artifacts: Real-Centric Envelope Modeling for Reliable AI-Generated Image Detection** · Ruiqi Liu et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/3942) · [[arXiv]](https://arxiv.org/abs/2512.20937) · [[code]](https://github.com/handsome-rich/REM) · `Image`
- **FAIR: Feature-Augmented Implicit Regularization for AI-generated Fake Image Detection** · Md Redwanul Haque et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/5812) · [[arXiv]](https://arxiv.org/abs/2607.22087) · `Image`
- **GLARE: Towards Generalizable Detection of Latent Diffusion Images with Global-Local Reconstruction Error** · Jiangtao Yan et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/4623) · `Image`
- **LoRC: Detecting AI-Generated Images via Low-Rank Collapse in the Semantic-Residual Space** · Haozhen Yan et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/5654) · `Image`
- **Prior-Conditioned Gaussian Discriminants for Generalizable AI-generated Image Detection** · Shashank Kotyan et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/5771) · [[arXiv]](https://arxiv.org/abs/2608.18523) · `Image`
- **REVEAL: Reasoning-Enhanced Forensic Evidence Analysis for Explainable AI-Generated Image Detection** · Huangsen Cao et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/5044) · [[arXiv]](https://arxiv.org/abs/2511.23158) · `Image` `MLLM`
- **Revisiting Deepfake Detection: BCNet for Robust Generalization Beyond Semantic Dependence** · Jian Yang et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/5201) · `Image`
- **SafeGuard: A Multi-Agent Perception-Reasoning Framework for Social-Risk AI-Generated Video Detection** · Wenlin Wu et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/4333) · [[arXiv]](https://arxiv.org/abs/2607.03069) · [[code]](https://github.com/williamw99/SafeGuard) · `Video` `MLLM`
- **SPLIT: Training-Free AI-Generated and Partially Edited Video Detection via Spatial Patch‑Level Incoherence and Temporal Roughness** · Jongyeop Hyun et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/5677) · [[arXiv]](https://arxiv.org/abs/2607.02886) · [[code]](https://github.com/mldljyh/SPLIT) · `Video` · ✅
- **UC-VLM: Consistency-Driven Learning for AI-Generated Image Detection with Vision-Language Large Models** · Lei Tan et al. · [[paper]](https://eccv.ecva.net/virtual/2026/poster/5204) · [[arXiv]](https://arxiv.org/abs/2608.15238) · `Image` `MLLM`

### ECCV 2024

- **Contrasting Deepfakes Diffusion via Contrastive Learning and Global-Local Similarities** · Lorenzo Baraldi et al. · [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8009_ECCV_2024_paper.php) · [[arXiv]](https://arxiv.org/abs/2407.20337) · [[code]](https://github.com/aimagelab/CoDE) · `Image`
- **Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection** · Christos Koutlis et al. · [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9160_ECCV_2024_paper.php) · [[arXiv]](https://arxiv.org/abs/2402.19091) · [[code]](https://github.com/mever-team/rine) · `Image`
- **Zero-Shot Detection of AI-Generated Images** · Davide Cozzolino et al. · [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2665_ECCV_2024_paper.php) · [[arXiv]](https://arxiv.org/abs/2409.15875) · `Image`

### ECCV 2022

- **Detecting Generated Images by Real Images** · Bo Liu et al. · [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5017_ECCV_2022_paper.php) · [[arXiv]](https://arxiv.org/abs/2311.00962) · `Image`
- **Discovering Transferable Forensic Features for CNN-Generated Images Detection** · Keshigeyan Chandrasegaran et al. · [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3181_ECCV_2022_paper.php) · [[arXiv]](https://arxiv.org/abs/2208.11342) · `Image`
- **FingerprintNet: Synthesized Fingerprints for Generated Image Detection** · Yonghyun Jeong et al. · [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4708_ECCV_2022_paper.php) · `Image`

### ECCV 2020

- **What makes fake images detectable? Understanding properties that generalize** · Lucy Chai et al. · [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5308_ECCV_2020_paper.php) · [[arXiv]](https://arxiv.org/abs/2008.10588) · `Image`

## NeurIPS

### NeurIPS 2026

- **Beyond Real or Fake: A Dual-Channel Authenticity and Reasoning Protocol for Photographic Assessment** · Xiaoxiao Li et al. · [[paper]](https://openreview.net/forum?id=CsqD0OgleJ) · [[arXiv]](https://arxiv.org/abs/2611.10908) · `Image`
- **BIAS-ID: A Framework for Analyzing Transformation Biases in AI-Generated Image Detectors** · Jonas Ricker et al. · [[paper]](https://neurips.cc/virtual/2026/poster/139597) · [[arXiv]](https://arxiv.org/abs/2605.31153) · [[code]](https://github.com/jonasricker/biasid) · `Image` `Benchmark`
- **BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation** · Haiquan Wen et al. · [[paper]](https://openreview.net/forum?id=coouK1prvK) · [[arXiv]](https://arxiv.org/abs/2505.12620) · [[code]](https://github.com/l8cv/BusterX) · `Video` `MLLM`
- **Can Pixels Alone Reveal Image Origin? Minimax Limits and Learnable Interfaces for Passive Provenance** · Kai Yao · [[paper]](https://openreview.net/forum?id=DQ0b1sTnB8) · [[arXiv]](https://arxiv.org/abs/2609.30997) · [[code]](https://github.com/kaikaiyao/pixels-alone-provenance) · `Image`
- **DEDCA: Test-Time Adaptation for Generalized AI-Generated Image Detection** · Hao Tan et al. · [[paper]](https://openreview.net/forum?id=WP9ROPHHjk) · [[arXiv]](https://arxiv.org/abs/2611.16307) · `Image`
- **FakeParts: a New Family of AI-Generated Video Forgeries** · Ziyi LIU et al. · [[paper]](https://neurips.cc/virtual/2026/poster/138979) · `Video` `Benchmark` `Localization`
- **Less is More: Fewer Tokens and Blocks Make AIGI Detection Faster and More Generalizable** · Haipeng Fang et al. · [[paper]](https://openreview.net/forum?id=l7AUTUgcKZ) · `Image`
- **Less Supervision, Better Generalization: Weakly Supervised Fake Region Localization in Diffusion-Edited Images** · Junhee Lee et al. · [[paper]](https://openreview.net/forum?id=H5AFWh3aMx) · [[arXiv]](https://arxiv.org/abs/2609.36882) · `Image` `Localization`
- **ManipShield: A Unified Framework for Image Manipulation Detection, Localization and Explanation** · Zitong Xu et al. · [[paper]](https://openreview.net/forum?id=umrmVI1xvK) · [[arXiv]](https://arxiv.org/abs/2511.14259) · [[code]](https://github.com/IntMeGroup/ManipShield) · `Image` `Benchmark` `Localization` `MLLM`
- **MIRROR: Manifold Ideal Reference ReconstructOR for AI-Generated Image Detection** · Ruiqi Liu et al. · [[paper]](https://openreview.net/forum?id=rNJxI4kA0N) · `Image` · ✅
- **MixForensics: Blend Before You Encode for Generalizable AI-Generated Video Detection** · Ruiqi Liu et al. · [[paper]](https://openreview.net/forum?id=BzzvL3RHzi) · `Video`
- **Post-hoc Selective Classification for Reliable Synthetic Image Detection** · Kaixiang Zheng et al. · [[paper]](https://openreview.net/forum?id=WYncw2up3x) · [[arXiv]](https://arxiv.org/abs/2605.08574) · `Image`
- **Rethinking the Readout: Unlocking Video Backbones for AI-Generated Video Detection** · Manni Cui et al. · [[paper]](https://openreview.net/forum?id=rXN1Faijz6) · [[arXiv]](https://arxiv.org/abs/2607.15321) · [[code]](https://github.com/manic-cui/Rethinking-the-readout) · `Video`
- **SAGE: Semantic-Anchor-Guided Embedding for Generalized AI-Generated Image Detection** · Seongho Kim et al. · [[paper]](https://openreview.net/forum?id=sCLdMcBvzN) · `Image`
- **SALART-VQA: Diagnosing Whether VLMs Understand Salient Artifacts in Generated Images** · Xiaoxiao Sun et al. · [[paper]](https://neurips.cc/virtual/2026/poster/139139) · [[arXiv]](https://arxiv.org/abs/2606.12671) · `Image` `Benchmark` `MLLM`
- **SIGMA: Semantic-Difference Instruction-Grounding Mask Annotator for Text-Driven Image Manipulation Localization** · Peiyu Zhuang et al. · [[paper]](https://openreview.net/forum?id=YE3wLDLd1p) · [[arXiv]](https://arxiv.org/abs/2605.27924) · `Image` `Localization`
- **Stability-Weighted Direction Regularization Disentangles Generator Shortcuts from Detection Signal** · Jang-Ho Choi et al. · [[paper]](https://openreview.net/forum?id=2VpqRkpXM9) · `Image`
- **SynthForensics: Benchmarking and Evaluating People-Centric Synthetic Video Deepfakes** · Roberto Leotta et al. · [[paper]](https://neurips.cc/virtual/2026/poster/139072) · [[arXiv]](https://arxiv.org/abs/2602.04939) · `Video` `Benchmark`
- **Temporal Slice Learning for AI-Generated Video Detection with 400× Fewer FLOPs** · Wei Shang et al. · [[paper]](https://openreview.net/forum?id=CNFgZ3PsmY) · [[arXiv]](https://arxiv.org/abs/2610.35148) · `Video`
- **The road reaches every place, the short cut only one: Self-Adversarial Shortcut Mitigation for AI-Generated Image Detection** · Haifeng Zhang et al. · [[paper]](https://openreview.net/forum?id=bO6CYKnaZo) · [[code]](https://github.com/oceanzhf/SA-AFL) · `Image`
- **Too Aligned to be Real: Detecting AI-Generated Images via Cross-modal Alignment Shift** · Haifeng Zhang et al. · [[paper]](https://openreview.net/forum?id=k72oka2LYR) · [[code]](https://github.com/oceanzhf/CASNet) · `Image`
- **Unified Forensic Preference Learning for Generalizable Synthetic Image Detection** · Laixin Zhang et al. · [[paper]](https://openreview.net/forum?id=fGUaeL5g1k) · `Image`
- **Video Forensic Self-Descriptions: Leveraging Temporally Distributed Forensic Microstructures for Zero-Shot Detection of AI-Generated Videos** · Aref Azizpour et al. · [[paper]](https://openreview.net/forum?id=pnZvRHgjWk) · `Video`

### NeurIPS 2025

- **AI-Generated Video Detection via Perceptual Straightening** · Christian Internò et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1d9a43752c2819e03967c5c1b708169c-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2507.00583) · [[code]](https://github.com/ChristianInterno/ReStraV) · `Video` · ✅
- **Breaking Latent Prior Bias in Detectors for Generalizable AIGC Image Detection** · Yue Zhou et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2c2be72635379ec8600eea13df4743e3-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2506.00874) · `Image`
- **Denoising Trajectory Biases for Zero-Shot AI-Generated Image Detection** · Yachao Liang et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dfd12fe50b18505e3c912c4426707cc7-Abstract-Conference.html) · `Image`
- **Detecting Generated Images by Fitting Natural Image Distributions** · Yonggang Zhang et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2cfa9b0d9be8a5c01cf3eb7f21b4f2b8-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2511.01293) · [[code]](https://github.com/tmlr-group/ConV) · `Image`
- **Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable** · Ruoxin Chen et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/991c9f1799cfeebb4217baaacc462d86-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2505.14359) · [[code]](https://github.com/roy-ch/Dual-Data-Alignment) · `Image` · ✅
- **Epistemic Uncertainty for Generated Image Detection** · Jun Nie et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/927dfe0d77fb20d9822bacd6737375a8-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2412.05897) · [[code]](https://github.com/tmlr-group/WePe) · `Image`
- **FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies** · Shuqiao Liang et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3bfcc45018b4156b92197db47e9742bd-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2509.20890) · [[code]](https://github.com/xigua7105/FerretNet) · `Image`
- **ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization** · Bo Du et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5d8a47fc1cb0a143f149ee07321621c9-Abstract-Datasets_and_Benchmarks_Track.html) · [[arXiv]](https://arxiv.org/abs/2505.11003) · [[code]](https://github.com/scu-zjz/ForensicHub) · `Image` `Benchmark` `Localization`
- **Is Artificial Intelligence Generated Image Detection a Solved Problem?** · Ziqiang Li et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb693c67f61e5321746ffce8b6fdd2d0-Abstract-Datasets_and_Benchmarks_Track.html) · [[arXiv]](https://arxiv.org/abs/2505.12335) · [[code]](https://github.com/HorizonTEL/AIGIBench) · `Image` `Benchmark`
- **MLEP: Multi-granularity Local Entropy Patterns for Generalized AI-generated Image Detection** · Lin Yuan et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/63db94e06ce85dde10871c99dd39ad98-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2504.13726) · [[code]](https://github.com/fkeufss/MLEP) · `Image`
- **NeuroRenderedFake: A Challenging Benchmark to Detect Fake Images Generated by Advanced Neural Rendering Methods** · Chengdong Dong et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/56bdf726a96d43ee1e66172d14c63a61-Abstract-Datasets_and_Benchmarks_Track.html) · `Image` `Benchmark`
- **Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection** · Shuhai Zhang et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ff7c9c90030f1eb3cbf5c81b6fbd9a05-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2510.08073) · [[code]](https://github.com/ZSHsh98/NSG-VD) · `Video` · ✅
- **Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation** · Riccardo Corvi et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/25e92e33ac8c35fd49f394c37f21b6da-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2506.16802) · [[code]](https://github.com/grip-unina/WaveRep-SyntheticVideoDetection) · `Video` · ✅
- **Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation** · Siwei Wen et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/551d275ae5cb1bb1ebe6e37fbe201db8-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2503.14905) · [[code]](https://github.com/opendatalab/FakeVLM) · `Image` `MLLM` · ✅
- **Towards Generalizable Detector for Generated Image** · Qianshu Cai et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/87f5d94a4234e93e7854b45ebd04d2ad-Abstract-Conference.html) · `Image`
- **Towards Reliable Identification of Diffusion-based Image Manipulations** · Alex Costanzino et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36721d1209a059dcb7a090dd543f34c4-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2506.05466) · `Image` `Localization`
- **Training-free Detection of AI-generated images via Cropping Robustness** · Sungik Choi et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1d8f05e4da49a4e1e1b052a3046bceac-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2511.14030) · `Image`

### NeurIPS 2024

- **Breaking Semantic Artifacts for Generalized AI-generated Image Detection** · Chende Zheng et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html) · [[code]](https://github.com/Zig-HS/FakeImageDetection) · `Image`
- **On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection** · Xiufeng Song et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dccbeb7a8df3065c4646928985edf435-Abstract-Conference.html) · [[arXiv]](https://arxiv.org/abs/2410.23623) · [[code]](https://github.com/SparkleXFantasy/MM-Det) · `Video`
- **Semi-Truths: A Large-Scale Dataset of AI-Augmented Images for Evaluating Robustness of AI-Generated Image detectors** · Anisha Pal et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d5cdf7e56422f2a229c497dd89c3b995-Abstract-Datasets_and_Benchmarks_Track.html) · [[arXiv]](https://arxiv.org/abs/2411.07472) · [[code]](https://github.com/J-Kruk/SemiTruths) · `Image` `Benchmark`

### NeurIPS 2023

- **GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image** · Mingjian Zhu et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html) · [[arXiv]](https://arxiv.org/abs/2306.08571) · [[code]](https://github.com/GenImage-Dataset/GenImage) · `Image` `Benchmark`
- **Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images** · Zeyu Lu et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/505df5ea30f630661074145149274af0-Abstract-Datasets_and_Benchmarks.html) · [[arXiv]](https://arxiv.org/abs/2304.13023) · [[code]](https://github.com/Inf-imagine/Sentry) · `Image` `Benchmark`

### NeurIPS 2020

- **Fourier Spectrum Discrepancies in Deep Network Generated Images** · Tarik Dzanic et al. · [[paper]](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1f8d87e1161af68b81bace188a1ec624-Abstract.html) · [[arXiv]](https://arxiv.org/abs/1911.06465) · `Image`

## ICLR

### ICLR 2026

- **All Patches Matter, More Patches Better: Enhance AI-Generated Image Detection via Panoptic Patch Learning** · Zheng Yang et al. · [[paper]](https://openreview.net/forum?id=ob7PJs8kPU) · [[arXiv]](https://arxiv.org/abs/2504.01396) · `Image`
- **Detective SAM: Adaptive AI-Image Forgery Localization** · Gert Lek et al. · [[paper]](https://openreview.net/forum?id=GKJHPHNFIx) · [[code]](https://github.com/Gertlek/DetectiveSAM) · `Image` `Localization`
- **Enabling Your Forensic Detector Know How Well It Performs on Distorted Samples** · Bin Li et al. · [[paper]](https://openreview.net/forum?id=Jz5SA2KoFt) · `Image`
- **FakeXplain: AI-Generated Image Detection via Human-Aligned Grounded Reasoning** · Yikun Ji et al. · [[paper]](https://openreview.net/forum?id=UcpTOa8OnG) · [[code]](https://github.com/Gennadiyev/FakeXplain) · `Image` `MLLM`
- **HSIC Bottleneck for Cross-Generator and Domain-Incremental Synthetic Image Detection** · Chin-Chia Yang et al. · [[paper]](https://openreview.net/forum?id=msLnKDvhBx) · `Image`
- **No Pixel Left Behind: A Detail-Preserving Architecture for Robust High-Resolution AI-Generated Image Detection** · Lianrui Mu et al. · [[paper]](https://openreview.net/forum?id=9QQ3Kc2hj6) · [[arXiv]](https://arxiv.org/abs/2508.17346) · `Image`
- **Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale** · Zhengcen Li et al. · [[paper]](https://openreview.net/forum?id=XD43lfRCg6) · [[arXiv]](https://arxiv.org/abs/2604.04634) · `Video`
- **Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images** · Chuangchuang Tan et al. · [[paper]](https://openreview.net/forum?id=0iN4UKZwgn) · [[arXiv]](https://arxiv.org/abs/2510.10231) · [[code]](https://github.com/chuangchuangtan/Semantic-Visual-Anomaly-Detection-and-Reasoning) · `Image` `MLLM`
- **Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection** · Yao Xiao et al. · [[paper]](https://openreview.net/forum?id=Tk8ujiOgHM) · [[arXiv]](https://arxiv.org/abs/2601.19430) · [[code]](https://github.com/Coxy7/X-AIGD) · `Image` `Benchmark` `Localization`
- **Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning** · Hao Tan et al. · [[paper]](https://openreview.net/forum?id=5VXJPS1HoM) · [[arXiv]](https://arxiv.org/abs/2508.21048) · [[code]](https://github.com/EricTan7/Veritas) · `Image` `Benchmark` `MLLM`
- **VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL** · Kyoungjun Park et al. · [[paper]](https://openreview.net/forum?id=gXjOsBcXIR) · [[arXiv]](https://arxiv.org/abs/2510.02282) · [[code]](https://github.com/kyoungjunpark/VidGuard-R1) · `Video` `MLLM`

### ICLR 2025

- **A Sanity Check for AI-generated Image Detection** · Shilin Yan et al. · [[paper]](https://openreview.net/forum?id=ODRHZrkOQM) · [[arXiv]](https://arxiv.org/abs/2406.19435) · [[code]](https://github.com/shilinyan99/AIDE) · `Image` `Benchmark` · ✅
- **Aligned Datasets Improve Detection of Latent Diffusion-Generated Images** · Anirudh Sundara Rajan et al. · [[paper]](https://openreview.net/forum?id=doBkiqESYq) · [[arXiv]](https://arxiv.org/abs/2410.11835) · `Image`
- **FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models** · Zhipei Xu et al. · [[paper]](https://openreview.net/forum?id=pAQzEY7M03) · [[arXiv]](https://arxiv.org/abs/2410.02761) · [[code]](https://github.com/zhipeixu/FakeShield) · `Image` `Localization` `MLLM`
- **LOKI: A Comprehensive Synthetic Data Detection Benchmark using Large Multimodal Models** · Junyan Ye et al. · [[paper]](https://openreview.net/forum?id=z8sxoCYgmd) · [[arXiv]](https://arxiv.org/abs/2410.09732) · `Image` `Video` `Benchmark` `MLLM`
- **Manifold Induced Biases for Zero-shot and Few-shot Detection of Generated Images** · Jonathan Brokman et al. · [[paper]](https://openreview.net/forum?id=7gGl6HB5Zd) · [[arXiv]](https://arxiv.org/abs/2504.15470) · `Image`

## ICML

### ICML 2026

- **Beyond Pixels: Mining Compressed Domain Artifacts for Efficient AI-Generated Video Detection** · Anran Zhu et al. · [[paper]](https://proceedings.mlr.press/v306/zhu26ap.html) · `Video`
- **BPL: Generalizable Deepfake Detection via Bias-only Pair-aware Learning** · Yuxiang Xu et al. · [[paper]](https://proceedings.mlr.press/v306/xu26s.html) · `Image`
- **Breaking Manifold Continuity: Vector Quantized Modeling for Real-Centric Deepfake Detection** · Changshuo Wang et al. · [[paper]](https://proceedings.mlr.press/v306/wang26it.html) · `Image`
- **Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction** · Bo Du et al. · [[paper]](https://proceedings.mlr.press/v306/du26g.html) · [[arXiv]](https://arxiv.org/abs/2602.06676) · [[code]](https://github.com/venus-guangjian/SICA_OpenMMSec) · `Image`
- **Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models** · Kaiqing Lin et al. · [[paper]](https://proceedings.mlr.press/v306/lin26an.html) · [[arXiv]](https://arxiv.org/abs/2606.15880) · [[code]](https://github.com/KQL11/Deep-VRM) · `Image` `MLLM`
- **DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection** · Jiazhen Yan et al. · [[paper]](https://proceedings.mlr.press/v306/yan26h.html) · [[arXiv]](https://arxiv.org/abs/2511.13108) · `Image`
- **Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection** · Dahye Kim et al. · [[paper]](https://proceedings.mlr.press/v306/kim26k.html) · [[arXiv]](https://arxiv.org/abs/2606.10309) · [[code]](https://github.com/anti-fake/dear) · `Image`
- **Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection** · Xiaolu Kang et al. · [[paper]](https://proceedings.mlr.press/v306/kang26o.html) · [[arXiv]](https://arxiv.org/abs/2606.01885) · [[code]](https://github.com/kxl0825/DiCoME) · `Image`
- **DNA: Uncovering Universal Latent Forgery Knowledge** · Jingtong Dou et al. · [[paper]](https://proceedings.mlr.press/v306/dou26b.html) · [[arXiv]](https://arxiv.org/abs/2601.22515) · `Image`
- **Explainable Forensics of Manipulated Segments in Untrimmed Long Videos** · Yue Feng et al. · [[paper]](https://proceedings.mlr.press/v306/feng26g.html) · [[arXiv]](https://arxiv.org/abs/2606.02402) · `Video` `Benchmark` `Localization` `MLLM`
- **FakeWorld 1.0: An Omni-modal Benchmark for Fake Media and Content** · Yifeng Gao et al. · [[paper]](https://proceedings.mlr.press/v306/gao26g.html) · `Image` `Video` `Benchmark`
- **Fleet: Few Shots Lead Effective AI-generated Image Detection** · Jiaan Wang et al. · [[paper]](https://proceedings.mlr.press/v306/wang26eo.html) · [[arXiv]](https://arxiv.org/abs/2606.31082) · [[code]](https://github.com/ICTMCG/Fleet) · `Image` `Benchmark`
- **ForensicConcept: Transferable Forensic Concepts for AIGI Detection** · Menyanshu Zhou et al. · [[paper]](https://proceedings.mlr.press/v306/zhou26bt.html) · [[arXiv]](https://arxiv.org/abs/2606.07034) · `Image`
- **GenShield: Unified Detection and Artifact Correction for AI-Generated Images** · Zhipei Xu et al. · [[paper]](https://proceedings.mlr.press/v306/xu26cl.html) · [[arXiv]](https://arxiv.org/abs/2605.16122) · [[code]](https://github.com/zhipeixu/GenShield) · `Image` `MLLM`
- **Hermes: An Evidence-Driven Agentic Framework for Trustworthy and Explainable AI-Generated Video Detection** · Shuaibo Li et al. · [[paper]](https://proceedings.mlr.press/v306/li26be.html) · `Video` `MLLM`
- **OmniAID: Decoupling Semantics and Artifacts for Universal AI-Generated Image Detection in the Wild** · Yuncheng Guo et al. · [[paper]](https://proceedings.mlr.press/v306/guo26ag.html) · [[arXiv]](https://arxiv.org/abs/2511.08423) · [[code]](https://github.com/yunncheng/OmniAID) · `Image` · ✅
- **Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization** · Yiming Wang et al. · [[paper]](https://proceedings.mlr.press/v306/wang26iy.html) · [[arXiv]](https://arxiv.org/abs/2606.02178) · [[code]](https://github.com/phoenixnir/FLAME) · `Image` `Localization`
- **PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection** · Xiaoyu Zhou et al. · [[paper]](https://proceedings.mlr.press/v306/zhou26h.html) · [[arXiv]](https://arxiv.org/abs/2605.21207) · [[code]](https://github.com/xiaoyu6868/PGC) · `Image` · ✅
- **PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection** · Tuan Nguyen et al. · [[paper]](https://proceedings.mlr.press/v306/nguyen26f.html) · [[arXiv]](https://arxiv.org/abs/2509.26272) · `Image` `MLLM`
- **RA-Det: Towards Universal Detection of AI-Generated Images via Robustness Asymmetry** · Xinchang Wang et al. · [[paper]](https://proceedings.mlr.press/v306/wang26ad.html) · [[arXiv]](https://arxiv.org/abs/2603.01544) · `Image`
- **Real Data Lies: Unveiling and Closing the Quality Shortcut in Generalizable AI-Generated Video Detection** · Ziyuan Fang et al. · [[paper]](https://proceedings.mlr.press/v306/fang26s.html) · `Video`
- **TranX-Adapter: Bridging Artifacts and Semantics within MLLMs for Robust AI-generated Image Detection** · Wenbin Wang et al. · [[paper]](https://proceedings.mlr.press/v306/wang26cj.html) · [[arXiv]](https://arxiv.org/abs/2602.21716) · [[code]](https://github.com/DreamMr/TranX-Adapter) · `Image` `MLLM`
- **VideoVeritas: AI-Generated Video Detection via Perception Pretext Reinforcement Learning** · Hao Tan et al. · [[paper]](https://proceedings.mlr.press/v306/tan26b.html) · [[arXiv]](https://arxiv.org/abs/2602.08828) · [[code]](https://github.com/EricTan7/VideoVeritas) · `Video` `MLLM`
- **Where Detectors Fail: Probing Generative Space for Generalizable AI-Generated Image Detection** · Zijie Cao et al. · [[paper]](https://proceedings.mlr.press/v306/cao26s.html) · [[arXiv]](https://arxiv.org/abs/2605.24906) · [[code]](https://github.com/Amamiya-C/PROBE-AIGI-Detection) · `Image` · ✅

### ICML 2025

- **Are High-Quality AI-Generated Images More Difficult for Models to Detect?** · Yao Xiao et al. · [[paper]](https://proceedings.mlr.press/v267/xiao25g.html) · `Image`
- **Few-Shot Learner Generalizes Across AI-Generated Image Detection** · Shiyu Wu et al. · [[paper]](https://proceedings.mlr.press/v267/wu25r.html) · [[arXiv]](https://arxiv.org/abs/2501.08763) · [[code]](https://github.com/teheperinko541/Few-Shot-AIGI-Detector) · `Image`
- **Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection** · Zhiyuan Yan et al. · [[paper]](https://proceedings.mlr.press/v267/yan25b.html) · [[arXiv]](https://arxiv.org/abs/2411.15633) · [[code]](https://github.com/YZY-stack/Effort-AIGI-Detection) · `Image` · ✅
- **PiD: Generalized AI-Generated Images Detection with Pixelwise Decomposition Residuals** · Xinghe Fu et al. · [[paper]](https://proceedings.mlr.press/v267/fu25i.html) · `Image`
- **Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection** · Anirudh Sundara Rajan et al. · [[paper]](https://proceedings.mlr.press/v267/sundara-rajan25a.html) · [[arXiv]](https://arxiv.org/abs/2502.07778) · `Image`
- **STD-FD: Spatio-Temporal Distribution Fitting Deviation for AIGC Forgery Identification** · Hengrui Lou et al. · [[paper]](https://proceedings.mlr.press/v267/lou25a.html) · `Image`

### ICML 2024

- **DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images** · Baoying Chen et al. · [[paper]](https://proceedings.mlr.press/v235/chen24ay.html) · `Image` `Benchmark`

### ICML 2020

- **Leveraging Frequency Analysis for Deep Fake Image Recognition** · Joel Frank et al. · [[paper]](https://proceedings.mlr.press/v119/frank20a.html) · [[arXiv]](https://arxiv.org/abs/2003.08685) · [[code]](https://github.com/RUB-SysSec/GANDCTAnalysis) · `Image`
- **T-GD: Transferable GAN-generated Images Detection Framework** · Hyeonseong Jeon et al. · [[paper]](https://proceedings.mlr.press/v119/jeon20a.html) · [[arXiv]](https://arxiv.org/abs/2008.04115) · `Image`

## AAAI

### AAAI 2026

- **Aggregating Diverse Cue Experts for AI-Generated Image Detection** · Lei Tan et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37890) · [[arXiv]](https://arxiv.org/abs/2601.08790) · `Image`
- **Beyond Binary Classification: A Semi-supervised Approach to Generalized AI-generated Image Detection** · Hong-Hanh Nguyen-Le et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/40886) · [[arXiv]](https://arxiv.org/abs/2511.19499) · `Image`
- **Beyond Semantic Features: Pixel-level Mapping for Generalized AI-Generated Image Detection** · Chenming Zhou et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/40927) · [[arXiv]](https://arxiv.org/abs/2512.17350) · `Image`
- **CausalCLIP: Causally-Informed Feature Disentanglement and Filtering for Generalizable Detection of Generated Images** · Bo Liu et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37642) · [[arXiv]](https://arxiv.org/abs/2512.13285) · `Image`
- **Decision-Driven Orthogonal Learning with Complementary Feature Mining for Robust Synthetic Image Detection** · Kai Li et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37553) · `Image`
- **Dual-Branch Asymmetric Discrepancy Learning Based on Fake Image Pattern-Coexistence for AI-Generated Image Detection** · Chunli Song et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37071) · `Image`
- **DySy-Det: A Synergistic Framework with Dynamic Reconstruction-Path Consistency for AI-Generated Image Detection** · Fanli Jin et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/40868) · `Image`
- **Explainable Synthetic Image Detection Through Diffusion Timestep Ensembling** · Yixin Wu et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/38060) · [[arXiv]](https://arxiv.org/abs/2503.06201) · [[code]](https://github.com/Shadowlized/ESIDE) · `Image`
- **FIND: A Simple Yet Effective Baseline for Diffusion-Generated Image Detection** · Jie Li et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37547) · [[arXiv]](https://arxiv.org/abs/2603.14220) · `Image`
- **GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection** · Zhenliang Ni et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/38587) · [[arXiv]](https://arxiv.org/abs/2501.11340) · `Video` `Benchmark`
- **Leveraging Failed Samples: A Few-Shot and Training-Free Framework for Generalized Deepfake Detection** · Shibo Yao et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/39998) · [[arXiv]](https://arxiv.org/abs/2508.09475) · `Image`
- **MIRAGE: Towards AI-Generated Image Detection in the Wild** · OuCheng Huang et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37421) · [[arXiv]](https://arxiv.org/abs/2508.13223) · [[code]](https://github.com/och-alibaba/mirage) · `Image` `Benchmark` `MLLM`
- **RealNet: Efficient and Unsupervised Detection of AI-Generated Images via Real-Only Representation Learning** · Shuaibo Li et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/41234) · `Image`
- **SynerDetect: Hierarchical Synergistic Learning for Generalizable AI-Generated Image Detection** · Shuaibo Li et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37568) · `Image`
- **Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated** · Muli Yang et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/38146) · [[arXiv]](https://arxiv.org/abs/2602.01973) · [[code]](https://github.com/muliyangm/AIGI-Det-Calib) · `Image`
- **Zooming In on Fakes: A Novel Dataset for Localized AI-Generated Image Detection with Forgery Amplification Approach** · Lvpan Cai et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/37240) · [[arXiv]](https://arxiv.org/abs/2504.11922) · [[code]](https://github.com/clpbc/BR-Gen) · `Image` `Benchmark` `Localization`

### AAAI 2025

- **C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection** · Chuangchuang Tan et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32772) · [[arXiv]](https://arxiv.org/abs/2408.09647) · [[code]](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection) · `Image`
- **GIM: A Million-scale Benchmark for Generative Image Manipulation Detection and Localization** · Yirui Chen et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32231) · [[arXiv]](https://arxiv.org/abs/2406.16531) · [[code]](https://github.com/chenyirui/GIM) · `Image` `Benchmark` `Localization`
- **InpDiffusion: Image Inpainting Localization via Conditional Diffusion Models** · Kai Wang et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32837) · [[arXiv]](https://arxiv.org/abs/2501.02816) · `Image` `Localization`
- **Revisiting Tampered Scene Text Detection in the Era of Generative AI** · Chenfan Qu et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32051) · [[arXiv]](https://arxiv.org/abs/2407.21422) · [[code]](https://github.com/qcf-568/OSTF) · `Image` `Localization`
- **Transfer Learning of Real Image Features with Soft Contrastive Loss for Fake Image Detection** · Ziyou Liang et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34826) · [[arXiv]](https://arxiv.org/abs/2403.16513) · `Image`
- **WildFake: A Large-Scale and Hierarchical Dataset for AI-Generated Images Detection** · Yan Hong et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32363) · `Image` `Benchmark`

### AAAI 2024

- **Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning** · Chuangchuang Tan et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28310) · [[code]](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) · `Image`

### AAAI 2022

- **FrePGAN: Robust Deepfake Detection Using Frequency-Level Perturbations** · Yonghyun Jeong et al. · [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/19990) · [[arXiv]](https://arxiv.org/abs/2202.03347) · `Image`

## IJCAI

### IJCAI 2026

- **Fast and Generalizable AI-Generated Image Detection via Model-Agnostic Feature Reconstruction** · Qinghui He et al. · [[paper]](https://www.ijcai.org/proceedings/2026/131) · `Image`
- **PURE: Purging Unrelated Representations for Content-Agnostic Forgery Detection** · Xinyu Wu et al. · [[paper]](https://www.ijcai.org/proceedings/2026/207) · `Image`

### IJCAI 2025

- **DcDsDiff: Dual-Conditional and Dual-Stream Diffusion Model for Generative Image Tampering Localization** · Qixian Hao et al. · [[paper]](https://www.ijcai.org/proceedings/2025/120) · `Image` `Localization`
- **Unlocking the Potential of Lightweight Quantized Models for Deepfake Detection** · Renshuai Tao et al. · [[paper]](https://www.ijcai.org/proceedings/2025/59) · `Image`

### IJCAI 2023

- **Towards Robust Gan-Generated Image Detection: A Multi-View Completion Representation** · Chi Liu et al. · [[paper]](https://www.ijcai.org/proceedings/2023/52) · [[arXiv]](https://arxiv.org/abs/2306.01364) · `Image`

### IJCAI 2021

- **Beyond the Spectrum: Detecting Deepfakes via Re-Synthesis** · Yang He et al. · [[paper]](https://www.ijcai.org/proceedings/2021/349) · [[arXiv]](https://arxiv.org/abs/2105.14376) · [[code]](https://github.com/SSAW14/BeyondtheSpectrum) · `Image`

### IJCAI 2020

- **FakeSpotter: A Simple yet Robust Baseline for Spotting AI-Synthesized Fake Faces** · Run Wang et al. · [[paper]](https://www.ijcai.org/proceedings/2020/476) · [[arXiv]](https://arxiv.org/abs/1909.06122) · `Image`

## ACM MM

### ACM MM 2026

- **AGIDefect-4K: A Richly Annotated Dataset for AI-Generated Image Defect Detection, Localization and Explanation** · Xiangfei Sheng et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3838634) · [[arXiv]](https://arxiv.org/abs/2608.20713) · [[code]](https://github.com/sxfly99/AGIDefect-4K) · `Image` `Benchmark` `Localization` `MLLM`
- **Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable AI-Generated Image Detection** · Ruoxin Chen et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836274) · [[code]](https://github.com/GzZning/Align-Dont-Blend) · `Image` `MLLM` · ✅
- **Anchor-Regularized Adaptation for Generalizable AI-Generated Image Detection with DINOv3** · Hyeongjun Choi et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836598) · [[arXiv]](https://arxiv.org/abs/2608.15196) · `Image`
- **DDL: A Large-Scale Dataset for Deepfake Detection and Localization in Diversified Real-World Scenarios** · Changtao Miao et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3838639) · [[arXiv]](https://arxiv.org/abs/2506.23292) · `Image` `Benchmark` `Localization`
- **Defake-o3: From Speculative Rationales to Verifiable Evidence for Explainable AIGI Detection** · Bowen Deng et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3835683) · [[arXiv]](https://arxiv.org/abs/2608.16259) · `Image` `MLLM`
- **Deviating from Real and Aligning with Fake: An Asymmetric Evidence Comparison for RAG-based AI-Generated Image Detection** · WeiXu Zhang et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3835836) · `Image` `MLLM`
- **DVAR: Adversarial Multi-Agent Debate for Video Authenticity Detection** · Hongyuan Qi et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836490) · [[arXiv]](https://arxiv.org/abs/2604.16987) · `Video` `MLLM`
- **From Cheap Fakes to Pure Synthesis: Addressing the New Era of T2V Fake News Videos** · Yifeng Luo et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836334) · [[arXiv]](https://arxiv.org/abs/2608.06732) · `Video` `Benchmark`
- **HydraPrompt: An Adaptive and Asymmetric Framework of Vision-Language Models for Synthetic Image Detection** · Senyuan Shi et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836508) · [[arXiv]](https://arxiv.org/abs/2605.26421) · `Image`
- **LaP-Forensics: Latent-Pixel Consistency Guided Multimodal Reasoning for Deepfake Detection** · Can Wang et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836367) · [[arXiv]](https://arxiv.org/abs/2607.25962) · `Image` `Localization` `MLLM`
- **Learning to Falsify: Hierarchical Multiple Instance Learning for High Resolution AI-Generated Image Detection** · Haoran Chen et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836365) · [[arXiv]](https://arxiv.org/abs/2611.04529) · [[code]](https://github.com/HuiGuanLab/LTF) · `Image` · ✅
- **LEGO: LoRA-Enabled Generator-Oriented Framework for Synthetic Image Detection** · Yutong Xiao et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3835877) · [[arXiv]](https://arxiv.org/abs/2605.04445) · `Image`
- **MARE: Multimodal Alignment and Reinforcement for Explainable Deepfake Detection via Vision-Language Models** · Wenbo Xu et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3835666) · [[arXiv]](https://arxiv.org/abs/2601.20433) · `Image` `MLLM`
- **Mind the Rift: Cross-Scale Coupling Mismatch for AI-Generated Video Detection** · Siyu Li et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3835693) · [[arXiv]](https://arxiv.org/abs/2609.00742) · [[code]](https://github.com/Litsay/RIFT) · `Video`
- **REPSR: Reconstruction Error Phase Structure Response for AI-Generated Image Detection** · Guochang Li · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836542) · `Image`
- **Smoothing the Semantic Landscape: Generalizable AI-Generated Image Detection via Text-Induced Flatness** · Qinghui He et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3836264) · `Image`
- **SphereVideo: Prototype-anchored Hyperspherical Boundary for Continual AI-generated Video Detection** · Fei Li et al. · [[paper]](https://dl.acm.org/doi/10.1145/3767308.3835218) · [[arXiv]](https://arxiv.org/abs/2608.01334) · `Video`

### ACM MM 2025

- **AEGIS: Authenticity Evaluation Benchmark for AI-Generated Video Sequences** · Jieyu Li et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3758295) · [[arXiv]](https://arxiv.org/abs/2508.10771) · `Video` `Benchmark`
- **BrokenVideos: A Benchmark Dataset for Fine-Grained Artifact Localization in AI-Generated Videos** · Jiahao Lin et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3758305) · [[arXiv]](https://arxiv.org/abs/2506.20103) · `Video` `Benchmark` `Localization`
- **Detecting Synthetic Image by Cross-Modal Commonality Interaction** · Kai Li et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755049) · `Image`
- **DFBench: Benchmarking Deepfake Image Detection Capability of Large Multimodal Models** · Jiarui Wang et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3758204) · [[arXiv]](https://arxiv.org/abs/2506.03007) · [[code]](https://github.com/IntMeGroup/DFBench) · `Image` `Benchmark` `MLLM`
- **ExDA: Towards Universal Detection and Plug-and-Play Attribution of AI-Generated Ex-Regulatory Images** · Wenpeng Mu et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755434) · `Image`
- **Frequency-aware Correlation Discovering and Spatial Forgery Clue Distilling for Synthetic Image Detection** · Jiehua Zhang et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755815) · `Image`
- **MiraGe: Multimodal Discriminative Representation Learning for Generalizable AI-Generated Image Detection** · Kuo Shi et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755142) · [[arXiv]](https://arxiv.org/abs/2508.01525) · `Image`
- **RAIDX: A Retrieval-Augmented Generation and GRPO Reinforcement Learning Framework for Explainable Deepfake Detection** · Tianxiao Li et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3754798) · [[arXiv]](https://arxiv.org/abs/2508.04524) · `Image` `MLLM`
- **RealHD: A High-Quality Dataset for Robust Detection of State-of-the-Art AI-Generated Images** · Hanzhe Yu et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755207) · [[arXiv]](https://arxiv.org/abs/2602.10546) · `Image` `Benchmark`
- **SpecXNet: A Dual-Domain Convolutional Network for Robust Deepfake Detection** · Inzamamul Alam et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755707) · [[arXiv]](https://arxiv.org/abs/2509.22070) · [[code]](https://github.com/inzamamulDU/SpecXNet) · `Image`
- **Towards Explainable Fake Image Detection with Multi-Modal Large Language Models** · Yikun Ji et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755421) · [[arXiv]](https://arxiv.org/abs/2504.14245) · [[code]](https://github.com/Gennadiyev/mllm-defake) · `Image` `MLLM`
- **Towards Good Generalizations for Diffusion Generated Image Detection Using Multiple Reconstruction Contrastive Learning** · Wanyi Zhuang et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3754567) · `Image`
- **Towards Open-world Generalized Deepfake Detection: General Feature Extraction via Unsupervised Domain Adaptation** · Midou Guo et al. · [[paper]](https://dl.acm.org/doi/10.1145/3746027.3755495) · [[arXiv]](https://arxiv.org/abs/2505.12339) · `Image`

### ACM MM 2024

- **Rethinking Image Editing Detection in the Era of Generative AI Revolution** · Zhihao Sun et al. · [[paper]](https://dl.acm.org/doi/10.1145/3664647.3681445) · [[arXiv]](https://arxiv.org/abs/2311.17953) · `Image` `Benchmark` `Localization`
- **SemGIR: Semantic-Guided Image Regeneration Based Method for AI-generated Image Detection and Attribution** · Xiao Yu et al. · [[paper]](https://dl.acm.org/doi/10.1145/3664647.3680776) · `Image`

## KDD

### KDD 2026

- **FakeI2V-Bench: Benchmarking the Applicability of Image-level Deepfake Detectors for Deepfake Video Detection** · Pei Li et al. · [[paper]](https://dl.acm.org/doi/10.1145/3770855.3817509) · [[arXiv]](https://arxiv.org/abs/2608.03096) · [[code]](https://github.com/CryptoAILab/FakeI2V-Bench) · `Video` `Benchmark`

### KDD 2025

- **Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective** · Ouxiang Li et al. · [[paper]](https://dl.acm.org/doi/10.1145/3690624.3709392) · [[arXiv]](https://arxiv.org/abs/2408.06741) · [[code]](https://github.com/Ouxiang-Li/SAFE) · `Image` · ✅

## WACV

### WACV 2026

- **ExDDV: A New Dataset for Explainable Deepfake Detection in Video** · Vlad Hondru et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2026/html/Hondru_ExDDV_A_New_Dataset_for_Explainable_Deepfake_Detection_in_Video_WACV_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2503.14421) · [[code]](https://github.com/vladhondru25/ExDDV) · `Video` `Benchmark` `MLLM`
- **General and Domain-Specific Zero-shot Detection of Generated Images via Conditional Likelihood** · Roy Betser et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2026/html/Betser_General_and_Domain-Specific_Zero-shot_Detection_of_Generated_Images_via_Conditional_WACV_2026_paper.html) · [[arXiv]](https://arxiv.org/abs/2512.05590) · `Image`
- **SVD-Det: A Lightweight Framework for Video Forgery Detection Using Semantic and Visual Defect Cues** · Tsung-Shan Yang et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2026/html/Yang_SVD-Det_A_Lightweight_Framework_for_Video_Forgery_Detection_Using_Semantic_WACV_2026_paper.html) · `Video`
- **Training-free Detection of Text-to-video Generations via Over-coherence** · Jonathan Brokman et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2026/html/Brokman_Training-free_Detection_of_Text-to-video_Generations_via_Over-coherence_WACV_2026_paper.html) · `Video`

### WACV 2025

- **DeCLIP: Decoding CLIP Representations for Deepfake Localization** · Stefan Smeu et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2025/html/Smeu_DeCLIP_Decoding_CLIP_Representations_for_Deepfake_Localization_WACV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2409.08849) · `Image` `Localization`
- **Reducing the Content Bias for AI-Generated Image Detection** · Seoyeon Gye et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2025/html/Gye_Reducing_the_Content_Bias_for_AI-Generated_Image_Detection_WACV_2025_paper.html) · [[arXiv]](https://arxiv.org/abs/2502.17105) · `Image`

### WACV 2024

- **D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles** · Ashish Hooda et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2024/html/Hooda_D4_Detection_of_Adversarial_Diffusion_Deepfakes_Using_Disjoint_Ensembles_WACV_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2202.05687) · `Image`
- **Deep Image Fingerprint: Towards Low Budget Synthetic Image Detection and Model Lineage Analysis** · Sergey Sinitsa et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2024/html/Sinitsa_Deep_Image_Fingerprint_Towards_Low_Budget_Synthetic_Image_Detection_and_WACV_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2303.10762) · `Image`
- **Weakly-Supervised Deepfake Localization in Diffusion-Generated Images** · Dragoș-Constantin Țânțaru et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2024/html/Tantaru_Weakly-Supervised_Deepfake_Localization_in_Diffusion-Generated_Images_WACV_2024_paper.html) · [[arXiv]](https://arxiv.org/abs/2311.04584) · `Image` `Localization`

### WACV 2023

- **A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials** · Chuqiao Li et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Li_A_Continual_Deepfake_Detection_Benchmark_Dataset_Methods_and_Essentials_WACV_2023_paper.html) · [[arXiv]](https://arxiv.org/abs/2205.05467) · [[code]](https://github.com/Coral79/CDDB) · `Image` `Benchmark`
- **CYBORG: Blending Human Saliency Into the Loss Improves Deep Learning-Based Synthetic Face Detection** · Aidan Boyd et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Boyd_CYBORG_Blending_Human_Saliency_Into_the_Loss_Improves_Deep_Learning-Based_WACV_2023_paper.html) · `Image`

### WACV 2022

- **BiHPF: Bilateral High-Pass Filters for Robust Deepfake Detection** · Yonghyun Jeong et al. · [[paper]](https://openaccess.thecvf.com/content/WACV2022/html/Jeong_BiHPF_Bilateral_High-Pass_Filters_for_Robust_Deepfake_Detection_WACV_2022_paper.html) · [[arXiv]](https://arxiv.org/abs/2109.00911) · `Image`

## Journals

### TPAMI 2026

- **CoDA: Color Distribution Probing for Efficient and Generalizable AI-Generated Image Detection** · Zexi Jia et al. · [[paper]](https://ieeexplore.ieee.org/document/11535584/) · [[arXiv]](https://arxiv.org/abs/2605.24306) · `Image`
- **Penny-Wise and Pound-Foolish in AI-Generated Image Detection** · Yabin Wang et al. · [[paper]](https://ieeexplore.ieee.org/document/11395536/) · `Image`
- **Self-Supervised AI-Generated Image Detection: A Camera Metadata Perspective** · Nan Zhong et al. · [[paper]](https://ieeexplore.ieee.org/document/11353045/) · [[arXiv]](https://arxiv.org/abs/2512.05651) · [[code]](https://github.com/Ekko-zn/SDAIE) · `Image`

### IJCV 2026

- **AvatarShield: Visual Reinforcement Learning for Human-Centric Synthetic Video Detection** · Zhipei Xu et al. · [[paper]](https://link.springer.com/article/10.1007/s11263-026-02954-5) · [[arXiv]](https://arxiv.org/abs/2505.15173) · `Video` `MLLM`

### IJCV 2025

- **Language-Guided Hierarchical Fine-Grained Image Forgery Detection and Localization** · Xiao Guo et al. · [[paper]](https://link.springer.com/article/10.1007/s11263-024-02255-9) · [[arXiv]](https://arxiv.org/abs/2410.23556) · `Image` `Localization`

---

**Maintenance.** The list is refreshed monthly: new proceedings and accepted-paper lists are searched by keyword, each candidate's abstract is read for scope, and only papers with an official acceptance record are added. Papers that are only on arXiv move here once their venue publishes them.
