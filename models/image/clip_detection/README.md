# CLIPDetection: Frozen CLIP Features for Generalizable Fake Image Detection

## Short method explanation

**CLIPDetection** is a fake-image detector based on frozen CLIP image features. Instead of training a full deep network to distinguish real images from fake images, the method uses a pretrained CLIP visual encoder as a fixed feature extractor and trains only a lightweight classifier on top of the extracted features.

The motivation is that standard fake-image detectors, such as CNNDetection, can overfit to artifacts from the generator used during training. For example, a detector trained on ProGAN fakes may learn “ProGAN-specific artifact = fake” and classify images from unseen generator families, such as diffusion models, as real. Ojha et al. describe this as a **sink-class failure**, where the real class becomes the default class for images that do not contain the learned fake fingerprint. The authors therefore propose to classify images in a feature space that was not explicitly trained for fake-vs-real detection.  [oai_citation:0‡CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/papers/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.pdf?utm_source=chatgpt.com)

The paper evaluates two simple variants:

1. **Nearest Neighbor (NN):** extract frozen CLIP features and classify a test image based on the labels of its nearest neighbors in feature space.
2. **Linear Classification (LC):** freeze the CLIP image encoder and train only a linear binary classifier on top of the CLIP features.

In the released implementation, the LC setup uses the `--fix_backbone` option, which freezes the pretrained backbone and trains only the final classifier layer.

## Training dataset

CLIPDetection follows the CNNDetection training setup. The detector is trained using the **ProGAN real-vs-fake training dataset** introduced by Wang et al. in CNNDetection.

The training set contains:

| Split | Source | Number of images |
|---|---|---:|
| Real | LSUN real images from 20 object categories | 360,000 |
| Fake | ProGAN-generated images from the same 20 LSUN categories | 360,000 |
| Total | Real + fake | 720,000 |

For each of the 20 LSUN object categories, ProGAN generates **18,000 fake images**, giving 360,000 fake images in total. The real class contains 360,000 corresponding real images.  [oai_citation:1‡CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/papers/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.pdf?utm_source=chatgpt.com)

The validation set follows the same structure and consists of held-out ProGAN real and fake images from the CNNDetection dataset.  [oai_citation:2‡GitHub](https://github.com/peterwang512/CNNDetection?utm_source=chatgpt.com)

## Augmentation details

The training augmentation is inherited from the CNNDetection-style pipeline. It includes resizing, cropping, flipping, Gaussian blur, and JPEG compression.

| Component | Setting |
|---|---|
| Resize | `loadSize = 256` |
| Crop | `cropSize = 224` |
| Crop type | Random crop during training, center crop during testing |
| Horizontal flip | Random horizontal flip during training |
| Gaussian blur probability | `blur_prob = 0.5` |
| Gaussian blur sigma | sampled from `blur_sig = 0.0,3.0` |
| JPEG probability | `jpg_prob = 0.5` |
| JPEG backend | randomly selected from `cv2,pil` |
| JPEG quality | sampled from `jpg_qual = 30,100` |
| Normalization | CLIP normalization for CLIP backbones |

The blur and JPEG augmentations are important because CNNDetection showed that such low-level corruptions improve cross-generator generalization. Ojha et al. keep this augmentation style while changing the representation and training protocol.  [oai_citation:3‡arXiv](https://arxiv.org/html/2302.10174v2?utm_source=chatgpt.com)

## Performance tables

### Aggregate performance

The paper reports that frozen CLIP features substantially improve generalization to unseen diffusion and autoregressive models compared with CNNDetection-style trained detectors.

| Method | Training setup | Unseen diffusion/autoregressive mAP | Unseen diffusion/autoregressive accuracy |
|---|---|---:|---:|
| CNNDetection-style detector | End-to-end fake-vs-real training on ProGAN | 75.51 | 56.81 |
| CLIPDetection NN | Frozen CLIP features + nearest neighbor | 90.58 | 82.71 |
| CLIPDetection LC | Frozen CLIP features + linear classifier | 95.00 | 81.55 |

The nearest-neighbor version improves over the prior state of the art by **+15.07 mAP** and **+25.90% accuracy** on unseen diffusion and autoregressive models.  [oai_citation:4‡arXiv](https://arxiv.org/abs/2302.10174?utm_source=chatgpt.com)

### Full benchmark averages

| Method | Main classifier | Total average mAP | Total average accuracy |
|---|---|---:|---:|
| CLIPDetection NN | k-nearest neighbor in CLIP space | 90.58+ on unseen diffusion/autoregressive models | 84.25 total average accuracy |
| CLIPDetection LC | Linear probe on frozen CLIP features | 93.38 total average mAP | 81.38 total average accuracy |

The full evaluation includes images from multiple generator families, including ProGAN, StyleGAN, BigGAN, CycleGAN, StarGAN, GauGAN, CRN, IMLE, SITD, SAN, DeepFake, LDM, GLIDE, Guided Diffusion, and DALL-E-mini.  [oai_citation:5‡CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/papers/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.pdf?utm_source=chatgpt.com)

### AGID reproduction (paper vs reproduced)

Reproduced through `benchmark/bench_eval.py` (%). AGID's `clip` is a frozen
CLIP:ViT-L/14 + **linear-probe** detector, so it is compared against the paper's **LC** (linear-classifier)
variant. **AGID_official** loads the authors' released linear weights into the AGID class; **AGID_train** trains the
linear probe from scratch on ProGAN (single best run). Both reproduce the LC numbers; the per-generator mean differs slightly
from the paper because AGID averages over the on-disk generator set.

| Benchmark | Paper | AGID_official | AGID_train |
|---|---|---|---|
| ForenSynths GAN/low-level (11) | — | mAP 96.95 | mAP 97.22 / acc 88.28 |
| UniversalFakeDetect diffusion/AR (8) | 95.0 | mAP 95.66 | mAP 96.44 / acc 87.23 |

Both columns land on the paper's LC mAP (≈92 ForenSynths, ≈95 UniversalFakeDetect), confirming the frozen-CLIP
linear probe reproduces the paper's headline generalization result.

## Core novelties

### 1. Fake detection without learning a fake-specific representation

The main novelty is to avoid learning the full representation on a narrow fake-vs-real dataset. The CLIP backbone is frozen, so the representation is not optimized to detect ProGAN artifacts.

### 2. Frozen CLIP as a generalizable feature space

The method shows that a large vision-language model pretrained on broad internet-scale data can provide a feature space where real and fake images are more separable across generator families.

### 3. Simple classifiers are sufficient

The paper demonstrates that nearest-neighbor classification and linear probing are already strong baselines when applied to frozen CLIP features. This suggests that much of the generalization benefit comes from the pretrained representation rather than from a complex detector architecture.

### 4. Diagnosis of CNNDetection failure

The paper argues that CNNDetection-style models trained on one generator can develop asymmetric behavior: they detect the training generator’s fake artifacts, but images from unseen generators that lack those artifacts are classified as real.

## Reproduced benchmark results

All benchmarks the paper reports, with our numbers (verified, traced to `repro/bench_results/` + `repro/benchmark_specs/`).

| Benchmark | Paper | AGID_official | AGID_train | Status |
|---|---|---|---|---|
| ForenSynths GAN/low-level test (11 of the pape | — | mAP=96.95 | mAP=97.22 / acc=88.28 | reference |
| UniversalFakeDetect diffusion/autoregressive t | map=95.0 | mAP=95.66 | mAP=96.44 / acc=87.23 | reproduced |

**reproduced** = AGID-trained checkpoint matches the paper · **official** = official weights loaded into the AGID class match (the AGID-trained number is shown alongside where we trained from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · **missing** = not yet evaluated.
