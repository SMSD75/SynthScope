# Licenses and attribution

The **SynthScope framework** — the training/eval pipeline, the unified config system, the dataset
modules, the metrics/transform/logging stack, the `synthscope` Python API, and the web interface — is
released under the **Apache License 2.0** (see [LICENSE](LICENSE)).

SynthScope **re-implements** a range of published detectors and, for a few methods, **vendors** the
authors' original network code so their released weights can be loaded faithfully. Those
third-party components, and every **pretrained checkpoint / backbone** SynthScope loads, remain under
**their original licenses**. SynthScope does not relicense them.

> ⚠️ **Before redistributing any model weights** (e.g. via a public model zoo / Hugging Face Hub),
> confirm that the upstream license permits redistribution and your intended use. Several
> AI-generated-image-detection paper repos and some pretrained backbones are **research- or
> non-commercial-only**, and some weights are **gated** (require accepting an upstream license).
> The table below is a starting point — verify each entry against its source. (`⚠ verify` = please
> confirm and fill in from the official repo before relying on it.)

## Detectors

| `model_type` | Method (venue) | Upstream code | Code license | Released weights SynthScope can load | Weights license |
| --- | --- | --- | --- | --- | --- |
| `cnn` | CNNDetection (CVPR'20) | PeterWang512/CNNDetection | ⚠ verify | ProGAN-trained (SynthScope-trained here) | ⚠ verify |
| `clip` | UnivFD (CVPR'23) | WisconsinAIVision/UniversalFakeDetect | ⚠ verify | — (SynthScope-trained probe) | n/a |
| `npr` | NPR (CVPR'24) | chuangchuangtan/NPR-DeepfakeDetection | no license in the repo; **written permission** from the first author (C. Tan, 2026-10-11) to include SynthScope's implementation under Apache-2.0, with credit and citation | **official** ProGAN weights (`NPR.pth`) | same permission: benchmarking and the free, non-commercial public demo, crediting the paper and repository |
| `fat_former` | FatFormer (CVPR'24) | Michel-liu/FatFormer (verify) | ⚠ verify | **official** weights | ⚠ verify |
| `aide` | AIDE (ICLR'25) | ⚠ verify (official repo) | ⚠ verify | **official** weights | ⚠ verify |
| `co_spy` | CO-SPY (CVPR'25) | ⚠ verify (official repo) | ⚠ verify | **official** weights | ⚠ verify |
| `spai` | SPAI (CVPR'25) | mever-team/spai (verify) | ⚠ verify | **official** weights | ⚠ verify |
| `vib_net` | VIB-Net (CVPR'25) | ⚠ verify (official repo) | ⚠ verify | — (SynthScope-trained) | n/a |
| `safe` | SAFE (KDD'25) | ⚠ verify (official repo) | ⚠ verify | — (SynthScope-trained) | n/a |
| `pgc` | PGC | ⚠ verify (official repo) | ⚠ verify | — (SynthScope-trained) | n/a |
| `cure` | CuRe (arXiv 2609.14316) | manic-cui/CuRe | Apache-2.0 (repo `LICENSE`) | **official** 25 MB adapter (`cure_adapter.pt`, loads as `cure.pth`) | Apache-2.0 (shipped in the repo; verify before redistributing) |
| `dinov2` | DINOv2 probe | facebookresearch/dinov2 | Apache-2.0 (verify) | DINOv2 backbone | Apache-2.0 (verify) |
| `dinov3` | DINOv3 probe | facebookresearch/dinov3 | DINOv3 License (gated) | DINOv3 backbone | **gated** — DINOv3 License |
| `adapted_dino` | Adapter-tuned DINOv2 (SynthScope) | this repo | Apache-2.0 | — (SynthScope-trained) | n/a |
| `d3` | D3 (ICCV'25) | ⚠ verify (official repo) | ⚠ verify | training-free (no weights) | n/a |
| `restrav` | ReStraV | ⚠ verify (official repo) | ⚠ verify | — (SynthScope-trained) | n/a |
| `stall` | STALL (CVPR'26) | ⚠ verify (official repo) | ⚠ verify | training-free; needs DINOv3 backbone | gated (DINOv3) |

## Backbones / foundation models (downloaded on first use)

| Backbone | Used by | License |
| --- | --- | --- |
| OpenAI CLIP (ViT-L/14) | `clip`, `fat_former`, `d3` | MIT (verify) |
| OpenCLIP (ConvNeXt-XXL, LAION) | `aide` | code MIT; weights per LAION / model card (verify) |
| DINOv2 (ViT-L/14, ViT-S/14) | `dinov2`, `adapted_dino`, `restrav` | Apache-2.0 (verify) |
| DINOv3 (ViT-L/16) | `dinov3`, `stall` | **gated** — DINOv3 License (non-commercial; accept upstream) |
| SigLIP | `co_spy` | Apache-2.0 (verify) |
| MFM ViT-B/16 | `spai` | ⚠ verify |
| Stable Diffusion VAE | `co_spy` | CreativeML Open RAIL-M (verify) |
| Perception Encoder Core L/14-336 (`facebook/PE-Core-L14-336`) | `cure` | Apache-2.0 (Meta); `official_arch.py` vendors the CuRe/Meta backbone code — see its header |

## Datasets

The dataset auto-downloaders fetch third-party datasets (GenImage, ForenSynths/CNNDetection,
AIGIBench, Co-Spy-Bench, DiTFake, UniversalFakeDetect, GenVideo, …). Each dataset is governed by
its **own** terms / license at the source; SynthScope only automates retrieval and restructuring. Review
and comply with each dataset's terms before use or redistribution. Some are restricted: RealChain is CC BY-NC 4.0
(non-commercial), BFree-Online's URL list is for informational and nonprofit use only (GRIP license), and SynthWildX and
BFree-Online point to images hosted by third parties (X and other websites).

---

*This file is a good-faith attribution map, not legal advice. Maintainers: replace every `⚠ verify`
with the confirmed upstream repository URL and license before publishing a model zoo or any
redistributed weights.*
