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
| `cnn` | CNNDetection (CVPR'20) | PeterWang512/CNNDetection | ⚠ verify | — (SynthScope-trained on ProGAN) | n/a (backbone: see below) |
| `clip` | UnivFD (CVPR'23) | WisconsinAIVision/UniversalFakeDetect | MIT (repo `LICENSE`) | — (SynthScope-trained probe) | n/a (backbone: see below) |
| `npr` | NPR (CVPR'24) | chuangchuangtan/NPR-DeepfakeDetection | no license in the repo; **written permission** from the first author (C. Tan, 2026-10-11) to include SynthScope's implementation under Apache-2.0, with credit and citation | **official** ProGAN weights (`NPR.pth`) | same permission: benchmarking and the free, non-commercial public demo, crediting the paper and repository |
| `fat_former` | FatFormer (CVPR'24) | Michel-liu/FatFormer | Apache-2.0 (repo `LICENSE`) | **official** weights | ⚠ verify |
| `aide` | AIDE (ICLR'25) | shilinyan99/AIDE | MIT (repo `LICENSE`) | **official** weights | ⚠ verify |
| `co_spy` | CO-SPY (CVPR'25) | Megum1/CO-SPY | ⚠ verify | — (SynthScope-trained on GenImage SD v1.4; the official checkpoints are used only as references) | n/a (backbones: see below) |
| `spai` | SPAI (CVPR'25) | mever-team/spai | Apache-2.0 (repo `LICENSE`) | **official** weights | Apache-2.0 (repo README: code and weights); its MFM backbone is under the S-Lab License (below) |
| `vib_net` | VIB-Net (CVPR'25) | ⚠ verify (official repo) | ⚠ verify | — (SynthScope-trained) | n/a (backbone: see below) |
| `vib_net_official` | VIB-Net, official model (CVPR'25) | ⚠ verify (official repo) | ⚠ verify | **official** checkpoints (restricted Google Drive; access requested), or a SynthScope-trained one | ⚠ verify; not redistributed |
| `safe` | SAFE (KDD'25) | ⚠ verify (official repo) | ⚠ verify | — (SynthScope-trained) | n/a |
| `pgc` | PGC | xiaoyu6868/PGC | ⚠ verify | **official** ProGAN checkpoint (`PGC_train_progan_ckpt.pth`, loaded as `pgc.pth`) | ⚠ verify |
| `cure` | CuRe (arXiv 2609.14316) | manic-cui/CuRe | Apache-2.0 (repo `LICENSE`) | **official** 25 MB adapter (`cure_adapter.pt`, loads as `cure.pth`) | Apache-2.0 (shipped in the repo; verify before redistributing) |
| `d3qe` | D3QE (ICCV'25) | Zhangyr2022/D3QE | MIT (repo `LICENSE`); `official/` vendors its network code — see the file headers | **official** checkpoint (`model_epoch_best.pth` from HF `Yanran21/D3QE`, loads as `d3qe.pth`) | MIT (HF model card license tag) |
| `ltf` | LTF (arXiv 2611.04529) | HuiGuanLab/LTF | MIT (repo `LICENSE`, `NOTICE`); `official/` vendors `ltf/model.py`, `ltf/token_ops.py`, `ltf/mil/*.py` and `ltf/data/images.py` with the license in `official/LICENSE` — see the file headers | **official** GenImage checkpoint (`weights/epoch_20.pth` from HF `MuskAI/LTF-v3.3` @ `a2a2e29c`, loads as `ltf.pth`) | **no license tag** on the HF model card (`MuskAI/LTF-v3.3`, checked 2026-11-05); only the code repo is MIT. The weights are not uploaded to the zoo and `ltf` is not on the Space until the authors confirm the weights terms |
| `dda` | DDA (NeurIPS'25) | roy-ch/Dual-Data-Alignment | Apache-2.0 (repo `LICENSE`); `official_arch.py` vendors its `lora.py` — see its header | **official** checkpoint (`DDA_ckpt.pth` from HF `Junwei-Xi/Dual-Data-Alignment`, loads as `dda.pth`) | Apache-2.0 (HF model card license tag) |
| `omniaid`, `omniaid_v1`, `omniaid_genimage` | OmniAID (arXiv 2511.08423) | yunncheng/OmniAID | MIT (README badge; no LICENSE file in the repo); `official_arch.py` vendors `models/OmniAID.py` — see its header | **official** checkpoints (`ckpt/checkpoint_omniaid_{v2,v1,genimage_sd14}.pth` from HF `Yunncheng/OmniAID`, loaded as `<key>.pth`) | MIT (HF model card license tag) |
| `bfree` | B-Free (CVPR'25) | grip-unina/B-Free | GRIP-UNINA license: informational and **nonprofit use only** (notices kept); `official_arch.py` vendors `code/networks/wrapper5crops.py` — see its header | **official** checkpoint (`BFREE_dino2reg4/model_epoch_best.pth` from grip.unina.it, loaded as `bfree.pth`) | GRIP-UNINA license (nonprofit use only) |
| `mirror` | MIRROR (NeurIPS'26) | handsome-rich/MIRROR | **unclear**: the README badge says MIT, but the repo has no LICENSE file (the badge's link 404s); `official_arch.py` vendors `models/mirror.py`, and `preprocess.py` vendors the eval transform of `data/datasets.py` (see the file headers) | **official** phase-2 checkpoint (`checkpoint-h-cur.pth`, authors' Google Drive, loaded as `mirror.pth`; it contains the DINOv3 ViT-H+ weights) | **unclear** (no licence stated); not redistributed: not in the model zoo or on the Space |
| `moejepa` | MoE-JEPA (ECCV'26 W) | ALCOR-Lab-DIAG/MoE-JEPA | MIT; `official_arch.py` vendors the blocks of `model/model.py` — see its header | **official** checkpoint (`moejepa_sidset.safetensors`, v1.0 GitHub release, loaded as `moejepa.safetensors`) | MIT |
| `effort`, `effort_chameleon` | Effort (ICML'25) | YZY-stack/Effort-AIGI-Detection | CC BY-NC 4.0 (README badge); `official_arch.py` vendors the SVD-residual layers of `DeepfakeBench/training/detectors/effort_detector.py` — see its header | **official** checkpoints (`effort_clip_L14_trainOn_{sdv14,chameleon}.pth` from the authors' Google Drive, loaded as `<key>.pth`); the SynthScope-trained ProGAN model (config `image/effort_ufd`) is not a served key and stays on the shared filesystem | CC BY-NC 4.0 (non-commercial) |
| `dinov2` | DINOv2 probe | facebookresearch/dinov2 | Apache-2.0 (verify) | DINOv2 backbone | Apache-2.0 (verify) |
| `dinov3` | DINOv3 probe | facebookresearch/dinov3 | DINOv3 License (gated) | DINOv3 backbone | **gated** — DINOv3 License |
| `adapted_dino` | Adapter-tuned DINOv2 (SynthScope) | this repo | Apache-2.0 | — (SynthScope-trained) | n/a (backbone: see below) |
| `d3` | D3 (ICCV'25) | Zig-HS/D3 | MIT (repo `LICENSE`) | training-free (no weights) | n/a |
| `split` | SPLIT (ECCV'26) | mldljyh/SPLIT | MIT (LICENSE added 2026-11-02); no code copied: `models/video/split/split.py` is SynthScope's own implementation of the paper's equations | training-free (no weights) | n/a |
| `restrav` | ReStraV | ⚠ verify (official repo) | ⚠ verify | — (SynthScope-trained) | n/a (backbone: see below) |
| `stall` | STALL (CVPR'26) | OmerBenHayun/STALL | CC BY-NC 4.0 (`LICENSE.txt`) | training-free; needs the DINOv3 backbone (and ships the VATEX calibration `.npz`) | DINOv3 License (gated download) |

## Backbones / foundation models (downloaded on first use)

| Backbone | Used by | License |
| --- | --- | --- |
| OpenAI CLIP (ViT-L/14) | `clip`, `fat_former`, `d3`, `d3qe`, `ltf` (the pinned `backbones/ViT-L-14.pt` of HF `MuskAI/LTF-v3.3`, the same file) | MIT (verify) |
| OpenAI CLIP ViT-L/14@336 (HF `openai/clip-vit-large-patch14-336`) | `omniaid`, `omniaid_v1`, `omniaid_genimage` | MIT |
| OpenCLIP (ConvNeXt-XXL, LAION) | `aide` | code MIT; weights per LAION / model card (verify) |
| DINOv2 (ViT-L/14, ViT-S/14) | `dinov2`, `adapted_dino`, `restrav`, `dda` | Apache-2.0 (verify) |
| DINOv3 (ViT-L/16) | `dinov3`, `stall` | **gated** download — DINOv3 License (use and redistribution allowed with the license and attribution; trade-control restrictions) |
| DINOv3 ViT-H+/16 (HF `facebook/dinov3-vith16plus-pretrain-lvd1689m`) | `mirror` (its weights are inside the MIRROR checkpoint; the gated download is only needed for training) | **gated** - DINOv3 License |
| SigLIP | `co_spy` | Apache-2.0 (verify) |
| MFM ViT-B/16 | `spai` | S-Lab License 1.0 (non-commercial use; shipped as `mfm/LICENSE_S-Lab.txt` in the weights repo) |
| Stable Diffusion VAE | `co_spy` | CreativeML Open RAIL-M (verify) |
| V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`) | `moejepa` | MIT (HF model card) |
| X-CLIP B/16 (`microsoft/xclip-base-patch16`) | `d3`, `split` | MIT (HF model card) |
| LlamaGen VQ-16 tokenizer (`vq_ds16_c2i.pt`, HF `FoundationVision/LlamaGen`) | `d3qe` | MIT |
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
