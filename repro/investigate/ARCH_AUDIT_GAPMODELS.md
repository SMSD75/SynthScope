# Architecture + Preprocessing Fidelity Audit — Gap Models

Audit of the four AGID detectors whose trained checkpoints score below the
official weights/paper, to determine for each whether the gap is **architectural**
(needs a code change) or **closeable by retraining** (data/recipe).

Method: read the AGID class, clone + read the official repo, diff backbone /
head / resolution / normalization / key preprocessing / loss. Also cross-checked
against the vendored "official" evals in `repro/official_evals/`.

Official repos cloned to `/hddstore/ssalehi/AGID/official_repos/`:
- Co-Spy  → `Megum1/Co-Spy`
- AIDE    → `shilinyan99/AIDE`
- VIB-Net → `oceanzhf/VIBAIGCDetect`
- SPAI    → existing clone at `/ivi/zfs/s0/original_homes/ssalehi/spai_official` (`mever-team/spai`)

---

## 1. SPAI

| Aspect | Official (`spai/models/sid.py` `PatchBasedMFViT`, `configs/spai.yaml`) | AGID (`models/image/spai/spai.py`) | Match? |
|---|---|---|---|
| Backbone | MFM-pretrained ViT-B/16, embed 768, depth 12, heads 12, APE, mean-pool, intermediate layers `[0..11]` (N=12) (`configs/spai.yaml:10-24`) | Same VIT, `intermediate_layers=tuple(range(12))` (N=12), MFM ckpt (`spai.py:156-168`) | YES |
| Normalization | ImageNet, applied **inside** model on [0,1] input (`sid.py:441-443,469-471`) | ImageNet inside model on [0,1] (`spai.py:14-15,153,172`) | YES |
| Per-feature projector | `FeatureSpecificProjector` (PATCH_PROJECTION_PER_FEATURE=True), proj_dim 1024, 2 layers, last-act Identity (`configs/spai.yaml:21-23,27`; `sid.py:837-858`) | per-layer `nn.ModuleList([Projector])`, dim 1024, 2 layers, last_act Identity default (`spai.py:126-127,31-46`) | YES |
| SRS (recon-similarity) | cosine-sim mean+std over 3 pairs → `6*N`; order `[xlow_m,xlow_s,xhi_m,xhi_s,lowhi_m,lowhi_s]` (`sid.py:772-790`) | `_srs` cosine mean+std, same 3 pairs, same order, `6*N` (`spai.py:176-184`) | YES |
| SCV (spectral context) | `FeatureImportanceProjector`: softmax(alpha)·[mean;std]-proj → proj_dim (`sid.py:918-943`) | `SpectralContext`: identical softmax-alpha over [mean;std] → dim (`spai.py:49-61`) | YES |
| `cls_vector_dim` | `6*N + proj_dim = 72 + 1024 = 1096` (`sid.py:1209-1214`) | `z_dim = 6*n + dim = 1096` (`spai.py:129`) | YES |
| Spectral-Context-Attention (SCA) | cross-attn: learnable aggregator `(12,1,128)`, `to_kv: Linear(1096,1536*2,bias=False)`, `to_out: Linear(1536,1096)`, heads 12, attn_embed 1536 (`sid.py:74-86`) | `SpectralContextAttention` identical dims/heads/embed (`spai.py:75-105,135-138`) | YES (faithful port) |
| Post-SCA norm | `LayerNorm(1096)` (`sid.py:88,171`) | `attn_norm = LayerNorm(z_dim)` (`spai.py:139,192`) | YES |
| Classifier head | `ClassificationHead` mlp_ratio=3: `Lin(1096,3288)-ReLU-Dp-Lin(3288,3288)-ReLU-Dp-Lin(3288,1)` (`configs/spai.yaml:30`; `sid.py:894-915`) | `hidden=3*1096=3288`, same 3-layer head (`spai.py:129,141-145`) | YES |
| Loss | BCE (`configs/spai.yaml:50`) | BCEWithLogitsLoss (`spai.py:154`) | YES |
| **Resolution / patchify** | `RESOLUTION_MODE: arbitrary`, **native/original resolution**, tile into all 224 patches that fit (variable L), `MINIMUM_PATCHES=4` with `five_crop` fallback; `TEST.ORIGINAL_RESOLUTION: True` (`configs/spai.yaml:8,32,52`; `sid.py:176-237`) | **224x224 crop** fed in (`experiments/image/spai.sh:15-16` = `resize_short_256,center_crop_224`), then `patchify_image(224,patch=224,stride=224)` → **L=1** single patch (`spai.py:64-72,190`) | **NO** |

**The architecture is a faithful port** — head dims, projector structure, SCA, SCV,
classifier all match the published config exactly. The one real divergence is
**input resolution / patch count**:

- AGID's `spai.yaml` *intends* a 448 input → 2×2 grid (L=4), and the SCA only
  does useful cross-patch aggregation when L>1. But `experiments/image/spai.sh`
  feeds a **224 center-crop** (`spai.sh:15-16`), so `patchify_image` yields
  **L=1** and the SCA degenerates to a no-op single-patch aggregator.
- The official runs **arbitrary resolution at the image's native size**
  (`RESOLUTION_MODE: arbitrary`, `MINIMUM_PATCHES=4`, `TEST.ORIGINAL_RESOLUTION`),
  i.e. dozens of 224 patches per image. The vendored eval
  (`repro/official_evals/spai_official_eval.py:11-14,60-61`) reproduces this as
  the fixed-resolution 448→2×2 simplification.

This is a **training/eval-recipe gap, not an architecture bug**: the model class
can already do the multi-patch path (`use_sca=True`, `patchify_image`). Closing it
needs (a) feeding ≥448 inputs so L≥4 (e.g. swap the transform to
`resize_short_448,random_crop_448` / `center_crop_448`), ideally the native-res
arbitrary path with `minimum_patches`/`five_crop` fallback, and (b) retraining.
No new module is required.

**VERDICT: DIVERGES-closeable-by-retrain.** Code change is to the *recipe*, not the
class: set SPAI's transform to 448 crops (`experiments/image/spai.sh:15-16` →
`resize_short_448 … random_crop_448 / center_crop_448`) so `patchify_image`
produces the L=4 grid the SCA needs, and retrain. For full fidelity, add a
native-resolution arbitrary-res eval path with the `MINIMUM_PATCHES=4` /
`five_crop` fallback that the official uses.

---

## 2. Co-Spy

Official has two variants under `detectors/`: `progan` (CLIP-L 768 + NPR-residual
artifact) and `sd-v1_4` (SigLIP-384 + SD-VAE-residual artifact). The default /
benchmark variant is `sd-v1_4` (`main.py:81 default="sd-v1_4"`). AGID implements
the **sd-v1_4** variant.

| Aspect | Official `sd-v1_4` (`detectors/sd-v1_4/*.py`) | AGID (`models/image/co_spy/co_spy.py`) | Match? |
|---|---|---|---|
| Semantic backbone | open_clip `ViT-SO400M-14-SigLIP-384` / `webli`, frozen, 1152-d; mean/std=0.5; res 384 (`semantic.py:13-15,28-33`) | same model/pretrained, 1152-d (auto), `SEMANTIC_MEAN/STD=0.5`, res 384 (`co_spy.py:150-167`) | YES |
| Artifact preprocessing | SD-v1.4 VAE: `x - vae_recon(x)`, scale `/7*100` (`artifact.py:181-194`) | `(x - vae_dec(vae_enc.mean)) * 100/7` (`co_spy.py:245-251`, scale=100/7=14.2857) | YES |
| Artifact encoder | ResNet-50 stem: conv1 3×3 s2 → bn → maxpool → layer1[3] → layer2[4] → avgpool → 512-d (`artifact.py:137-208`, `_make_layer` Bottleneck) | identical `_ArtifactResNetStem` 512-d (`co_spy.py:80-130`) | YES |
| Fusion | `cat([sem_coeff·sem_feat, art_coeff·art_feat])`, coeff = per-branch fc logit; train-time coeff dropout zeroes one branch w.p. 0.3 (`end2end.py:89-109`) | identical `_fuse` + `_apply_coeff_dropout` (p=0.3) (`co_spy.py:287-308`) | YES |
| Head | `Linear(sem_dim+art_dim, 1)` = `Linear(1152+512,1)` (`end2end.py:25`) | `fc = Linear(sem_dim+art_dim, 1)` (`co_spy.py:192`) | YES (score-via-feature-fusion) |
| Feature interpolation | `Beta(α,α)` mix on a `ratio` of the batch w/ soft labels (`train.py:106-123,150-163`) | `_mix_semantic_features` Beta(0.2,0.2), prob 0.5 (`co_spy.py:253-285`) | YES |
| Loss | BCEWithLogits (with optional label smoothing) | `_LabelSmoothingBCEWithLogits` (`co_spy.py:133-144,203`) | YES |
| Art-branch resolution | semantic 384; artifact `Resize(224, antialias=False)` then Normalize(0) of the 384 tensor (`end2end.py:32-35`) | `_resize_if_needed(x,224)` for artifact, 384 for semantic (`co_spy.py:177,239,246`) | ~YES (AGID uses `antialias` default True via F.interpolate bilinear, minor) |

**This is the correct two-branch, two-frozen-encoder architecture**, not a
single-backbone simplification. Backbone (SigLIP-384), artifact branch
(SD-VAE residual, not NPR), and the score/coefficient-fusion head all match the
official `sd-v1_4` detector and the vendored eval
(`repro/official_evals/cospy_official_eval.py:22-29`). Only nit: the official art
branch resizes 384→224 with `antialias=False`; AGID's `F.interpolate(...,
mode="bilinear")` applies antialiasing — a tiny preprocessing detail that shifts
the VAE-residual statistics slightly and is closeable by passing `antialias=False`.

**VERDICT: MATCHES** (the sd-v1_4 variant). Any remaining gap is data/recipe +
the trivial `antialias=False` nit on the 384→224 artifact resize
(`co_spy.py:230`); no architectural change needed.

---

## 3. AIDE

| Aspect | Official (`models/AIDE.py`, `data/dct.py`, `data/datasets.py`) | AGID (`models/image/aide/aide.py`) | Match? |
|---|---|---|---|
| Semantic backbone | OpenCLIP `convnext_xxlarge` (`laion2b_…_soup`) trunk, frozen, global_pool/flatten→Identity (`AIDE.py:230-247`) | same convnext_xxlarge trunk, frozen (`aide.py:159-172`) | YES |
| Semantic proj | avgpool → `Linear(3072,256)` (`AIDE.py:241-244`) | avgpool → `Linear(3072,256)` (`aide.py:175-176`) | YES |
| SRM HPF | 30 fixed SRM kernels `Conv2d(3→30,5×5)`, frozen (`AIDE.py:9-34`, `srm_filter_kernel.py`) | `_HPF` identical 30-kernel `Conv2d(3→30)` (`aide.py:39-58`); SRM list verified bit-identical | YES |
| Patch ResNets | two custom `ResNet(Bottleneck,[3,4,6,3])`, conv1 `Conv2d(30→64)`, fc removed → 2048-d; `model_min`,`model_max` (`AIDE.py:119-186,212-213`) | `model_min`,`model_max` = torchvision resnet50 w/ conv1 `Conv2d(30→64)`, fc=Identity (`aide.py:125-157`) | ~YES (arch equiv; AGID uses torchvision ResNet-50 + ImageNet-V2 init vs official custom ResNet) |
| Head | `Mlp(2048+256, 1024, 2)` (`AIDE.py:228`) | `Lin(2048+256,1024)-GELU-Lin(1024,2)` (`aide.py:178-184`) | YES |
| Loss | (Soft/LabelSmoothing) CrossEntropy (`main_finetune.py:252,304-308`) | CrossEntropyLoss + label_smoothing (`aide.py:186-188`) | YES |
| **DCT patch selection** | `DCT_base_Rec_Module`: unfold 32×32 s16 → DCT → **level-filter band → IDCT reconstruct** (`level_x_unfold`) → grade by `2^k`·log\|DCT\| → top-2/bottom-2 → **fold0 keeps 32×32 patches** (`dct.py:82-144`) | `_DCTPatchSelector` **skips the level-filter DCT/IDCT round-trip**, picks raw pixel patches from `nn.Unfold` directly (explicit, `aide.py:64-69,98-122`) | **NO (simplified)** |
| **Input pipeline** | `ToTensor()` only — **native resolution**, no resize before DCT; then each of 4 patches + full image `x_0` → `Resize([256,256])` + **ImageNet** normalize → `[5,3,256,256]` (`datasets.py:34-51,122-139`) | dataset feeds a single **224×224 CLIP-normalized** tensor (`aide.sh:15-16`); model undoes CLIP→pixel, **resizes to 256**, runs DCT on the 256 image, upsamples 32×32 patches→256 (`aide.py:208-220,222-225`) | **NO** |
| Semantic input res | `x_0` = 256×256 ImageNet-norm, remapped ImageNet→CLIP **inside** model → convnext sees 256 → `[B,3072,8,8]` (`AIDE.py:266-279`) | convnext fed the **224 CLIP-norm** image directly (`aide.py:199-204`) → 7×7, no remap needed | **NO (224 vs 256)** |

The AGID class **does** implement the genuine 5-view (4 DCT patches + semantic)
ConvNeXt-XXL + dual-ResNet + 30-kernel-SRM AIDE — it is **not** a CLIP
simplification. But there are three faithful-preprocessing divergences vs the
official and the vendored eval (`repro/official_evals/aide_official_eval.py:20-26`,
which keeps `ToTensor`-only → DCT-on-native → `Resize(256)+ImageNet`):

1. **Native-resolution loss**: AGID feeds a fixed 224 crop upstream, so all
   native-resolution frequency cues are destroyed *before* the DCT selector
   (`aide.sh:15-16`). Official runs DCT on the native-size image.
2. **Skipped DCT/IDCT level-filter reconstruction** (`aide.py:64-69`). With the
   default all-pass level filter (`level_fliter=[0]`, band `0..2*window`) the IDCT
   is near-identity, so this is *numerically close* but not bit-exact, and the
   patch *grading* differs because AGID grades a 256-resized image whereas
   official grades the native image.
3. **Normalization + semantic res**: AGID uses 224 CLIP-normalized (convnext
   7×7); official uses 256 ImageNet-normalized remapped to CLIP (convnext 8×8).

None of these requires a different *architecture* — the modules are all present
and correctly wired — but matching the official needs **preprocessing/recipe
changes**: feed native-resolution (ToTensor-only) images, normalize with
ImageNet (not CLIP), run the DCT at native res, keep patches at 32→256 via the
official Resize, and (optionally) restore the level-filter IDCT round-trip.

**VERDICT: DIVERGES-closeable-by-retrain.** The class is architecturally faithful
(5-view DCT/SRM + ConvNeXt-XXL); the gap is preprocessing. Code changes:
(a) switch AIDE's dataset transform to native-res `to_tensor`-only + ImageNet
normalize (drop the 224 crop and `normalize_clip`); (b) drop the `images*std+mean`
un-normalize and feed the 256 ImageNet image to both branches as the official does;
(c) optionally re-add the level-filter DCT/IDCT reconstruction in `_DCTPatchSelector`.
Then retrain.

---

## 4. VIB-Net

The official repo ships **only the CLIP path and the released checkpoints are
CLIP-based** — there is **no FLAVA anywhere** in `oceanzhf/VIBAIGCDetect`
(`grep -ri flava` → no hits; `options/base_options.py:32` default
`--arch CLIP:ViT-L/14`; `models/__init__.py` VALID_NAMES are CLIP/Imagenet only).
The paper's Fig. 5 backbone ablation may show FLAVA, but the *released code +
weights* use CLIP ViT-L/14.

| Aspect | Official (`models/clip_models.py` `CLIPModel`) | AGID (`models/image/vib_net/vib_net.py`) | Match? |
|---|---|---|---|
| Backbone | OpenAI CLIP `ViT-L/14`, frozen, 768-d (`clip_models.py:11-19`; `base_options.py:32`) | `CLIP:ViT-L/14` default, frozen, 768-d (`vib_net.py:64-67`; `conf/model/image/vib_net.yaml`) | YES (CLIP path) |
| Normalization | CLIP (eval `RealFakeDataset` Normalize CLIP) | CLIP (`vib_net.sh:16-17` `normalize_clip`) | YES |
| Input | 224 (loadSize 256 → crop 224, `base_options.py:56-57`) | 224 (`resize_short_256,center_crop_224`) | YES |
| **Feature pre-norm** | dropout 0.5 on **raw** CLIP features, **no BatchNorm/LayerNorm** (`clip_models.py:33`) | **`BatchNorm1d(feat_dim)` then dropout** (`vib_net.py:46-47,77,104`) | **NO (AGID adds BN)** |
| VIB head | `fc_1:768→1024 · ReLU · fc_2:1024→1024 · ReLU · fc_3:1024→512` (`clip_models.py:21-38`) | `fc_1:768→1024 · ReLU · fc_2:1024→1024 · ReLU · fc_3:1024→512` (`vib_net.py:48-51`) | YES |
| Latent | k=256: `mu=stats[:,:256]` (`clip_models.py:22,39`) | `latent_dim=256` (`vib_net.py:36,112`) | YES |
| std parameterization | `std = softplus(stats[:,256:] - 5)`, **no floor** (`clip_models.py:40`) | `softplus(… - std_shift(=5)).clamp(min=1e-4)` (`vib_net.py:116`) | ~YES (+1e-4 floor) |
| Reparameterization | `mu + eps*std`, eps ~ N(0,1) (`clip_models.py:42,50-62`) | identical `_reparameterize` (`vib_net.py:95-100,118`) | YES |
| Decode | `Linear(256, 1)` (`clip_models.py:24,43`) | `Linear(256, 1)` (`vib_net.py:52,119`) | YES |
| Loss | inference-only repo; head returns `(mu,std),logit` → VIB = BCE + β·KL(N(μ,σ²)‖N(0,I)) (paper) | `BCE + vib_beta(1e-3)·KL` to N(0,I) (`vib_net.py:53-54,79-83,122-132`) | YES (matches the standard VIB objective; β not in official code) |

**For the CLIP backbone, AGID's VIB head is essentially identical** (same
fc_1/2/3 dims, k=256, softplus-5 std, reparam, decode, KL-to-N(0,I)), and the
AGID CLIP checkpoint reportedly works (GenImage mAP 83.26). The **two AGID
additions** are:

1. **`BatchNorm1d(feat_dim)` on the frozen feature** before the head
   (`vib_net.py:46`) — **not in the official**, which applies dropout to raw CLIP
   features. For CLIP this is a benign extra normalization; for FLAVA the AGID
   comment says it was *required* to stop a head-only probe from collapsing.
2. **`clamp(min=1e-4)` on std** (`vib_net.py:116`) — a NaN guard absent from the
   official; harmless.

On the **FLAVA non-convergence**: this is **not an AGID head bug**. The VIB head
is mathematically the standard, correct VIB (KL term, reparameterization, decode
all match the official CLIP model). The instability is a genuine consequence of
training a high-capacity VIB head on a **frozen FLAVA** encoder whose pooled
[CLS] output is tanh-squashed to a small, per-dimension-uneven scale — exactly
why AGID had to insert the BatchNorm. Critically, **FLAVA is not the official
backbone at all** — the released VIB-Net is CLIP ViT-L/14 — so chasing FLAVA is
chasing a configuration the official authors never shipped. The gap to the
paper/official is therefore best closed on the **CLIP path** by matching the
training data/recipe, not by fixing the head.

**VERDICT: MATCHES** (CLIP path; the head is a faithful VIB). The CLIP-vs-official
gap is data/recipe, closeable by retraining on the official ProGAN/SD-v1.4 data
with their schedule. The only code-level deltas are AGID's extra `BatchNorm1d`
(`vib_net.py:46`) and std `clamp` (`vib_net.py:116`); to be bit-faithful to the
official CLIP model, drop the BatchNorm (apply dropout to raw features as in
`clip_models.py:33`). The FLAVA non-convergence is a method/backbone-choice
instability, **not** a VIB head bug, and FLAVA is not the shipped official backbone.

---

## Summary of VERDICTs

- **SPAI** — DIVERGES-closeable-by-retrain. Architecture (backbone, projectors,
  SRS/SCV, SCA, head) is a faithful port; the gap is the input being a **224 crop
  (L=1, SCA no-op)** vs the official **native/arbitrary resolution (L≥4)**. Fix
  the transform to 448 crops (or native-res arbitrary) and retrain — no class
  change.
- **Co-Spy** — MATCHES (sd-v1_4 variant: SigLIP-384 + SD-VAE residual +
  coefficient-fusion). Only nit: artifact 384→224 resize uses antialias where the
  official passes `antialias=False`.
- **AIDE** — DIVERGES-closeable-by-retrain. The full 5-view DCT/SRM +
  ConvNeXt-XXL is implemented (not a CLIP simplification), but preprocessing
  diverges: **224 CLIP-normalized fixed crop** (vs native-res ToTensor +
  **ImageNet** normalize + 256 patches) and the **skipped DCT/IDCT level-filter
  reconstruction**. Fix preprocessing and retrain.
- **VIB-Net** — MATCHES (CLIP ViT-L/14 path; VIB head is faithful). Official ships
  **CLIP, not FLAVA**; the FLAVA non-convergence is a frozen-FLAVA training
  instability (why AGID added BatchNorm), not a head bug. CLIP-vs-paper gap is
  data/recipe. AGID-only deltas: extra `BatchNorm1d` + std `clamp`.
