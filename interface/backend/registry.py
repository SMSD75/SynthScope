"""Model registry: the single source of truth for every detector the interface can run.

For each detector we record:
  * how to compose its SynthScope config (Hydra overrides, exactly like ``main.py``),
  * its evaluation preprocessing (the ``eval_transform_list`` of its model config, matched to the
    official eval script or the verified official-weights eval under repro/),
  * which checkpoint file backs it and how that checkpoint loads (strict SynthScope vs. an
    official-weights key remap), and
  * how to turn a forward pass into a fake-probability.

Label convention (dataset.py:_get_label): index 0 = real, index 1 = fake.
"""
import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from omegaconf import OmegaConf

from transformations import IMAGENET_MEAN, IMAGENET_STD, AIGITransformations   # IMAGENET_*: the video_norm below

MODEL_CONF_DIR = Path(__file__).resolve().parents[2] / "conf" / "model"


@functools.lru_cache(maxsize=None)
def model_transform(config_name: str) -> AIGITransformations:
    """The test-time preprocessing of model config ``conf/model/<config_name>.yaml`` (its ``eval_transform_list``):
    the detector's official evaluation transform, PIL image in, model input tensor out. Cached: specs share the
    instance, so it must not be mutated."""
    return AIGITransformations(list(OmegaConf.load(MODEL_CONF_DIR / f"{config_name}.yaml").eval_transform_list))


# ---------------------------------------------------------------------------
# Official-weights -> SynthScope state_dict remaps (copied verbatim from the verified repro
# eval scripts in /hddstore/ssalehi/SynthScope/repro/{fatformer,aide,npr}_eval.py).
# ---------------------------------------------------------------------------
def remap_fatformer(sd: dict) -> dict:
    drop_pref = ("image_encoder.",)
    drop_exact = {
        "language_guided_alignment.token_prefix",
        "language_guided_alignment.token_suffix",
    }
    return {
        k: v for k, v in sd.items()
        if not k.startswith(drop_pref) and k not in drop_exact
    }


def remap_aide(sd: dict) -> dict:
    """Official AIDE checkpoint -> SynthScopeAIDEModel. The SynthScope class now wraps the GENUINE arch as
    ``self.net`` (key-for-key identical to the official ``AIDE_Model``), so we just strip any
    DataParallel ``module.`` prefix and add the ``net.`` prefix; every official key loads strict."""
    return {("net." + (k[len("module."):] if k.startswith("module.") else k)): v
            for k, v in sd.items()}


def remap_npr(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("conv1."):
            nk = "backbone.0." + k[len("conv1."):]
        elif k.startswith("bn1."):
            nk = "backbone.1." + k[len("bn1."):]
        elif k.startswith("layer1."):
            nk = "backbone.4." + k[len("layer1."):]
        elif k.startswith("layer2."):
            nk = "backbone.5." + k[len("layer2."):]
        elif k.startswith("fc1."):
            nk = "head." + k[len("fc1."):]
        else:
            nk = k
        out[nk] = v
    return out


def remap_vibnet(sd: dict) -> dict:
    """Official VIB-Net checkpoint (CLIPModel: the OpenAI CLIP under ``model.``) -> SynthScopeVIBNetModel (``backbone.``);
    already-SynthScope keys pass through."""
    return {("backbone." + k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}


def remap_bfree(sd: dict) -> dict:
    """Official B-Free checkpoint -> SynthScopeBFreeModel: the official wrapper keeps the patch embedding outside the
    timm model (``patch_embed.*``) and stores the rest of the timm model without a prefix (``model.*`` here)."""
    return {(k if k.startswith("patch_embed.") else "model." + k): v for k, v in sd.items()}


def remap_effort(sd: dict) -> dict:
    """Official Effort checkpoint (saved from a DataParallel EffortDetector) -> SynthScopeEffortModel: drop ``module.``,
    as the official demo does; the remaining ``backbone.*`` / ``head.*`` names are identical."""
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


# Saved groups of the released LTF checkpoint that its forward path never reads (official ltf/checkpoint.py).
LTF_UNUSED_PREFIXES = (
    "baseline_core.cls_token",
    "baseline_core.classifier.",
    "baseline_core.jpeg_estimate.",
    "baseline_core.patch_classifier.",
    "baseline_core.tile_transformer.",
)


def remap_ltf(sd: dict) -> dict:
    """Official LTF checkpoint (an ``LTFCore`` state dict) -> SynthScopeLTFModel: drop the 27 disconnected baseline
    tensors, as the official ``load_core``, and put the 37 retained ones under ``core.`` (then checked strictly)."""
    return {"core." + k: v for k, v in sd.items() if not k.startswith(LTF_UNUSED_PREFIXES)}


def remap_safe(sd: dict) -> dict:
    """Official SAFE (Ouxiang-Li/SAFE) → SynthScopeSAFEModel. The custom ResNet (conv1/bn1/layer1/layer2)
    matches our class verbatim; only the classifier head differs: official names it ``fc1``, ours
    ``head``. (The engine already unwraps the ``model`` wrapper before this.)"""
    out = {}
    for k, v in sd.items():
        out[("head." + k[len("fc1."):]) if k.startswith("fc1.") else k] = v
    return out


@dataclass
class ModelSpec:
    key: str                       # menu id / dispatch handle
    name: str                      # human-friendly display name
    paper: str                     # short citation
    modality: str                  # "image" (also runs frame-by-frame on video) or "video"
    overrides: list[str]           # Hydra overrides for config_utils.compose_config
    predict_kind: str              # ce | bce | bce_npr | vib | video
    checkpoint: Optional[str]      # filename inside CHECKPOINTS_DIR (None = no weights file)
    ckpt_kind: str                 # synthscope | official_fatformer | official_aide | official_npr | none
    transform: Optional[AIGITransformations] = None   # image preprocessing (None for pure-video models)
    # Video-clip sampling (used when modality == "video").
    num_frames: int = 16
    fps: Optional[float] = None
    frame_size: int = 224
    video_norm: Optional[tuple] = None  # (mean, std) applied to the decoded [0,1] clip
    window_center: bool = False         # center the sampling window (ReStraV) vs start (D3/STALL)
    resize_interp: str = "bilinear"     # frame resize interpolation ("bicubic" for ReStraV)
    edge_crop: float = 0.0              # center edge-crop fraction per frame before resize (D3: 0.1)
    edge_crop_long_side: bool = False   # crop only the longer side (D3's crop_center_by_percentage)
    short_clip_frames: int = 0          # frames to use when a video is too short for num_frames (D3: 8)
    fps_nearest: bool = False           # frame nearest to each k/fps, as ffmpeg's fps filter (D3)
    bgr: bool = False                   # feed BGR channels (D3's official reader never converts cv2's BGR)
    notes: str = ""
    # Which weights back the DEFAULT inference checkpoint, surfaced in the web UI:
    #   "official"      = authors' released weights (loaded into the SynthScope class)
    #   "synthscope_trained"  = our from-scratch SynthScope checkpoint
    #   "training_free" = no weights (frozen-feature / training-free detector)
    weights: str = "synthscope_trained"
    # Score at native/arbitrary resolution one image at a time (bs=1) instead of stacking a
    # fixed-size batch — required by SPAI's any-res spectral path (variable patch count per image).
    native_res: bool = False
    # Interpretability map the detector can expose in the web UI (engine._interpret_map dispatches on it):
    #   "" = none (black-box) · "safe_hf" = DWT HH-band artifact map · "npr_residual" = pixel-residual map.
    interpret: str = ""
    remap: Optional[Callable[[dict], dict]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------
MODELS: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec):
    MODELS[spec.key] = spec


# --- Image detectors ---------------------------------------------------------
_add(ModelSpec(
    key="clip", name="UnivFD / CLIP", paper="Ojha et al., CVPR'23",
    modality="image", predict_kind="ce", checkpoint="clip.pth", ckpt_kind="synthscope",
    overrides=["model@ModelConfig=image/clip"],
    transform=model_transform("image/clip"),
    notes="Frozen CLIP ViT-L/14 features + linear real/fake head (trained on ProGAN).",
))
_add(ModelSpec(
    key="cnn", name="CNNDetection (ResNet-50)", paper="Wang et al., CVPR'20",
    modality="image", predict_kind="ce", checkpoint="cnn.pth", ckpt_kind="synthscope",
    overrides=["model@ModelConfig=image/cnn"],
    transform=model_transform("image/cnn"),
    notes="ImageNet ResNet-50 fine-tuned as a real/fake classifier (trained on ProGAN).",
))
_add(ModelSpec(
    key="dinov2", name="DINOv2 probe", paper="Oquab et al., 2023",
    modality="image", predict_kind="ce", checkpoint="dinov2.pth", ckpt_kind="synthscope",
    overrides=["model@ModelConfig=image/dinov2"],
    transform=model_transform("image/dinov2"),
    notes="Frozen DINOv2 ViT-L/14 + linear head (trained on ProGAN).",
))
_add(ModelSpec(
    key="dinov3", name="DINOv3 probe", paper="Meta AI, 2025",
    modality="image", predict_kind="ce", checkpoint="dinov3.pth", ckpt_kind="synthscope",
    overrides=["model@ModelConfig=image/dinov3"],
    transform=model_transform("image/dinov3"),
    notes="Frozen DINOv3 ViT-L/16 + linear head. Requires DINOv3 backbone weights.",
))
_add(ModelSpec(
    key="adapted_dino", name="Adapter-tuned DINOv2", paper="SynthScope",
    modality="image", predict_kind="ce", checkpoint="adapted_dino.pth", ckpt_kind="synthscope",
    overrides=["model@ModelConfig=image/adapted_dino"],
    transform=model_transform("image/adapted_dino"),
    notes="DINOv2 with per-block bottleneck adapters; backbone frozen.",
))
_add(ModelSpec(
    key="safe", name="SAFE", paper="Li et al., KDD'25",
    modality="image", predict_kind="ce", checkpoint="safe.pth",
    ckpt_kind="synthscope", remap=None, weights="synthscope_trained",
    overrides=["model@ModelConfig=image/safe"],
    transform=model_transform("image/safe"),
    interpret="safe_hf",   # DWT HH-band high-frequency artifact map
    notes="High-frequency (DWT bior1.3 HH-band) artifact detector. Serves the SynthScope-TRAINED checkpoint "
          "(full 40-epoch SAFE crop recipe, safe_full_v2/model_38), which reproduces the paper "
          "(ForenSynths-8 97.79 / UFD-8 98.44 / GenImage-8 99.51 / DiTFake 99.96 mAP) and beats the official "
          "checkpoint on GenImage and full-13 ForenSynths. Center-crop-256, no resize/no norm (a resize "
          "corrupts the high-freq signal -> all-fake).",
))
_add(ModelSpec(
    key="fat_former", name="FatFormer", paper="Liu et al., CVPR'24",
    modality="image", predict_kind="ce", checkpoint="fat_former.pth",
    ckpt_kind="official_fatformer", remap=remap_fatformer, weights="official",
    # The released checkpoint has FAA at blocks [7,15,23] => num_vit_adapter=3 (not the yaml's 8).
    overrides=["model@ModelConfig=image/fat_former", "ModelConfig.num_vit_adapter=3"],
    transform=model_transform("image/fat_former"),
    notes="Frozen CLIP + forgery-aware adapters + learnable prompts. Official ProGAN weights.",
))
_add(ModelSpec(
    key="aide", name="AIDE", paper="Yan et al., ICLR'25",
    modality="image", predict_kind="ce", checkpoint="aide.pth",
    ckpt_kind="official_aide", remap=remap_aide, weights="official",
    # Genuine 5-view AIDE arch; the official checkpoint supplies the ConvNeXt trunk (build it random
    # via convnext_pretrained=null to skip the 4.8GB download, then load the ckpt strict).
    overrides=["model@ModelConfig=image/aide", "ModelConfig.convnext_pretrained=null"],
    transform=model_transform("image/aide"),  # NATIVE-res DCT 5-view stack [5,3,256,256] (official preprocessing)
    interpret="aide_dct",   # native-res per-window DCT energy = the patch-selection saliency
    notes="ConvNeXt-XXL semantic + DCT/SRM low-level cues. The DCT patch selector runs at NATIVE "
          "resolution in preprocessing (resizing to 256 before the DCT costs ~12 mAP on the same "
          "weights); only the 4 selected 32x32 patches + the full image are resized to 256.",
))
_add(ModelSpec(
    key="spai", name="SPAI", paper="Karageorgiou et al., CVPR'25",
    modality="image", predict_kind="bce", checkpoint="spai.pth", ckpt_kind="synthscope",
    weights="official",  # shipped spai.pth = official SPAI weights remapped into the SynthScope class
    overrides=["model@ModelConfig=image/spai"],
    transform=model_transform("image/spai"),  # NATIVE resolution (raw [0,1], cap long side 1024) — the faithful any-res path
    native_res=True,                # score one image at a time (variable patch count)
    interpret="spai_patch",   # per-patch spectral fake-evidence over the native 224-grid
    notes="Spectral learning (MFM ViT-B/16) + Spectral Context Attention at NATIVE/arbitrary resolution: each "
          "image is patchified into a variable 224-grid (five_crop fallback for tiny images). Serves the "
          "official SPAI weights remapped into the SynthScope class (GenImage-8 94.22 / Synthbuster 96.75 mAP on "
          "SynthScope's protocol). A fixed 448 resize/crop destroys the native-res spectral statistics.",
))
_add(ModelSpec(
    key="vib_net", name="VIB-Net", paper="Zhang et al., CVPR'25",
    modality="image", predict_kind="vib", checkpoint="vib_net.pth", ckpt_kind="synthscope",
    weights="synthscope_trained",
    # vib_net.pth = trained in SynthScope with the reconstructed recipe (experiments/image/vib_net.sh, job 267846);
    # reproduces the paper (15 generators: 94.08 AP / 85.22 acc vs 95.53 / 85.83). Official model + test transform.
    overrides=["model@ModelConfig=image/vib_net_official"],
    transform=model_transform("image/vib_net_official"),  # official main.py: CenterCrop(224), ToTensor, CLIP norm (no resize)
    notes="Variational information bottleneck on frozen CLIP ViT-L/14 features, trained in SynthScope on GenImage "
          "SD v1.4 with a recipe reconstructed from the paper (reproduces its 15-generator benchmark). The central "
          "224x224 pixels are scored; one latent sample per image, as the official model.",
))
_add(ModelSpec(
    key="vib_net_official", name="VIB-Net (official model)", paper="Zhang et al., CVPR'25",
    modality="image", predict_kind="vib", checkpoint="vib_net_official.pth", ckpt_kind="official_vibnet",
    remap=remap_vibnet, weights="official",
    overrides=["model@ModelConfig=image/vib_net_official"],
    transform=model_transform("image/vib_net_official"),  # official main.py: CenterCrop(224), ToTensor, CLIP norm (no resize)
    notes="The official VIB-Net model (fp32 CLIP ViT-L/14, no feature BatchNorm; one latent sample per image as the "
          "official code) with the official test transform. Loads the released checkpoints or a SynthScope-trained "
          "one (experiments/image/vib_net.sh).",
))
_add(ModelSpec(
    key="bfree", name="B-Free", paper="Guillaro et al., CVPR'25",
    modality="image", predict_kind="bce", checkpoint="bfree.pth", ckpt_kind="official_bfree", remap=remap_bfree,
    weights="official",
    # bfree.pth = the official BFREE_dino2reg4/model_epoch_best.pth unchanged (grip.unina.it); full model, strict load.
    overrides=["model@ModelConfig=image/bfree"],
    native_res=True,                # official main_bfree.py: one image at a time at its native resolution
    transform=model_transform("image/bfree"),   # official get_list_norm('resnet'): ToTensor + ImageNet norm
    notes="DINOv2 ViT-B/14 with registers, trained end to end on bias-free data (inpainted fakes that share content "
          "and format with their real images). Uses the OFFICIAL released checkpoint (GRIP license: nonprofit use). "
          "Five 504x504 crops of the native-resolution image are scored and averaged, so nothing is resized.",
))
for _key, _name, _ckpt, _note in [
    ("effort", "Effort", "effort.pth", "trained on GenImage SD v1.4 (the paper's GenImage table)"),
    ("effort_chameleon", "Effort (Chameleon)", "effort_chameleon.pth", "trained on Chameleon"),
]:
    _add(ModelSpec(
        key=_key, name=_name, paper="Yan et al., ICML'25",
        modality="image", predict_kind="ce", checkpoint=_ckpt, ckpt_kind="official_effort", remap=remap_effort,
        weights="official",
        # <key>.pth = the official effort_clip_L14_trainOn_{sdv14,chameleon}.pth unchanged (Google Drive, CC BY-NC 4.0).
        overrides=["model@ModelConfig=image/effort"],
        transform=model_transform("image/effort"),
        notes=f"Orthogonal subspace decomposition: CLIP ViT-L/14 fine-tuned only along the minor SVD direction of each "
              f"attention projection, keeping the pretrained principal subspace. Official checkpoint, {_note}. The "
              f"whole image is squashed to 224x224 (cv2 bicubic), as the official DeepfakeBench test loader.",
    ))
_add(ModelSpec(
    key="moejepa", name="MoE-JEPA", paper="Teglia & Amerini, ECCV'26 W",
    modality="image", predict_kind="ce", checkpoint="moejepa.safetensors", ckpt_kind="official_moejepa",
    weights="official",
    # moejepa.safetensors = the official moejepa_sidset.safetensors (v1.0 GitHub release, MIT), unchanged; it holds only
    # the trained tensors, the frozen V-JEPA 2 ViT-L comes from HF facebook/vjepa2-vitl-fpc64-256.
    overrides=["model@ModelConfig=image/moejepa"],
    transform=model_transform("image/moejepa"),
    notes="Frozen V-JEPA 2 ViT-L (the image as a static 4-frame clip) with a residual mixture of experts in its last "
          "three blocks, gated-attention MIL pooling and a BayarConv noise branch fused by a learned gate. Official "
          "checkpoint, trained on SID-Set (social-media images: real, fully synthetic, tampered); P(fake) = P(fully "
          "synthetic) + P(tampered). Short side resized to 292, center crop 256.",
))
_add(ModelSpec(
    key="co_spy", name="CO-SPY", paper="Cheng et al., CVPR'25",
    modality="image", predict_kind="bce", checkpoint="co_spy.pth", ckpt_kind="synthscope",
    overrides=["model@ModelConfig=image/co_spy"],
    transform=model_transform("image/co_spy"),  # raw [0,1]; per-branch norm inside the model
    # co_spy.pth = the SynthScope retrain on GenImage SD v1.4 (repro/retrain_cospy_full_mgpu.sbatch,
    # checkpoints_retrain/cospy_full/model_1.pth, promoted 2026-06-08), not an official checkpoint.
    notes="SigLIP semantic + SD-VAE-residual artifact streams, feature fusion (trained in SynthScope on GenImage SD v1.4).",
))
_add(ModelSpec(
    key="pgc", name="PGC", paper="Zhou et al., 2026",
    modality="image", predict_kind="bce", checkpoint="pgc.pth", ckpt_kind="synthscope",
    weights="official",  # official PGC ProGAN weights load remap-free; deploy PGC_train_progan_ckpt.pth as pgc.pth (FS-8 96.41 / UFD 99.17 mAP, jobs 266282/266660; the official recipe trained in SynthScope, step 500, gives 97.41 / 98.99)
    overrides=["model@ModelConfig=image/pgc"],
    transform=model_transform("image/pgc"),  # official PGC eval = PadCenterCrop(224), NO resize
    interpret="pgc_peak",   # peak-guided residual fake-evidence map (per-location score)
    notes="LoRA-DINOv2 RGB + YCbCr-residual CNN, peak-guided calibration. Uses OFFICIAL released weights. "
          "Eval is center-crop-224 NO resize (official PadCenterCrop): the YCbCr residual is resolution-"
          "sensitive, and a Resize wipes it on super-res/low-light generators (san/seeingdark). "
          "See repro/investigate/pgc_gap_investigation.md.",
))
_add(ModelSpec(
    key="cure", name="CuRe", paper="arXiv 2609.14316 (manic-cui/CuRe)",
    modality="image", predict_kind="ce", checkpoint="cure.pth", ckpt_kind="synthscope",
    weights="official",  # cure.pth = the official 25MB cure_adapter.pt unchanged; frozen PE-Core base fetched from HF
    overrides=["model@ModelConfig=image/cure"],
    transform=model_transform("image/cure"),  # official: PNG->JPEG96, SmallestMaxSize(390)+CenterCrop+Resize(336), JPEG70, mean=std=0.5
    notes="Frozen Perception-Encoder Core L/14 @336 + Q/V LoRA + source-response subspace + linear head, "
          "trained by regressing real-generated mixing ratios on GenImage SD-v1.4. Uses the OFFICIAL released "
          "adapter. Eval preprocessing is part of the method (PNG sources get a JPEG q96 round-trip and every "
          "image a JPEG q70 pass), so use the registered transform — see models/image/cure/README.md.",
))
_add(ModelSpec(
    key="dda", name="DDA", paper="Chen et al., NeurIPS'25 Spotlight",
    modality="image", predict_kind="bce", checkpoint="dda.pth", ckpt_kind="official_dda", weights="official",
    # dda.pth = the official DDA_ckpt.pth unchanged (HF Junwei-Xi/Dual-Data-Alignment); it holds the full model,
    # so the DINOv2 weights are not fetched separately.
    overrides=["model@ModelConfig=image/dda", "ModelConfig.pretrained_backbone=false"],
    transform=model_transform("image/dda"),  # official Inference/inference.py: CenterCrop(336), ToTensor, CLIP norm
    notes="DINOv2 ViT-L/14 + LoRA (rank 8) + linear head, trained on DDA's pixel- and frequency-aligned "
          "VAE-reconstructed fakes. Uses the OFFICIAL released checkpoint. The official test transform is a 336 "
          "center crop with no resize (smaller images are zero-padded), so large images are judged on their center.",
))
_add(ModelSpec(
    key="d3qe", name="D3QE", paper="Zhang et al., ICCV'25",
    modality="image", predict_kind="bce", checkpoint="d3qe.pth", ckpt_kind="official_d3qe", weights="official",
    # d3qe.pth = the official model_epoch_best.pth unchanged (HF Yanran21/D3QE); the frozen CLIP ViT-L/14 and LlamaGen
    # VQ-16 are fetched from their own releases (the model's pretrained_prefixes), all other keys load strictly.
    overrides=["model@ModelConfig=image/d3qe"],
    transform=model_transform("image/d3qe"),  # official test path: Resize(256) short side, CenterCrop(256), ToTensor, no norm
    notes="Detector for autoregressive (token-based) image generators: the quantization error of a frozen LlamaGen VQ "
          "tokenizer, read by a small transformer, fused with frozen CLIP ViT-L/14 features. Trained on LlamaGen vs "
          "ImageNet only. Uses the OFFICIAL released checkpoint.",
))
_add(ModelSpec(
    key="ltf", name="LTF", paper="Chen et al., arXiv 2611.04529",
    modality="image", predict_kind="ce", checkpoint="ltf.pth", ckpt_kind="official_ltf", remap=remap_ltf,
    weights="official",
    # ltf.pth = the official weights/epoch_20.pth unchanged (HF MuskAI/LTF-v3.3 @ a2a2e29c; code MIT); the frozen CLIP
    # ViT-L/14 is the pinned backbones/ViT-L-14.pt of the same revision, loaded by the model (pretrained_prefixes).
    overrides=["model@ModelConfig=image/ltf"],
    transform=model_transform("image/ltf"),   # official prepare_image: global view + native-res 224 tiles
    native_res=True,                # the view count depends on the image size: score one image at a time
    notes="Hierarchical multiple-instance learning on frozen CLIP ViT-L/14 tokens: the image is cut into 224x224 tiles "
          "at its own resolution, plus one resized global view. Patch tokens are pooled within each tile, tiles and "
          "views are pooled across the image, and a gated fusion head decides. Uses the OFFICIAL GenImage checkpoint.",
))
_add(ModelSpec(
    key="mirror", name="MIRROR", paper="Liu et al., NeurIPS'26",
    modality="image", predict_kind="ce", checkpoint="mirror.pth", ckpt_kind="official_mirror", weights="official",
    # mirror.pth = the official phase-2 checkpoint-h-cur.pth unchanged (authors' Google Drive); it holds the full model
    # (DINOv3 ViT-H+ backbone, LoRA, memory bank, head), so neither the gated DINOv3 weights nor mirror_phase1.pth are
    # needed at inference. Strict load; the ViT block prefix is renamed for transformers 4/5 (official_arch.py).
    overrides=["model@ModelConfig=image/mirror"],
    # official data/datasets.py eval: short side >512 -> 512, center 224, PNG/BMP -> JPEG q96
    transform=model_transform("image/mirror"),
    notes="Reference-comparison detector: a frozen memory bank (real-image prototypes in the paper; in the released "
          "checkpoint, the code's seeded default initialization) reconstructs an 'ideal "
          "reference' of the DINOv3 ViT-H+/16 (+ LoRA) features; the residual, the retrieval uncertainty and the CLS "
          "token give the score. Uses the OFFICIAL checkpoint. PNG/BMP inputs get a JPEG q96 round trip (the paper's "
          "format-aligned inputs), so use the registered transform. Runs fp32, the official default.",
))
# OmniAID's official eval preprocessing: v2 resizes straight to 336 (data/datasets.py); the v1-era releases used a
# double resize (512x512, then 336x336) - the commented-out pipeline that the official code says to use for v1 weights.
for _key, _name, _ckpt, _note in [
    ("omniaid", "OmniAID", "omniaid.pth",
     "the recommended v2 release, trained on Mirage-Train (the authors' modern in-the-wild dataset)"),
    ("omniaid_v1", "OmniAID (v1)", "omniaid_v1.pth",
     "the v1 release trained on Mirage-Train, the paper's OmniAID-Mirage"),
    ("omniaid_genimage", "OmniAID (GenImage)", "omniaid_genimage.pth",
     "the release trained on GenImage SD v1.4 only, for the paper's academic GenImage protocol"),
]:
    _add(ModelSpec(
        key=_key, name=_name, paper="Guo et al., arXiv 2511.08423",
        modality="image", predict_kind="ce", checkpoint=_ckpt, ckpt_kind="official_omniaid", weights="official",
        # <key>.pth = the official ckpt/checkpoint_omniaid_*.pth unchanged (HF Yunncheng/OmniAID); full model, strict load.
        overrides=[f"model@ModelConfig=image/{_key}"],
        transform=model_transform(f"image/{_key}"),
        notes="Hybrid mixture of experts on CLIP ViT-L/14@336: a router picks semantic experts (human, animal, object, "
              "scene, ...) and an always-on artifact expert sees every image. Uses the OFFICIAL checkpoint: " + _note + ".",
    ))
_add(ModelSpec(
    key="npr", name="NPR", paper="Tan et al., CVPR'24",
    modality="image", predict_kind="bce_npr", checkpoint="npr.pth",
    ckpt_kind="official_npr", remap=remap_npr, weights="official",
    overrides=["model@ModelConfig=image/npr"],
    transform=model_transform("image/npr"),
    interpret="npr_residual",   # neighboring-pixel residual map (up-sampling artifacts)
    notes="Neighboring-pixel-residual ResNet. Official ProGAN weights.",
))

# --- Video detectors ---------------------------------------------------------
_add(ModelSpec(
    key="d3", name="D3 (video)", paper="ICCV'25",
    modality="video", predict_kind="video", checkpoint=None, ckpt_kind="none", weights="training_free",
    overrides=["model@ModelConfig=video/d3", "ModelConfig.backbone_name=xclip",
               "ModelConfig.first_order=l2"],
    num_frames=16, fps=8.0, frame_size=224, video_norm=(IMAGENET_MEAN, IMAGENET_STD), edge_crop=0.1,
    edge_crop_long_side=True, short_clip_frames=8, fps_nearest=True, bgr=True,
    notes="Training-free: frozen X-CLIP-B/16 per-frame features; 2nd-order trajectory std is the score. "
          "Official preprocessing (D3 utils/video2frame.py + data/datasets.py): 16 frames at 8 fps (8 for clips under "
          "2 s), BGR channels as cv2 reads them, 10% crop of the longer side, resize 224, ImageNet normalization.",
))
_add(ModelSpec(
    key="split", name="SPLIT (video)", paper="ECCV'26",
    modality="video", predict_kind="video", checkpoint=None, ckpt_kind="none", weights="training_free",
    overrides=["model@ModelConfig=video/split"],
    num_frames=16, fps=8.0, frame_size=224, video_norm=(IMAGENET_MEAN, IMAGENET_STD), edge_crop=0.1,
    edge_crop_long_side=True, short_clip_frames=8, fps_nearest=True, bgr=True,
    notes="Training-free: frozen X-CLIP-B/16 patch tokens; two-step temporal roughness (gamma 8) times spatial "
          "patch-level motion incoherence is the score. Official preprocessing (the same video2frame / read_video as "
          "D3): 16 frames at 8 fps (8 for clips under 2 s), BGR, 10% crop of the longer side, resize 224, ImageNet "
          "normalization.",
))
_add(ModelSpec(
    key="restrav", name="ReStraV (video)", paper="2025",
    modality="video", predict_kind="video", checkpoint="restrav.pth", ckpt_kind="synthscope",
    overrides=["model@ModelConfig=video/restrav", "ModelConfig.backbone_name=dinov2_vits14"],
    num_frames=24, fps=None, frame_size=224, video_norm=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    window_center=True, resize_interp="bicubic",
    notes="Frozen DINOv2 ViT-S/14 trajectory-geometry + standardized MLP. Raw [0,1], center window + BICUBIC (official ReStraV).",
))
# DINOv3 ViT-L/16 gated weights. STALL is training-free but needs them: $DINO_V3_WEIGHTS (the Space sets it to the copy
# fetched from the weights repo), else the maintainers' local copy.
DINOV3_WEIGHTS = os.environ.get("DINO_V3_WEIGHTS") or \
    "/hddstore/ssalehi/SynthScope/dinov3_checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
_add(ModelSpec(
    key="stall", name="STALL (video)", paper="Ben Hayun et al., CVPR'26",
    modality="video", predict_kind="video", checkpoint=None, ckpt_kind="none", weights="training_free",
    overrides=["model@ModelConfig=video/stall", f"ModelConfig.dino_weights={DINOV3_WEIGHTS}"],
    num_frames=16, fps=None, frame_size=224, video_norm=(IMAGENET_MEAN, IMAGENET_STD),
    notes="Training-free spatial+temporal likelihood vs real-video stats (DINOv3 ViT-L/16, VATEX calibration).",
))


# ---------------------------------------------------------------------------
# What each detector does to the input before scoring, shown in the UI as a short label + explanation. These describe
# the eval_transform_list of each model config (conf/model/), which mirrors the paper's evaluation (changing it would
# break reproduction).
# ---------------------------------------------------------------------------
_CENTER = ("Resize + center-crop", "The shorter side is scaled to {r} px and the central {c}×{c} square is used. "
           "The aspect ratio is kept; anything outside the square is not seen by the detector.")
INPUT_NOTES: dict[str, tuple[str, str]] = {
    "clip": (_CENTER[0] + " 224", _CENTER[1].format(r=256, c=224)),
    "cnn": (_CENTER[0] + " 224", _CENTER[1].format(r=256, c=224)),
    "dinov2": (_CENTER[0] + " 224", _CENTER[1].format(r=256, c=224)),
    "adapted_dino": (_CENTER[0] + " 224", _CENTER[1].format(r=256, c=224)),
    "vib_net": ("Center-crop 224, no resize", "The central 224×224 pixels are used exactly as they are - no scaling, so fine "
            "pixel patterns are preserved. The rest of the image is not seen."),
    "dinov3": (_CENTER[0] + " 256", _CENTER[1].format(r=256, c=256)),
    "co_spy": (_CENTER[0] + " 384", _CENTER[1].format(r=384, c=384)),
    "safe": ("Center-crop 256, no resize", "The central 256×256 pixels are used exactly as they are - no scaling, so fine "
             "pixel patterns are preserved. The rest of the image is not seen."),
    "pgc": ("Center-crop 224, no resize", "The central 224×224 pixels are used exactly as they are - no scaling, so fine "
            "pixel patterns are preserved. The rest of the image is not seen."),
    "vib_net_official": ("Center-crop 224, no resize", "The central 224×224 pixels are used exactly as they are - no scaling, so fine "
            "pixel patterns are preserved. The rest of the image is not seen."),
    "fat_former": ("Squashed to a square", "The whole image is stretched to 256×256 (the aspect ratio changes), then the "
                   "central 224×224 is used."),
    "npr": ("Squashed to 256×256", "The whole image is stretched to a 256×256 square (the aspect ratio changes)."),
    "spai": ("Original resolution", "Scored at its original size and aspect ratio; it is only downscaled if the long side "
             "exceeds 1024 px."),
    "aide": ("Original resolution, 5 views", "Builds five 256×256 views, shown side by side: four patches cut from the "
             "original pixels (picked by their frequency content) plus one resized view of the whole image."),
    "dda": ("Center-crop 336, no resize", "The central 336×336 pixels are used exactly as they are - no scaling. Smaller "
            "images are padded with black; the rest of a larger image is not seen."),
    "d3qe": ("Resized to 256, center crop", "The shorter side is scaled to 256 and the central 256×256 square is used; "
             "its CLIP branch looks at the central 224×224 of that square."),
    "bfree": ("Native size, five 504 crops", "The image is used at its own resolution: five 504×504 crops (center and "
              "corners) are scored and averaged; a smaller image is tiled up to 504."),
    "moejepa": ("Short side 292, center 256×256", "The short side is resized to 292 and the central 256×256 is kept."),
    "effort": ("Squashed to 224×224", "The whole image is stretched to a 224×224 square (cv2 bicubic, the aspect ratio changes)."),
    "effort_chameleon": ("Squashed to 224×224", "The whole image is stretched to a 224×224 square (cv2 bicubic, the aspect ratio changes)."),
    "mirror": ("Center-crop 224 (short side ≤ 512)", "If the shorter side exceeds 512 px it is scaled to 512, then "
               "the central 224×224 is used (smaller images are padded with black). For PNG/BMP files that 224×224 "
               "crop is then re-saved as JPEG (quality 96), as in the paper's format-aligned evaluation."),
    "omniaid": ("Squashed to 336×336", "The whole image is stretched to a 336×336 square (the aspect ratio changes)."),
    "omniaid_v1": ("Squashed to 512, then 336", "The image is stretched to 512×512 and then to 336×336, as in the v1 release."),
    "omniaid_genimage": ("Squashed to 512, then 336", "The image is stretched to 512×512 and then to 336×336, as in the v1-era releases."),
    "ltf": ("Native-res 224 tiles + global view", "The image is cut into 224×224 tiles at its own resolution (no "
            "scaling; edge tiles are aligned to the border, a side under 224 px is padded with black), plus one view of "
            "the whole image resized to 224 and center-cropped."),
    "cure": ("Paper-specific pipeline", "As in the paper: PNG images are re-saved as JPEG, the shorter side is scaled to "
             "390 and center-cropped, resized to 336, then JPEG-compressed again (quality 70)."),
}


def input_note(spec: ModelSpec) -> dict:
    """{label, explanation} describing the preprocessing of ``spec`` (frames for video models)."""
    if spec.modality == "video":
        crop = f", after trimming {int(spec.edge_crop * 100)}% off every edge" if spec.edge_crop else ""
        return {"label": f"{spec.num_frames} sampled frames",
                "explanation": f"{spec.num_frames} frames are sampled from the clip and resized to "
                               f"{spec.frame_size}×{spec.frame_size}{crop}."}
    label, expl = INPUT_NOTES.get(spec.key, ("Model-specific", "Preprocessing follows the detector's paper."))
    return {"label": label, "explanation": expl + " This matches the paper's evaluation."}


# ---------------------------------------------------------------------------
# Paper-vs-reproduced numbers, shown per model in the UI. Every "ours" value traces to a logged
# eval; official-weights rows are cross-checked against the paper, from-scratch/probe rows are our
# reproductions. Honest caveats are kept in `note` (see README repro section + session findings).
# ---------------------------------------------------------------------------
# Full per-method benchmark table (ALL reported benchmarks: paper vs SynthScope-trained vs official-weights
# + status), generated by build_repro_full.py from repro/benchmark_specs/ + repro/bench_results/.
# Loaded FIRST so the headline cards below DERIVE from the same data and can never drift out of sync.
import json as _json, pathlib as _pathlib
try:
    REPRO_FULL = _json.loads((_pathlib.Path(__file__).resolve().parent / "repro_full.json").read_text())
except Exception:
    REPRO_FULL = {}

# SynthScope-only probe variants (no paper benchmark table of their own) — hand-kept headline cards.
# "ours" = the full ForenSynths-13 test set (repro/bench_results/<model>_synthscope_forensynths.json).
_PROBE_REPRO: dict[str, dict] = {
    "dinov2": {"benchmark": "ForenSynths-13", "paper": "~97 mAP (CLIP-probe ref.)", "ours": "87.37 mAP / 87.41 AUROC",
               "match": "ours-only", "note": "frozen DINOv2 ViT-L/14 probe (8-GAN subset: 90.61 mAP)."},
    "dinov3": {"benchmark": "ForenSynths-13", "paper": "~97 mAP (CLIP-probe ref.)", "ours": "89.02 mAP / 88.23 AUROC",
               "match": "ours-only", "note": "frozen DINOv3 ViT-L/16 probe (8-GAN subset: 91.23 mAP)."},
    "adapted_dino": {"benchmark": "ForenSynths-13", "paper": "— (SynthScope variant)", "ours": "95.35 mAP / 94.81 AUROC",
                     "match": "ours-only", "note": "DINOv2 + per-block bottleneck adapters (8-GAN subset: 98.78 mAP)."},
}

# Each model's headline = its primary paper benchmark, derived from REPRO_FULL (single source of truth).
_PRIMARY = {
    "co_spy": "Co-Spy-Bench", "fat_former": "4-class", "npr": "Table 1", "aide": "GenImage",
    "clip": "UniversalFakeDetect", "cnn": "ForenSynths", "safe": "GenImage", "spai": "GenImage",
    "vib_net": "Diffusion-source (SDv1.4-trained), AP", "pgc": "GenImage", "d3": "GenVideo",
    "restrav": "GenVidBench", "stall": "GenVideo",
}
_MATCH = {"reproduced": "match", "official": "match", "below": "below",
          "reference": "partial", "missing": "pending"}


def _headline(model: str):
    rows = REPRO_FULL.get(model, [])
    if not rows:
        return None
    sub = _PRIMARY.get(model, "")
    row = next((r for r in rows if sub.lower() in r["benchmark"].lower()), rows[0])
    return {"benchmark": row["benchmark"], "paper": row["paper"], "ours": row["ours"],
            "match": _MATCH.get(row["status"], row["status"]), "note": row.get("note", "")}


REPRO: dict[str, dict] = dict(_PROBE_REPRO)
for _m in REPRO_FULL:
    _h = _headline(_m)
    if _h:
        REPRO[_m] = _h


def list_specs() -> list[ModelSpec]:
    return list(MODELS.values())


def get_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(MODELS)}")
    return MODELS[key]
