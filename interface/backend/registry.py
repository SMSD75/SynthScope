"""Model registry: the single source of truth for every detector the interface can run.

For each detector we record:
  * how to compose its AGID config (Hydra overrides, exactly like ``main.py``),
  * its evaluation preprocessing (matched to the training/eval recipe or the verified
    official-weights eval script under repro/),
  * which checkpoint file backs it and how that checkpoint loads (strict AGID vs. an
    official-weights key remap), and
  * how to turn a forward pass into a fake-probability.

Label convention (dataset.py:_get_label): index 0 = real, index 1 = fake.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Normalization constants (mirror transformations.py).
# ---------------------------------------------------------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # == normalize_dinov2
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV3_MEAN = (0.430, 0.411, 0.296)
DINOV3_STD = (0.213, 0.156, 0.143)


def _norm(mean, std):
    return T.Normalize(mean=list(mean), std=list(std))


def tf_resize_crop(resize: int, crop: int, mean=None, std=None) -> T.Compose:
    """Resize shorter side -> ``resize``, center-crop ``crop``, ToTensor, optional Normalize."""
    steps = [T.Resize(resize), T.CenterCrop(crop), T.ToTensor()]
    if mean is not None:
        steps.append(_norm(mean, std))
    return T.Compose(steps)


def tf_square_resize(resize: int, crop: Optional[int], mean=None, std=None) -> T.Compose:
    """Square Resize((resize,resize)), optional center-crop, ToTensor, optional Normalize.

    Matches the official FatFormer/NPR/AIDE eval preprocessing (a square resize, not a
    shorter-side resize)."""
    steps = [T.Resize((resize, resize))]
    if crop is not None:
        steps.append(T.CenterCrop(crop))
    steps.append(T.ToTensor())
    if mean is not None:
        steps.append(_norm(mean, std))
    return T.Compose(steps)


# ---------------------------------------------------------------------------
# Official-weights -> AGID state_dict remaps (copied verbatim from the verified repro
# eval scripts in /hddstore/ssalehi/AGID/repro/{fatformer,aide,npr}_eval.py).
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
    out = {}
    for k, v in sd.items():
        if k.startswith("model_min.fc.") or k.startswith("model_max.fc."):
            continue  # AGID uses fc=Identity
        if k.startswith("hpf.hpf."):
            nk = "hpf.conv." + k[len("hpf.hpf."):]
        elif k.startswith("openclip_convnext_xxl."):
            nk = "convnext." + k[len("openclip_convnext_xxl."):]
        elif k.startswith("convnext_proj.0."):
            nk = "semantic_proj." + k[len("convnext_proj.0."):]
        elif k.startswith("fc.fc1."):
            nk = "head.0." + k[len("fc.fc1."):]
        elif k.startswith("fc.fc2."):
            nk = "head.2." + k[len("fc.fc2."):]
        else:
            nk = k
        out[nk] = v
    return out


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


@dataclass
class ModelSpec:
    key: str                       # menu id / dispatch handle
    name: str                      # human-friendly display name
    paper: str                     # short citation
    modality: str                  # "image" (also runs frame-by-frame on video) or "video"
    overrides: list[str]           # Hydra overrides for config_utils.compose_config
    predict_kind: str              # ce | bce | bce_npr | vib | video
    checkpoint: Optional[str]      # filename inside CHECKPOINTS_DIR (None = no weights file)
    ckpt_kind: str                 # agid | official_fatformer | official_aide | official_npr | none
    transform: Optional[T.Compose] = None  # image preprocessing (None for pure-video models)
    # Video-clip sampling (used when modality == "video").
    num_frames: int = 16
    fps: Optional[float] = None
    frame_size: int = 224
    video_norm: Optional[tuple] = None  # (mean, std) applied to the decoded [0,1] clip
    window_center: bool = False         # center the sampling window (ReStraV) vs start (D3/STALL)
    resize_interp: str = "bilinear"     # frame resize interpolation ("bicubic" for ReStraV)
    notes: str = ""
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
    modality="image", predict_kind="ce", checkpoint="clip.pth", ckpt_kind="agid",
    overrides=["ModelConfig.model_type=clip", "ModelConfig.backbone_name=CLIP:ViT-L/14",
               "ModelConfig.use_adapter=false", "ModelConfig.bottleneck_dim=64",
               "ModelConfig.freeze_backbone=true"],
    transform=tf_resize_crop(256, 224, CLIP_MEAN, CLIP_STD),
    notes="Frozen CLIP ViT-L/14 features + linear real/fake head (trained on ProGAN).",
))
_add(ModelSpec(
    key="cnn", name="CNNDetection (ResNet-50)", paper="Wang et al., CVPR'20",
    modality="image", predict_kind="ce", checkpoint="cnn.pth", ckpt_kind="agid",
    overrides=["ModelConfig.model_type=cnn", "ModelConfig.freeze_backbone=false"],
    transform=tf_resize_crop(256, 224, IMAGENET_MEAN, IMAGENET_STD),
    notes="ImageNet ResNet-50 fine-tuned as a real/fake classifier (trained on ProGAN).",
))
_add(ModelSpec(
    key="dinov2", name="DINOv2 probe", paper="Oquab et al., 2023",
    modality="image", predict_kind="ce", checkpoint="dinov2.pth", ckpt_kind="agid",
    overrides=["ModelConfig.model_type=dinov2", "ModelConfig.backbone_name=dinov2_vitl14",
               "ModelConfig.use_adapter=false", "ModelConfig.bottleneck_dim=64",
               "ModelConfig.freeze_backbone=true"],
    transform=tf_resize_crop(256, 224, IMAGENET_MEAN, IMAGENET_STD),
    notes="Frozen DINOv2 ViT-L/14 + linear head (trained on ProGAN).",
))
_add(ModelSpec(
    key="dinov3", name="DINOv3 probe", paper="Meta AI, 2025",
    modality="image", predict_kind="ce", checkpoint="dinov3.pth", ckpt_kind="agid",
    overrides=["ModelConfig.model_type=dinov3", "ModelConfig.backbone_name=dinov3_vitl16",
               "ModelConfig.use_adapter=false", "ModelConfig.bottleneck_dim=64",
               "ModelConfig.freeze_backbone=true"],
    transform=tf_resize_crop(256, 256, DINOV3_MEAN, DINOV3_STD),
    notes="Frozen DINOv3 ViT-L/16 + linear head. Requires DINOv3 backbone weights.",
))
_add(ModelSpec(
    key="adapted_dino", name="Adapter-tuned DINOv2", paper="AGID",
    modality="image", predict_kind="ce", checkpoint="adapted_dino.pth", ckpt_kind="agid",
    overrides=["ModelConfig.model_type=adapted_dino", "ModelConfig.backbone_name=dinov2_vitl14",
               "ModelConfig.use_adapter=true", "ModelConfig.bottleneck_dim=64",
               "ModelConfig.freeze_backbone=true"],
    transform=tf_resize_crop(256, 224, IMAGENET_MEAN, IMAGENET_STD),
    notes="DINOv2 with per-block bottleneck adapters; backbone frozen.",
))
_add(ModelSpec(
    key="safe", name="SAFE", paper="Li et al., KDD'25",
    modality="image", predict_kind="ce", checkpoint="safe.pth", ckpt_kind="agid",
    overrides=["model@ModelConfig=image/safe"],
    transform=tf_resize_crop(256, 256, None, None),
    notes="High-frequency (DWT) artifact detector trained from scratch. Raw [0,1] input.",
))
_add(ModelSpec(
    key="fat_former", name="FatFormer", paper="Liu et al., CVPR'24",
    modality="image", predict_kind="ce", checkpoint="fat_former.pth",
    ckpt_kind="official_fatformer", remap=remap_fatformer,
    # The released checkpoint has FAA at blocks [7,15,23] => num_vit_adapter=3 (not the yaml's 8).
    overrides=["model@ModelConfig=image/fat_former", "ModelConfig.num_vit_adapter=3"],
    transform=tf_square_resize(256, 224, IMAGENET_MEAN, IMAGENET_STD),
    notes="Frozen CLIP + forgery-aware adapters + learnable prompts. Official ProGAN weights.",
))
_add(ModelSpec(
    key="aide", name="AIDE", paper="Yan et al., ICLR'25",
    modality="image", predict_kind="ce", checkpoint="aide.pth",
    ckpt_kind="official_aide", remap=remap_aide,
    # AGID-trained faithful AIDE: build the SAME frozen laion2b ConvNeXt-XXL used in training (the
    # checkpoint stores only trainable params); resnet built random then overwritten by the ckpt.
    overrides=["model@ModelConfig=image/aide", "ModelConfig.resnet_pretrained=false"],
    transform=tf_square_resize(256, None, None, None),
    notes="ConvNeXt semantic + DCT/SRM low-level cues. Raw [0,1] 256 in; model applies CLIP norm "
          "(semantic) + ImageNet norm (SRM patch branch) internally, per official AIDE preprocessing.",
))
_add(ModelSpec(
    key="spai", name="SPAI", paper="Karageorgiou et al., CVPR'25",
    modality="image", predict_kind="bce", checkpoint="spai.pth", ckpt_kind="agid",
    overrides=["model@ModelConfig=image/spai"],
    transform=tf_resize_crop(448, 448, None, None),
    notes="Spectral learning (MFM ViT-B/16) + Spectral Context Attention over a 2x2 grid of 224 patches "
          "(448 raw [0,1] input). Set ModelConfig.use_sca=false + a 224 transform for the old single-patch model.",
))
_add(ModelSpec(
    key="vib_net", name="VIB-Net", paper="Zhang et al., CVPR'25",
    modality="image", predict_kind="vib", checkpoint="vib_net.pth", ckpt_kind="agid",
    overrides=["model@ModelConfig=image/vib_net"],
    transform=tf_resize_crop(256, 224, CLIP_MEAN, CLIP_STD),
    notes="Variational information bottleneck on frozen CLIP features (trained on GenImage).",
))
_add(ModelSpec(
    key="co_spy", name="CO-SPY", paper="Cheng et al., CVPR'25",
    modality="image", predict_kind="bce", checkpoint="co_spy.pth", ckpt_kind="agid",
    overrides=["model@ModelConfig=image/co_spy"],
    transform=tf_resize_crop(384, 384, None, None),  # raw [0,1]; per-branch norm inside the model
    notes="SigLIP semantic + SD-VAE-residual artifact streams, feature fusion (trained on ProGAN).",
))
_add(ModelSpec(
    key="pgc", name="PGC", paper="Zhou et al., 2026",
    modality="image", predict_kind="bce", checkpoint="pgc.pth", ckpt_kind="agid",
    overrides=["model@ModelConfig=image/pgc"],
    transform=tf_resize_crop(256, 224, IMAGENET_MEAN, IMAGENET_STD),
    notes="LoRA-DINOv2 RGB + YCbCr-residual CNN, peak-guided calibration (residual computed in-model).",
))
_add(ModelSpec(
    key="npr", name="NPR", paper="Tan et al., CVPR'24",
    modality="image", predict_kind="bce_npr", checkpoint="npr.pth",
    ckpt_kind="official_npr", remap=remap_npr,
    overrides=["model@ModelConfig=image/npr"],
    transform=tf_square_resize(256, None, IMAGENET_MEAN, IMAGENET_STD),
    notes="Neighboring-pixel-residual ResNet. Official ProGAN weights.",
))

# --- Video detectors ---------------------------------------------------------
_add(ModelSpec(
    key="d3", name="D3 (video)", paper="ICCV'25",
    modality="video", predict_kind="video", checkpoint=None, ckpt_kind="none",
    overrides=["model@ModelConfig=video/d3", "ModelConfig.backbone_name=clip",
               "ModelConfig.first_order=l2"],
    num_frames=16, fps=8.0, frame_size=224, video_norm=(CLIP_MEAN, CLIP_STD),
    notes="Training-free: frozen CLIP per-frame features; 2nd-order trajectory std is the score.",
))
_add(ModelSpec(
    key="restrav", name="ReStraV (video)", paper="2025",
    modality="video", predict_kind="video", checkpoint="restrav.pth", ckpt_kind="agid",
    overrides=["model@ModelConfig=video/restrav", "ModelConfig.backbone_name=dinov2_vits14"],
    num_frames=24, fps=None, frame_size=224, video_norm=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    window_center=True, resize_interp="bicubic",
    notes="Frozen DINOv2 ViT-S/14 trajectory-geometry + standardized MLP. Raw [0,1], center window + BICUBIC (official ReStraV).",
))
# DINOv3 ViT-L/16 gated weights (downloaded locally). STALL is training-free but needs them.
DINOV3_WEIGHTS = "/hddstore/ssalehi/AGID/dinov3_checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
_add(ModelSpec(
    key="stall", name="STALL (video)", paper="Ben Hayun et al., CVPR'26",
    modality="video", predict_kind="video", checkpoint=None, ckpt_kind="none",
    overrides=["model@ModelConfig=video/stall", f"ModelConfig.dino_weights={DINOV3_WEIGHTS}"],
    num_frames=16, fps=None, frame_size=224, video_norm=(IMAGENET_MEAN, IMAGENET_STD),
    notes="Training-free spatial+temporal likelihood vs real-video stats (DINOv3 ViT-L/16, VATEX calibration).",
))


# ---------------------------------------------------------------------------
# Paper-vs-reproduced numbers, shown per model in the UI. Every "ours" value traces to a logged
# eval; official-weights rows are cross-checked against the paper, from-scratch/probe rows are our
# reproductions. Honest caveats are kept in `note` (see README repro section + session findings).
# ---------------------------------------------------------------------------
REPRO: dict[str, dict] = {
    "fat_former": {"benchmark": "ForenSynths-8", "paper": "98.4 acc / 99.7 mAP",
                   "ours": "98.39 acc / 99.70 mAP", "match": "match",
                   "note": "official ProGAN weights — reproduces the paper."},
    "co_spy": {"benchmark": "Co-Spy-Bench (110 subsets)", "paper": "87.1 acc / 96.0 AP",
               "ours": "official 83.88 acc / 86.65 mAP; AGID 56.6 / 72.81", "match": "partial",
               "note": "coverage 66→110 (added Flickr30k/COCO proxy reals); 94.6 mAP on the 3 shipped-real sources (≈paper); AIGCDetect official 94.59. An 8-epoch from-scratch AGID retrain reproduced the same 72.7 mAP ceiling, confirming the gap is architectural (single backbone vs official SigLIP-384 + SD-VAE fusion), not the training recipe."},
    "npr": {"benchmark": "ForenSynths-8 / GANGen", "paper": "92.5 acc / 96.1 mAP (FS); 96.6 (GANGen)",
            "ours": "91.83 / 92.81 (FS-8); 94.81 / 96.49 (GANGen)",
            "match": "match", "note": "official ProGAN weights. GANGen fixed via --native-res eval (resize injects NPR's own up-sampling artifact: 65→96 mAP)."},
    "aide": {"benchmark": "GenImage (SD-v1.4 train)", "paper": "86.9 acc", "ours": "official 85.91 acc / 97.12 mAP; AGID-trained 91.75 mAP",
             "match": "match", "note": "faithful ConvNeXt-XXL + 5-view DCT, sd14_train ckpt. ForenSynths 94.40, AIGCDetect 97.94 (paper 92.8). AGID-trained from scratch on GenImage reaches 91.75 mAP in-domain; the ProGAN retrain for ForenSynths diverged, so official 94.40 is the ForenSynths reference."},
    "clip": {"benchmark": "ForenSynths", "paper": "~98 mAP", "ours": "97.2 mAP / 97.0 AUROC",
             "match": "match", "note": "frozen CLIP ViT-L/14 probe (ProGAN-trained)."},
    "cnn": {"benchmark": "ForenSynths", "paper": "~96–98 mAP", "ours": "96.8 mAP / 96.6 AUROC",
            "match": "match", "note": "ResNet-50 fine-tune (ProGAN)."},
    "dinov2": {"benchmark": "ForenSynths", "paper": "~97 mAP (CLIP-probe ref.)", "ours": "95.03 mAP / 94.86 AUROC",
               "match": "close", "note": "frozen DINOv2 ViT-L/14 probe."},
    "dinov3": {"benchmark": "ForenSynths-8", "paper": "~97 mAP (CLIP-probe ref.)", "ours": "93.68 mAP / 92.92 AUROC",
               "match": "close", "note": "frozen DINOv3 ViT-L/16 probe (8 ForenSynths gens, n=400/class)."},
    "adapted_dino": {"benchmark": "ForenSynths-8", "paper": "— (AGID variant)", "ours": "98.96 mAP / 98.77 AUROC",
                     "match": "ours-only", "note": "DINOv2 + per-block bottleneck adapters (8 gens, n=400/class)."},
    "safe": {"benchmark": "ForenSynths", "paper": "— (no comparable number)", "ours": "94.13 mAP / 93.68 AUROC",
             "match": "ours-only", "note": "trained from scratch on ProGAN (paper uses AIGIBench)."},
    "spai": {"benchmark": "GenImage", "paper": "~91 AUC (any-res set)", "ours": "official 91.33 / AGID 88.80 mAP (retrained)",
             "match": "gap-closed", "note": "GAP-CLOSING: ported the official any-resolution Spectral-Context-Attention into the AGID class + fixed a deleted-reals data bug (training had 0 reals -> all-fake 50.0 collapse) -> GenImage 80.75->88.80 mAP (promoted to spai.pth); residual ~2.5 mAP is re-impl divergence."},
    "vib_net": {"benchmark": "GenImage SD-v1.4", "paper": "95.5 AP", "ours": "83.26 mAP (retrained)",
                "match": "below", "note": "no official weights released. CLIP-backbone retrain = 83.26 mAP; a FLAVA-backbone retrain (the paper's headline backbone) did not converge (val loss stayed at chance), so the residual gap is architectural and documented, not closed."},
    "pgc": {"benchmark": "GenImage", "paper": "98.3 acc / 100.0 AP", "ours": "78.6 acc / 97.62 mAP (retrained SD-v1.4)", "match": "gap-closed",
            "note": "GAP-CLOSING: AGID retrain on the paper's SD-v1.4 source took GenImage mAP 69.68→97.62 (promoted to pgc.pth); CommGen15 73.5→98.6, UnivFD 97.3→99.0."},
    "d3": {"benchmark": "GenVideo-Val", "paper": "98.5 mAP", "ours": "89.00 mAP / 93.87 AUROC", "match": "below",
           "note": "training-free; full GenVideo-Val (10 gens) + EvalCrafter 91.2 mAP. CLIP-B/16 encoder vs paper XCLIP accounts for the ~7-9 mAP gap."},
    "restrav": {"benchmark": "GenVideo-Val", "paper": "in-dist ~high-90s; unseen-gen 97.3 mAP / Sora 92.9",
                "ours": "in-dist AUROC ~1.00; unseen-gen ~1.00; cross-dataset ~0.48–0.72 (unstable)",
                "match": "partial", "note": "in-dist & paper-style unseen-generator reproduce; cross-DATASET (real-source shift) is unstable — deployed restrav_data/test ≈ AUROC 0.48 (chance)."},
    "stall": {"benchmark": "GenVideo-Val", "paper": "~80 AUC", "ours": "82.38 mAP / 83.51 AUROC",
              "match": "match", "note": "training-free; DINOv3 + calibration recomputed on GenVideo reals. VideoFeedback 83.1 mAP, ComGenVid 82.6 mAP (all ≈ paper)."},
}


# Full per-method benchmark table (ALL reported benchmarks: paper vs ours + status), generated by
# build_repro_full.py from repro/benchmark_specs/ + repro/bench_results/. Shown per selected method in the UI.
import json as _json, pathlib as _pathlib
try:
    REPRO_FULL = _json.loads((_pathlib.Path(__file__).resolve().parent / "repro_full.json").read_text())
except Exception:
    REPRO_FULL = {}


def list_specs() -> list[ModelSpec]:
    return list(MODELS.values())


def get_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(MODELS)}")
    return MODELS[key]
