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
import functools
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


def tf_crop(crop: int, mean=None, std=None) -> T.Compose:
    """Center-crop ``crop`` with NO resize, ToTensor, optional Normalize. Official SAFE
    preprocessing (scripts/eval.sh: --input_size 256 --transform_mode 'crop', no normalization) —
    a resize would alter the high-frequency signal SAFE keys on."""
    steps = [T.CenterCrop(crop), T.ToTensor()]
    if mean is not None:
        steps.append(_norm(mean, std))
    return T.Compose(steps)


# AIDE's official native-resolution 5-view preprocessing (DCT patch selection on the native-size
# image). Importing this is light (no open_clip at module level); the heavy net is built lazily.
from models.image.aide.official_arch import AIDEFiveView  # noqa: E402
aide_five_view = AIDEFiveView()


def _cap_long_side(im, max_side: int):
    """Downscale a PIL image (keeping aspect, min side 224) only if its long side exceeds
    ``max_side`` — otherwise return it untouched at native resolution."""
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(int(w * s), 224), max(int(h * s), 224)))
    return im


def tf_native_cap(max_side: int = 1024, mean=None, std=None) -> T.Compose:
    """Native/arbitrary-resolution preprocessing: keep the image at its NATIVE size, only
    downscaling (keeping aspect, min side 224) when the long side exceeds ``max_side`` to bound
    memory. ToTensor (raw [0,1]), optional Normalize. This is SPAI's official any-resolution
    path — a fixed Resize/CenterCrop would destroy the native-res spectral statistics it learns.
    NB: produces variable-size tensors, so a ``native_res=True`` spec must be scored ONE image at
    a time (bs=1); the web ``predict_image`` already does this, and bench_eval forces bs=1."""
    steps = [T.Lambda(functools.partial(_cap_long_side, max_side=max_side)), T.ToTensor()]
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
    """Official AIDE checkpoint -> AGIDAIDEModel. The AGID class now wraps the GENUINE arch as
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


def remap_safe(sd: dict) -> dict:
    """Official SAFE (Ouxiang-Li/SAFE) → AGIDSAFEModel. The custom ResNet (conv1/bn1/layer1/layer2)
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
    ckpt_kind: str                 # agid | official_fatformer | official_aide | official_npr | none
    transform: Optional[T.Compose] = None  # image preprocessing (None for pure-video models)
    # Video-clip sampling (used when modality == "video").
    num_frames: int = 16
    fps: Optional[float] = None
    frame_size: int = 224
    video_norm: Optional[tuple] = None  # (mean, std) applied to the decoded [0,1] clip
    window_center: bool = False         # center the sampling window (ReStraV) vs start (D3/STALL)
    resize_interp: str = "bilinear"     # frame resize interpolation ("bicubic" for ReStraV)
    edge_crop: float = 0.0              # center edge-crop fraction per frame before resize (D3: 0.1)
    notes: str = ""
    # Which weights back the DEFAULT inference checkpoint, surfaced in the web UI:
    #   "official"      = authors' released weights (loaded into the AGID class)
    #   "agid_trained"  = our from-scratch AGID checkpoint
    #   "training_free" = no weights (frozen-feature / training-free detector)
    weights: str = "agid_trained"
    # Score at native/arbitrary resolution one image at a time (bs=1) instead of stacking a
    # fixed-size batch — required by SPAI's any-res spectral path (variable patch count per image).
    native_res: bool = False
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
    modality="image", predict_kind="ce", checkpoint="safe.pth",
    ckpt_kind="agid", remap=None, weights="agid_trained",
    overrides=["model@ModelConfig=image/safe"],
    transform=tf_crop(256),
    notes="High-frequency (DWT bior1.3 HH-band) artifact detector. Serves the AGID-TRAINED checkpoint "
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
    transform=tf_square_resize(256, 224, IMAGENET_MEAN, IMAGENET_STD),
    notes="Frozen CLIP + forgery-aware adapters + learnable prompts. Official ProGAN weights.",
))
_add(ModelSpec(
    key="aide", name="AIDE", paper="Yan et al., ICLR'25",
    modality="image", predict_kind="ce", checkpoint="aide.pth",
    ckpt_kind="official_aide", remap=remap_aide, weights="official",
    # Genuine 5-view AIDE arch; the official checkpoint supplies the ConvNeXt trunk (build it random
    # via convnext_pretrained=null to skip the 4.8GB download, then load the ckpt strict).
    overrides=["model@ModelConfig=image/aide", "ModelConfig.convnext_pretrained=null"],
    transform=aide_five_view,  # NATIVE-res DCT 5-view stack [5,3,256,256] (official preprocessing)
    notes="ConvNeXt-XXL semantic + DCT/SRM low-level cues. The DCT patch selector runs at NATIVE "
          "resolution in preprocessing (resizing to 256 before the DCT costs ~12 mAP on the same "
          "weights); only the 4 selected 32x32 patches + the full image are resized to 256.",
))
_add(ModelSpec(
    key="spai", name="SPAI", paper="Karageorgiou et al., CVPR'25",
    modality="image", predict_kind="bce", checkpoint="spai.pth", ckpt_kind="agid",
    weights="official",  # shipped spai.pth = official SPAI weights remapped into the AGID class
    overrides=["model@ModelConfig=image/spai"],
    transform=tf_native_cap(1024),  # NATIVE resolution (raw [0,1], cap long side 1024) — the faithful any-res path
    native_res=True,                # score one image at a time (variable patch count)
    notes="Spectral learning (MFM ViT-B/16) + Spectral Context Attention at NATIVE/arbitrary resolution: each "
          "image is patchified into a variable 224-grid (five_crop fallback for tiny images). Serves the "
          "official SPAI weights remapped into the AGID class; reproduces the paper (GenImage-8 90.74 / "
          "Synthbuster 96.07 mAP). A fixed 448 resize/crop destroys the native-res spectral statistics.",
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
    weights="official",  # official PGC ProGAN weights load remap-free (FS 96.25 / hard-gens 94.35 vs our from-scratch 75.87); deploy PGC_train_progan_ckpt.pth as pgc.pth
    overrides=["model@ModelConfig=image/pgc"],
    transform=tf_resize_crop(256, 224, IMAGENET_MEAN, IMAGENET_STD),
    notes="LoRA-DINOv2 RGB + YCbCr-residual CNN, peak-guided calibration. Uses OFFICIAL released weights "
          "(better than our from-scratch PGC on the ForenSynths hard generators; see repro/investigate/pgc_gap_investigation.md).",
))
_add(ModelSpec(
    key="npr", name="NPR", paper="Tan et al., CVPR'24",
    modality="image", predict_kind="bce_npr", checkpoint="npr.pth",
    ckpt_kind="official_npr", remap=remap_npr, weights="official",
    overrides=["model@ModelConfig=image/npr"],
    transform=tf_square_resize(256, None, IMAGENET_MEAN, IMAGENET_STD),
    notes="Neighboring-pixel-residual ResNet. Official ProGAN weights.",
))

# --- Video detectors ---------------------------------------------------------
_add(ModelSpec(
    key="d3", name="D3 (video)", paper="ICCV'25",
    modality="video", predict_kind="video", checkpoint=None, ckpt_kind="none", weights="training_free",
    overrides=["model@ModelConfig=video/d3", "ModelConfig.backbone_name=xclip",
               "ModelConfig.first_order=l2"],
    num_frames=16, fps=8.0, frame_size=224, video_norm=(CLIP_MEAN, CLIP_STD), edge_crop=0.1,
    notes="Training-free: frozen X-CLIP-B/16 per-frame features; 2nd-order trajectory std is the score. "
          "Official preprocessing: 10% center edge-crop per frame before resize.",
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
    modality="video", predict_kind="video", checkpoint=None, ckpt_kind="none", weights="training_free",
    overrides=["model@ModelConfig=video/stall", f"ModelConfig.dino_weights={DINOV3_WEIGHTS}"],
    num_frames=16, fps=None, frame_size=224, video_norm=(IMAGENET_MEAN, IMAGENET_STD),
    notes="Training-free spatial+temporal likelihood vs real-video stats (DINOv3 ViT-L/16, VATEX calibration).",
))


# ---------------------------------------------------------------------------
# Paper-vs-reproduced numbers, shown per model in the UI. Every "ours" value traces to a logged
# eval; official-weights rows are cross-checked against the paper, from-scratch/probe rows are our
# reproductions. Honest caveats are kept in `note` (see README repro section + session findings).
# ---------------------------------------------------------------------------
# Full per-method benchmark table (ALL reported benchmarks: paper vs AGID-trained vs official-weights
# + status), generated by build_repro_full.py from repro/benchmark_specs/ + repro/bench_results/.
# Loaded FIRST so the headline cards below DERIVE from the same data and can never drift out of sync.
import json as _json, pathlib as _pathlib
try:
    REPRO_FULL = _json.loads((_pathlib.Path(__file__).resolve().parent / "repro_full.json").read_text())
except Exception:
    REPRO_FULL = {}

# AGID-only probe variants (no paper benchmark table of their own) — hand-kept headline cards.
_PROBE_REPRO: dict[str, dict] = {
    "dinov2": {"benchmark": "ForenSynths", "paper": "~97 mAP (CLIP-probe ref.)", "ours": "95.03 mAP / 94.86 AUROC",
               "match": "close", "note": "frozen DINOv2 ViT-L/14 probe."},
    "dinov3": {"benchmark": "ForenSynths-8", "paper": "~97 mAP (CLIP-probe ref.)", "ours": "93.68 mAP / 92.92 AUROC",
               "match": "close", "note": "frozen DINOv3 ViT-L/16 probe (8 ForenSynths gens, n=400/class)."},
    "adapted_dino": {"benchmark": "ForenSynths-8", "paper": "— (AGID variant)", "ours": "98.96 mAP / 98.77 AUROC",
                     "match": "ours-only", "note": "DINOv2 + per-block bottleneck adapters (8 gens, n=400/class)."},
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
