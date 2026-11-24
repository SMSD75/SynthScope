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

from models.image.adb_full.adb_full import ADB_FULL, FusionConfig
from models.image.adb_vlm.adb_vlm import ADB_VLM
from models.image.fakevlm.fakevlm import FAKEVLM
from models.image.veritaspp.veritaspp import VERITASPP
from models.video.wevis.wevis import WEVIS, WeVisConfig
from models.vlm.vlm import VLMConfig
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
    DataParallel ``module.`` prefix and add the ``net.`` prefix; every official key loads strict. A SynthScope-TRAINED
    checkpoint (the trainer saves the model's own state dict) already carries ``net.`` and keeps it, so
    ``bench_eval --model aide --ckpt <trained>`` loads it; no official key starts with ``net.``."""
    out = {}
    for k, v in sd.items():
        k = k[len("module."):] if k.startswith("module.") else k
        out[k if k.startswith("net.") else "net." + k] = v
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


def remap_vina(sd: dict) -> dict:
    """Official VINA checkpoint ({"model_state_dict": ...}) -> SynthScopeVINAModel: the official
    util.load_checkpoint rule (unwrap ``model_state_dict``, drop ``module.``); the names are the official ones (the
    block prefix is renamed by a load hook)."""
    sd = sd.get("model_state_dict", sd)
    return {k.replace("module.", ""): v for k, v in sd.items()}


def remap_waverep(sd: dict) -> dict:
    """Official WaveRep checkpoint -> SynthScopeWaveRepModel: the released weights_dinov2_G{1,4}.ckpt are flat timm
    state dicts; the official create_model also accepts a Lightning ``{"state_dict": {"model.<timm key>": ...}}`` (it keeps
    the keys starting with ``model`` and drops that prefix). Either way the timm keys go under ``model.``."""
    if "state_dict" in sd:
        sd = {k[6:]: v for k, v in sd["state_dict"].items() if k.startswith("model")}
    return {"model." + k: v for k, v in sd.items()}


def remap_nsg_vd(sd: dict) -> dict:
    """Official NSG-VD checkpoint ({standard,unbalance}-{Pika,SEINE}-{mp,d}.pth: a ``deep_MMD`` state dict, the kernel
    parameters + buffers and the TALL discriminator under ``net.``) -> SynthScopeNSGVDModel, which holds the official
    ``deep_MMD`` as ``mmd``: add the ``mmd.`` prefix (test_dMMD.py loads the same dict strictly)."""
    return {"mmd." + k: v for k, v in sd.items()}


def remap_probe(sd: dict) -> dict:
    """Official PROBE checkpoint ({"model_state_dict": ...}) -> SynthScopePROBEModel / SynthScopePROBEResNetModel: the
    official evaluate_dino.py / evaluate_resnet.py load ``checkpoint['model_state_dict']`` strictly; the names are the
    official ones."""
    return sd.get("model_state_dict", sd)


def remap_pgc(sd: dict) -> dict:
    """Official PGC checkpoint -> SynthScopePGCModel: the official ``PGCNetwork`` keeps the DINOv2 backbone inside its
    ``RgbStream`` wrapper (``rgb_stream.backbone.*``); the SynthScope class holds it as ``self.backbone``. Strip the
    wrapper prefix. ``residual_stream`` / ``pgcm`` / ``classifier`` and the LoRA layout (``original_layer`` +
    ``lora.lora_{A,B}`` on ``mlp.fc1`` / ``mlp.fc2``) are already identical. SynthScope-trained checkpoints
    (``backbone.*``) pass through unchanged."""
    return {(k[len("rgb_stream."):] if k.startswith("rgb_stream.") else k): v for k, v in sd.items()}


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
    predict_kind: str              # ce | bce | bce_npr | vib | video | vlm | video_agent | fusion
    checkpoint: Optional[str]      # filename inside CHECKPOINTS_DIR (None = no weights file; a folder for "vlm")
    ckpt_kind: str                 # synthscope | official_* (strict) | none
    transform: Optional[AIGITransformations] = None   # image preprocessing (None for pure-video and VLM models)
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
    # "ffmpeg_fps_center" (VINA): ffmpeg writes JPEG frames at ``fps``, the central ``num_frames`` are read back and
    # each goes through ``transform`` (the image eval preprocessing) instead of the clip settings above.
    # "cv2_frames" (WaveRep): the first ``num_frames`` frames OpenCV decodes (0: every frame), each through
    # ``transform`` and scored on its own; P(fake) = sigmoid of the mean frame logit (Engine.video_frame_logits).
    # "ffmpeg_select_jpeg" (NSG-VD): ffmpeg writes every k-th frame (k = frame count // num_frames) as JPEGs, the first
    # ``num_frames`` are read back and each goes through ``transform``.
    video_sampler: str = ""
    # video output -> P(fake): softmax[:, 1] of [1, 2] logits, "sigmoid", or "mmd" (NSG-VD: the model's raw statistic
    # through ``model.fake_probability``; the statistic is reported as ``logit``).
    video_prob: str = "softmax"
    # Extra files the model reads from the checkpoint folder: {Hydra config key: file name in CHECKPOINTS_DIR}; the
    # engine passes their paths as overrides and reports the detector unavailable while one is missing.
    aux_files: dict = field(default_factory=dict)
    # A detector whose official protocol depends on the ffmpeg build pins it (NSG-VD: JPEG encoder + frame selection):
    # ``ffmpeg_version`` is the required "major.minor" and ``ffmpeg_env`` an environment variable naming the binary for
    # this detector only. The engine resolves the binary from ``ffmpeg_env``, else as every other detector
    # ($SYNTHSCOPE_FFMPEG, ffmpeg on the PATH, imageio-ffmpeg), records its version in each result and warns when it is
    # not the pinned one. Empty for every other detector (their behaviour is unchanged).
    ffmpeg_version: str = ""
    ffmpeg_env: str = ""
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
    # predict_kind "vlm": a released vision-language model that writes a rationale and a verdict (models/vlm/vlm.py).
    # Its preprocessing is the official image loader + the model's own processor, so ``transform`` is None and
    # ``overrides`` is empty: the config (with its model family) carries the processor, prompt, decoding, answer
    # marker, verdict words, parser and pixel cap. ``checkpoint`` is the Hugging Face folder under CHECKPOINTS_DIR
    # (ckpt_kind "official_vlm": strict ``from_pretrained``, local files only), or a LoRA adapter folder when the config
    # names a local ``base_model`` (relative to CHECKPOINTS_DIR; kept unmerged). Still images only.
    vlm: Optional[VLMConfig] = field(default=None, repr=False)
    # predict_kind "video_agent": an agentic video detector run by the official code in its own process (WeVisForensics,
    # models/video/wevis/). ``checkpoint`` is the adapter folder; the config names the base model, the OCR models, the
    # official code and the isolated runtime. No ``transform``, no ``overrides``: the official agent samples its frames.
    agent: Optional[WeVisConfig] = field(default=None, repr=False)
    # predict_kind "fusion": an official decision rule over two registered detectors (Align, Don't Blend's full method:
    # its VLM branch OR its pixel expert, models/image/adb_full/). No weights of its own (``checkpoint`` None,
    # ``ckpt_kind`` "fusion"): it is available when both components are, and each loads and caches under its own key.
    fusion: Optional[FusionConfig] = field(default=None, repr=False)

    @property
    def has_vlm(self) -> bool:
        """A VLM detector, or a fusion with one: it gets the decoded file without an RGB conversion, scores one still
        image at a time (seconds on a GPU) and no video."""
        return self.vlm is not None or self.fusion is not None

    def eval_batch_size(self, batch_size: int) -> int:
        """Batch size an evaluation loop may use: 1 for a ``native_res`` spec (its inputs differ in size and cannot be
        stacked), else ``batch_size``."""
        return 1 if self.native_res else batch_size


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
    # The released checkpoint has FAA at blocks [7,15,23] => num_vit_adapter=3 (also the yaml's value now; the official argparse default is 8).
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
    modality="image", predict_kind="bce", checkpoint="pgc.pth", ckpt_kind="official_pgc", remap=remap_pgc,
    # pgc.pth = the official PGC ProGAN checkpoint (PGC_train_progan_ckpt.pth). Its keys carry the official
    # ``rgb_stream.`` wrapper prefix, which remap_pgc strips (a SynthScope-trained pgc.pth, ``backbone.*``, is unchanged);
    # every key then loads strictly. Without the remap only the 28 head tensors loaded and all LoRA tensors were dropped
    # silently (fixed 2026-11-09, see repro/investigate/pgc_gap_investigation.md).
    weights="official",
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
# PROBE: two released detectors (Table 1 of the paper has both rows), each scored as the official evaluate_*.py does:
# the mean logit of the image's non-overlapping crop-size patches at native resolution.
_PROBE_NOTES = ("Trained with PROBE, which steers a generative model through small edits of its internal "
                "representations to find realistic images the detector gets wrong, then refines the detector on them. ")
_add(ModelSpec(
    key="probe", name="PROBE (DINOv2)", paper="Cao et al., ICML'26",
    modality="image", predict_kind="bce", checkpoint="probe.pth", ckpt_kind="official_probe", remap=remap_probe,
    weights="official",
    # probe.pth = the official DINOv2_best_model_step_34999.pth unchanged (ModelScope
    # shuinishaojiu/PROBE-AIGI-Detection, sha256 c8caa9ba...450156); it holds the full model (DINOv2-reg ViT-L/14 +
    # head), so no backbone download.
    overrides=["model@ModelConfig=image/probe"],
    transform=model_transform("image/probe"),   # official evaluate_dino.py: repeat-pad to 336, 336 patches, ImageNet
    native_res=True,                # the patch count depends on the image size: score one image at a time
    notes=_PROBE_NOTES + "DINOv2 ViT-L/14 with registers + a linear head on the CLS token, trained on the paper's "
                         "reconstruction training set (SD 2.1 reconstructions of real images) plus PROBE samples. Uses "
                         "the OFFICIAL checkpoint (licence unclear: not in the zoo or on the Space). The image is cut "
                         "into 336x336 patches at its own resolution and the patch logits are averaged.",
))
_add(ModelSpec(
    key="probe_resnet", name="PROBE (ResNet-50)", paper="Cao et al., ICML'26",
    modality="image", predict_kind="bce", checkpoint="probe_resnet.pth", ckpt_kind="official_probe",
    remap=remap_probe, weights="official",
    # probe_resnet.pth = the official ResNet50_best_model_step_19343.pth unchanged (same ModelScope repo, sha256
    # 71ef257e...20f0507c).
    overrides=["model@ModelConfig=image/probe_resnet"],
    transform=model_transform("image/probe_resnet"),   # official evaluate_resnet.py: black-pad to 224, 224 patches
    native_res=True,
    notes=_PROBE_NOTES + "ResNet-50 with a linear head, trained on GenImage SD v1.4 plus PROBE samples. Uses the "
                         "OFFICIAL checkpoint (licence unclear: not in the zoo or on the Space). The image is cut into "
                         "224x224 patches at its own resolution and the patch logits are averaged.",
))
# VINA: one model for images and videos. `vina` scores images (and, like every image detector, sampled frames of a video
# with the mean P(fake)); `vina_video` is the official video protocol, so the paper's video benchmarks run through it.
_VINA_NOTES = ("DINOv3 ViT-L/16 fine-tuned end to end with a linear head, trained on images and video frames together "
               "(video as a natural augmentation), so one model detects AI-generated images and videos. Uses the "
               "OFFICIAL checkpoint VINA_DINOv3-L_140k+DDA (licence unclear: not in the zoo or on the Space). ")
_add(ModelSpec(
    key="vina", name="VINA", paper="arXiv 2605.21977",
    modality="image", predict_kind="bce", checkpoint="vina.pth", ckpt_kind="official_vina", remap=remap_vina,
    weights="official",
    # vina.pth = the official VINA_DINOv3-L_140k+DDA.pth unchanged (HF mgiant/VINA_checkpoints); it holds the full model
    # (DINOv3 ViT-L/16 + head), so the gated DINOv3 weights are not needed. Strict load; the ViT block prefix is renamed
    # for transformers 4/5 (official_arch.py).
    overrides=["model@ModelConfig=image/vina"],
    transform=model_transform("image/vina"),   # official 'default' eval: Resize(256, bicubic), crop 256, ImageNet
    notes=_VINA_NOTES + "Images: short side resized to 256 (bicubic), center crop 256. Runs fp32 (official default).",
))
_add(ModelSpec(
    key="vina_video", name="VINA (video)", paper="arXiv 2605.21977",
    modality="video", predict_kind="video", checkpoint="vina.pth", ckpt_kind="official_vina", remap=remap_vina,
    weights="official",
    overrides=["model@ModelConfig=image/vina"],
    transform=model_transform("image/vina"),
    # official protocol: preprocess/video2frame.py (ffmpeg -vf fps=2 -> JPEG frames), then the test loader reads the
    # central select_frame_nums=1 frame(s) (configs/VINA_DINOv3.yaml); P(fake) = sigmoid(mean frame logit)
    num_frames=1, fps=2.0, frame_size=256, video_sampler="ffmpeg_fps_center", video_prob="sigmoid",
    notes=_VINA_NOTES + "Videos: the official protocol - ffmpeg writes the clip as JPEG frames at 2 fps and the "
                        "central frame is scored with the image preprocessing (needs ffmpeg).",
))
# WaveRep: two released checkpoints. The demo README says G1 is trained on "a single generator" and G4 on "four generators"
# (the demo's default); we infer the paper's Pyramid Flow detector (Tables 1-2) and its Table 5 set (+ CogVideoX 1.5,
# Allegro, Open-Sora Plan).
_WAVEREP_NOTES = ("DINOv2 ViT-B/14 with registers, fine-tuned end to end on single video frames with a forensic "
                  "augmentation: a fake frame's low-pass and horizontal / vertical wavelet bands are replaced by a "
                  "real frame's, steering the detector to the diagonal high-frequency bands. Uses the OFFICIAL checkpoint "
                  "{ckpt} (GRIP license: nonprofit use). The first 64 frames (the paper's protocol) are decoded with "
                  "OpenCV; each frame's central 504x504 pixels are scored (no resize; a smaller frame is zero-padded) "
                  "and P(fake) is the sigmoid of the mean frame logit.")
for _key, _name, _ckpt, _note in [
    ("waverep", "WaveRep (video)", "waverep.pth", "weights_dinov2_G1, trained on one generator (Pyramid Flow, as we infer)"),
    ("waverep_g4", "WaveRep-G4 (video)", "waverep_g4.pth", "weights_dinov2_G4, trained on four generators"),
]:
    _add(ModelSpec(
        key=_key, name=_name, paper="Corvi et al., NeurIPS'25",
        modality="video", predict_kind="video", checkpoint=_ckpt, ckpt_kind="official_waverep", remap=remap_waverep,
        weights="official",
        # <key>.pth = the official weights_dinov2_G{1,4}.ckpt unchanged (grip.unina.it, md5 as the demo README); a flat
        # timm state dict holding the full model, strict load.
        overrides=["model@ModelConfig=video/waverep"],
        transform=model_transform("video/waverep"),   # official create_transform: CenterCrop(504), ToTensor, ImageNet
        # official demo/utils.py: ReadVideoIteratorCV (every cv2 frame, BGR->RGB) limited to the paper's first 64
        # frames (the demo's --limit 64; without it the demo scores every frame), mean logit, expit.
        num_frames=64, fps=None, frame_size=504, video_sampler="cv2_frames", video_prob="sigmoid",
        notes=_WAVEREP_NOTES.format(ckpt=_note),
    ))
# NSG-VD: the paper's method is the multi-population (MMD-MP) objective (Sec. 3.3, Eq. 13), which the authors' README
# also recommends; its three result tables match the three released MMD-MP checkpoints: Table 1 Kinetics-400 + Pika,
# Table 2 Kinetics-400 + SEINE, Table 3 data-imbalanced (Kinetics-400 + 1,000 SEINE). Default: SEINE-mp, the best
# released AUROC among the MMD-MP checkpoints (assets/nsg-vd-test_results). The MMD-D checkpoints (*-d.pth, only in the
# released results files, no paper table) are not registered. The protocol is pinned to ffmpeg 4.2 (parity build);
# $SYNTHSCOPE_FFMPEG_NSG_VD points these keys at a binary.
_NSGVD_NOTES = ("Physics-driven: a pre-trained diffusion model (guided-diffusion 256 uncond) estimates the Normalized "
                "Spatiotemporal Gradient (spatial score over temporal change) of 8 frames; a deep-kernel MMD between the "
                "clip's NSG features and 100 real MSR-VTT reference videos (the official reference set) is the score, "
                "fake above 1. Uses the OFFICIAL checkpoint {ckpt}. Every k-th frame (k = frames // 8) is extracted as "
                "JPEG by ffmpeg and resized to 224x224 (LANCZOS); needs ffmpeg.")
for _key, _name, _ckpt, _note in [
    ("nsg_vd", "NSG-VD (video)", "nsg_vd.pth", "standard-SEINE-mp (Kinetics-400 + SEINE, MMD-MP)"),
    ("nsg_vd_pika", "NSG-VD-Pika (video)", "nsg_vd_pika.pth", "standard-Pika-mp (Kinetics-400 + Pika, MMD-MP)"),
    ("nsg_vd_unbalance", "NSG-VD-unbalanced (video)", "nsg_vd_unbalance.pth",
     "unbalance-SEINE-mp (Kinetics-400 + 1,000 SEINE, MMD-MP)"),
]:
    _add(ModelSpec(
        key=_key, name=_name, paper="Zhang et al., NeurIPS'25",
        modality="video", predict_kind="video", checkpoint=_ckpt, ckpt_kind="official_nsg_vd", remap=remap_nsg_vd,
        weights="official",
        # <key>.pth = the official ckpts/<name>.pth unchanged (a deep_MMD state dict, strict load under mmd.).
        overrides=["model@ModelConfig=video/nsg_vd"],
        aux_files={"ModelConfig.diffusion_weights": "256x256_diffusion_uncond.pt",
                   "ModelConfig.reference_path": "nsg_vd_reference_msrvtt_val100.safetensors"},
        transform=model_transform("video/nsg_vd"),   # official _preprocess_frame_from_path: LANCZOS 224, /255
        num_frames=8, fps=None, frame_size=224, video_sampler="ffmpeg_select_jpeg", video_prob="mmd",
        ffmpeg_version="4.2", ffmpeg_env="SYNTHSCOPE_FFMPEG_NSG_VD",
        notes=_NSGVD_NOTES.format(ckpt=_note),
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

# --- VLM detectors (inference only, official weights; one image at a time) ----------------------------------------
_add(ModelSpec(
    key="veritaspp", name="Veritas++", paper="Tan et al., arXiv 2607.27113",
    modality="image", predict_kind="vlm", checkpoint="veritaspp", ckpt_kind="official_vlm", weights="official",
    # veritaspp = the official HF folder EricTan7/VeritasPP @ 22ba6c6b unchanged (Apache-2.0; 4 safetensors shards,
    # 17.5 GB bf16, 750 tensors), loaded strictly into transformers' Qwen3VLForConditionalGeneration.
    overrides=[], vlm=VERITASPP, native_res=True,
    notes="Qwen3-VL-8B trained to reason about an image (a written rationale, about 400-650 tokens) and then answer "
          "<answer> real/fake </answer>. Uses the OFFICIAL weights and the official prompt; scored with greedy "
          "decoding (the authors' inference script samples at temperature 0.7). P(fake) = sigmoid of the fake-minus-real logit at the "
          "answer token, so it is near 0 or 1; the verdict is the parsed answer. Still images only. On an RTX A5000 "
          "(bf16): about 17 s per image and 18.9 GB peak up to 2048x2048, about 38 s and 23 GB at the processor's "
          "16.8 MP cap (repro/veritaspp_fidelity.json). Not on the Space.",
))
_add(ModelSpec(
    key="adb_vlm", name="Align, Don't Blend (VLM)", paper="Chen et al., ACM MM'26 (arXiv 2512.06746)",
    modality="image", predict_kind="vlm", checkpoint="adb_vlm", ckpt_kind="official_vlm", weights="official",
    # adb_vlm = the official LoRA adapter folder HF Gaffeyzz/Align-Dont-Blend-VLM @ 49975ccd unchanged (adapter_config.json +
    # adapter_model.safetensors, 392 tensors), kept unmerged on the local base folder qwen2.5-vl-7b-instruct (HF
    # Qwen/Qwen2.5-VL-7B-Instruct @ cc594898, loaded strictly into transformers' Qwen2_5_VLForConditionalGeneration).
    overrides=[], vlm=ADB_VLM, native_res=True,
    notes="The VLM branch of Align, Don't Blend: Qwen2.5-VL-7B with the authors' LoRA writes a short rationale and then "
          "answers \"This is an authentic / AI-generated image.\" inside <answer> tags. Uses the OFFICIAL adapter, prompt "
          "and greedy decoding (repetition penalty 1.05 from the base model). P(fake) = sigmoid of the AI-minus-"
          "authentic logit at the verdict word the official parser reads (the answer tag, or a tag-less final line, "
          "about 2% of outputs); mostly near 0 or 1 (2.7% of benchmark images between 0.1 and 0.9). The verdict is the "
          "parsed answer. This is the VLM branch alone; the paper's full method ORs it with the DDA pixel expert "
          "(adb_full). Still images only. Apache-2.0 code and adapter; not on the Space (a 7B VLM needs a GPU).",
))
_add(ModelSpec(
    key="adb_full", name="Align, Don't Blend (VLM + DDA)", paper="Chen et al., ACM MM'26 (arXiv 2512.06746)",
    modality="image", predict_kind="fusion", checkpoint=None, ckpt_kind="fusion", weights="official",
    # The official final decision of predict.py (GzZning/Align-Dont-Blend @ 5d83e40): adb_vlm's parsed verdict OR dda's
    # P(fake) >= 0.5. The expert is the released DDA_ckpt.pth (= dda.pth); no weights of its own.
    overrides=[], fusion=ADB_FULL, native_res=True,
    notes="The full Align, Don't Blend detector: the VLM branch (adb_vlm: Qwen2.5-VL-7B + the authors' LoRA judges "
          "semantic plausibility) OR the pixel-artifact expert (dda: the released DDA checkpoint, DINOv2 ViT-L/14 + "
          "LoRA on the central 336x336 pixels), the official rule of predict.py: fake if the VLM answers "
          "\"AI-generated\" or DDA's P(fake) >= 0.5. P(fake) = max(the VLM's P(fake) clamped to its verdict's side of "
          "0.5, DDA's P(fake)), so P(fake) >= 0.5 exactly when the official decision is fake. Both components are loaded "
          "(18.5 GB peak in bf16 on an RTX A6000, about 5 s per image; repro/adb_full_fidelity.json). Still images only. Apache-2.0; not on the Space (a 7B VLM needs a GPU).",
))

_add(ModelSpec(
    key="fakevlm", name="FakeVLM", paper="Wen et al., NeurIPS'25 (arXiv 2503.14905)",
    modality="image", predict_kind="vlm", checkpoint="fakevlm", ckpt_kind="official_vlm", weights="official",
    # fakevlm = the official HF folder lingcco/fakeVLM @ dce9e88d unchanged (no licence; 3 safetensors shards, 14.1 GB
    # bf16), loaded strictly into transformers' LlavaForConditionalGeneration; the processor from the local folder
    # llava-1.5-7b-hf_a272c74_processor (llava-hf/llava-1.5-7b-hf @ a272c74, tokenizer + processor files), as eval.py.
    overrides=[], vlm=FAKEVLM, native_res=True,
    notes="LLaVA-1.5-7B fine-tuned on FakeClue to answer \"Does the image looks real/fake?\" with \"This is a real / "
          "fake image.\" and an explanation of the artifacts. Uses the OFFICIAL weights, processor, prompt and greedy "
          "decoding (256 tokens). P(fake) = sigmoid of the fake-minus-real logit at the first \"fake\" / \"real\" "
          "word it writes; the verdict is the official parse of the answer. The image is seen as a 336x336 centre "
          "crop. On an RTX A5000 (bf16): about 0.6 s for a bare verdict (9 tokens), up to 8.3 s with a full explanation "
          "(256 tokens); 15.1 GB (repro/fakevlm_fidelity.json). GenImage is in its training data (FakeClue). Still "
          "images only. "
          "Not on the Space (no licence).",
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
_add(ModelSpec(
    key="wevis", name="WeVisForensics (video agent)", paper="Du et al., arXiv 2611.21373",
    modality="video", predict_kind="video_agent", checkpoint="wevisforensics_lora", ckpt_kind="official_wevis",
    weights="official",
    # wevisforensics_lora = the official HF folder tencent/WeVisForensics @ 9fa025a0 unchanged (Apache-2.0; five LoRA
    # adapters r64 + skill_card.json) on the local base folder qwen3.5-4b (HF Qwen/Qwen3.5-4B @ 851bf6e8, Apache-2.0); the
    # official code Tencent/WeVisForensics @ 43ae1c0 runs them (demo/wvf_local.py, unmodified) in its own process under
    # transformers 5.23.0. Every file's sha256 is pinned in models/video/wevis/manifest.json.
    overrides=[], agent=WEVIS,
    notes="An agent investigates the video: a Qwen3.5-4B investigator (LoRA) looks at a 1 fps overview (at most 16 "
          "frames), calls up to four inspectors (visual appearance, physical consistency, common sense on a time "
          "window, text/OCR at a time point; the same base with their own LoRAs) and answers real or fake. Uses the "
          "OFFICIAL code and weights, run unmodified with greedy decoding in a separate process; audio tools are off "
          "(every video is run as silent). P(fake) = sigmoid of the fake-minus-real logit at the investigator's final "
          "verdict token, so it is near 0 or 1; the verdict is the agent's answer (no valid answer: unparsed, counted "
          "wrong). About 47 s per video (20-71 s) and up to 22.4 GB on an RTX A5000 (repro/wevis_fidelity.json); long HD clips can run out of memory there. Not on the Space.",
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
    "probe": ("Native-res 336 patches, mean logit", "The image is cut into non-overlapping 336×336 patches at its own "
              "resolution (no scaling; a strip at the right / bottom edge narrower than 336 px is not seen); a side "
              "under 336 px is first tiled with copies of the image. Each patch is scored and the scores are "
              "averaged."),
    "probe_resnet": ("Native-res 224 patches, mean logit", "The image is cut into non-overlapping 224×224 patches at "
                     "its own resolution (no scaling; a strip at the right / bottom edge narrower than 224 px is not "
                     "seen); an image with a side under 224 px is centered on a black 224×224 square instead (its "
                     "other side, if longer, is cut to the central 224 px). Each patch is scored and the scores are "
                     "averaged."),
    "vina": ("Resize + center-crop 256 (bicubic)", "The shorter side is scaled to 256 px (bicubic) and the central "
             "256×256 square is used. The aspect ratio is kept; anything outside the square is not seen by the "
             "detector."),
    "veritaspp": ("Native resolution (Qwen3-VL processor)", "As the official server loads it, the image is turned "
                  "upright by its EXIF orientation and any transparency is filled with white; then the whole image goes "
                  "to the vision-language model at its own resolution, resized only to a multiple of 32 px (between "
                  "0.07 and 16.8 megapixels)."),
    "adb_vlm": ("Resized to at most 1.0 MP (multiples of 28)", "As the official ms-swift pipeline loads it, the image is "
                "converted to RGB (no EXIF rotation; any transparency channel is dropped); an image above 1,003,520 "
                "pixels is first shrunk to about that size (bilinear), then the whole image is resized (bicubic) so "
                "both sides are multiples of 28 px, keeping the aspect ratio, and goes to the vision-language model."),
    "adb_full": ("Two views: whole image ≤1.0 MP + center-crop 336", "The vision-language model sees the whole image as "
                 "the official ms-swift pipeline loads it (RGB, no EXIF rotation, transparency dropped; shrunk to at "
                 "most 1,003,520 pixels, sides multiples of 28 px). The pixel expert (DDA) sees the central 336×336 "
                 "pixels exactly as they are, with no scaling (a smaller image is padded with black)."),
    "fakevlm": ("Resize + center-crop 336", "As the official code loads it, the image is converted to RGB (no EXIF "
                "rotation; any transparency channel is dropped), the shorter side is scaled to 336 px (bicubic) and the "
                "central 336x336 square goes to the vision-language model; anything outside it is not seen."),
    "wevis": ("Agent: 1 fps overview + inspected windows", "As the official agent samples them: a 1 fps overview of at "
              "most 16 frames, each resized to at most 393,216 pixels (multiples of 32), then the frames of each time "
              "window an inspector is asked about at 2 fps (at most 72), and for the text check one full-resolution "
              "frame read by PP-OCRv5. Audio is not used."),
    "cure": ("Paper-specific pipeline", "As in the paper: PNG images are re-saved as JPEG, the shorter side is scaled to "
             "390 and center-cropped, resized to 336, then JPEG-compressed again (quality 70)."),
}


# Paper and official-code links shown (clickable) with each detector on the web page, as credit to the authors (a
# condition of serving LTF's weights in the demo). Sources: the model READMEs and the upstream-repository column of
# LICENSES.md; every arXiv id was checked against its abstract page and every GitHub repository resolved (2026-11-24).
# A detector without a verified link (or a SynthScope-only probe) has none.
_GH, _AX = "https://github.com/", "https://arxiv.org/abs/"
_LINKS: dict[tuple[str, ...], tuple[str | None, str | None]] = {
    ("cnn",): (_AX + "1912.11035", _GH + "PeterWang512/CNNDetection"),
    ("clip",): (_AX + "2302.10174", _GH + "WisconsinAIVision/UniversalFakeDetect"),
    ("npr",): (_AX + "2312.10461", _GH + "chuangchuangtan/NPR-DeepfakeDetection"),
    ("fat_former",): (_AX + "2312.16649", _GH + "Michel-liu/FatFormer"),
    ("aide",): (_AX + "2406.19435", _GH + "shilinyan99/AIDE"),
    ("co_spy",): (_AX + "2503.18286", _GH + "Megum1/CO-SPY"),
    ("spai",): (_AX + "2411.19417", _GH + "mever-team/spai"),
    ("safe",): (_AX + "2408.06741", _GH + "Ouxiang-Li/SAFE"),
    ("pgc",): (None, _GH + "xiaoyu6868/PGC"),
    ("cure",): (_AX + "2609.14316", _GH + "manic-cui/CuRe"),
    ("d3qe",): (_AX + "2510.05891", _GH + "Zhangyr2022/D3QE"),
    ("ltf",): (_AX + "2611.04529", _GH + "HuiGuanLab/LTF"),
    ("dda",): (_AX + "2505.14359", _GH + "roy-ch/Dual-Data-Alignment"),
    ("omniaid", "omniaid_v1", "omniaid_genimage"): (_AX + "2511.08423", _GH + "yunncheng/OmniAID"),
    ("bfree",): (_AX + "2412.17671", _GH + "grip-unina/B-Free"),
    ("mirror",): (_AX + "2602.02222", None),             # the code repository is 404 since 2026-11-12
    ("vina", "vina_video"): (_AX + "2605.21977", _GH + "mgiant/VINA"),
    ("probe", "probe_resnet"): (_AX + "2605.24906", _GH + "Amamiya-C/PROBE-AIGI-Detection"),
    ("veritaspp",): (_AX + "2607.27113", _GH + "EricTan7/VeritasPP"),
    ("adb_vlm", "adb_full"): (_AX + "2512.06746", _GH + "GzZning/Align-Dont-Blend"),
    ("fakevlm",): (_AX + "2503.14905", _GH + "opendatalab/FakeVLM"),
    ("moejepa",): (_AX + "2609.16778", _GH + "ALCOR-Lab-DIAG/MoE-JEPA"),
    ("effort", "effort_chameleon"): (_AX + "2411.15633", _GH + "YZY-stack/Effort-AIGI-Detection"),
    ("d3",): (None, _GH + "Zig-HS/D3"),
    ("split",): (_AX + "2607.02886", _GH + "mldljyh/SPLIT"),
    ("waverep", "waverep_g4"): (_AX + "2506.16802", _GH + "grip-unina/WaveRep-SyntheticVideoDetection"),
    ("nsg_vd", "nsg_vd_pika", "nsg_vd_unbalance"): (_AX + "2510.08073", _GH + "ZSHsh98/NSG-VD"),
    ("wevis",): (_AX + "2611.21373", _GH + "Tencent/WeVisForensics"),
    ("stall",): (None, _GH + "OmerBenHayun/STALL"),
}
DETECTOR_LINKS: dict[str, dict[str, str]] = {
    key: {kind: url for kind, url in (("paper", paper), ("code", code)) if url}
    for keys, (paper, code) in _LINKS.items() for key in keys}


def detector_links(key: str) -> dict:
    """{"paper": url, "code": url} (either may be missing) for the web page's detector credit line."""
    return dict(DETECTOR_LINKS.get(key, {}))


def input_note(spec: ModelSpec) -> dict:
    """{label, explanation} describing the preprocessing of ``spec`` (frames for video models)."""
    if spec.video_sampler == "ffmpeg_fps_center":
        one = spec.num_frames == 1
        frames = "the central frame" if one else f"the central {spec.num_frames} frames"
        return {"label": f"{frames.capitalize()} at {spec.fps:g} fps",
                "explanation": f"The clip is written as JPEG frames at {spec.fps:g} fps (ffmpeg) and {frames} of "
                               f"them {'is' if one else 'are'} scored like an image: the shorter side is scaled to "
                               f"{spec.frame_size} px (bicubic) and the central {spec.frame_size}×{spec.frame_size} "
                               "square is used. This matches the paper's evaluation."}
    if spec.video_sampler == "ffmpeg_select_jpeg":
        return {"label": f"{spec.num_frames} evenly spaced frames, {spec.frame_size}×{spec.frame_size}",
                "explanation": f"{spec.num_frames} frames are taken at an even spacing from the start of the clip "
                               "(every k-th frame, k = number of frames ÷ "
                               f"{spec.num_frames}), saved as JPEG by ffmpeg and stretched to "
                               f"{spec.frame_size}×{spec.frame_size} (the aspect ratio changes). A clip with fewer than "
                               f"{spec.num_frames} frames cannot be scored. This matches the paper's evaluation."}
    if spec.video_sampler == "cv2_frames":
        frames = f"The first {spec.num_frames} frames" if spec.num_frames else "Every frame"
        return {"label": f"{frames}, center {spec.frame_size}",
                "explanation": f"{frames} of the clip {'are' if spec.num_frames else 'is'} scored one by one: the "
                               f"central {spec.frame_size}×{spec.frame_size} pixels of each frame are used exactly "
                               "as they are (no scaling; a smaller frame is padded with black), and the frame scores "
                               "are averaged. This matches the paper's evaluation."}
    if spec.predict_kind == "video_agent":
        label, expl = INPUT_NOTES[spec.key]
        return {"label": label, "explanation": expl + " This matches the official inference."}
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
    # WeVisForensics: the held-out headline, not the in-distribution GenBuster-Bench (the spec's first row).
    "wevis": "4 held subsets",
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
