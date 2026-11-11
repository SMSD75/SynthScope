"""NSG-VD (Zhang et al., "Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection", NeurIPS 2025).

Official repo: github.com/ZSHsh98/NSG-VD (Apache-2.0). A video is scored against a fixed set of real reference videos
(official test_dMMD.py, configs/nsg-vd-224x224/test.yaml):

    frames  = 8 frames, every k-th with k = n_frames // 8 (ffmpeg JPEGs), LANCZOS-resized to 224x224, in [0, 1]
    s_t     = -eps_theta(2 x_t - 1, 999 t) / std(t), first 3 channels, t = (5 + 1) / 1000        (guided-diffusion 256
              uncond, fp16)
    NSG_t   = s_t / (<s_t, dx_t/dt> + 1e-10)        (central frame differences; one-sided at the ends)
    x       = NSG with its first and last frames multiplied by 100                               [8, 3, 224, 224]
    f       = TALL/Swin discriminator feature of x                                               [300]
    score   = MMD_batch2: 1/n^2 sum_ij K(r_i, r_j) - 2/n sum_i K(r_i, x) + K(x, x) over the n = 100 reference videos r,
              K = deep kernel exp(-|f-f'|^2 / sigma0 - |x-x'|^2 / sigma) (+ epsilon smoothing; trained sigma, sigma0,
              epsilon from the checkpoint);  fake if score > 1 (the official threshold).

The reference set is the official one: the first 100 ids of the released MSR-VTT val split (assets/split.zip,
``ref_mode: val``, ``ref_load_len: 100``) in the official loader's order. Its NSG features ``x`` (100 x 8 x 3 x 224 x 224
float32, the end frames already scaled) ship as a precomputed file built by models/video/nsg_vd/build_reference.py; the
reference discriminator features ``f`` depend on the checkpoint and are computed from it at the first score.

``SynthScopeNSGVDModel`` holds: ``unet`` - the diffusion model (weights from ``ModelConfig.diffusion_weights``, the
official 256x256_diffusion_uncond.pt, loaded in ``__init__``; ``pretrained_prefixes`` tells the checkpoint loader it is
not in the NSG-VD checkpoints); ``mmd`` - the official ``deep_MMD`` (kernel parameters + the TALL discriminator ``net``)
whose state dict is the released checkpoint's, key for key (``registry.remap_nsg_vd`` adds the ``mmd.`` prefix); the
reference NSG features (``ModelConfig.reference_path``, a non-persistent buffer).

Input contract: ``forward`` / ``score`` take [B, T=8, 3, 224, 224] frames in [0, 1] (media.ffmpeg_select_frames + the
``nsgvd_frame_224`` transform) and return the official statistic [B] (higher = more likely generated).
``fake_probability`` maps it to P(fake) = sigmoid((score - 1) / prob_scale): 0.5 at the official threshold.

Precision: as the official code, the UNet's convolutions run in float16 (``convert_to_fp16``). The engine casts a model
to float32 after loading, so the conversion is (re)applied before each diffusion pass. On a CPU, where float16
convolutions are unsupported or slow, ``cpu_fp32`` (default true) runs the UNet in float32 instead (a deviation; scores
then differ slightly from the official GPU run).

Training: not supported (official train_dMMD.py); ``compute_loss`` raises.
"""
import contextlib
import io
import os
import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file as load_safetensors

from models.video.nsg_vd.official_mmd import MMD_batch2, deep_MMD
from models.video.nsg_vd.official_nsg import UNET_CONFIG, build_unet, scale_end_frames, score_features
from models.video.nsg_vd.official_tall import SingleSwinBlockDiscriminator

THRESHOLD = 1.0                # test_dMMD.py: predict = (raw_predict > 1)


def load_reference(path: str) -> torch.Tensor:
    """The shipped reference NSG features: a safetensors file with ``ref_data`` [N, 8, 3, 224, 224] float32."""
    return load_safetensors(path)["ref_data"]


class SynthScopeNSGVDModel(nn.Module):
    """NSG-VD's detector (diffusion NSG estimator + deep-kernel MMD against a real reference set)."""

    pretrained_prefixes = ("unet.",)
    higher_is_fake = True

    def __init__(self, config: DictConfig):
        super().__init__()
        mc = config.ModelConfig
        self.num_frames = int(mc.get("num_frames", 8))
        self.diffuse_steps = int(mc.get("diffuse_steps", 5))
        self.resolution_size = int(mc.get("resolution_size", 224))
        self.threshold = float(mc.get("threshold", THRESHOLD))
        self.prob_scale = float(mc.get("prob_scale", 0.1))
        self.process_batch = int(mc.get("process_batch", 1))   # videos per diffusion pass (official extraction: 3)
        self.cpu_fp32 = bool(mc.get("cpu_fp32", True))
        unet_overrides = dict(mc.get("unet_overrides") or {})   # tests: a tiny UNet
        self.use_fp16 = bool(unet_overrides.get("use_fp16", UNET_CONFIG["use_fp16"]))
        self.unet = build_unet(**unet_overrides)
        if mc.get("diffusion_weights"):
            self.unet.load_state_dict(torch.load(str(mc.diffusion_weights), map_location="cpu", weights_only=True))
        self.unet.eval()
        # The vendored SwinTransformer prints its layout and calls torch.meshgrid without ``indexing`` (the default "ij",
        # a deprecation warning): silenced here, the module itself is unchanged.
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"torch\.meshgrid", category=UserWarning)
            discriminator = SingleSwinBlockDiscriminator(num_features=int(mc.get("feature_dim", 300)))
        self.mmd = deep_MMD(discriminator=discriminator,
                            sigma=mc.get("sigma", 1000), sigma0=mc.get("sigma0", 0.1), epsilon=mc.get("epsilon", 10),
                            img_size=int(mc.get("img_size", 224)), is_yy_zero=bool(mc.get("is_yy_zero", True)),
                            is_smooth=bool(mc.get("is_smooth", True)))
        ref = mc.get("reference_path")
        self.register_buffer("ref_data", load_reference(str(ref)) if ref and os.path.exists(str(ref)) else None,
                             persistent=False)
        self.register_buffer("feature_ref", None, persistent=False)   # net(ref_data), per checkpoint, computed lazily

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.feature_ref = None                 # new discriminator weights: recompute the reference features
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_reference(self, ref_data: torch.Tensor):
        """Replace the reference NSG features [N, T, 3, 224, 224] (e.g. a domain-matched real set)."""
        self.ref_data = ref_data.to(next(self.mmd.parameters()).device, torch.float32)
        self.feature_ref = None

    def _unet_dtype(self, device: torch.device):
        """float16 convolutions as the official ``convert_to_fp16`` (undone by a ``.float()`` cast; on a CPU, float32
        when ``cpu_fp32``)."""
        half = self.use_fp16 and not (device.type == "cpu" and self.cpu_fp32)
        want = torch.float16 if half else torch.float32
        if self.unet.input_blocks[0][0].weight.dtype != want:
            self.unet.convert_to_fp16() if half else self.unet.convert_to_fp32()
        self.unet.dtype = want

    @torch.no_grad()
    def nsg(self, clips: torch.Tensor) -> torch.Tensor:
        """[B, T, 3, H, W] frames in [0, 1] -> the official NSG features [B, T, 3, 224, 224] (end frames x100)."""
        self._unet_dtype(clips.device)
        out = []
        for chunk in clips.split(self.process_batch):
            b = chunk.shape[0]
            _, velocity = score_features(self.unet, chunk.reshape(-1, *chunk.shape[2:]).float(), b,
                                         diffuse_steps=self.diffuse_steps, resolution_size=self.resolution_size)
            out.append(scale_end_frames(velocity))
        return torch.cat(out)

    @torch.no_grad()
    def reference_features(self) -> torch.Tensor:
        """get_ref_features: the discriminator features of the whole reference set in one pass [N, 300]."""
        if self.ref_data is None:
            raise RuntimeError("NSG-VD needs its reference set: set ModelConfig.reference_path to the shipped "
                               "nsg_vd_reference_msrvtt_val100.safetensors (models/video/nsg_vd/build_reference.py).")
        if self.feature_ref is None:
            _, self.feature_ref = self.mmd.net(self.ref_data, out_feature=True)
        return self.feature_ref

    @torch.no_grad()
    def score_nsg(self, x: torch.Tensor) -> torch.Tensor:
        """test_dMMD's statistic for NSG features x [B, T, 3, 224, 224] -> [B]."""
        feature_ref, ref_data = self.reference_features(), self.ref_data
        _, feature = self.mmd.net(x, out_feature=True)
        return MMD_batch2(torch.cat([feature_ref, feature], dim=0), feature_ref.shape[0],
                          torch.cat([ref_data, x], dim=0).view(ref_data.shape[0] + x.shape[0], -1),
                          self.mmd.sigma, self.mmd.sigma0_u, self.mmd.ep, is_smooth=self.mmd.is_smooth)

    @torch.no_grad()
    def score(self, clips: torch.Tensor) -> torch.Tensor:
        """[B, T, 3, 224, 224] frames in [0, 1] -> the official raw statistic [B] (fake if > 1)."""
        return self.score_nsg(self.nsg(clips))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.score(clips)

    def is_fake(self, score: torch.Tensor) -> torch.Tensor:
        """The official decision: fake iff score > threshold (test_dMMD.py ``raw_predict > 1``; a score of exactly 1 is
        real)."""
        return score > self.threshold

    def fake_probability(self, score: torch.Tensor) -> torch.Tensor:
        """P(fake) = sigmoid((score - threshold) / prob_scale): 0.5 exactly at the official threshold (1)."""
        return torch.sigmoid((score - self.threshold) / self.prob_scale)

    def compute_loss(self, batch):
        raise NotImplementedError("SynthScope does not train NSG-VD; use the official train_dMMD.py "
                                  "(github.com/ZSHsh98/NSG-VD) and load its checkpoint.")
