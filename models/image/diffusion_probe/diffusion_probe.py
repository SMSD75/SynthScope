"""Diffusion-feature linear probe for AI-generated image detection.

Idea: AGID already hosts CLIP / DINOv2 linear-probe detectors (UnivFD-style) — a
frozen self-supervised backbone + a trainable ``Linear(C, 2)`` head. This module
asks the same question for a **diffusion model**: are the internal features of a
pretrained text-to-image diffusion model linearly separable into real vs.
AI-generated?

Two frozen feature sources are supported (set ``ModelConfig.backbone``):

``unet`` (default, "DIFT"-style — Tang et al., *Emergent Correspondence from
    Image Diffusion*, NeurIPS'23):
        x in [0,1] --(*2-1)--> VAE.encode -> latent z0
        z_t = scheduler.add_noise(z0, eps, t)             # one fixed timestep t
        run UNet(z_t, t, empty-prompt context) ONE step, capture the output
        feature map of ``unet.up_blocks[up_ft_index]`` via a forward hook,
        spatially pool it -> a feature vector.
``vae``:
        global feature = the SD-VAE latent pooled to a small ``GxG`` grid and
        flattened (a deliberately cheap baseline, much weaker than ``unet``).

The backbone is frozen; only ``head: Linear(C, 2)`` trains. The model follows the
AGID detector contract (``__init__(config)`` / ``forward(images)->logits`` /
``compute_loss(batch)->{"loss","outputs"}``) with ``predict_kind="ce"`` so AGID's
engine reads ``softmax(logits)[:, 1]`` as the fake-probability.

Input contract: ``forward`` expects images in **[0, 1]** (the AGID ``to_dtype``
transform; do *not* CLIP/ImageNet-normalize upstream — the diffusion-specific
``*2-1`` mapping happens inside ``forward``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler


def _spatial_pool(feat: torch.Tensor, mode: str) -> torch.Tensor:
    """[B,C,H,W] -> [B,C] (mean) or [B,2C] (mean concat std)."""
    mean = feat.mean(dim=(2, 3))
    if mode == "mean":
        return mean
    if mode == "mean_std":
        std = feat.flatten(2).std(dim=2)
        return torch.cat([mean, std], dim=1)
    raise ValueError(f"Unknown pool mode: {mode!r}")


class AGIDDiffusionProbeModel(nn.Module):
    """Frozen Stable-Diffusion features + a linear real/fake head."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        mc = config.ModelConfig

        self.backbone_kind = str(mc.get("backbone", "unet")).lower()
        self.sd_id = str(mc.get("sd_id", "CompVis/stable-diffusion-v1-4"))
        self.image_size = int(mc.get("image_size", 256))
        self.timestep = int(mc.get("timestep", 50))
        self.up_ft_index = int(mc.get("up_ft_index", 1))
        self.pool = str(mc.get("pool", "mean"))
        self.vae_pool_grid = int(mc.get("vae_pool_grid", 8))
        self.eval_fixed_noise = bool(mc.get("eval_fixed_noise", True))
        self.noise_seed = int(mc.get("noise_seed", 0))

        # ---- frozen VAE (both backbones need it) ----
        self.vae = AutoencoderKL.from_pretrained(self.sd_id, subfolder="vae")
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.latent_channels = int(self.vae.config.latent_channels)

        self._feat_buffer: dict[str, torch.Tensor] = {}

        if self.backbone_kind == "unet":
            self.unet = UNet2DConditionModel.from_pretrained(self.sd_id, subfolder="unet")
            self.scheduler = DDIMScheduler.from_pretrained(self.sd_id, subfolder="scheduler")
            # Empty-prompt context (DIFT uses the "" prompt). Encode it once, then drop the
            # text encoder so it is not part of the (frozen) module / checkpoint.
            self.register_buffer("empty_context", self._build_empty_context(), persistent=True)
            n_up = len(self.unet.up_blocks)
            if not 0 <= self.up_ft_index < n_up:
                raise ValueError(f"up_ft_index {self.up_ft_index} out of range [0,{n_up})")
            self._hook = self.unet.up_blocks[self.up_ft_index].register_forward_hook(self._capture_hook)
        elif self.backbone_kind != "vae":
            raise ValueError(f"backbone must be 'unet' or 'vae', got {self.backbone_kind!r}")

        self._freeze_backbone()

        feature_dim = self._infer_feature_dim()
        self.feature_dim = feature_dim
        self.head = nn.Linear(feature_dim, 2)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.TrainerConfig.get("label_smoothing", 0.0)
        )

    # ------------------------------------------------------------------ setup
    def _build_empty_context(self) -> torch.Tensor:
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(self.sd_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.sd_id, subfolder="text_encoder")
        text_encoder.eval()
        tokens = tokenizer(
            "", padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            context = text_encoder(tokens.input_ids)[0]  # [1, 77, cross_dim]
        return context.detach()

    def _capture_hook(self, module, inputs, output):
        # ResNet/CrossAttn up-blocks return the hidden-state tensor (some versions a tuple).
        self._feat_buffer["up"] = output[0] if isinstance(output, tuple) else output

    def _freeze_backbone(self):
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        if self.backbone_kind == "unet":
            for p in self.unet.parameters():
                p.requires_grad = False
            self.unet.eval()

    @torch.no_grad()
    def _infer_feature_dim(self) -> int:
        dummy = torch.zeros(1, 3, self.image_size, self.image_size)
        return int(self._extract_features(dummy).shape[1])

    # --------------------------------------------------------------- features
    def _encode_latent(self, x01: torch.Tensor) -> torch.Tensor:
        x = x01 * 2.0 - 1.0  # [0,1] -> [-1,1], the SD VAE input convention
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        latent = self.vae.encode(x).latent_dist.mean * self.scaling_factor
        return latent

    def _sample_noise(self, latent: torch.Tensor) -> torch.Tensor:
        if (not self.training) and self.eval_fixed_noise:
            gen = torch.Generator(device=latent.device).manual_seed(self.noise_seed)
            return torch.randn(latent.shape, generator=gen, device=latent.device, dtype=latent.dtype)
        return torch.randn_like(latent)

    @torch.no_grad()
    def _extract_features(self, x01: torch.Tensor) -> torch.Tensor:
        latent = self._encode_latent(x01)

        if self.backbone_kind == "vae":
            pooled = F.adaptive_avg_pool2d(latent, (self.vae_pool_grid, self.vae_pool_grid))
            return pooled.flatten(1)  # [B, latent_channels * grid^2]

        # ---- DIFT-style UNet features ----
        bsz = latent.shape[0]
        noise = self._sample_noise(latent)
        t = torch.full((bsz,), self.timestep, device=latent.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(latent, noise, t)
        context = self.empty_context.to(latent.dtype).expand(bsz, -1, -1)
        self._feat_buffer.clear()
        self.unet(noisy, t, encoder_hidden_states=context)
        feat = self._feat_buffer["up"]  # [B, C, h, w]
        return _spatial_pool(feat, self.pool)

    # ----------------------------------------------------------------- AGID API
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self._extract_features(images)
        return self.head(feats.float())

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        return {"loss": loss, "outputs": outputs}
