"""CO-SPY integrated into AGID.

Paper: "CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic
Images by AI" (Cheng et al., CVPR 2025).

The detector has three trainable + two frozen sub-modules:

1. Frozen semantic backbone -- an OpenCLIP visual encoder (default
   `ViT-SO400M-14-SigLIP-384` / `webli`). It produces a CLS-pooled
   semantic feature `v_sem`. A linear `sem_fc(v_sem) -> 1` acts both as
   per-branch logit and as the semantic regulator coefficient `alpha`.

2. Frozen Stable Diffusion v1.4 VAE (`AutoencoderKL`). For each input
   `x` in `[0, 1]^{3xHxW}` we compute `x' = VAE_dec(VAE_enc(x).latent_dist.mean)`
   and the per-pixel residual `r = (x - x') * 100 / 7`. A small ResNet-50
   style stem (`conv1 + bn1 + maxpool + layer1 + layer2 + avgpool`) maps
   `r` to a 512-d artifact feature `v_art`. A linear `art_fc(v_art) -> 1`
   acts both as per-branch logit and as the artifact regulator coefficient
   `beta`.

3. Adaptive fusion: the fused feature is `[alpha * v_sem, beta * v_art]`.
   During training, with probability `coeff_dropout` we randomly zero one
   of `alpha` or `beta` (never both). A final linear head maps the fused
   feature to a single real/fake logit.

Training loss is `BCEWithLogits` with optional label smoothing, on a
single logit per sample. The two-class outputs returned to AGID metrics
are `stack([-logit, +logit], dim=1)`, matching the convention used by
`npr.py` and `vib_net.py`.

Optionally, the semantic feature `v_sem` can be augmented by the paper's
feature-space interpolation (Eq. 2): for a random subset of the batch,
`v_sem` is mixed with a random partner and the binary label is replaced
by a soft target in `[0, 1]`. Mixing weights are drawn from
`Beta(alpha=feat_interp_alpha, beta=feat_interp_alpha)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from diffusers import AutoencoderKL
from omegaconf import DictConfig


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ArtifactResNetStem(nn.Module):
    """The first three stages of a ResNet-50 used as the artifact encoder.

    Output is a 512-d feature (`128 * Bottleneck.expansion`). This matches
    the official CO-SPY artifact encoder verbatim.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(_Bottleneck, 64, 3)
        self.layer2 = self._make_layer(_Bottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_weights()

    @property
    def out_dim(self) -> int:
        return 128 * _Bottleneck.expansion

    def _make_layer(self, block: type[_Bottleneck], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class _LabelSmoothingBCEWithLogits(nn.Module):
    """BCEWithLogits with optional symmetric label smoothing."""

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if self.smoothing > 0.0:
            target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


class AGIDCoSpyModel(nn.Module):
    """CO-SPY adapted to AGID's `compute_loss(batch) -> {loss, outputs}` API."""

    SEMANTIC_MEAN = (0.5, 0.5, 0.5)
    SEMANTIC_STD = (0.5, 0.5, 0.5)

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        sem_model_name = str(
            config.ModelConfig.get("semantic_model_name", "ViT-SO400M-14-SigLIP-384")
        )
        sem_pretrained = str(config.ModelConfig.get("semantic_pretrained", "webli"))
        clip_model, _, _ = open_clip.create_model_and_transforms(
            sem_model_name, pretrained=sem_pretrained,
        )
        self.semantic = clip_model
        self.semantic_image_size = int(
            config.ModelConfig.get("semantic_image_size", 384)
        )

        vae_id = str(
            config.ModelConfig.get("vae_id", "CompVis/stable-diffusion-v1-4")
        )
        vae_subfolder = config.ModelConfig.get("vae_subfolder", "vae")
        if vae_subfolder is None or vae_subfolder == "":
            self.vae = AutoencoderKL.from_pretrained(vae_id)
        else:
            self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder=str(vae_subfolder))
        self.artifact_image_size = int(
            config.ModelConfig.get("artifact_image_size", 224)
        )
        self.artifact_residual_scale = float(
            config.ModelConfig.get("artifact_residual_scale", 100.0 / 7.0)
        )

        self.artifact_encoder = _ArtifactResNetStem()

        sem_dim = int(self._infer_semantic_dim())
        art_dim = int(self.artifact_encoder.out_dim)
        self.semantic_dim = sem_dim
        self.artifact_dim = art_dim
        self.sem_fc = nn.Linear(sem_dim, 1)
        self.art_fc = nn.Linear(art_dim, 1)
        self.fc = nn.Linear(sem_dim + art_dim, 1)

        self.coeff_dropout = float(config.ModelConfig.get("coeff_dropout", 0.3))
        self.feat_interp_prob = float(config.ModelConfig.get("feat_interp_prob", 0.5))
        self.feat_interp_alpha = float(config.ModelConfig.get("feat_interp_alpha", 0.2))

        for param in self.semantic.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        self.criterion = _LabelSmoothingBCEWithLogits(
            smoothing=float(config.TrainerConfig.get("label_smoothing", 0.0))
        )

        nn.init.normal_(self.sem_fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.sem_fc.bias)
        nn.init.normal_(self.art_fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.art_fc.bias)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def _infer_semantic_dim(self) -> int:
        visual = self.semantic.visual
        for attr in ("output_dim", "embed_dim"):
            value = getattr(visual, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        proj = getattr(visual, "proj", None)
        if isinstance(proj, torch.Tensor):
            return int(proj.shape[-1])
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.semantic_image_size, self.semantic_image_size)
            return int(self.semantic.encode_image(dummy).shape[-1])

    @staticmethod
    def _resize_if_needed(x: torch.Tensor, target: int) -> torch.Tensor:
        if x.shape[-1] != target or x.shape[-2] != target:
            return F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
        return x

    def _normalize_semantic(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.SEMANTIC_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.as_tensor(self.SEMANTIC_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _encode_semantic(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_if_needed(x, self.semantic_image_size)
        x = self._normalize_semantic(x)
        with torch.no_grad():
            features = self.semantic.encode_image(x)
        return features.float()

    def _encode_artifact(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_if_needed(x, self.artifact_image_size)
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mean
            recon = self.vae.decode(latent).sample
        residual = (x - recon) * self.artifact_residual_scale
        return self.artifact_encoder(residual.float())

    def _mix_semantic_features(
        self, v_sem: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature-space interpolation on a random subset of the batch.

        For a fraction `feat_interp_prob` of the batch we mix `v_sem` with
        a random partner using `delta ~ Beta(alpha, alpha)` and produce a
        soft target `delta * label_a + (1 - delta) * label_b`. Other
        samples keep their hard labels.
        """

        if self.feat_interp_prob <= 0.0:
            return v_sem, labels.float()

        bsz = v_sem.size(0)
        device = v_sem.device
        soft_labels = labels.float().clone()
        mix_mask = torch.rand(bsz, device=device) < self.feat_interp_prob
        n_mix = int(mix_mask.sum().item())
        if n_mix == 0:
            return v_sem, soft_labels

        mix_idx = torch.where(mix_mask)[0]
        partner = torch.randint(0, bsz, (n_mix,), device=device)
        beta = torch.distributions.Beta(self.feat_interp_alpha, self.feat_interp_alpha)
        delta = beta.sample((n_mix,)).to(device).to(v_sem.dtype)

        mixed = v_sem.clone()
        d2 = delta.unsqueeze(1)
        mixed[mix_idx] = d2 * v_sem[mix_idx] + (1.0 - d2) * v_sem[partner]

        soft_labels[mix_idx] = delta * soft_labels[mix_idx] + (1.0 - delta) * soft_labels[partner]
        return mixed, soft_labels

    def _apply_coeff_dropout(
        self, alpha: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """During training, with prob `coeff_dropout` zero one of (alpha, beta)."""

        if not self.training or self.coeff_dropout <= 0.0:
            return alpha, beta
        bsz = alpha.size(0)
        device = alpha.device
        drop_mask = torch.rand(bsz, device=device) < self.coeff_dropout
        which = torch.randint(0, 2, (bsz,), device=device)
        drop_alpha = drop_mask & (which == 0)
        drop_beta = drop_mask & (which == 1)
        alpha = alpha * (~drop_alpha).to(alpha.dtype).unsqueeze(1)
        beta = beta * (~drop_beta).to(beta.dtype).unsqueeze(1)
        return alpha, beta

    def _fuse(self, v_sem: torch.Tensor, v_art: torch.Tensor) -> torch.Tensor:
        alpha = self.sem_fc(v_sem)
        beta = self.art_fc(v_art)
        alpha, beta = self._apply_coeff_dropout(alpha, beta)
        return torch.cat([alpha * v_sem, beta * v_art], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        v_sem = self._encode_semantic(images)
        v_art = self._encode_artifact(images)
        fused = self._fuse(v_sem, v_art)
        return self.fc(fused).squeeze(-1)

    def compute_loss(self, batch):
        images, labels, _folder_ids = batch
        v_sem = self._encode_semantic(images)
        v_art = self._encode_artifact(images)

        if self.training and self.feat_interp_prob > 0.0:
            v_sem, soft_labels = self._mix_semantic_features(v_sem, labels)
        else:
            soft_labels = labels.float()

        fused = self._fuse(v_sem, v_art)
        logits = self.fc(fused).squeeze(-1)
        loss = self.criterion(logits, soft_labels)

        outputs = torch.stack([-logits, logits], dim=1)
        return {"loss": loss, "outputs": outputs}
