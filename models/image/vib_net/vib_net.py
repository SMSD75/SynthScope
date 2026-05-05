from numbers import Number

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class AGIDVIBNetModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        backbone_name = config.ModelConfig.backbone_name
        if backbone_name.startswith("CLIP:"):
            backbone_name = backbone_name.removeprefix("CLIP:")

        self.backbone, _ = clip.load(backbone_name)
        if config.ModelConfig.freeze_backbone:
            self._freeze_backbone()

        self.latent_dim = config.ModelConfig.get("vib_latent_dim", 256)
        hidden_dim = config.ModelConfig.get("vib_hidden_dim", 1024)
        dropout = config.ModelConfig.get("dropout", 0.5)
        self.std_shift = config.ModelConfig.get("std_shift", 5.0)

        self.dropout = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(self.backbone.visual.output_dim, hidden_dim)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, 2 * self.latent_dim)
        self.decode = nn.Linear(self.latent_dim, 1)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    @staticmethod
    def _expand_sample(v, num_samples):
        if isinstance(v, Number):
            return torch.tensor([v]).expand(num_samples, 1)
        return v.expand(num_samples, *v.size())

    def _reparameterize(self, mu, std, num_samples=1):
        if num_samples != 1:
            mu = self._expand_sample(mu, num_samples)
            std = self._expand_sample(std, num_samples)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs):
        image_features = self.backbone.encode_image(inputs)
        if image_features.dim() > 2:
            image_features = image_features.view(image_features.size(0), -1)
        image_features = self.dropout(image_features.float())

        statistics = self.fc_1(image_features)
        statistics = self.relu(statistics)
        statistics = self.fc_2(statistics)
        statistics = self.relu(statistics)
        statistics = self.fc_3(statistics)

        mu = statistics[:, :self.latent_dim]
        std = F.softplus(statistics[:, self.latent_dim:] - self.std_shift, beta=1)
        encoding = self._reparameterize(mu, std, num_samples=1)
        logit = self.decode(encoding)
        return (mu, std), logit
