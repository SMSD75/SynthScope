from torch._numpy import torch
import torch.nn as nn
from omegaconf import DictConfig
import clip
import torch

class AGIDVIBNetModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.backbone = clip.load(config.ModelConfig.backbone_name)
        if config.ModelConfig.freeze_backbone:
            self._freeze_backbone()
        self.vib_head = nn.Sequential(*[
            nn.Linear(self.backbone.visual.output_dim, config.ModelConfig.vib_head_dim),
            nn.ReLU(),
            nn.Linear(config.ModelConfig.vib_head_dim, config.ModelConfig.vib_head_dim),
            nn.ReLU(),
            nn.Linear(config.ModelConfig.vib_head_dim, config.ModelConfig.vib_output_dim)
        ])
        self.mu_head = nn.Linear(config.ModelConfig.output_dim, config.ModelConfig.output_dim)
        self.sigma_head = nn.Linear(config.ModelConfig.output_dim, config.ModelConfig.output_dim)
        self.final_classifier = nn.Linear(config.ModelConfig.output_dim, 2)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        image_features = self.backbone.encode_image(inputs)
        mu, sigma = self.mu_head(self.vib_head(image_features)), self.sigma_head(self.vib_head(image_features))
        y = mu + sigma * torch.randn_like(mu)
        return self.final_classifier(y)