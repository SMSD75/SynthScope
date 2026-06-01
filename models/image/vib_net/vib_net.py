from numbers import Number

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import torch.nn.functional as f

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
        self.loss_1 = nn.BCEWithLogitsLoss(reduction="mean")
        self.loss_2 = self._kl_to_standard_normal

    def _kl_to_standard_normal(self, mu, std, eps=1e-8):
        """KL( N(mu, std^2 I) || N(0, I) ), mean over batch, sum over latent dims."""
        var = std.pow(2).clamp(min=eps)
        kl = 0.5 * (mu.pow(2) + var - torch.log(var + eps) - 1.0)
        return kl.sum(dim=1).mean()

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

        fc1_out = self.fc_1(image_features)
        relu1_out = self.relu(fc1_out)
        fc2_out = self.fc_2(relu1_out)
        relu2_out = self.relu(fc2_out)
        fc3_out = self.fc_3(relu2_out)

        mu = fc3_out[:, :self.latent_dim]
        # Floor std away from 0: training can drive the pre-softplus input very negative,
        # collapsing std->0 -> var->0 -> KL's -log(var)->+inf -> NaN (caused a CUDA
        # device-side assert after the first epoch). 1e-4 floor keeps the VIB stable.
        std = F.softplus(fc3_out[:, self.latent_dim:] - self.std_shift, beta=1).clamp(min=1e-4)

        encoding = self._reparameterize(mu, std, num_samples=1)
        logit = self.decode(encoding)
        return (mu, std), logit.squeeze(-1)
        
    def compute_loss(self, batch):
        images, labels, folder_ids = batch
        (mu, std), logit = self.forward(images)
        loss_1 = self.loss_1(logit, labels.float())
        loss_2 = self.loss_2(mu, std)
        beta = self.config.ModelConfig.get("vib_beta", 1e-3)
        loss = loss_1 + beta * loss_2
        pos = logit                            
        neg = -logit                            
        outputs = torch.stack([neg, pos], dim=1)
        return {"loss": loss, "outputs": outputs}
