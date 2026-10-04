import torch.nn as nn
from omegaconf import DictConfig
import torchvision
from torchvision.models import ResNet50_Weights
import torch

class SynthScopeCNNModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if config.ModelConfig.freeze_backbone:
            self._freeze_backbone()
        self.config = config
        self.head = nn.Linear(self.backbone.fc.in_features, 2)
        self.backbone.fc = nn.Identity()
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config.TrainerConfig.get("label_smoothing", 0.0)
        )
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        return self.head(out)
    
    def compute_loss(self, batch):
        images, labels, folder_ids = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        return {"loss": loss, "outputs": outputs}