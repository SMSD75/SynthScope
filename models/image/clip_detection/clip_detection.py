import torch.nn as nn
from omegaconf import DictConfig
import clip
import torch

class SynthScopeCLIPModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        backbone_name = config.ModelConfig.backbone_name
        if backbone_name.startswith("CLIP:"):
            backbone_name = backbone_name.removeprefix("CLIP:")

        self.backbone, _ = clip.load(backbone_name)
        if config.ModelConfig.freeze_backbone:
            self._freeze_backbone()
        self.config = config
        self.head = nn.Linear(self.backbone.visual.output_dim, 2)
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config.TrainerConfig.get("label_smoothing", 0.0)
        )

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        image_features = self.backbone.encode_image(images)
        return self.head(image_features.float())
    
    def compute_loss(self, batch):
        images, labels, folder_ids = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        return {"loss": loss, "outputs": outputs}
