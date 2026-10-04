import torch
import torch.nn as nn
from omegaconf import DictConfig
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, conv1x1

def make_bottleneck(in_ch, planes, stride):
    """Single Bottleneck block with a matching downsample on the skip."""
    downsample = nn.Sequential(
        conv1x1(in_ch, planes * Bottleneck.expansion, stride),
        nn.BatchNorm2d(planes * Bottleneck.expansion),
    )
    return Bottleneck(in_ch, planes, stride=stride, downsample=downsample)


def make_bottleneck_layer(in_ch, planes, blocks, stride=1):
    layers = [make_bottleneck(in_ch, planes, stride)]
    out_ch = planes * Bottleneck.expansion
    for _ in range(1, blocks):
        layers.append(Bottleneck(out_ch, planes))
    return nn.Sequential(*layers), out_ch


class NPR(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        layer1, layer1_ch = make_bottleneck_layer(64, 64, blocks=3)
        layer2, layer2_ch = make_bottleneck_layer(layer1_ch, 128, blocks=4, stride=2)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            layer1,
            layer2,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(layer2_ch, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
    

    def NPR_transform(self, x: torch.Tensor) -> torch.Tensor:
        # The residual compares each pixel with its 2x2 block, so odd sizes are cropped by one row/column (as in the
        # official code's commented-out lines). The registered 256x256 input is never odd; native-resolution scoring can be.
        x = x[:, :, : x.shape[2] - x.shape[2] % 2, : x.shape[3] - x.shape[3] % 2]
        x_down = F.interpolate(
            x,
            scale_factor=0.5,
            mode="nearest",
            recompute_scale_factor=True,
        )
        x_up = F.interpolate(
            x_down,
            scale_factor=2.0,
            mode="nearest",
            recompute_scale_factor=True,
        )
        return (x - x_up) * (2.0 / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)
    
    def compute_loss(self, batch):
        images, labels, folder_ids = batch
        logits = self.forward(self.NPR_transform(images)).squeeze(-1)
        pos = logits
        neg = -logits
        outputs = torch.stack([neg, pos], dim=1)
        loss = self.criterion(logits, labels.float())
        return {"loss": loss, "outputs": outputs}
