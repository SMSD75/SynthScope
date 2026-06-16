"""Genuine AIDE architecture + native-resolution 5-view preprocessing (Yan et al., ICLR'25).

Ported VERBATIM from the official repo (shilinyan99/AIDE: ``data/dct.py`` + ``models/AIDE.py`` +
``data/datasets.py``) so the released checkpoints load key-for-key and the AGID class trains the
SAME network. This is the single source of truth for the AIDE arch; ``aide.py`` wraps it for the
AGID ``compute_loss`` API and ``repro/official_evals/aide_official_eval.py`` mirrors it for the
standalone official eval.

The decisive detail (see repro: native-res DCT vs resize-256 is a ~12 mAP swing on the SAME weights):
the DCT patch selector runs on the **native-resolution** image; only the SELECTED 32x32 patches are
then resized to 256. ``AIDEFiveView`` produces the official ``[5,3,256,256]`` view stack per image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .srm_kernels import all_normalized_hpf_list

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_DINOV2_MEAN = (0.485, 0.456, 0.406)
_DINOV2_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- DCT 5-view selection
def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) *
          np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super().__init__()
        self.use_learnable = use_learnable
        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)),
                                 requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                requires_grad=False)

    def forward(self, x):
        filt = self.base + norm_sigma(self.learnable) if self.use_learnable else self.base
        return x * filt / self.ft_num if self.norm else x * filt


class DCT_base_Rec_Module(nn.Module):
    """x: [C, H, W] (native size) -> 4 patches each [C, window, window] (window_size=32)."""

    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()
        assert output % window_size == 0
        assert len(level_fliter) > 0
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)

        self._DCT_patch = nn.Parameter(
            torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
        self.fold0 = nn.Fold(output_size=(window_size, window_size),
                             kernel_size=(window_size, window_size), stride=window_size)

        level_f = [Filter(window_size, 0, window_size * 2)]
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([
            Filter(window_size, window_size * 2. / grade_N * i,
                   window_size * 2. / grade_N * (i + 1), norm=True) for i in range(grade_N)])

    def forward(self, x):
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)

        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size)
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)

        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1, 2, 3])
            grade += w * _x
            w *= k

        _, idx = torch.sort(grade)
        max_idx = torch.flip(idx, dims=[0])[:N]
        maxmax_idx = max_idx[0]
        maxmax_idx1 = max_idx[0] if len(max_idx) == 1 else max_idx[1]

        min_idx = idx[:N]
        minmin_idx = idx[0]
        minmin_idx1 = idx[0] if len(min_idx) == 1 else idx[1]

        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx)
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx)
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1)
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1)

        x_minmin = x_minmin.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)

        return (self.fold0(x_minmin), self.fold0(x_maxmax),
                self.fold0(x_minmin1), self.fold0(x_maxmax1))


# --------------------------------------------------------------------------- network
class HPF(nn.Module):
    def __init__(self):
        super().__init__()
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode="constant")
            all_hpf_list_5x5.append(hpf_item)
        hpf_weight = torch.Tensor(np.array(all_hpf_list_5x5)).view(30, 1, 5, 5).contiguous()
        hpf_weight = nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, x):
        return self.hpf(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AIDE_Model(nn.Module):
    """Genuine AIDE: 5-view input ``[B,5,3,256,256]`` -> logits ``[B,2]``.

    ``convnext_pretrained=None`` builds the OpenCLIP ConvNeXt-XXL trunk with random weights (the
    official checkpoint supplies them on strict load); pass ``"laion2b_s34b_b82k_augreg_soup"`` to
    download+freeze the trunk for training from scratch.
    """

    def __init__(self, convnext_pretrained=None, convnext_model_name="convnext_xxlarge"):
        super().__init__()
        import open_clip
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])

        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            convnext_model_name, pretrained=convnext_pretrained)
        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()
        self.openclip_convnext_xxl.eval()
        # Trunk feature channels (convnext_xxlarge -> 3072, matching the official checkpoint;
        # smaller trunks for tests resolve to their own width).
        feat_dim = int(getattr(self.openclip_convnext_xxl, "num_features", 3072))
        self.fc = Mlp(2048 + 256, 1024, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(nn.Linear(feat_dim, 256))
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False
        # The ResNet ``fc`` heads are unused (the streams stop at avgpool); they exist only so the
        # official checkpoint loads strict. Freeze them so the optimizer never touches dead params.
        for stream in (self.model_min, self.model_max):
            for p in stream.fc.parameters():
                p.requires_grad = False

        self.register_buffer("clip_mean", torch.tensor(_CLIP_MEAN).view(3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor(_CLIP_STD).view(3, 1, 1), persistent=False)
        self.register_buffer("dinov2_mean", torch.tensor(_DINOV2_MEAN).view(3, 1, 1), persistent=False)
        self.register_buffer("dinov2_std", torch.tensor(_DINOV2_STD).view(3, 1, 1), persistent=False)

    def forward(self, x):
        # x: [B, 5, 3, 256, 256] -> [minmin, maxmax, minmin1, maxmax1, tokens]
        x_minmin = self.hpf(x[:, 0])
        x_maxmax = self.hpf(x[:, 1])
        x_minmin1 = self.hpf(x[:, 2])
        x_maxmax1 = self.hpf(x[:, 3])
        tokens = x[:, 4]

        with torch.no_grad():
            # FROZEN trunk only: re-normalize the ImageNet/DINOv2-normalized tokens into CLIP space.
            local = self.openclip_convnext_xxl(
                tokens * (self.dinov2_std / self.clip_std)
                + (self.dinov2_mean - self.clip_mean) / self.clip_std)
            local = self.avgpool(local).view(tokens.size(0), -1)   # [B, feat_dim]
        x_0 = self.convnext_proj(local)   # trainable projection (outside no_grad so it learns)

        x_1 = (self.model_min(x_minmin) + self.model_max(x_maxmax)
               + self.model_min(x_minmin1) + self.model_max(x_maxmax1)) / 4
        return self.fc(torch.cat([x_0, x_1], dim=1))


# --------------------------------------------------------------------------- 5-view preprocessing
_transform_before = transforms.Compose([transforms.ToTensor()])                     # native, [0,1]
_transform_view = transforms.Compose([transforms.Resize([256, 256]),
                                       transforms.Normalize(mean=list(_DINOV2_MEAN),
                                                            std=list(_DINOV2_STD))])


class AIDEFiveView:
    """Official AIDE TestDataset preprocessing: PIL -> ``[5,3,256,256]`` = the 4 DCT-selected
    native-resolution patches (each resized to 256) + the full image (tokens). Runs the DCT on the
    NATIVE-size tensor (the key fidelity detail). Picklable + stateless aside from the DCT module."""

    def __init__(self):
        self.dct = DCT_base_Rec_Module()

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        image = _transform_before(pil_img)  # [3, H, W] native, [0,1]
        a, b, c, d = self.dct(image)
        return torch.stack([_transform_view(a), _transform_view(b), _transform_view(c),
                            _transform_view(d), _transform_view(image)], dim=0)
