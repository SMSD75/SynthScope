"""MoE-JEPA architecture blocks, vendored unchanged from the official repository.

Source: https://github.com/ALCOR-Lab-DIAG/MoE-JEPA, ``model/model.py`` (v1.0, 2026-10-30): ``BayarConv2d``,
``NoiseStream``, ``GatedAttentionPooling``, ``Expert`` and ``MoEJEPALayer``, copied verbatim (the Lightning module,
losses and data helpers are not needed here). Original license:

    MIT License

    Copyright (c) 2026 ALCOR Lab - DIAG

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConv2d(nn.Module):
    """
    A constrained convolution layer that learns to extract high-frequency noise 
    residuals while suppressing image content.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(BayarConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        center = kernel_size // 2
        mask[:, :, center, center] = 0
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        
        weight_sum = masked_weight.sum(dim=(2, 3), keepdim=True)
        norm_weight = masked_weight / (weight_sum + 1e-5)
        
        center = self.kernel_size // 2
        final_weight = norm_weight.clone()
        final_weight[:, :, center, center] = -1.0
        
        return F.conv2d(x, final_weight, stride=1, padding=self.padding)

class NoiseStream(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bayar = BayarConv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten() 
        )
        
    def forward(self, x):        
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        else:
            B = x.shape[0]
            T = 1
        
        noise_map = self.bayar(x)
        features = self.cnn(noise_map)
        
        if T > 1:
            features = features.view(B, T, -1)
            features = features.mean(dim=1)
            
        return features

class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        self.projector_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.projector_u = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        v = self.projector_v(x)
        u = self.projector_u(x)
        
        gated_features = v * u
        scores = self.w(gated_features) / self.temperature
        
        attn_weights = F.softmax(scores, dim=1)
        
        weighted_features = torch.sum(x * attn_weights, dim=1)
        
        return weighted_features, attn_weights


class Expert(nn.Module):
    def __init__(self, weights=None, train=True, dim=1024, intermediate_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, dim)

        if weights is not None:
            self.load_state_dict(weights)

        if not train:
            for param in self.parameters():
                param.requires_grad = False 
        else:
            for param in self.parameters():
                param.requires_grad = True


    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MoEJEPALayer(nn.Module):
    def __init__(self, weights=None, num_trainable_experts=3, top_k=3, router_temperature=1.0, hidden_dim=1024, intermediate_dim=4096):
        super(MoEJEPALayer, self).__init__()

        self.num_trainable_experts = num_trainable_experts
        self.top_k = top_k
        self.router_temperature = router_temperature

        self.router = nn.Linear(hidden_dim, self.num_trainable_experts)
        self.shared_expert = Expert(weights=weights, train=False, dim=hidden_dim, intermediate_dim=intermediate_dim)
        self.routed_experts = nn.ModuleList([Expert(weights=weights, train=True, dim=hidden_dim, intermediate_dim=intermediate_dim) for _ in range(self.num_trainable_experts)])

        for expert in self.routed_experts:
            for param in expert.parameters():
                param.data += torch.randn_like(param.data) * 1e-4

        self.current_confidence = None
        self.current_weights = None

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        shared_out = self.shared_expert(x_flat)

        logits = self.router(x_flat)
        logits = logits / self.router_temperature

        experts_prob = torch.sigmoid(logits)

        confidence = experts_prob.max(dim=-1)[0].mean()
        self.current_confidence = confidence
        self.current_weights = experts_prob

        top_k_weights, top_k_indices = torch.topk(experts_prob, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = (top_k_indices == i)
            if mask.any():
                token_idx, k_idx = torch.where(mask)
                expert_out = expert(x_flat[token_idx])
                weight = top_k_weights[token_idx, k_idx].unsqueeze(-1)
                routed_out[token_idx] += weight * expert_out

        final_output = shared_out + routed_out

        return final_output.view(batch_size, seq_len, hidden_dim)
