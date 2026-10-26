# Vendored from github.com/Zhangyr2022/D3QE (MIT License, Copyright (c) 2025 Yanran Zhang), file networks/D3QE.py.
# Unchanged below this header, so the released checkpoint loads key-for-key.
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import math

from .clip import clip
from .vq_model import VQ_models


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "vae": [0.5, 0.5, 0.5],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "vae": [0.5, 0.5, 0.5],
}


def init_weights(m):
    """Unified initialization for common layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class D3QE(nn.Module):
    """D3QE model integrating VQ-VAE quantized features and CLIP features for classification."""

    def __init__(self, vqvae_path=None):
        """
        Args:
            vqvae_path: Path to a pretrained VQ-VAE checkpoint.
        """
        super().__init__()
        if vqvae_path is None:
            raise ValueError("vqvae_path is required")

        # VQ-VAE model (frozen)
        self.vq_model = VQ_models["VQ-16"](
            codebook_size=16384,
            codebook_embed_dim=8,
        )
        vq_ckpt = vqvae_path
        self.vq_model.load_state_dict(torch.load(vq_ckpt, map_location="cpu")["model"])
        for param in self.vq_model.parameters():
            param.requires_grad = False

        # CLIP visual encoder (frozen)
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Codebook frequency stats (single-scale)
        self.register_buffer("real_codebook_count", torch.zeros(16384))
        self.register_buffer("fake_codebook_count", torch.zeros(16384))
        self.freq_log_counter = 0
        self.trans_dim = 512
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Conv2d(8, self.trans_dim, 3, 1, 1),
            nn.BatchNorm2d(self.trans_dim),
            nn.ReLU(),
            D3AT(dim=self.trans_dim, num_heads=8, depth=2, token_num=256),
        )

        # Spatial pooling
        self.feature_pooler = nn.Sequential(
            nn.Conv2d(self.trans_dim, self.trans_dim, 1),
            nn.BatchNorm2d(self.trans_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.feature_processor.apply(init_weights)
        self.feature_pooler.apply(init_weights)
        self.fusion_dim = 256 if self.trans_dim >= 256 else self.trans_dim

        # Projection heads
        self.mixed_proj = nn.Sequential(
            nn.BatchNorm1d(self.trans_dim),
            nn.Linear(self.trans_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.clip_proj = nn.Sequential(
            nn.Linear(768, self.fusion_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim * 2),
            nn.Linear(self.fusion_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        # Initialize heads
        self.mixed_proj.apply(init_weights)
        self.clip_proj.apply(init_weights)
        self.fc.apply(init_weights)
        print(
            "Trainable parameters:",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def update_codebook_frequency(self, indices, labels):
        """Update per-codebook index usage frequency conditioned on labels.

        Args:
            indices: Flattened code indices of shape [B, H*W]
            labels: Labels of shape [B] (0 = real, 1 = fake)
        """
        with torch.no_grad():
            expanded_labels = labels.repeat_interleave(indices.shape[1])

            # Update counts for real samples (label == 0)
            real_mask = expanded_labels == 0
            if real_mask.any():
                real_indices = indices.flatten()[real_mask]
                unique_indices, counts = torch.unique(real_indices, return_counts=True)
                self.real_codebook_count.scatter_add_(0, unique_indices, counts.float())

            # Update counts for fake samples (label == 1)
            fake_mask = expanded_labels == 1
            if fake_mask.any():
                fake_indices = indices.flatten()[fake_mask]
                unique_indices, counts = torch.unique(fake_indices, return_counts=True)
                self.fake_codebook_count.scatter_add_(0, unique_indices, counts.float())

            self.freq_log_counter += 1

            # if self.freq_log_counter % 2000 == 0 and self.training:
            #     print(
            #         "Real codebook top-20:",
            #         torch.sort(self.real_codebook_count, descending=True)[0][:20]
            #         .cpu()
            #         .numpy(),
            #     )
            #     print(
            #         "Fake codebook top-20:",
            #         torch.sort(self.fake_codebook_count, descending=True)[0][:20]
            #         .cpu()
            #         .numpy(),
            #     )

    def forward(self, x, labels=None):
        """Forward pass.

        Args:
            x: Input images [B, C, H, W]
            labels: Optional labels [B]; used to update codebook frequency during training.
        Returns:
            Logits of shape [B, 1]
        """
        B, _, H, W = x.shape
        vae_norm = transforms.Normalize(mean=MEAN["vae"], std=STD["vae"])
        clip_norm = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(mean=MEAN["clip"], std=STD["clip"]),
            ]
        )

        clip_x = torch.stack([clip_norm(img) for img in x])
        vae_x = vae_norm(x)

        clip_results = self.clip_model.visual(clip_x, return_dict=True)
        clip_feat, clip_visual = clip_results["x"], clip_results["patch"]

        # VQ-VAE encode and quantize (frozen)
        with torch.no_grad():
            results = self.vq_model.encode(x)
            z, z_q, indices = results[0][1], results[0][0], results[2][2].reshape(B, -1)

        # Residual features: quantized - pre-quantization
        residual = z_q - torch.einsum("b h w c -> b c h w", z)

        # Compute frequency difference signal
        real_freq = self.real_codebook_count / (self.real_codebook_count.sum() + 1e-8)
        fake_freq = self.fake_codebook_count / (self.fake_codebook_count.sum() + 1e-8)
        freq_diff = (
            torch.zeros_like(real_freq)
            if self.freq_log_counter < 200
            else real_freq - fake_freq
        )

        # Local feature processing with D3AT
        feat = self.feature_processor[:-1](residual)
        freq_diff_expanded = (
            freq_diff[indices]
            .reshape(indices.shape[0], residual.shape[-2], residual.shape[-1])
            .unsqueeze(1)
        )
        feat = self.feature_processor[-1](feat, freq_diff_expanded)

        # Global pooling and projections
        mixed_features = self.feature_pooler(feat).squeeze(-1).squeeze(-1)
        mixed_features = self.mixed_proj(mixed_features)
        clip_feat = self.clip_proj(clip_feat)

        # Fusion and classification
        fused_features = torch.cat([mixed_features, clip_feat], dim=1)
        out = self.fc(fused_features)

        # Update codebook frequency statistics during training
        if self.training and labels is not None:
            self.update_codebook_frequency(indices, labels)

        return out


class D3AT_blocks(nn.Module):
    """Transformer block with frequency-aware attention bias."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Q/K/V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.freq_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Learnable gate for frequency bias
        self.freq_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._reset_parameters()
        # Per-head transformations for frequency bias
        self.W_q = nn.Linear(self.head_dim, self.head_dim)
        self.W_k = nn.Linear(self.head_dim, self.head_dim)

    def _reset_parameters(self):
        """Specialized initialization for transformer components."""
        nn.init.ones_(self.freq_scale)

        if hasattr(self, "qkv"):
            nn.init.xavier_uniform_(self.qkv.weight, gain=1 / math.sqrt(2))
            if self.qkv.bias is not None:
                nn.init.zeros_(self.qkv.bias)

        if hasattr(self, "out_proj"):
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

        if hasattr(self, "mlp"):
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, freq_diff):
        """
        Args:
            x: [B, L, C]
            freq_diff: [B, L]
        """
        shortcut = x
        x = self.norm1(x)

        B, L, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Frequency-aware bias
        freq_bias = self.freq_proj(freq_diff.unsqueeze(-1))  # [B, L, C]
        freq_bias = freq_bias.view(B, L, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        query_bias = self.W_q(freq_bias)  # [B, num_heads, L, head_dim]
        key_bias = self.W_k(freq_bias)  # [B, num_heads, L, head_dim]

        freq_attn_bias = query_bias @ key_bias.transpose(-2, -1)  # [B, heads, L, L]

        # Attention with gated frequency bias
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.freq_scale * freq_attn_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.out_proj(x)
        x = self.dropout(x)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


class D3AT(nn.Module):
    """Stacked transformer that consumes spatial features with frequency guidance."""

    def __init__(self, dim, num_heads, depth, token_num=256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, token_num, dim))

        self.layers = nn.ModuleList(
            [
                D3AT_blocks(
                    embed_dim=dim, num_heads=num_heads, ff_dim=dim * 4, dropout=0.1
                )
                for _ in range(depth)
            ]
        )

        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x, freq_diff):
        """
        Args:
            x: [B, C, H, W]
            freq_diff: [B, H, W]
        """
        B, C, H, W = x.shape

        # To sequence
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        x = x + self.pos_embed[:, : H * W]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, freq_diff.view(B, H * W))

        # To spatial
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return x
