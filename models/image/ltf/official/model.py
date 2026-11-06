# Vendored from github.com/HuiGuanLab/LTF (MIT License, Copyright (c) 2026 HuiGuanLab), commit a12bab9, file ltf/model.py.
# Unchanged below this header, so the released checkpoint loads key-for-key.
"""Hierarchical MIL and gated fusion used by the released LTF checkpoint."""

from typing import Optional, Tuple
import torch
from torch import nn
from .mil import MILCore
from .token_ops import BagTokenShuffle


class LTFCore(nn.Module):
    """Pool [batch, views, tokens, channels], with the global view at index 0."""

    def __init__(
        self,
        embed_dim=1024,
        num_classes=2,
        dropout=0.1,
        mil_heads=8,
        fusion_hidden=512,
        l1_cross_shuffle_p: float = 0.9,
        l1_replace_ratio_range: Tuple[float, float] = (0.5, 0.9),
        l1_with_replacement_p: float = 0.5,
        l2_cross_shuffle_p: float = 0.5,
        l2_replace_ratio_range: Tuple[float, float] = (0.1, 0.5),
        l2_with_replacement_p: float = 0.5,
        bag_token_shuffle_p: float = 0.5,
        cross_shuffle_min_keep_original: int = 0,
        use_mil_cross_shuffle: bool = True,
        pool_gate_hidden: int = 16,
        use_pool_gate_fusion: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_mil_cross_shuffle = bool(use_mil_cross_shuffle)
        self.l1_cross_shuffle_p = float(l1_cross_shuffle_p)
        self.l1_replace_ratio_range = (
            float(l1_replace_ratio_range[0]),
            float(l1_replace_ratio_range[1]),
        )
        self.l1_with_replacement_p = float(l1_with_replacement_p)
        self.l2_cross_shuffle_p = float(l2_cross_shuffle_p)
        self.l2_replace_ratio_range = (
            float(l2_replace_ratio_range[0]),
            float(l2_replace_ratio_range[1]),
        )
        self.l2_with_replacement_p = float(l2_with_replacement_p)
        self.bag_token_shuffle_p = float(bag_token_shuffle_p)
        self.cross_shuffle_min_keep_original = int(cross_shuffle_min_keep_original)
        self.use_pool_gate_fusion = bool(use_pool_gate_fusion)
        mil_kw = dict(
            attn_hidden=min(256, embed_dim),
            attn_dropout=dropout,
            attn_bag_mode="embed_cls",
        )
        self.mil_level1 = MILCore(
            d_in=embed_dim,
            num_classes=num_classes,
            mil_head_pooling="attn",
            mil_head_kwargs=mil_kw,
            use_cross_shuffle=self.use_mil_cross_shuffle,
            trans=BagTokenShuffle(),
            allow_trans_token_drop=False,
        )
        self.mil_level2 = MILCore(
            d_in=embed_dim,
            num_classes=num_classes,
            mil_head_pooling="attn",
            mil_head_kwargs=mil_kw,
            use_cross_shuffle=self.use_mil_cross_shuffle,
            trans=BagTokenShuffle(),
            allow_trans_token_drop=False,
        )
        self.level1_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        mh = mil_heads if embed_dim % mil_heads == 0 else 8
        while embed_dim % mh != 0:
            mh -= 1
        if mh < 1:
            mh = 1
        self.level1_aggregate_attn = nn.MultiheadAttention(
            embed_dim, mh, dropout=dropout, batch_first=True
        )
        self.head_level1 = nn.Linear(embed_dim, num_classes)
        self.head_level2 = nn.Linear(embed_dim, num_classes)
        self.tile_inst_head = nn.Linear(embed_dim, num_classes)
        fusion_in = embed_dim + embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, num_classes),
        )
        gate_in = 2 * num_classes
        self.pool_gate = nn.Sequential(
            nn.Linear(gate_in, int(pool_gate_hidden)),
            nn.ReLU(),
            nn.Linear(int(pool_gate_hidden), 1),
        )
        nn.init.trunc_normal_(self.level1_query, std=0.02)

    @staticmethod
    def _sample_replace_ratio(low: float, high: float, device: torch.device) -> float:
        lo, hi = (min(low, high), max(low, high))
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if hi <= lo:
            return float(lo)
        u = torch.rand((), device=device)
        return float(lo + (hi - lo) * u)

    @staticmethod
    def _logit_pool_from_mil_head(mil_head_out: dict) -> torch.Tensor:
        inst = mil_head_out.get("inst_logits")
        attn = mil_head_out.get("attn")
        if inst is None or attn is None:
            raise RuntimeError(
                "logit_pool requires MILHead outputs: inst_logits and attn."
            )
        return (attn * inst).sum(dim=1)

    def forward(
        self,
        x,
        labels: Optional[torch.Tensor] = None,
        x_mil: Optional[torch.Tensor] = None,
        return_viz: bool = False,
    ):
        x_feat = x
        if x_mil is not None:
            if x_mil.shape != x.shape:
                raise ValueError(
                    f"x_mil shape {tuple(x_mil.shape)} != x shape {tuple(x.shape)}"
                )
            x_mil_feat = x_mil
        else:
            x_mil_feat = x_feat
        b, n, t, d = x_mil_feat.shape
        if n < 2:
            raise ValueError(f"LTFCore expects N>=2 (global + tiles), got N={n}")
        nt = n - 1
        train_shuffle = self.training and labels is not None
        p_bag_shuffle = self.bag_token_shuffle_p if train_shuffle else 0.0
        # L1: pool patch tokens within each local tile, then aggregate tile bags.
        patch_only = x_mil_feat[:, 1:, 1:, :].reshape(b * nt, t - 1, d)
        do1 = (
            train_shuffle
            and float(torch.rand((), device=x.device)) < self.l1_cross_shuffle_p
        )
        rr1 = self._sample_replace_ratio(*self.l1_replace_ratio_range, x.device)
        wr1 = bool(float(torch.rand((), device=x.device)) < self.l1_with_replacement_p)
        y1 = (
            labels.unsqueeze(1).expand(-1, nt).reshape(-1)
            if labels is not None
            else None
        )
        mk = self.cross_shuffle_min_keep_original
        mil1 = self.mil_level1(
            patch_only,
            y=y1,
            return_dict=True,
            do_cross_shuffle=do1,
            replace_ratio=rr1,
            with_replacement=wr1,
            detach_inserted=True,
            cross_shuffle_min_keep=mk,
            p_shuffle=p_bag_shuffle,
        )
        be = mil1["mil_head"]["bag_embed"]
        if be is None:
            raise RuntimeError("MIL level1 expected bag_embed (use attn + embed_cls).")
        bag_seq = be.reshape(b, nt, d)
        q = self.level1_query.expand(b, -1, -1)
        agg_out, attn_w = self.level1_aggregate_attn(
            q, bag_seq, bag_seq, need_weights=True, average_attn_weights=False
        )
        level1_emb = agg_out.squeeze(1)
        if attn_w is None:
            raise RuntimeError("Expected attention weights from level1_aggregate_attn.")
        tile_attn = attn_w.mean(dim=1).squeeze(1)
        tile_inst_logits = self.tile_inst_head(bag_seq)
        level1_logit_pool = (tile_attn.unsqueeze(-1) * tile_inst_logits).sum(dim=1)
        # L2 includes the global CLS token and every local tile CLS token.
        cls_seq = x_mil_feat[:, :, 0, :]
        do2 = (
            train_shuffle
            and float(torch.rand((), device=x.device)) < self.l2_cross_shuffle_p
        )
        rr2 = self._sample_replace_ratio(*self.l2_replace_ratio_range, x.device)
        wr2 = bool(float(torch.rand((), device=x.device)) < self.l2_with_replacement_p)
        mil2 = self.mil_level2(
            cls_seq,
            y=labels,
            return_dict=True,
            do_cross_shuffle=do2,
            replace_ratio=rr2,
            with_replacement=wr2,
            detach_inserted=True,
            cross_shuffle_min_keep=mk,
            p_shuffle=p_bag_shuffle,
        )
        mil2_head = mil2["mil_head"]
        be2 = mil2_head["bag_embed"]
        if be2 is None:
            raise RuntimeError("MIL level2 expected bag_embed (use attn + embed_cls).")
        level2_emb = be2
        level2_logit_pool = self._logit_pool_from_mil_head(mil2_head)
        logits_level1 = self.head_level1(level1_emb)
        logits_level2 = self.head_level2(level2_emb)
        if self.use_pool_gate_fusion:
            # Gate features using pooled logits; this is not probability averaging.
            gate_in = torch.cat([level1_logit_pool, level2_logit_pool], dim=-1)
            g = torch.sigmoid(self.pool_gate(gate_in))
            fusion_in = torch.cat([g * level1_emb, (1.0 - g) * level2_emb], dim=-1)
        else:
            fusion_in = torch.cat([level1_emb, level2_emb], dim=-1)
        logits_fusion = self.fusion_head(fusion_in)
        out = {
            "logits_level1": logits_level1,
            "logits_level2": logits_level2,
            "logits_fusion": logits_fusion,
        }
        if return_viz:
            mil1_head = mil1.get("mil_head", {})
            out["viz"] = {
                "logits_level1": logits_level1,
                "logits_level2": logits_level2,
                "logits_fusion": logits_fusion,
                "level1_logit_pool": level1_logit_pool,
                "level2_logit_pool": level2_logit_pool,
                "level1_emb": level1_emb,
                "level2_emb": level2_emb,
                "fusion_in": fusion_in,
                "tile_attn": tile_attn,
                "tile_inst_logits": tile_inst_logits,
                "level1_patch_attn": mil1_head.get("attn"),
                "level1_patch_inst_logits": mil1_head.get("inst_logits"),
                "level2_attn": mil2_head.get("attn"),
                "level2_inst_logits": mil2_head.get("inst_logits"),
                "num_tiles": torch.full(
                    (b, 1), nt, device=logits_fusion.device, dtype=torch.int32
                ),
                "num_patches_per_tile": torch.full(
                    (b, 1), t - 1, device=logits_fusion.device, dtype=torch.int32
                ),
                "gate": g
                if self.use_pool_gate_fusion
                else torch.full(
                    (b, 1), 0.5, device=logits_fusion.device, dtype=logits_fusion.dtype
                ),
                "use_pool_gate_fusion": torch.tensor(
                    [1 if self.use_pool_gate_fusion else 0],
                    device=logits_fusion.device,
                    dtype=logits_fusion.dtype,
                ),
            }
        return out


class LTF(nn.Module):
    """Frozen CLIP tokens followed by patch MIL, view MIL and gated fusion."""

    def __init__(self, clip_visual, feature_layer=None, **core_options):
        super().__init__()
        self.visual = clip_visual
        self.visual.requires_grad_(False)
        self.visual.eval()
        self.input_resolution = int(self.visual.input_resolution)
        self.embed_dim = self.visual.conv1.out_channels
        self.feature_layer = (
            len(self.visual.transformer.resblocks)
            if feature_layer is None
            else feature_layer
        )
        if not 1 <= self.feature_layer <= len(self.visual.transformer.resblocks):
            raise ValueError("feature_layer is outside the CLIP transformer.")
        self.core = LTFCore(embed_dim=self.embed_dim, **core_options)
        self.eval_chunk_size = 8

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        return self

    def encode_tokens(self, images):
        """Read hidden tokens before CLIP final normalization and projection."""
        visual = self.visual
        x = visual.conv1(images.to(visual.conv1.weight.dtype))
        x = x.flatten(2).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = visual.ln_pre(
            torch.cat((cls, x), dim=1) + visual.positional_embedding.to(x.dtype)
        )
        x = x.permute(1, 0, 2)
        for block in list(visual.transformer.resblocks)[: self.feature_layer]:
            x = block(x)
        return x.permute(1, 0, 2)

    def forward(self, x, labels=None, chunk_size=256, return_viz=False):
        batch, views, channels, height, width = x.shape
        images = x.reshape(batch * views, channels, height, width)
        step = self.eval_chunk_size
        with torch.no_grad():
            tokens = torch.cat(
                [
                    self.encode_tokens(images[start : start + step])
                    for start in range(0, len(images), step)
                ],
                dim=0,
            )
        features = tokens.reshape(
            batch, views, tokens.shape[1], tokens.shape[2]
        ).float()
        return self.core(features, labels=labels, return_viz=return_viz)
