# Vendored from github.com/HuiGuanLab/LTF (MIT License, Copyright (c) 2026 HuiGuanLab), commit a12bab9, file ltf/mil/head.py.
# Unchanged below this header, so the released checkpoint loads key-for-key.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MILHead(nn.Module):
    """Instance classification and bag pooling. LTF always selects gated attention."""

    def __init__(
        self,
        d_in: int = 1024,
        num_classes: int = 2,
        pooling: str = "lsep",
        lsep_temp: float = 1.0,
        lsep_norm: bool = True,
        topk: int = None,
        topk_ratio: float = 0.1,
        attn_hidden: int = 128,
        attn_dropout: float = 0.0,
        attn_bag_mode: str = "embed_cls",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooling = pooling.lower()
        self.lsep_temp = float(lsep_temp)
        self.lsep_norm = bool(lsep_norm)
        self.topk = topk
        self.topk_ratio = float(topk_ratio)
        self.attn_bag_mode = attn_bag_mode.lower()
        if self.pooling not in {"max", "lsep", "topk", "attn"}:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        if self.attn_bag_mode not in {"logit_pool", "embed_cls"}:
            raise ValueError(f"Unknown attn_bag_mode: {self.attn_bag_mode}")
        self.classifier = nn.Linear(d_in, num_classes)
        if self.pooling == "attn":
            self.attn_V = nn.Linear(d_in, attn_hidden, bias=True)
            self.attn_U = nn.Linear(d_in, attn_hidden, bias=True)
            self.attn_w = nn.Linear(attn_hidden, 1, bias=False)
            self.attn_drop = nn.Dropout(attn_dropout)
            if self.attn_bag_mode == "embed_cls":
                self.bag_classifier = nn.Linear(d_in, num_classes)
            else:
                self.bag_classifier = None
        else:
            self.bag_classifier = None

    @staticmethod
    def _masked_fill_min(x, mask, fill_value=None):
        if mask is None:
            return x
        if fill_value is None:
            fill_value = torch.finfo(x.dtype).min
        return x.masked_fill(~mask, fill_value)

    def _pool_max(self, logits, mask=None):
        logits = self._masked_fill_min(
            logits, mask.unsqueeze(-1) if mask is not None else None
        )
        bag = logits.max(dim=1).values
        return bag

    def _pool_lsep(self, logits, mask=None):
        z = logits / self.lsep_temp
        if mask is not None:
            neg_inf = torch.finfo(z.dtype).min
            z = z.masked_fill(~mask.unsqueeze(-1), neg_inf)
        bag = torch.logsumexp(z, dim=1)
        if self.lsep_norm:
            if mask is None:
                n = logits.size(1)
                bag = bag - math.log(max(n, 1))
            else:
                n_eff = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                bag = bag - n_eff.log()
        return bag

    def _pool_topk(self, logits, mask=None):
        B, N, C = logits.shape
        if mask is not None:
            logits = self._masked_fill_min(logits, mask.unsqueeze(-1))
            n_eff = mask.sum(dim=1)
        else:
            n_eff = logits.new_full((B,), N, dtype=torch.long)
        k = self.topk
        if k is None:
            k = torch.clamp((n_eff.float() * self.topk_ratio).long(), min=1)
        else:
            k = torch.clamp(torch.full_like(n_eff, k), min=1, max=N)
        scores = logits.transpose(1, 2)
        topk_vals = []
        for b in range(B):
            kb = int(min(int(k[b].item()), int(n_eff[b].item())))
            vals, _ = scores[b].topk(kb, dim=-1)
            topk_vals.append(vals.mean(dim=-1))
        bag = torch.stack(topk_vals, dim=0)
        return bag

    def _compute_attn(self, feats, mask=None):
        Vh = self.attn_V(feats)
        Uh = self.attn_U(feats)
        gate = torch.tanh(Vh) * torch.sigmoid(Uh)
        gate = self.attn_drop(gate)
        attn = self.attn_w(gate)
        if mask is not None:
            neg_inf = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(~mask.unsqueeze(-1), neg_inf)
        attn = F.softmax(attn, dim=1)
        return attn

    def _pool_attn(self, feats, logits, mask=None):
        attn = self._compute_attn(feats, mask)
        bag_embed = (attn * feats).sum(dim=1)
        if self.attn_bag_mode == "logit_pool":
            bag_logits = (attn * logits).sum(dim=1)
        elif self.attn_bag_mode == "embed_cls":
            bag_logits = self.bag_classifier(bag_embed)
        else:
            raise ValueError(f"Unknown attn_bag_mode: {self.attn_bag_mode}")
        return (bag_logits, attn, bag_embed)

    def forward(self, feats, mask=None):
        inst_logits = self.classifier(feats)
        attn = None
        bag_embed = None
        if self.pooling == "max":
            bag_logits = self._pool_max(inst_logits, mask)
        elif self.pooling == "lsep":
            bag_logits = self._pool_lsep(inst_logits, mask)
        elif self.pooling == "topk":
            bag_logits = self._pool_topk(inst_logits, mask)
        elif self.pooling == "attn":
            bag_logits, attn, bag_embed = self._pool_attn(feats, inst_logits, mask)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        return {
            "inst_logits": inst_logits,
            "bag_logits": bag_logits,
            "attn": attn,
            "bag_embed": bag_embed,
        }
