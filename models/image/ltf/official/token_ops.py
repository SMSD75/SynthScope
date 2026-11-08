# Vendored from github.com/HuiGuanLab/LTF (MIT License, Copyright (c) 2026 HuiGuanLab), commit a12bab9, file ltf/token_ops.py.
# Unchanged below this header, so the released checkpoint loads key-for-key.
import torch
from torch import nn


class BagTokenShuffle(nn.Module):
    def forward(
        self,
        x,
        p_shuffle: float = 0.0,
        p_drop: float = 0.0,
        key_padding_mask=None,
        **kwargs,
    ):
        del p_drop, key_padding_mask, kwargs
        if not self.training or p_shuffle <= 0:
            return x
        if float(torch.rand((), device=x.device)) >= float(p_shuffle):
            return x
        B, N, D = x.shape
        if N <= 1:
            return x
        perm = torch.rand(B, N, device=x.device).argsort(dim=1)
        return x.gather(1, perm.unsqueeze(-1).expand(B, N, D))
