# Vendored from github.com/HuiGuanLab/LTF (MIT License, Copyright (c) 2026 HuiGuanLab), commit a12bab9, file ltf/mil/cross_shuffle.py.
# Unchanged below this header, so the released checkpoint loads key-for-key.
import torch
import torch.nn as nn
from typing import Optional


class CrossTokenShuffle(nn.Module):
    """Insert real-bag tokens; real bags draw donors only from other images.

    min_keep_original caps replacement so fake bags retain original instances.
    Inserted tokens are detached by the training recipe.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        replace_ratio: float = 0.5,
        with_replacement: bool = True,
        detach_inserted: bool = True,
        generator: Optional[torch.Generator] = None,
        min_keep_original: int = 0,
    ) -> torch.Tensor:
        assert x.dim() == 3, "x must be (B, N, D)"
        B, N, D = x.shape
        if B == 0 or N == 0:
            return x
        device = x.device
        dtype = x.dtype
        K = int(N * replace_ratio)
        mk = int(min_keep_original)
        if mk > 0:
            K = min(K, max(0, N - mk))
        if K <= 0:
            return x
        x_flat = x.reshape(B * N, D)
        y = y.to(device).view(B)
        token_labels = y.repeat_interleave(N)
        real_flat = torch.nonzero(token_labels == 0, as_tuple=False).view(-1)
        M = int(real_flat.numel())
        if M == 0:
            return x.clone()
        owner = real_flat // N
        b_ar = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1)
        rows_fake = (y == 1).unsqueeze(1)
        allowed_m = rows_fake | (owner.unsqueeze(0) != b_ar)
        has_cand = allowed_m.any(dim=1)
        r_pos = torch.rand(B, N, device=device, generator=generator)
        replace_pos_idx = r_pos.argsort(dim=1)[:, :K].long()
        cand_flat_idx = torch.zeros(B, K, device=device, dtype=torch.long)
        if with_replacement:
            rows_with = has_cand
            if rows_with.any():
                weights = allowed_m[rows_with].float()
                idx_m = torch.multinomial(
                    weights, num_samples=K, replacement=True, generator=generator
                )
                cand_flat_idx[rows_with] = real_flat[idx_m]
        else:
            scores = torch.where(
                allowed_m,
                torch.zeros(B, M, device=device, dtype=torch.float32),
                torch.full((B, M), float("-inf"), device=device, dtype=torch.float32),
            )
            U = torch.rand(B, M, device=device, generator=generator).clamp_(
                1e-06, 1 - 1e-06
            )
            g = -torch.log(-torch.log(U))
            cand_m = (scores + g).topk(k=K, dim=1).indices
            cand_flat_idx = real_flat[cand_m]
        inserted_tokens = x_flat.index_select(0, cand_flat_idx.reshape(-1)).reshape(
            B, K, D
        )
        if detach_inserted:
            inserted_tokens = inserted_tokens.detach()
        x_out = x.clone()
        if has_cand.any():
            rows = torch.where(has_cand)[0]
            pos_idx = replace_pos_idx.index_select(0, rows)
            batch_idx = rows.unsqueeze(1).expand(-1, pos_idx.size(1))
            ins = inserted_tokens.index_select(0, rows)
            x_out[batch_idx, pos_idx, :] = ins
        return x_out.to(dtype=dtype, device=device)
