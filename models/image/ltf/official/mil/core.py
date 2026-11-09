# Vendored from github.com/HuiGuanLab/LTF (MIT License, Copyright (c) 2026 HuiGuanLab), commit a12bab9, file ltf/mil/core.py.
# Unchanged below this header, so the released checkpoint loads key-for-key.
from __future__ import annotations
import inspect
from typing import Any, Dict, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
from .cross_shuffle import CrossTokenShuffle
from .head import MILHead


def _forward_param_names(module: nn.Module) -> Set[str]:
    try:
        return set(inspect.signature(module.forward).parameters.keys())
    except (TypeError, ValueError):
        return set()


class CrossShuffleStage(nn.Module):
    def __init__(self, use: bool = True) -> None:
        super().__init__()
        self._use = use
        self.op: Optional[CrossTokenShuffle]
        if use:
            self.op = CrossTokenShuffle()
        else:
            self.op = None

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        *,
        apply: bool,
        replace_ratio: float = 0.5,
        with_replacement: bool = False,
        detach_inserted: bool = True,
        min_keep_original: int = 0,
    ) -> torch.Tensor:
        if not self._use or not apply or self.op is None or (y is None):
            return x
        if y.dim() != 1 or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"y must be 1D [B], got {tuple(y.shape)} for B={x.shape[0]}"
            )
        return self.op(
            x,
            y,
            replace_ratio=replace_ratio,
            with_replacement=with_replacement,
            detach_inserted=detach_inserted,
            min_keep_original=min_keep_original,
        )


class TransStage(nn.Module):
    def __init__(self, core: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.core = core
        self._params = _forward_param_names(core) if core is not None else set()

    def forward(
        self,
        x: torch.Tensor,
        *,
        p_shuffle: float = 0.0,
        p_drop: float = 0.0,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.core is None:
            return x
        if "p_shuffle" in self._params:
            call_kw: Dict[str, Any] = {}
            for name, val in (
                ("p_shuffle", p_shuffle),
                ("p_drop", p_drop),
                ("key_padding_mask", key_padding_mask),
            ):
                if name in self._params:
                    call_kw[name] = val
            for k, v in kwargs.items():
                if k in self._params:
                    call_kw[k] = v
            return self.core(x, **call_kw)
        kw2: Dict[str, Any] = {}
        if key_padding_mask is not None and "src_key_padding_mask" in self._params:
            kw2["src_key_padding_mask"] = key_padding_mask
        for k, v in kwargs.items():
            if k in self._params:
                kw2[k] = v
        return self.core(x, **kw2) if kw2 else self.core(x)


class MILHeadStage(nn.Module):
    def __init__(self, **mil_kw: Any) -> None:
        super().__init__()
        self.head = MILHead(**mil_kw)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        return self.head(x, mask=mask)


class MILCore(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_classes: int = 2,
        mil_head_pooling: str = "attn",
        mil_head_kwargs: Optional[Dict[str, Any]] = None,
        *,
        use_cross_shuffle: bool = True,
        trans: Optional[nn.Module] = None,
        allow_trans_token_drop: bool = False,
    ) -> None:
        super().__init__()
        self.allow_trans_token_drop = bool(allow_trans_token_drop)
        self.shuffle = CrossShuffleStage(use=use_cross_shuffle)
        self.trans = TransStage(core=trans)
        mil_kw: Dict[str, Any] = {
            "d_in": d_in,
            "num_classes": num_classes,
            "pooling": mil_head_pooling,
        }
        if mil_head_kwargs:
            mil_kw.update(mil_head_kwargs)
        self.head = MILHeadStage(**mil_kw)

    def forward(
        self,
        tokens: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        do_cross_shuffle: bool = False,
        replace_ratio: float = 0.5,
        with_replacement: bool = False,
        detach_inserted: bool = True,
        cross_shuffle_min_keep: int = 0,
        p_shuffle: float = 0.0,
        p_drop: float = 0.0,
        trans_key_padding_mask: Optional[torch.Tensor] = None,
        trans_extra: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        x = self.shuffle(
            tokens,
            y,
            apply=do_cross_shuffle,
            replace_ratio=replace_ratio,
            with_replacement=with_replacement,
            detach_inserted=detach_inserted,
            min_keep_original=cross_shuffle_min_keep,
        )
        trans_p_drop = float(p_drop) if self.allow_trans_token_drop else 0.0
        tex = dict(trans_extra or {})
        trans_out = self.trans(
            x,
            p_shuffle=p_shuffle,
            p_drop=trans_p_drop,
            key_padding_mask=trans_key_padding_mask,
            **tex,
        )
        if isinstance(trans_out, tuple) and len(trans_out) == 2:
            x, trans_mil_mask = trans_out
            if mask is not None:
                raise ValueError(
                    "MILCore: `trans` returned (x, mil_valid); omit `mask` and use the trans-stage mask."
                )
            mask = trans_mil_mask
        else:
            x = trans_out
        if mask is not None and mask.shape != x.shape[:2]:
            raise ValueError(
                f"mask must be [B,N], got {tuple(mask.shape)} vs x {tuple(x.shape[:2])}"
            )
        mil_out = self.head(x, mask=mask)
        if not return_dict:
            return mil_out["bag_logits"]
        return {"bag_logits": mil_out["bag_logits"], "mil_head": mil_out, "features": x}

    def forward_logits(
        self,
        tokens: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        extra = dict(kwargs)
        extra.pop("return_dict", None)
        return self.forward(tokens, y, mask, return_dict=False, **extra)


CrossShuffleTransMIL = MILCore
