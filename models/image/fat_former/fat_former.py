"""FatFormer integrated into AGID without vendoring a CLIP fork.

The original FatFormer (Liu et al., CVPR 2024) is built on a modified copy of
OpenAI's CLIP that adds:

1. A Forgery-Aware Adapter (FAA) injected into a subset of visual
   transformer blocks.
2. A `return_full` flag on the visual encoder so the full patch-token
   sequence is exposed.
3. A Transformer that returns a per-layer dict alongside the final hidden
   state.

Only (1) is a real architectural change. (2) is a 10-line re-implementation of
stock CLIP's `VisionTransformer.forward`, and (3) is cosmetic (only the final
hidden state is ever consumed). So this file rebuilds FatFormer on top of the
already-installed `clip` package, by attaching FAA modules to selected blocks
and rebinding their `forward`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer
from omegaconf import DictConfig
from pytorch_wavelets import DWTForward, DWTInverse


_tokenizer = SimpleTokenizer()


class ForgeryAwareAdapterLayer(nn.Module):
    """Forgery-Aware Adapter (verbatim layout from FatFormer's clip/model.py).

    Combines a wavelet-domain intra/inter-band self-attention branch with a
    bottleneck-conv branch. The two outputs are summed (the wavelet branch is
    scaled by a learnable `freq_scale` initialised to zero so the adapter
    starts as a no-op).
    """

    def __init__(
        self,
        d_model: int = 1024,
        bottleneck: int = 64,
        dropout: float = 0.1,
        kernel_size: int = 1,
        adapter_scalar: str = "0.1",
        head: int = 8,
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.d_model = d_model
        self.scale = float(adapter_scalar)

        self.first_conv_layer = nn.Conv1d(self.n_embd, self.down_size, kernel_size)
        self.non_linear_func = nn.ReLU()
        self.second_conv_layer = nn.Conv1d(self.down_size, self.n_embd, kernel_size)
        self.dropout = dropout

        self.freq_scale = nn.Parameter(torch.zeros(1))
        self.dwt_transform = DWTForward(J=1, wave="haar")
        self.idwt_transform = DWTInverse(wave="haar")

        self.dwt_norm = nn.GroupNorm(128, d_model)
        self.intra_band = nn.MultiheadAttention(d_model, head)
        self.dropout_intra = nn.Dropout(dropout)
        self.norm_intra = nn.LayerNorm(d_model)

        self.inter_band = nn.MultiheadAttention(d_model, head)
        self.dropout_inter = nn.Dropout(dropout)
        self.norm_inter = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward_freq(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        nq = x.shape[2]
        q = k = v = x.transpose(0, 1).flatten(1, 2)
        tgt2 = self.intra_band(q, k, v)[0].reshape(C, B, nq, self.d_model).transpose(0, 1)
        x = self.norm_intra(x + self.dropout_intra(tgt2))

        q = k = v = x.flatten(0, 1).transpose(0, 1)
        tgt2 = self.inter_band(q, k, v)[0].transpose(0, 1).reshape(B, C, nq, self.d_model)
        x = self.norm_inter(x + self.dropout_inter(tgt2))

        return self.forward_ffn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nq, bs, md = x.shape
        side = int(math.sqrt(nq))

        img_patches = (
            x[1:].transpose(0, 1).reshape(bs, side, side, md).permute(0, 3, 1, 2)
        )
        dwt_l, dwt_h = self.dwt_transform(img_patches)
        dwt_img = torch.cat([dwt_l[:, :, None], dwt_h[0]], dim=2)
        hh, ww = dwt_img.shape[-2:]
        dwt_img = dwt_img.flatten(-2)
        dwt_img = self.dwt_norm(dwt_img).permute(0, 2, 3, 1)
        dwt_feature = self.forward_freq(dwt_img)
        dwt_feature = dwt_feature.reshape(bs, 4, hh, ww, md).permute(0, 4, 1, 2, 3)
        dwt_feature = (
            self.idwt_transform((dwt_feature[:, :, 0], [dwt_feature[:, :, 1:]]))
            .flatten(-2)
            .permute(2, 0, 1)
        )
        dwt_feature = torch.cat([torch.zeros_like(x[:1]), dwt_feature], dim=0)

        down = self.first_conv_layer(x.permute(1, 2, 0))
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.second_conv_layer(down).permute(2, 0, 1) * self.scale

        return up + dwt_feature * self.freq_scale


def _attach_faa(block: nn.Module, d_model: int) -> None:
    """Attach an FAA to a stock CLIP `ResidualAttentionBlock` and rebind forward."""

    block.forgery_aware_adapter = ForgeryAwareAdapterLayer(d_model=d_model)

    def faa_forward(x: torch.Tensor) -> torch.Tensor:
        x = x + block.attention(block.ln_1(x))
        adapt_x = block.forgery_aware_adapter(x)
        x = x + block.mlp(block.ln_2(x)) + adapt_x
        return x

    block.forward = faa_forward  # type: ignore[assignment]


def _visual_forward(
    visual: nn.Module, image: torch.Tensor, return_full: bool = False
) -> torch.Tensor:
    """Re-implements stock CLIP `VisionTransformer.forward` with optional full-token output."""

    x = visual.conv1(image)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat(
        [
            visual.class_embedding.to(x.dtype)
            + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x,
        ],
        dim=1,
    )
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x) if return_full else visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        x = x @ visual.proj
    return x


class TextEncoder(nn.Module):
    """Mirror of FatFormer's TextEncoder, using stock CLIP's `Transformer(x) -> x`."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot_idx = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection
        return x


class LanguageGuidedAlignment(nn.Module):
    """Patch-based prompt learner with text-guided enhancer (FatFormer LGA)."""

    def __init__(self, clip_model: nn.Module, classnames, args):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = int(args.num_context_embedding)
        ctx_init = str(args.init_context_embedding) if args.init_context_embedding else ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :], persistent=False)
        self.register_buffer("tokenized_prompts", tokenized_prompts, persistent=False)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens

        d_model = clip_model.ln_final.weight.shape[0]
        d_ffn = d_model * 4
        self.patch_basaed_enhancer = nn.MultiheadAttention(d_model, num_heads=12)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_FFN(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        return self.norm2(tgt)

    @staticmethod
    def construct_prompts(ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self, im_features: torch.Tensor) -> torch.Tensor:
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        tgt = ctx[:, None].repeat_interleave(im_features.shape[0], dim=1)
        tgt2 = self.patch_basaed_enhancer(
            tgt, im_features.transpose(0, 1), im_features.transpose(0, 1)
        )[0]
        tgt = self.norm1(tgt + tgt2)
        tgt = self.forward_FFN(tgt)

        ctx_shifted = tgt.transpose(0, 1)

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        return torch.stack(prompts)


class AGIDFatFormerModel(nn.Module):
    """FatFormer adapted to AGID's `compute_loss(batch) -> {loss, outputs}` API.

    The forward path mirrors FatFormer's `CLIPModel.forward`:

    1. Encode the image with the FAA-augmented visual transformer, returning
       the full patch-token sequence.
    2. Run Language-Guided Alignment to obtain class-specific text prompts
       conditioned on the patch features.
    3. Encode the prompts with the (frozen) CLIP text encoder.
    4. Compute zero-shot-style logits between the CLS image feature and each
       text feature (`logits`).
    5. Run a text-guided interactor over patch tokens to obtain augmented
       image features, and compute a second set of logits (`aug_logits`).
    6. Return `logits + aug_logits` as two-class real/fake logits.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        backbone_name = config.ModelConfig.backbone_name
        if backbone_name.startswith("CLIP:"):
            backbone_name = backbone_name.removeprefix("CLIP:")

        self.clip_model, _ = clip.load(backbone_name, device="cpu", jit=False)
        if config.ModelConfig.get("force_fp32", True):
            self.clip_model.float()

        resblocks = self.clip_model.visual.transformer.resblocks
        layers = len(resblocks)
        d_model_visual = resblocks[0].ln_1.normalized_shape[0]

        n_faa = int(config.ModelConfig.get("num_vit_adapter", 8))
        if n_faa > 0:
            assert layers % n_faa == 0, (
                f"visual transformer layers ({layers}) must be divisible by "
                f"num_vit_adapter ({n_faa})"
            )
            step = layers // n_faa
            for i in range(n_faa):
                _attach_faa(resblocks[(i + 1) * step - 1], d_model=d_model_visual)

        lga_args = SimpleNamespace(
            num_context_embedding=int(
                config.ModelConfig.get("num_context_embedding", 8)
            ),
            init_context_embedding=config.ModelConfig.get(
                "init_context_embedding", ""
            ),
        )
        classnames = list(config.ModelConfig.get("classnames", ["real", "fake"]))
        self.language_guided_alignment = LanguageGuidedAlignment(
            self.clip_model, classnames=classnames, args=lga_args,
        )
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale

        d_model_text = self.clip_model.ln_final.weight.shape[0]
        d_ffn = d_model_text * 4
        self.text_guided_interactor = nn.MultiheadAttention(d_model_text, num_heads=12)
        self.norm1 = nn.LayerNorm(d_model_text)
        self.linear1 = nn.Linear(d_model_text, d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model_text)
        self.norm2 = nn.LayerNorm(d_model_text)

        if config.ModelConfig.get("freeze_backbone", True):
            self._freeze_clip_keep_adapters()

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.TrainerConfig.get("label_smoothing", 0.0)
        )

    def _freeze_clip_keep_adapters(self) -> None:
        for name, p in self.clip_model.named_parameters():
            p.requires_grad = "forgery_aware_adapter" in name

    @property
    def dtype(self) -> torch.dtype:
        return self.clip_model.dtype

    def _forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        return self.norm2(tgt)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logit_scale = self.logit_scale.exp()

        image_features = _visual_forward(
            self.clip_model.visual, image.type(self.dtype), return_full=True,
        )
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.language_guided_alignment(image_features)
        tokenized_prompts = self.language_guided_alignment.tokenized_prompts

        logits = []
        text_feature_list = []
        for pts_i, imf_i in zip(prompts, image_features_norm):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_feature_list.append(text_features)
            text_features_n = text_features / text_features.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * imf_i[0] @ text_features_n.t())
        logits = torch.stack(logits)

        text_features = torch.stack(text_feature_list, dim=1)
        tgt = image_features[:, 1:].transpose(0, 1)
        tgt2 = self.text_guided_interactor(tgt, text_features, text_features)[0]
        tgt = self.norm1(tgt + tgt2)
        tgt = self._forward_ffn(tgt)

        aug_image_features = tgt.transpose(0, 1).mean(dim=1)
        aug_image_features = aug_image_features / aug_image_features.norm(dim=-1, keepdim=True)
        text_features_n = text_features / text_features.norm(dim=-1, keepdim=True)
        aug_logits = []
        for pts_i, imf_i in zip(aug_image_features, text_features_n.transpose(0, 1)):
            aug_logits.append(logit_scale * pts_i @ imf_i.t())
        aug_logits = torch.stack(aug_logits)

        return logits + aug_logits

    def compute_loss(self, batch):
        images, labels, folder_ids = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        return {"loss": loss, "outputs": outputs}
