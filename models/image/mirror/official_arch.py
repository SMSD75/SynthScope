# Vendored from github.com/handsome-rich/MIRROR (commit 7f108fd), file models/mirror.py. Licence UNCLEAR: the README
# badge says MIT, but the repo has no LICENSE file (the badge's link 404s); see LICENSES.md.
#
# The detector code below the marked section is unchanged, so the released checkpoint-h-cur.pth loads key-for-key with
# strict=True. Three forced changes, all in the backbone wrapper (`DINO`), none of which alter the computation:
#   1. peft is not a SynthScope dependency: `get_peft_model` / `LoraConfig` are replaced by a small module that
#      reproduces peft 0.21.2's LoRA Linear forward and its module layout (`layer.N.base_model.model.<...>.q_proj.
#      base_layer / lora_A.default / lora_B.default`), so the checkpoint keys are the official ones.
#   2. The DINOv3 ViT-H+ can be built from its config (`DINOV3_VITH16PLUS`) instead of `AutoModel.from_pretrained`:
#      the phase-2 checkpoint holds the full backbone (frozen DINOv3 weights + LoRA), so inference needs neither the
#      gated Hugging Face download nor the phase-1 memory-bank file. A directory path still uses from_pretrained.
#   3. transformers >= 5 moved the ViT blocks from `dino.layer` to `dino.model.layer`. The official
#      `load_detector_checkpoint` renames between the two; a load_state_dict pre-hook on `DINO` does the same rename
#      for every load (the engine's plain `load_state_dict` included).
import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, DINOv3ViTConfig

# config.json of facebook/dinov3-vith16plus-pretrain-lvd1689m (revision c807c9eeea853df70aec4069e6f56b28ddc82acc), the
# official `--backbone_path dinov3-huge`; attention runs as SDPA, what from_pretrained selects in the official env.
DINOV3_VITH16PLUS = dict(
    attention_dropout=0.0, drop_path_rate=0.0, hidden_act="silu", hidden_size=1280, image_size=224,
    initializer_range=0.02, intermediate_size=5120, key_bias=False, layer_norm_eps=1e-05, layerscale_value=1.0,
    mlp_bias=True, num_attention_heads=20, num_channels=3, num_hidden_layers=32, num_register_tokens=4, patch_size=16,
    pos_embed_jitter=None, pos_embed_rescale=2.0, pos_embed_shift=None, proj_bias=True, query_bias=True,
    rope_theta=100.0, use_gated_mlp=True, value_bias=True,
)


# --- peft-free LoRA (replaces `from peft import get_peft_model, LoraConfig`) -----------------------------------------
class LoraConfig:
    """The fields of peft's LoraConfig that MIRROR sets (r=8, lora_alpha=16; lora_dropout 0, no rsLoRA: peft defaults)."""

    def __init__(self, r=8, lora_alpha=16, target_modules=(), lora_dropout=0.0):
        self.r, self.lora_alpha, self.target_modules, self.lora_dropout = r, lora_alpha, list(target_modules), lora_dropout


class LoraLinear(nn.Module):
    """peft's lora.Linear for one adapter named "default": base_layer(x) + lora_B(lora_A(dropout(x))) * alpha / r, with
    peft's input cast to the LoRA weight dtype and the result cast back to the base output dtype."""

    def __init__(self, base_layer: nn.Linear, config: LoraConfig):
        super().__init__()
        self.base_layer = base_layer
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(config.lora_dropout) if config.lora_dropout > 0
                                           else nn.Identity()})
        self.lora_A = nn.ModuleDict({"default": nn.Linear(base_layer.in_features, config.r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(config.r, base_layer.out_features, bias=False)})
        self.scaling = config.lora_alpha / config.r
        nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))   # peft init_lora_weights=True
        nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        lora_A, lora_B = self.lora_A["default"], self.lora_B["default"]
        if x.dtype != lora_A.weight.dtype:
            x = x.to(dtype=lora_A.weight.dtype)
        result = result + lora_B(lora_A(self.lora_dropout["default"](x))) * self.scaling
        return result.to(torch_result_dtype)


class LoraModel(nn.Module):
    """peft's LoraModel: holds the wrapped module as `model` (with its target Linear layers swapped for LoraLinear)."""

    def __init__(self, model: nn.Module, config: LoraConfig):
        super().__init__()
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, nn.Linear) and name.split(".")[-1] in config.target_modules]
        for name in targets:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoraLinear(getattr(parent, child), config))
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PeftModel(nn.Module):
    """peft's PeftModel (no task type): `base_model` is the LoraModel and forward passes straight through."""

    def __init__(self, model: nn.Module, config: LoraConfig):
        super().__init__()
        self.base_model = LoraModel(model, config)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


def get_peft_model(model: nn.Module, config: LoraConfig) -> PeftModel:
    return PeftModel(model, config)


def _load_backbone(dinov3_path):
    """`AutoModel.from_pretrained(path)` for a directory / hub id; a config dict builds the same model untrained."""
    if isinstance(dinov3_path, Mapping):
        return AutoModel.from_config(DINOv3ViTConfig(**dinov3_path), attn_implementation="sdpa")
    return AutoModel.from_pretrained(dinov3_path, weights_only=False)


def _config_name(config: Mapping) -> str:
    return f"config (DINOv3 ViT, width {config['hidden_size']}, {config['num_hidden_layers']} blocks)"


def _rename_block_prefix(module, state_dict, prefix, *args):
    """load_state_dict pre-hook of `DINO`: official checkpoints name the ViT blocks `dino.layer.N` (transformers 4.x)
    or `dino.model.layer.N` (5.x); rename to whichever this backbone has (the official load_detector_checkpoint rule)."""
    has_model = hasattr(module.dino, "model") and hasattr(module.dino.model, "layer")
    src, dst = (prefix + "dino.layer.", prefix + "dino.model.layer.") if has_model else \
        (prefix + "dino.model.layer.", prefix + "dino.layer.")
    for name in [k for k in state_dict if k.startswith(src)]:
        state_dict[dst + name[len(src):]] = state_dict.pop(name)


# --- official models/mirror.py (unchanged except for the marked lines in DINO.__init__) ------------------------------
class Norm(nn.Module):
    def __init__(self, mode='clip'):
        super().__init__()
        self.mode = mode
        if mode == 'clip':
            self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class DINO(nn.Module):
    def __init__(self, dinov3_path, finetune=True):
        super(DINO, self).__init__()
        print(f"Loading Backbone from: {dinov3_path if isinstance(dinov3_path, str) else _config_name(dinov3_path)}")  # changed (2)
        self.dino = _load_backbone(dinov3_path)                                       # changed (2)
        self.dino.requires_grad_(False)
        self.register_load_state_dict_pre_hook(_rename_block_prefix)                  # changed (3)


        model_type = getattr(self.dino.config, 'model_type', '')
        self.is_v3 = model_type.startswith('dinov3') or hasattr(self.dino, "layer") or "dinov3" in str(dinov3_path).lower()

        if finetune:
            self._apply_lora()

    def _apply_lora(self):

        target_modules = ["q_proj", "k_proj", "v_proj"] if self.is_v3 else ["query", "key", "value"]
        config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules)


        if hasattr(self.dino, 'layer'):
            encoder_layers = self.dino.layer
        elif hasattr(self.dino, 'encoder') and hasattr(self.dino.encoder, 'layer'):
            encoder_layers = self.dino.encoder.layer
        elif hasattr(self.dino, 'model') and hasattr(self.dino.model, 'layer'):
            encoder_layers = self.dino.model.layer
        else:
            raise ValueError('The backbone does not expose supported transformer layers for LoRA')
        for i in range(len(encoder_layers)):
            encoder_layers[i] = get_peft_model(encoder_layers[i], config)

    def forward(self, x):
        outputs = self.dino(pixel_values=x)
        last_hidden_state = outputs[0]


        feat_cls = last_hidden_state[:, 0]
        feat_tokens = last_hidden_state[:, 1:]
        return feat_tokens, feat_cls


class MirrorMemoryBank(nn.Module):

    def __init__(self, feature_dim, mem_slots=4096, num_heads=8, top_k=128):
        super().__init__()
        if num_heads < 1 or feature_dim % num_heads:
            raise ValueError('feature_dim must be divisible by a positive num_heads')
        if not 1 <= top_k <= mem_slots:
            raise ValueError('top_k must be between 1 and mem_slots')
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k


        self.memory = nn.Parameter(torch.randn(mem_slots, feature_dim))


        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        B, N, C = x.shape
        M_slots, _ = self.memory.shape


        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)


        mem_ext = self.memory.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(mem_ext).reshape(B, M_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mem_ext).reshape(B, M_slots, self.num_heads, self.head_dim).transpose(1, 2)


        attn_logits = (q @ k.transpose(-2, -1)) * self.scale


        topk_indices = torch.topk(attn_logits, k=self.top_k, dim=-1).indices


        mask = torch.zeros_like(attn_logits, dtype=torch.bool).scatter_(-1, topk_indices, True)
        masked_logits = attn_logits.masked_fill(~mask, float('-inf'))
        if masked_logits.dtype in (torch.float16, torch.bfloat16):
            masked_logits = masked_logits.float()
        sparse_attn = torch.softmax(masked_logits, dim=-1).to(v.dtype)


        recon = (sparse_attn @ v).transpose(1, 2).reshape(B, N, C)
        recon = self.out_proj(recon)


        return recon, sparse_attn


class DualBranchClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()


        self.perplexity_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )


        self.residual_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256)
        )


        self.head = nn.Sequential(
            nn.Linear(64 + 256 + feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, attn_weights, f_last, f_recon, feat_cls):


        metric_weights = attn_weights.float() if attn_weights.dtype in (torch.float16, torch.bfloat16) else attn_weights
        max_scores = metric_weights.max(dim=-1)[0].mean(dim=[1, 2]).unsqueeze(-1)


        entropy = -(metric_weights * torch.log(metric_weights + 1e-9)).sum(dim=-1).mean(dim=[1, 2]).unsqueeze(-1)


        evidence = torch.cat([max_scores, entropy], dim=-1).to(self.perplexity_mlp[0].weight.dtype)
        v_per = self.perplexity_mlp(evidence)


        residual_map = f_last - f_recon

        residual_gap = residual_map.mean(dim=1)


        v_res = self.residual_mlp(residual_gap)


        combined = torch.cat([v_per, v_res, feat_cls], dim=-1)

        return self.head(combined)


class MIRROR_Detector(nn.Module):
    def __init__(self, dino_path, memory_path=None, feature_dim=None):
        super(MIRROR_Detector, self).__init__()


        self.norm = Norm(mode='imagenet')


        self.backbone = DINO(dino_path, finetune=True)
        backbone_dim = self.backbone.dino.config.hidden_size
        if feature_dim is None:
            feature_dim = backbone_dim
        elif feature_dim != backbone_dim:
            raise ValueError(f'feature_dim={feature_dim} does not match backbone hidden_size={backbone_dim}')


        self.memory_bank = MirrorMemoryBank(feature_dim=feature_dim)

        if memory_path:
            print(f"Loading Memory Bank from {memory_path}")
            state_dict = torch.load(memory_path, map_location='cpu', weights_only=False)
            load_memory_checkpoint(self.memory_bank, state_dict)


        self.memory_bank.eval()
        for param in self.memory_bank.parameters():
            param.requires_grad = False


        self.detector = DualBranchClassifier(feat_dim=feature_dim)

    def forward(self, x):

        x = self.norm(x)


        feat_tokens, feat_cls = self.backbone(x)


        f_recon, attn_weights = self.memory_bank(feat_tokens)


        logits = self.detector(attn_weights, feat_tokens, f_recon, feat_cls)

        return logits, f_recon, feat_tokens

def build_mirror(memory_path=None, backbone_path='./weight/dinov3-huge'):

    model = MIRROR_Detector(dino_path=backbone_path, memory_path=memory_path)
    return model

def _checkpoint_state(checkpoint, keys):
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must contain a state dictionary')
    state = checkpoint
    for key in keys:
        if isinstance(checkpoint.get(key), Mapping):
            state = checkpoint[key]
            break
    clean = {}
    for name, value in state.items():
        if not isinstance(name, str):
            raise ValueError('State dictionary keys must be strings')
        while name.startswith('module.'):
            name = name[len('module.'):]
        if name in clean:
            raise ValueError(f'Duplicate checkpoint parameter: {name}')
        clean[name] = value
    return clean

def load_memory_checkpoint(memory_bank, checkpoint):
    config = checkpoint.get('memory_config', {}) if isinstance(checkpoint, Mapping) else {}
    expected = {
        'feature_dim': memory_bank.feature_dim,
        'mem_slots': memory_bank.memory.shape[0],
        'num_heads': memory_bank.num_heads,
        'top_k': memory_bank.top_k,
    }
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise ValueError(f'Memory checkpoint {key}={config[key]} does not match {value}')
    state = _checkpoint_state(checkpoint, ('model_state_dict', 'model', 'state_dict'))
    if any(name.startswith('memory_bank.') for name in state):
        state = {name[len('memory_bank.'):]: value for name, value in state.items() if name.startswith('memory_bank.')}
    return memory_bank.load_state_dict(state, strict=True)

def load_detector_checkpoint(model, checkpoint):
    state = _checkpoint_state(checkpoint, ('model', 'state_dict', 'model_state_dict'))
    expected = model.state_dict()
    remapped = {}
    for name, value in state.items():
        target = name
        if name not in expected:
            for source, destination in (
                ('backbone.dino.layer.', 'backbone.dino.model.layer.'),
                ('backbone.dino.model.layer.', 'backbone.dino.layer.'),
            ):
                if name.startswith(source):
                    candidate = destination + name[len(source):]
                    if candidate in expected:
                        target = candidate
                        break
        if target in remapped:
            raise ValueError(f'Duplicate checkpoint parameter: {target}')
        remapped[target] = value
    return model.load_state_dict(remapped, strict=True)
