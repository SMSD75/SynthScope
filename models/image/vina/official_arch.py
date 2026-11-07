# Vendored from github.com/mgiant/VINA (commit 225d331), file models/DINOv3.py (class `DINOv3`). Licence UNCLEAR: the
# repository has no LICENSE file; see LICENSES.md.
#
# The `DINOv3` detector below is the official class, so the released VINA_DINOv3-L_140k+DDA.pth loads key-for-key with
# strict=True. Three forced changes, none of which alter the computation:
#   1. The backbone can be built from its config (`DINOV3_VITL16`) instead of `DINOv3ViTModel.from_pretrained`: the
#      checkpoint holds the full backbone, so inference needs no gated Hugging Face download. A directory path (the
#      official `weights/dinov3-vitl16-pretrain-lvd1689m`) still uses from_pretrained.
#   2. transformers >= 5 moved the ViT blocks from `backbone.layer` to `backbone.model.layer`; the checkpoint was saved
#      with transformers 4.x. A load_state_dict pre-hook renames between the two for every load.
#   3. `from peft import ...` is dropped: peft is only used by the unused `DINOv3_Lora` class, which is not vendored.
from collections.abc import Mapping

import torch
import torch.nn as nn

try:                                    # DINOv3 is in transformers >= 4.56; older versions can still import SynthScope
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
except ImportError:
    DINOv3ViTConfig = DINOv3ViTModel = None

# config.json of facebook/dinov3-vitl16-pretrain-lvd1689m (revision ea8dc2863c51be0a264bab82070e3e8836b02d51), the
# official `model_source: dinov3-vitl16-pretrain-lvd1689m` of configs/VINA_DINOv3.yaml.
DINOV3_VITL16 = dict(
    attention_dropout=0.0, drop_path_rate=0.0, hidden_act="gelu", hidden_size=1024, image_size=224,
    initializer_range=0.02, intermediate_size=4096, key_bias=False, layer_norm_eps=1e-05, layerscale_value=1.0,
    mlp_bias=True, num_attention_heads=16, num_channels=3, num_hidden_layers=24, num_register_tokens=4, patch_size=16,
    pos_embed_jitter=None, pos_embed_rescale=2.0, pos_embed_shift=None, proj_bias=True, query_bias=True,
    rope_theta=100.0, use_gated_mlp=False, value_bias=True,
)

CHECKPOINT = {
    'dinov3-vitb16-pretrain-lvd1689m': 'weights/dinov3-vitb16-pretrain-lvd1689m',
    'dinov3-vitl16-pretrain-lvd1689m': 'weights/dinov3-vitl16-pretrain-lvd1689m',
    'dinov3-vith16plus-pretrain-lvd1689m': 'weights/dinov3-vith16plus-pretrain-lvd1689m',
    'dinov3-vit7b16-pretrain-lvd1689m': 'weights/dinov3-vit7b16-pretrain-lvd1689m',
}


def _load_backbone(source, attn_implementation, dtype):
    """`DINOv3ViTModel.from_pretrained(path)` for a directory / hub id; a config dict builds the same model untrained."""
    if DINOv3ViTModel is None:
        raise ImportError("VINA needs transformers >= 4.56 (DINOv3); upgrade transformers")
    if isinstance(source, Mapping):
        model = DINOv3ViTModel(DINOv3ViTConfig(**source, attn_implementation=attn_implementation))
        return model.to(dtype)
    return DINOv3ViTModel.from_pretrained(source, attn_implementation=attn_implementation, dtype=dtype)


def _rename_block_prefix(module, state_dict, prefix, *args):
    """load_state_dict pre-hook of `DINOv3`: the released checkpoint names the ViT blocks `backbone.layer.N`
    (transformers 4.x); transformers 5.x has `backbone.model.layer.N`. Rename to whichever this backbone has."""
    has_model = hasattr(module.backbone, "model") and hasattr(module.backbone.model, "layer")
    src, dst = (prefix + "backbone.layer.", prefix + "backbone.model.layer.") if has_model else \
        (prefix + "backbone.model.layer.", prefix + "backbone.layer.")
    for name in [k for k in state_dict if k.startswith(src)]:
        state_dict[dst + name[len(src):]] = state_dict.pop(name)


# --- official models/DINOv3.py, class DINOv3 (unchanged except for the marked lines) ----------------------------------
class DINOv3(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        checkpoint_path = kwargs.get('backbone_source') or CHECKPOINT[kwargs.get('model_source')]   # changed (1)
        attn_implementation = kwargs.get('attn_implementation', 'eager')
        dtype = torch.bfloat16 if kwargs.get('use_bf16', False) else torch.float32
        # "flash_attention_2": flash_attention_forward,

        # config = AutoConfig.from_pretrained(checkpoint_path)
        self.backbone = _load_backbone(checkpoint_path, attn_implementation, dtype)                # changed (1)
        self.register_load_state_dict_pre_hook(_rename_block_prefix)                              # changed (2)
        tuning_mode = kwargs.get('tuning_mode', 'default')
        if tuning_mode == 'lp':
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def features(self, images) -> torch.tensor:
        feat = self.backbone(images).pooler_output
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def forward(self, pixel_values, **kwargs) -> dict:
        # get the features by backbone
        features = self.features(pixel_values)
        # get the prediction by classifier
        pred = self.classifier(features)
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'feat': features}
        return pred_dict

    def load_checkpoint(self, state_dict: dict)-> None:
        self.load_state_dict(state_dict)
