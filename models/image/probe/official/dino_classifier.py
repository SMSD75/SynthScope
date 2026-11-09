# Vendored from github.com/Amamiya-C/PROBE-AIGI-Detection (commit b145f71), file Detector/model/dino_classifier.py.
# Licence UNCLEAR: the repository has no LICENSE file and states no licence; see LICENSES.md.
#
# The `dino_classifier` below is the official class, so the released DINOv2_best_model_step_34999.pth loads key-for-key
# with strict=True. One forced change, which does not alter the computation:
#   1. The backbone can be built from its config (`DINOV2_REG_LARGE`, the config.json of
#      facebook/dinov2-with-registers-large) instead of `AutoModel.from_pretrained(model_name)`: the checkpoint holds
#      the full backbone, so inference needs no backbone download. A directory path or hub id (the official
#      `model_name`) still uses from_pretrained. Both build the same Dinov2WithRegistersModel with the transformers
#      default attention (sdpa).
from collections.abc import Mapping

import torch
from transformers import AutoModel, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

# config.json of facebook/dinov2-with-registers-large (revision e4c89a4e05589de9b3e188688a303d0f3c04d0f3), the backbone
# of the official `model_name` ('.../dinov2-l-register'; the commented-out default is the hub id).
DINOV2_REG_LARGE = dict(
    apply_layernorm=True, attention_probs_dropout_prob=0.0, drop_path_rate=0.0, hidden_act="gelu",
    hidden_dropout_prob=0.0, hidden_size=1024, image_size=518, initializer_range=0.02, interpolate_antialias=True,
    interpolate_offset=0.0, layer_norm_eps=1e-06, layerscale_value=1.0, mlp_ratio=4, num_attention_heads=16,
    num_channels=3, num_hidden_layers=24, num_register_tokens=4, patch_size=14, qkv_bias=True, use_swiglu_ffn=False,
)


def _load_backbone(source):
    """`AutoModel.from_pretrained(source)` for a directory / hub id; a config dict builds the same model untrained."""
    if isinstance(source, Mapping):
        return Dinov2WithRegistersModel(Dinov2WithRegistersConfig(**source))
    return AutoModel.from_pretrained(source)


# --- official Detector/model/dino_classifier.py (unchanged except for the marked line) --------------------------------
class dino_classifier(torch.nn.Module):
    def __init__(self,
                # model_name="facebook/dinov2-with-registers-large",
                model_name='/data_center/data2/dataset/czj/detector/model/dinov2-l-register',
                classifier_type='linear',
                mlp_hidden_size=512
                ):
        super(dino_classifier, self).__init__()
        self.backbone = _load_backbone(model_name)                                                      # changed (1)
        if classifier_type == 'linear':
            self.fc = torch.nn.Linear(self.backbone.config.hidden_size, 1)
        elif classifier_type == 'mlp':
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.backbone.config.hidden_size, mlp_hidden_size),
                torch.nn.ReLU(inplace=False),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(mlp_hidden_size, 1)
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        # Freeze backbone
        # for param in self.vision_model.parameters():
        #     param.requires_grad = False

    def extract_feature(self, input):
        outputs = self.backbone(input)
        sequence_output = outputs[0] # batch_size, sequence_length, hidden_size
        cls_token = sequence_output[:, 0]
        return cls_token

    def forward(self, input, return_feature=False):
        outputs = self.backbone(input)
        sequence_output = outputs[0] # batch_size, sequence_length, hidden_size
        cls_token = sequence_output[:, 0]
        feature = cls_token
        # patch_tokens = sequence_output[:, 1:]
        # linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        logits = self.fc(cls_token)
        if return_feature:
            return logits, feature
        else:
            return logits

