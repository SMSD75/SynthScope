# Vendored into SynthScope from NSG-VD (github.com/ZSHsh98/NSG-VD @ 9a60f57, libs/eps_ad/guided_diffusion/fp16_util.py):
# only the two module converters the UNet uses, copied unchanged (the training helpers and their logger import are
# left out). MIT licence of OpenAI guided-diffusion in LICENSE_GUIDED_DIFFUSION next to this file.
# ---------------------------------------------------------------
# Taken from the following link as is from:
# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/fp16_util.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_GUIDED_DIFFUSION).
# ---------------------------------------------------------------

"""
Helpers to train with 16-bit precision.
"""

import torch.nn as nn


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

