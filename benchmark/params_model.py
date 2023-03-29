import torch
from torchprofile import profile_macs

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.extend([".", ".."])

from models import convformer_s18, convformer_s12
from models.parcnetv2_5 import parcnetv2_5_s12, parcnetv2_5_s18
from models.parcnetv2_5_bgu import (
    parcnetv2_5_bgu_s12,
    parcnetv2_5_bgu4_s12,
    parcnetv2_5_bgu_s18,
    parcnetv2_5_bgu4_s18,
)
from models.parcnetv2 import (
    parcnetv2_xt,
    parcnetv2_tiny,
    parcnetv2_mlp,
    parcnetv2_iso_tiny,
    parcnetv2_lasthalf_tiny,
    parcnetv2_small,
    parcnetv2_iso_small,
    parcnetv2_base,
    parcnetv2_iso_base,
)
from models.parcnetv3 import (
    parcnetv3_xt,
)
from models.poolformer_bgu import poolformerv2_bgu_s12
from models.conv2former import conv2former_tiny
from models.convnext import convnext_xt, convnext_tiny, convnext_small, convnext_base
from models.swin import swin_tiny, swin_small, swin_base


def params(model):
    return sum([param.nelement() for param in model.parameters()])


inputs = torch.rand(1, 3, 224, 224)

models = {
    # "convformer_s18": convformer_s18(),
    # "convformer_s12": convformer_s12(),
    # "parcnetv2_5_s18": parcnetv2_5_s18(),
    # "parcnetv2_5_s12": parcnetv2_5_s12(),
    # "parcnetv2_5_bgu_s18": parcnetv2_5_bgu_s18(),
    # "parcnetv2_5_bgu4_s18": parcnetv2_5_bgu4_s18(),
    # "parcnetv2_5_bgu_s12": parcnetv2_5_bgu_s12(),
    # "parcnetv2_5_bgu4_s12": parcnetv2_5_bgu4_s12(),
    # "conv2former_tiny": conv2former_tiny(),
    "parcnetv2_xt": parcnetv2_xt(),
    "parcnetv3_xt": parcnetv3_xt(),
    "parcnetv2_tiny": parcnetv2_tiny(),
    # "parcnetv2_lasthalf_tiny": parcnetv2_lasthalf_tiny(),
    # "parcnetv2_mlp": parcnetv2_mlp(),
    "parcnetv2_iso_tiny": parcnetv2_iso_tiny(),
    # "parcnetv2_small": parcnetv2_small(),
    "parcnetv2_iso_small": parcnetv2_iso_small(),
    # "parcnetv2_base": parcnetv2_base(),
    "parcnetv2_iso_base": parcnetv2_iso_base(),
    # "swin_tiny": swin_tiny(),
    # "swin_small": swin_small(),
    # "swin_base": swin_base(),
    # "convnext_xt": convnext_xt(),
    # "convnext_tiny": convnext_tiny(),
    # "convnext_small": convnext_small(),
    # "convnext_base": convnext_base(),
}

for name, model in models.items():
    macs = profile_macs(model, inputs)
    print(f"{name.ljust(25)}: Params {params(model)}, macs {macs}")
