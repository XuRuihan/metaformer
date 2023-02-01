import torch
from torchprofile import profile_macs

import warnings
 
warnings.filterwarnings('ignore')

import sys

sys.path.extend([".", ".."])

from models import convformer_s18
from models.parcnet_v3 import parcnet_v3_s18, parcnet_v2_cvpr
from models.parcnet_v3_downsample import parcnet_v3_bgu_s18
from models.conv2former import conv2former_tiny
from models.swin import swin_tiny


def params(model):
    return sum([param.nelement() for param in model.parameters()])


inputs = torch.rand(1, 3, 224, 224)

models = {
    # "convformer_s18": convformer_s18(),
    # "parcnet_v3_s18": parcnet_v3_s18(),
    # "parcnet_v3_bgu_s18": parcnet_v3_bgu_s18(),
    # "conv2former_tiny": conv2former_tiny(),
    "parcnet_v2_cvpr": parcnet_v2_cvpr(),
    "swin_tiny": swin_tiny(),
}

for name, model in models.items():
    macs = profile_macs(model, inputs)
    print(f"{name.ljust(25)}: Params {params(model)}, macs {macs}")
