import torch
from torchprofile import profile_macs

import sys

sys.path.extend([".", ".."])

from models import convformer_s18, parcnet_v3_s18
from models.parcnet_v3_downsample import parcnet_v3_bgu_s18


def params(model):
    return sum([param.nelement() for param in model.parameters()])


inputs = torch.rand(1, 3, 224, 224)

for model in [convformer_s18(), parcnet_v3_s18(), parcnet_v3_bgu_s18()]:
    macs = profile_macs(model, inputs)
    print(params(model), macs)
