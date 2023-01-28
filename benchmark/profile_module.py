import torch
from ptflops import get_model_complexity_info

import sys

sys.path.extend([".", ".."])

from models import convformer_s18, parcnet_v3_s18
from models.parcnet_v3_downsample import parcnet_v3_bgu_s18


with torch.cuda.device(0):
    net = parcnet_v3_bgu_s18()
    macs, params = get_model_complexity_info(
        net, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))

