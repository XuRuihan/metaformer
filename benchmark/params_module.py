import torch
from ptflops import get_model_complexity_info

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.extend([".", ".."])

from models import convformer_s12, convformer_s18
from models.parcnet_v3 import parcnet_v3_s12, parcnet_v3_s18, parcnet_v2_cvpr
from models.parcnet_v3_bgu import parcnet_v3_bgu_s12, parcnet_v3_bgu_s18
from models.parcnet_v2_ideal import parcnet_v2_cvpr_ideal
from models.hornet import hornet_tiny_7x7


net = parcnet_v2_cvpr_ideal()
if torch.cuda.is_available():
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            net, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
        )
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
else:
    macs, params = get_model_complexity_info(
        net, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
