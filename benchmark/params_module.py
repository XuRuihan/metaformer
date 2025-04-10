import torch
from ptflops import get_model_complexity_info

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.extend([".", ".."])

from models import convformer_s12, convformer_s18
from models.parcnetv2_5 import parcnetv2_5_s12, parcnetv2_5_s18
from models.parcnetv2_5_bgu import parcnetv2_5_bgu_s12, parcnetv2_5_bgu_s18
from models.parcnetv2 import parcnetv2_xt, parcnetv2_tiny, parcnetv2_small
from models.poolformer_bgu import poolformerv2_bgu_s12
from models.hornet import hornet_tiny_7x7


net = parcnetv2_xt()
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
