import torch
import torch.utils.benchmark as benchmark

import sys

sys.path.extend([".", ".."])

from models.parcnet_v3 import parcnet_v3_s12, parcnet_v3_s18
from models.parcnet_v3_bgu import parcnet_v3_bgu_s18
from models.parcnetv2 import (
    parcnetv2_s12,
    parcnetv2_e2_s12,
    parcnetv2_s18,
    parcnetv2_44_tiny,
    parcnetv2_26_tiny,
    parcnetv2_tiny,
    parcnetv2_lasthalf_tiny,
    parcnetv2_mlp,
    parcnetv2_small,
)
from models.poolformer_bgu import poolformerv2_bgu_s12
from models.conv2former import conv2former_tiny
from models.swin import swin_tiny, swin_small, swin_base, swin_large
from models.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
from models.hornet import hornet_tiny_7x7


dim = 3
image_size = 224
use_cuda = True  # torch.cuda.is_available()
if use_cuda:
    device = "cuda"
    times = 100
    x = torch.rand(32, dim, image_size, image_size).to(device)
else:
    device = "cpu"
    times = 10
    x = torch.rand(1, dim, image_size, image_size).to(device)


models = {
    # "convnext_xt": convnext_xt().to(device),
    # "convnext_xt_nobottleneck": convnext_xt_nobottleneck().to(device),
    # "conv2former_tiny": conv2former_tiny().to(device),
    # "parcnet_v3_s12": parcnet_v3_s12().to(device),
    # "parcnet_v3_s18": parcnet_v3_s18().to(device),
    # "parcnet_v3_bgu_s18": parcnet_v3_bgu_s18().to(device),
    # "parcnetv2_s12": parcnetv2_s12().to(device),
    # "parcnetv2_e2_s12": parcnetv2_e2_s12().to(device),
    # "parcnetv2_s18": parcnetv2_s18().to(device),
    # "parcnetv2_tiny": parcnetv2_tiny().to(device),
    # "parcnetv2_lasthalf_tiny": parcnetv2_lasthalf_tiny().to(device),
    # "parcnetv2_26_tiny": parcnetv2_26_tiny().to(device),
    # "parcnetv2_44_tiny": parcnetv2_44_tiny().to(device),
    "parcnetv2_small": parcnetv2_small().to(device),
    # "convnext_xt": convnext_xt().to(device),
    # "convnext_tiny": convnext_tiny().to(device),
    "convnext_small": convnext_small().to(device),
    # "convnext_base": convnext_base().to(device),
    # "swin_tiny": swin_tiny().to(device),
    "swin_small": swin_small().to(device),
    # "swin_base": swin_base().to(device),
    # "poolformerv2_bgu_s12": poolformerv2_bgu_s12().to(device),
    # "hornet_tiny_7x7": hornet_tiny_7x7().to(device)
}

for name, model in models.items():
    timer = benchmark.Timer(
        stmt="model(x)",
        setup="from __main__ import model; model.eval()",
        globals={"x": x},
    )
    t = timer.timeit(times)
    print(f"{name.ljust(25)}: {t}")
