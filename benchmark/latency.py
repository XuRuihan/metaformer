import torch
import torch.utils.benchmark as benchmark

import sys

sys.path.extend([".", ".."])

from models import convnext_xt, convnext_xt_nobottleneck, convnext_tiny
from models.parcnet_v3 import parcnet_v3_s12, parcnet_v3_s18, parcnet_v2_cvpr
from models.parcnet_v3_bgu import parcnet_v3_bgu_s18
from models.parcnet_v2_ideal import parcnet_v2_s12, parcnet_v2_e2_s12, parcnet_v2_s18, parcnet_v2_e2_s18
from models.conv2former import conv2former_tiny
from models.swin import swin_tiny
from models.hornet import hornet_tiny_7x7


dim = 3
image_size = 224
if torch.cuda.is_available():
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
    # "parcnet_v2_s12": parcnet_v2_s12().to(device),
    # "parcnet_v2_e2_s12": parcnet_v2_e2_s12().to(device),
    "parcnet_v2_s18": parcnet_v2_s18().to(device),
    "parcnet_v2_e2_s18": parcnet_v2_e2_s18().to(device),
    # "convnext_xt": convnext_xt().to(device),
    # "convnext_tiny": convnext_tiny().to(device),
    # "swin_tiny": swin_tiny().to(device),
    "parcnet_v2_cvpr": parcnet_v2_cvpr().to(device),
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
