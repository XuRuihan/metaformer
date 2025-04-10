import os
import numpy as np
import torch
from torchvision.models import resnet18
import time

import sys

sys.path.extend([".", ".."])

from models.parcnetv2_5_bgu import parcnetv2_5_bgu_s18
from models.parcnetv2 import parcnetv2_tiny, parcnetv2_small, parcnetv2_base
from models.conv2former import conv2former_tiny
from models.swin import swin_tiny
from models.convnext import convnext_tiny
from models.hornet import hornet_tiny_7x7


if __name__ == "__main__":
    model = parcnetv2_tiny(pretrained=False)
    device = torch.device("cpu")
    model.eval()
    model.to(device)
    dump_input = torch.ones(1, 3, 224, 224).to(device)

    # Warm-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        end = time.time()
        print("Time:{}ms".format((end - start) * 1000))

    with torch.autograd.profiler.profile(
        enabled=True, use_cuda=False, record_shapes=False, profile_memory=False
    ) as prof:
        outputs = model(dump_input)
    print(prof.table())
    prof.export_chrome_trace("log/parcnetv2_tiny_profile.json")
