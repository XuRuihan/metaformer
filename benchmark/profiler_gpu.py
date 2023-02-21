import os
import numpy as np
import torch
from torchvision.models import resnet18
import time

import sys

sys.path.extend([".", ".."])

from models.parcnet_v3_bgu import parcnet_v3_bgu_s18
from models.parcnetv2 import (
    parcnetv2_s12,
    parcnetv2_e2_s12,
    parcnetv2_s18,
    parcnetv2_44_tiny,
    parcnetv2_tiny,
)
from models.poolformer_bgu import poolformerv2_bgu_s12
from models.conv2former import conv2former_tiny
from models.swin import swin_tiny
from models.convnext import convnext_tiny
from models.hornet import hornet_tiny_7x7


if __name__ == "__main__":
    # model = parcnetv2_tiny(pretrained=False)
    # log_file = "log/parcnetv2_tiny_ideal_profile.json"
    model = parcnetv2_44_tiny(pretrained=False)
    log_file = "log/parcnetv2_44_tiny_profile.json"
    # model = poolformerv2_bgu_s12(pretrained=False)
    # log_file = "log/poolformerv2_bgu_s12_profile.json"
    # model = hornet_tiny_7x7(pretrained=False)
    # log_file = "log/hornet_tiny_7x7_profile.json"
    # model = swin_tiny()
    # log_file = "log/swin_tiny_profile.json"
    # model = convnext_tiny()
    # log_file = "log/convnext_tiny_profile.json"

    device = torch.device("cuda")
    model.eval()
    model.to(device)
    dump_input = torch.ones(16, 3, 224, 224).to(device)

    # Warm-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print("Time:{}ms".format((end - start) * 1000))

    with torch.autograd.profiler.profile(
        enabled=True, use_cuda=True, record_shapes=False, profile_memory=False
    ) as prof:
        outputs = model(dump_input)
    print(prof.table())
    prof.export_chrome_trace(log_file)
