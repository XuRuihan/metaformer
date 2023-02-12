import os
import numpy as np
import torch
from torchvision.models import resnet18
import time

import sys

sys.path.extend([".", ".."])

from models.parcnet_v3_bgu import parcnet_v3_bgu_s18
from models.parcnet_v2 import parcnet_v2_cvpr
from models.hornet import hornet_tiny_7x7


if __name__ == '__main__':
    model = parcnet_v2_cvpr(pretrained=False)
    device = torch.device('cpu')
    model.eval()
    model.to(device)
    dump_input = torch.ones(1,3,224,224).to(device)

    # Warm-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False) as prof:
        outputs = model(dump_input)
    print(prof.table())
    prof.export_chrome_trace('log/parcnet_v2_cvpr_profile.json')
