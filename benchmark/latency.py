import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import sys

sys.path.extend([".", ".."])

from models import convnext_xt, convnext_xt_nobottleneck


def comparison():
    dim = 3
    image_size = 224
    device = "cuda:0"
    x = torch.rand(10, dim, image_size, image_size).to(device)

    t0 = benchmark.Timer(
        stmt="net(x)",
        setup="from __main__ import convnext_xt; net = convnext_xt(); net.cuda()",
        globals={"x": x},
    )

    t1 = benchmark.Timer(
        stmt=" net(x)",
        setup="from __main__ import convnext_xt_nobottleneck; net = convnext_xt_nobottleneck(); net.cuda()",
        globals={"x": x},
    )

    print("==================== ConvNeXt inference latent ====================")
    print(t0.timeit(100))
    print(t1.timeit(100))


if __name__ == "__main__":
    comparison()
