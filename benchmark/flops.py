import torch

import sys

sys.path.extend([".", ".."])

from models import convnext_xt, convnext_xt_nobottleneck


def torchprofile_benchmark(model):
    from torchprofile import profile_macs
    inputs = torch.randn(1, 3, 224, 224)

    params = sum(p.numel() for p in model.parameters())
    macs = profile_macs(model, inputs)
    print("{:<30}  {:<20}".format("Computational complexity: ", macs))
    print("{:<30}  {:<20}".format("Number of parameters: ", params))


def ptflops_benchmark(model):
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
        )
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))


benchmark = ptflops_benchmark
benchmark(convnext_xt())
benchmark(convnext_xt_nobottleneck())
