# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "convformer_s18": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth"
    ),
    "convformer_s18_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384.pth",
        input_size=(3, 384, 384),
    ),
    "convformer_s36": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36.pth"
    ),
    "convformer_s36_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384.pth",
        input_size=(3, 384, 384),
    ),
    "convformer_m36": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36.pth"
    ),
    "convformer_m36_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384.pth",
        input_size=(3, 384, 384),
    ),
    "convformer_b36": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36.pth"
    ),
    "convformer_b36_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384.pth",
        input_size=(3, 384, 384),
    ),
    "convformer_b36_in21ft1k": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21ft1k.pth"
    ),
    "convformer_b36_384_in21ft1k": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384_in21ft1k.pth",
        input_size=(3, 384, 384),
    ),
    "convformer_b36_in21k": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21k.pth",
        num_classes=21841,
    ),
    "caformer_s18": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth"
    ),
    "caformer_s18_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384.pth",
        input_size=(3, 384, 384),
    ),
    "caformer_s36": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth"
    ),
    "caformer_s36_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384.pth",
        input_size=(3, 384, 384),
    ),
    "caformer_m36": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth"
    ),
    "caformer_m36_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384.pth",
        input_size=(3, 384, 384),
    ),
    "caformer_b36": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth"
    ),
    "caformer_b36_384": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384.pth",
        input_size=(3, 384, 384),
    ),
    "caformer_b36_in21ft1k": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21ft1k.pth"
    ),
    "caformer_b36_384_in21ft1k": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384_in21ft1k.pth",
        input_size=(3, 384, 384),
    ),
    "caformer_b36_in21k": _cfg(
        url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21k.pth",
        num_classes=21841,
    ),
}


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        pre_norm=None,
        post_norm=None,
        pre_permute=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class StarGELU(nn.Module):
    def __init__(
        self,
        scale_value=1.0,
        bias_value=0.0,
        scale_learnable=False,
        bias_learnable=True,
        mode=None,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.gelu = nn.GELU()
        self.scale = nn.Parameter(
            scale_value * torch.ones(1), requires_grad=scale_learnable
        )
        self.bias = nn.Parameter(
            bias_value * torch.ones(1), requires_grad=bias_learnable
        )

    def forward(self, x):
        return self.scale * self.gelu(x) + self.bias


class OversizeConv2d(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, interpolate=False):
        super().__init__()

        if interpolate is True:
            padding = 0
        elif interpolate is False or interpolate is None:
            assert kernel_size % 2 == 1
            padding = kernel_size // 2
        else:
            print("interpolate:", interpolate)
            assert interpolate % 2 == 1
            padding = interpolate // 2
            interpolate = to_2tuple(interpolate)

        self.conv_h = nn.Conv2d(
            dim, dim, (kernel_size, 1), padding=(padding, 0), groups=dim, bias=bias
        )
        self.conv_w = nn.Conv2d(
            dim, dim, (1, kernel_size), padding=(0, padding), groups=dim, bias=bias
        )

        self.dim = dim
        self.kernel_size = kernel_size
        self.interpolate = interpolate
        self.padding = padding

    def get_instance_kernel(self, instance_kernel_size):
        h_weight = F.interpolate(
            self.conv_h.weight,
            [instance_kernel_size[0], 1],
            mode="bilinear",
            align_corners=True,
        )
        w_weight = F.interpolate(
            self.conv_w.weight,
            [1, instance_kernel_size[1]],
            mode="bilinear",
            align_corners=True,
        )
        return h_weight, w_weight

    def forward(self, x):
        if self.interpolate is True:
            H, W = x.shape[-2:]
            instance_kernel_size = 2 * H - 1, 2 * W - 1
            h_weight, w_weight = self.get_instance_kernel(instance_kernel_size)

            padding = H - 1, W - 1
            x = F.conv2d(
                x, h_weight, self.conv_h.bias, padding=(padding[0], 0), groups=self.dim
            )
            x = F.conv2d(
                x, w_weight, self.conv_w.bias, padding=(0, padding[1]), groups=self.dim
            )
        elif isinstance(self.interpolate, tuple):
            h_weight, w_weight = self.get_instance_kernel(self.interpolate)
            x = F.conv2d(
                x,
                h_weight,
                self.conv_h.bias,
                padding=(self.padding, 0),
                groups=self.dim,
            )
            x = F.conv2d(
                x,
                w_weight,
                self.conv_w.bias,
                padding=(0, self.padding),
                groups=self.dim,
            )
        else:
            x = self.conv_h(x)
            x = self.conv_w(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, kernel_size={self.kernel_size}"


class ParC_V3(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act_layer=StarGELU,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=True)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Linear(med_channels // 2, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x1, x2 = x.chunk(2, -1)
        x2 = self.act(x2)
        x = torch.cat([x1, x2], -1)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x1, x2 = x.chunk(2, -1)
        x = x1 * x2
        x = self.pwconv2(x)
        return x


class ParC_V3_add(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act_layer=StarGELU,
        bias=False,
        kernel_size=7,
        global_kernel_size=14,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=True)
        self.act = act_layer()
        self.dwconv1 = OversizeConv2d(med_channels, global_kernel_size, bias)
        self.dwconv2 = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels // 2, dim, 1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x1, x2 = x.chunk(2, 1)
        x2 = self.act(x2)
        x = torch.cat([x1, x2], 1)
        x = self.dwconv1(x) + self.dwconv2(x)
        x1, x2 = x.chunk(2, 1)
        x = x1 * x2
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 1)
        return x


class LayerNormGeneral(nn.Module):
    r"""General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(
        self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class Mlp(nn.Module):
    """MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class BGU(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=StarGELU,
        drop=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop1(x)
        x1, x2 = x.chunk(2, -1)
        x = x1 * self.act(x2)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """MLP classification head"""

    def __init__(
        self,
        dim,
        num_classes=1000,
        mlp_ratio=2,
        act_layer=StarGELU,
        norm_layer=nn.LayerNorm,
        head_dropout=0.0,
        bias=True,
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=True)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
        self,
        dim,
        token_mixer=nn.Identity,
        global_kernel_size=14,
        mlp=Mlp,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(
            dim=dim, drop=drop, global_kernel_size=global_kernel_size
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale1 = (
            Scale(dim=dim, init_value=layer_scale_init_value)
            if layer_scale_init_value
            else nn.Identity()
        )
        self.res_scale1 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale2 = (
            Scale(dim=dim, init_value=layer_scale_init_value)
            if layer_scale_init_value
            else nn.Identity()
        )
        self.res_scale2 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(
            self.drop_path1(self.token_mixer(self.norm1(x)))
        )
        x = self.res_scale2(x) + self.layer_scale2(
            self.drop_path2(self.mlp(self.norm2(x)))
        )
        return x


r"""
downsampling (stem) for the first stage is a layer of conv with k7, s4 and p2
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
# MetaFormer
DOWNSAMPLE_LAYERS_FOUR_STAGES = (
    [
        partial(
            Downsampling,
            kernel_size=7,
            stride=4,
            padding=2,
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
        )
    ]
    + [
        partial(
            Downsampling,
            kernel_size=3,
            stride=2,
            padding=1,
            pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
            pre_permute=True,
        )
    ]
    * 3
)
# MetaFormer_group4
DOWNSAMPLE_LAYERS_FOUR_STAGES_GROUP = (
    [
        partial(
            Downsampling,
            kernel_size=7,
            stride=4,
            padding=2,
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
        )
    ]
    + [
        partial(
            Downsampling,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=4,
            pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
            pre_permute=True,
        )
    ]
    * 3
)


class MetaFormer(nn.Module):
    r"""MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        global_kernel_sizes=[111, 55, 27, 13],
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [
                downsample_layers[i](down_dims[i], down_dims[i + 1])
                for i in range(num_stage)
            ]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = (
            nn.ModuleList()
        )  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers[i],
                        global_kernel_size=global_kernel_sizes[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2]))  # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def parcnet_v2_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 384, 672],
        # dims=[64, 144, 384, 640],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES_GROUP,
        token_mixers=ParC_V3,  # _add,
        mlps=BGU,
        head_fn=MlpHead,
        **kwargs,
    )
    model.default_cfg = default_cfgs["convformer_s18"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg["url"], map_location="cpu", check_hash=True
        )
        model.load_state_dict(state_dict)
    return model


@register_model
def parcnet_v2_e2_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[96, 192, 448, 672],
        # dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES_GROUP,
        token_mixers=ParC_V3,  # _add,
        mlps=partial(BGU, mlp_ratio=2),
        head_fn=MlpHead,
        **kwargs,
    )
    model.default_cfg = default_cfgs["convformer_s18"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg["url"], map_location="cpu", check_hash=True
        )
        model.load_state_dict(state_dict)
    return model


@register_model
def parcnet_v2_s18(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 384, 672],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES_GROUP,
        token_mixers=ParC_V3,  # _add,
        mlps=BGU,
        head_fn=MlpHead,
        **kwargs,
    )
    model.default_cfg = default_cfgs["convformer_s18"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg["url"], map_location="cpu", check_hash=True
        )
        model.load_state_dict(state_dict)
    return model


@register_model
def parcnet_v2_e2_s18(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 448, 672],
        # depths=[3, 3, 12, 3],
        # dims=[80, 160, 432, 640],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES_GROUP,
        token_mixers=ParC_V3,  # _add,
        mlps=partial(BGU, mlp_ratio=2),
        head_fn=MlpHead,
        **kwargs,
    )
    model.default_cfg = default_cfgs["convformer_s18"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg["url"], map_location="cpu", check_hash=True
        )
        model.load_state_dict(state_dict)
    return model


@register_model
def parcnet_v2_cvpr_ideal(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 12, 3],
        dims=[64, 128, 320, 512],
        # token_mixers=ParC_V3,
        token_mixers=ParC_V3_add,
        mlps=partial(BGU, mlp_ratio=5),
        **kwargs,
    )
    model.default_cfg = default_cfgs["convformer_s18"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg["url"], map_location="cpu", check_hash=True
        )
        model.load_state_dict(state_dict)
    return model
