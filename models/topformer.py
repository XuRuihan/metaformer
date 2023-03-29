import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import _load_checkpoint
from mmseg.utils import get_root_logger


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        bn_weight_init=1.0,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
        )
        bn = build_norm_layer(norm_cfg, out_channels)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0.0)
        self.add_module("bn", bn)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        attn_ratio=4,
        activation=None,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)
        )

    def forward(self, x):
        B, C, H, W = get_shape(x)

        qq = (
            self.to_q(x)
            .reshape(B, self.num_heads, self.key_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations=None,
        norm_cfg=dict(type="BN", requires_grad=True),
    ) -> None:
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, 1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend(
            [
                # dw
                Conv2d_BN(
                    hidden_dim,
                    hidden_dim,
                    ks,
                    stride=stride,
                    paddding=ks // 2,
                    groups=hidden_dim,
                    norm_cfg=norm_cfg,
                ),
                activations(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, norm_cfg=norm_cfg),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TokenPyramidModule(nn.Module):
    def __init__(
        self,
        cfgs,
        out_indices,
        in_channels=16,
        activation=nn.ReLU,
        norm_cfg=dict(type="BN", requires_grad=True),
        width_mult=1.0,
    ):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            Conv2d_BN(3, in_channels, 3, 2, 1, norm_cfg=norm_cfg), activation()
        )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * in_channels
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = "layer{}".format(i + 1)
            layer = InvertedResidual(
                in_channels,
                output_channel,
                ks=k,
                stride=s,
                expand_ratio=t,
                norm_cfg=norm_cfg,
                activations=activation,
            )
            self.add_module(layer_name, layer)
            in_channels = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer,
            norm_cfg=norm_cfg,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            norm_cfg=norm_cfg,
        )
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
        x = self.res_scale1(x) + self.drop_path(self.attn(x))
        x = self.res_scale2(x) + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        block_num,
        embedding_dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_layer=None,
    ):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_cfg=norm_cfg,
                    act_layer=act_layer,
                )
            )

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat(
            [nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1
        )


class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type="BN", requires_grad=True),
        activations=None,
    ) -> None:
        super(InjectionMultiSum, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.global_embedding = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.global_act = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(
            self.act(global_act), size=(H, W), mode="bilinear", align_corners=False
        )

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(
            global_feat, size=(H, W), mode="bilinear", align_corners=False
        )

        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSumCBR(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type="BN", requires_grad=True),
        activations=None,
    ) -> None:
        """
        local_embedding: conv-bn-relu
        global_embedding: conv-bn-relu
        global_act: conv
        """
        super(InjectionMultiSumCBR, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg
        )
        self.global_embedding = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg
        )
        self.global_act = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=None, act_cfg=None
        )
        self.act = nn.Hardsigmoid()

        self.out_channels = oup

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        # kernel
        global_act = self.global_act(x_g)
        global_act = F.interpolate(
            self.act(global_act), size=(H, W), mode="bilinear", align_corners=False
        )
        # feat_h
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(
            global_feat, size=(H, W), mode="bilinear", align_corners=False
        )
        out = local_feat * global_act + global_feat
        return out


class FuseBlockSum(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type="BN", requires_grad=True),
        activations=None,
    ) -> None:
        super(FuseBlockSum, self).__init__()
        self.norm_cfg = norm_cfg

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.fuse2 = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )

        self.out_channels = oup

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        kernel = self.fuse2(x_h)
        feat_h = F.interpolate(
            kernel, size=(H, W), mode="bilinear", align_corners=False
        )
        out = inp + feat_h
        return out


class FuseBlockMulti(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        norm_cfg=dict(type="BN", requires_grad=True),
        activations=None,
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.fuse2 = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.act = nn.Hardsigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(
            self.act(sig_act), size=(H, W), mode="bilinear", align_corners=False
        )
        out = inp * sig_act
        return out


SIM_BLOCK = {
    "fuse_sum": FuseBlockSum,
    "fuse_multi": FuseBlockMulti,
    "muli_sum": InjectionMultiSum,
    "muli_sum_cbr": InjectionMultiSumCBR,
}


class Topformer(nn.Module):
    def __init__(
        self,
        cfgs,
        channels,
        out_channels,
        embed_out_indice,
        decode_out_indices=[1, 2, 3],
        depths=4,
        key_dim=16,
        num_heads=8,
        attn_ratios=2,
        mlp_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.0,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_layer=nn.ReLU6,
        injection_type="muli_sum",
        init_cfg=None,
        injection=True,
    ):
        super().__init__()
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.injection = injection
        self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices
        self.init_cfg = init_cfg
        if self.init_cfg != None:
            self.pretrained = self.init_cfg["checkpoint"]

        self.tpm = TokenPyramidModule(
            cfgs=cfgs, out_indices=embed_out_indice, norm_cfg=norm_cfg
        )
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule
        self.trans = BasicLayer(
            block_num=depths,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            attn_ratio=attn_ratios,
            drop=0,
            attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,
            act_layer=act_layer,
        )

        # SemanticInjectionModule
        self.SIM = nn.ModuleList()
        inj_module = SIM_BLOCK[injection_type]
        if self.injection:
            for i in range(len(channels)):
                if i in decode_out_indices:
                    self.SIM.append(
                        inj_module(
                            channels[i],
                            out_channels[i],
                            norm_cfg=norm_cfg,
                            activations=act_layer,
                        )
                    )
                else:
                    self.SIM.append(nn.Identity())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location="cpu"
            )
            if "state_dict_ema" in checkpoint:
                state_dict = checkpoint["state_dict_ema"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        outputs = self.tpm(x)
        out = self.ppa(outputs)
        out = self.trans(out)

        if self.injection:
            xx = out.split(self.channels, dim=1)
            results = []
            for i in range(len(self.channels)):
                if i in self.decode_out_indices:
                    local_tokens = outputs[i]
                    global_semantics = xx[i]
                    out_ = self.SIM[i](local_tokens, global_semantics)
                    results.append(out_)
            return results
        else:
            outputs.append(out)
            return outputs


@register_model
def topformer_tiny(pretrained=False, **kwargs):
    model = Topformer(
        cfgs=[
            # k, t, c, s
            [3, 1, 16, 1],  # 1/2        0.464K  17.461M
            [3, 4, 16, 2],  # 1/4  1     3.44K   64.878M
            [3, 3, 16, 1],  #            4.44K   41.772M
            [5, 3, 32, 2],  # 1/8  3     6.776K  29.146M
            [5, 3, 32, 1],  #            13.16K  30.952M
            [3, 3, 64, 2],  # 1/16 5     16.12K  18.369M
            [3, 3, 64, 1],  #            41.68K  24.508M
            [5, 6, 96, 2],  # 1/32 7     0.129M  36.385M
            [5, 6, 96, 1],  #            0.335M  49.298M
        ],
        channels=[16, 32, 64, 96],
        out_channels=[None, 128, 128, 128],
        embed_out_indice=[2, 4, 6, 8],
        decode_out_indices=[1, 2, 3],
        depths=4,
        num_heads=4,
        c2t_stride=2,
        drop_path_rate=0.1,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        **kwargs,
    )
    return model
