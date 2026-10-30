"""Building blocks of decoded networks.

Norm / activation factories, the conv unit, SE, MaxBlurPool, the six block bodies, the three
connection types (none / add / concat), the stem and the head. The decoder (decoder.py) picks and
wires these; this module has no knowledge of genotypes.

Shapes are also computed analytically (conv_out, pool_out) so the decoder can repair a genotype before
building any module.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAKY_SLOPE = 0.01
HEAD_DROPOUT = 0.3

ACTS = {
    "relu": nn.ReLU,
    "leaky_relu": partial(nn.LeakyReLU, LEAKY_SLOPE),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "hardswish": nn.Hardswish,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}
RELU_LIKE = {"relu", "leaky_relu", "elu", "gelu", "silu", "mish", "hardswish"}  # Kaiming init; others Xavier
NORMS = ("bn", "ln", "none")  # GroupNorm left out of the search
BODIES = ("conv", "double_conv", "bottleneck", "dw_sep", "mbconv", "convnext")
FIXED_ORDER = {"mbconv", "convnext"}  # internal order is fixed; the order gene has no effect
POOLS = ("maxpool2", "maxpool3", "avgpool", "blurpool")
DOWNSAMPLES = (*POOLS, "strided_conv")
HEAD_POOLS = ("gap", "gmp", "gap_gmp", "flatten")


def make_act(name: str) -> nn.Module:
    return ACTS[name]()


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels only, at every pixel (ConvNeXt); NCHW in and out."""

    def forward(self, x):
        x = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def make_norm(name: str, c: int) -> nn.Module:
    if name == "bn":
        return nn.BatchNorm2d(c)
    if name == "ln":
        return LayerNorm2d(c)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {name!r}")


def round8(x: float) -> int:
    return max(8, int(round(x / 8)) * 8)


def conv_out(h: int, k: int, stride: int = 1, padding: str = "same") -> int:
    p = k // 2 if padding == "same" else 0
    return (h + 2 * p - k) // stride + 1


def pool_out(kind: str, h: int) -> int:
    if kind == "maxpool3":
        return (h - 1) // 2 + 1
    if kind == "strided_conv":
        return conv_out(h, 3, 2)
    return h // 2  # maxpool2, avgpool, blurpool


def body_out(kind: str, h: int, k: int, stride: int = 1, padding: str = "same") -> int:
    """Spatial size after a body: one k×k conv (with the stride), two for double_conv."""
    h = conv_out(h, k, stride, padding)
    return conv_out(h, k, 1, padding) if kind == "double_conv" else h


def init_conv(conv: nn.Conv2d, act: str):
    if act in RELU_LIKE:
        if act == "leaky_relu":
            nn.init.kaiming_normal_(conv.weight, a=LEAKY_SLOPE, nonlinearity="leaky_relu")
        else:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    else:
        nn.init.xavier_normal_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def make_conv(cin: int, cout: int, k: int, stride: int = 1, padding: str = "same", groups: int = 1,
              bias: bool = True, act: str = "relu") -> nn.Conv2d:
    conv = nn.Conv2d(cin, cout, k, stride, k // 2 if padding == "same" else 0, groups=groups, bias=bias)
    init_conv(conv, act)
    return conv


class ConvUnit(nn.Sequential):
    """conv -> norm -> act ("post") or norm -> act -> conv ("pre", pre-activation).

    use_act=False drops the activation of a post unit (the last unit of a residual branch: the
    activation comes after the addition). out_norm is the trailing norm of a post unit, if any.

    out_norm is a plain reference (object.__setattr__): assigning a module the usual way registers
    it a second time, and nn.Sequential would run it again after the activation.
    """

    def __init__(self, cin: int, cout: int, k: int, *, stride: int = 1, padding: str = "same", groups: int = 1,
                 norm: str = "bn", act: str = "relu", order: str = "post", use_act: bool = True):
        conv = make_conv(cin, cout, k, stride, padding, groups, bias=norm == "none", act=act)
        if order == "post":
            super().__init__(conv, make_norm(norm, cout), make_act(act) if use_act else nn.Identity())
            object.__setattr__(self, "out_norm", self[1] if norm != "none" else None)
        else:
            super().__init__(make_norm(norm, cin), make_act(act), conv)
            object.__setattr__(self, "out_norm", None)


class SE(nn.Module):
    """Squeeze-excitation: per-channel weights from global context; hidden size max(8, C/4)."""

    def __init__(self, c: int):
        super().__init__()
        h = max(8, c // 4)
        self.fc1 = nn.Conv2d(c, h, 1)
        self.fc2 = nn.Conv2d(h, c, 1)

    def forward(self, x):
        s = F.relu(self.fc1(x.mean((2, 3), keepdim=True)))
        return x * torch.sigmoid(self.fc2(s))


class MaxBlurPool(nn.Module):
    """Anti-aliased ×2 downsampling (Zhang 2019): stride-1 max pool, then [1 2 1] blur with stride 2."""

    def __init__(self):
        super().__init__()
        f = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("filt", (f[:, None] * f[None, :] / 16)[None, None], persistent=False)

    def forward(self, x):
        c = x.shape[1]
        x = F.pad(F.max_pool2d(x, 2, 1), (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.filt.to(x.dtype).expand(c, -1, -1, -1).contiguous(), stride=2, groups=c)


def make_pool(kind: str) -> nn.Module:
    if kind == "maxpool2":
        return nn.MaxPool2d(2)
    if kind == "maxpool3":
        return nn.MaxPool2d(3, 2, 1)
    if kind == "avgpool":
        return nn.AvgPool2d(2)
    if kind == "blurpool":
        return MaxBlurPool()
    raise ValueError(f"unknown pool {kind!r}")


def make_downsample(kind: str, c: int, norm: str, act: str, order: str) -> nn.Module:
    """A standalone ×2 downsampling layer (strided conv as a separate 3×3 stride-2 unit)."""
    if kind == "strided_conv":
        return ConvUnit(c, c, 3, stride=2, norm=norm, act=act, order=order)
    return make_pool(kind)


class Body(nn.Sequential):
    """A block body mapping cin -> cout channels; out_norm is its trailing norm (zero-init target),
    a plain reference like ConvUnit.out_norm."""

    def __init__(self, layers: list[nn.Module], out_norm: nn.Module | None):
        super().__init__(*layers)
        object.__setattr__(self, "out_norm", out_norm)


def make_body(kind: str, cin: int, cout: int, *, k: int, stride: int = 1, padding: str = "same",
              norm: str = "bn", act: str = "relu", order: str = "post", expansion: int = 4,
              se: bool = False, last_act: bool = True) -> Body:
    """The stride (1 or 2) is applied by the body's spatial conv (ResNet style)."""
    u = partial(ConvUnit, padding=padding, norm=norm, act=act, order=order)
    if kind == "conv":
        layers = [u(cin, cout, k, stride=stride, use_act=last_act)]
    elif kind == "double_conv":
        layers = [u(cin, cout, k, stride=stride), u(cout, cout, k, use_act=last_act)]
    elif kind == "bottleneck":
        mid = round8(cout / expansion)
        layers = [u(cin, mid, 1), u(mid, mid, k, stride=stride), u(mid, cout, 1, use_act=last_act)]
    elif kind == "dw_sep":
        layers = [u(cin, cin, k, stride=stride, groups=cin), u(cin, cout, 1, use_act=last_act)]
    elif kind == "mbconv":
        hid = cin * expansion
        p = partial(ConvUnit, padding=padding, norm=norm, act=act, order="post")
        layers = [p(cin, hid, 1), p(hid, hid, k, stride=stride, groups=hid), *([SE(hid)] if se else []),
                  p(hid, cout, 1, use_act=False)]
        return Body(layers, layers[-1].out_norm)
    elif kind == "convnext":
        hid = cout * expansion
        layers = [make_conv(cin, cin, k, stride, padding, groups=cin, act=act), make_norm(norm, cin),
                  make_conv(cin, hid, 1, act=act), make_act(act), make_conv(hid, cout, 1, act=act)]
    else:
        raise ValueError(f"unknown body {kind!r}")
    out_norm = layers[-1].out_norm if isinstance(layers[-1], ConvUnit) else None
    if se:
        layers.append(SE(cout))
    return Body(layers, out_norm)


class Residual(nn.Module):
    """body(x) + shortcut(x), then the activation in post order (ResNet v1) or nothing (v2)."""

    def __init__(self, body: Body, shortcut: nn.Module, post_act: nn.Module):
        super().__init__()
        self.body, self.shortcut, self.post_act = body, shortcut, post_act

    def forward(self, x):
        return self.post_act(self.body(x) + self.shortcut(x))


class Dense(nn.Module):
    """DenseNet connection: the body's output (growth channels) is concatenated to the input."""

    def __init__(self, body: Body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return torch.cat([x, self.body(x)], 1)


def make_shortcut(cin: int, cout: int, stride: int, norm: str, order: str) -> nn.Module:
    if cin == cout and stride == 1:
        return nn.Identity()
    conv = make_conv(cin, cout, 1, stride, bias=norm == "none")
    if order == "pre" or norm == "none":
        return conv
    return nn.Sequential(conv, make_norm(norm, cout))


def make_block(conn: str, kind: str, cin: int, width: int, *, k: int, stride: int = 1, padding: str = "same",
               norm: str = "bn", act: str = "relu", order: str = "post", expansion: int = 4, se: bool = False,
               growth: int = 32) -> tuple[nn.Module, int]:
    """One block; returns (module, output channels). add / concat need padding "same"; concat needs stride 1."""
    kw = dict(k=k, stride=stride, padding=padding, norm=norm, act=act, order=order, expansion=expansion, se=se)
    if conn == "none":
        return make_body(kind, cin, width, **kw), width
    if padding != "same":
        raise ValueError(f"{conn} connection needs same padding")
    if conn == "concat":
        if stride != 1:
            raise ValueError("concat connection needs stride 1")
        return Dense(make_body(kind, cin, growth, **kw)), cin + growth
    if conn == "add":
        post = order == "post" and kind not in FIXED_ORDER
        body = make_body(kind, cin, width, last_act=not post, **kw)
        if body.out_norm is not None:
            nn.init.zeros_(body.out_norm.weight)  # the block starts as its shortcut
        return Residual(body, make_shortcut(cin, width, stride, norm, order),
                        make_act(act) if post else nn.Identity()), width
    raise ValueError(f"unknown connection {conn!r}")


def make_transition(cin: int, width: int, norm: str, act: str, order: str) -> ConvUnit:
    """End of a concat stage: 1×1 back to the stage width."""
    return ConvUnit(cin, width, 1, norm=norm, act=act, order=order)


def make_stem(width: int, k: int, norm: str, act: str) -> ConvUnit:
    """1 -> width at 64×64, stride 1, always conv -> norm -> act."""
    return ConvUnit(1, width, k, norm=norm, act=act, order="post")


class Head(nn.Module):
    """pool -> [dropout -> Linear -> act] × n_fc -> dropout -> Linear (logits)."""

    def __init__(self, pool: str, c: int, h: int, w: int, n_fc: int, fc_width: int, act: str,
                 num_classes: int = 15, dropout: float = HEAD_DROPOUT):
        super().__init__()
        self.pool = pool
        feat = {"gap": c, "gmp": c, "gap_gmp": 2 * c, "flatten": c * h * w}[pool]
        layers = []
        for _ in range(n_fc):
            layers += [nn.Dropout(dropout), nn.Linear(feat, fc_width), make_act(act)]
            feat = fc_width
        layers += [nn.Dropout(dropout), nn.Linear(feat, num_classes)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.pool == "gap":
            x = x.mean((2, 3))
        elif self.pool == "gmp":
            x = x.amax((2, 3))
        elif self.pool == "gap_gmp":
            x = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        else:
            x = x.flatten(1)
        return self.mlp(x)
