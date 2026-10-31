"""Block library tests. Runs with pytest or directly:

    python tests/test_blocks.py
"""

import itertools
import sys
from unittest import SkipTest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.models.blocks import (ACTS, BODIES, DOWNSAMPLES, FIXED_ORDER, HEAD_POOLS, NORMS, Head,  # noqa: E402
                                  LayerNorm2d, body_out, make_block, make_downsample, pool_out)

CIN, WIDTH, GROWTH, H = 16, 32, 16, 16


def run(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = m(x)
    assert torch.isfinite(y).all()
    return y


def test_block_shapes():
    """Every body × connection × stride × order, input channels ≠ output channels."""
    x = torch.randn(2, CIN, H, H)
    for kind, conn, stride, order in itertools.product(BODIES, ("none", "add", "concat"), (1, 2), ("post", "pre")):
        if conn == "concat" and stride == 2:
            continue
        m, cout = make_block(conn, kind, CIN, WIDTH, k=3, stride=stride, order=order, growth=GROWTH, se=True)
        want_c = CIN + GROWTH if conn == "concat" else WIDTH
        y = run(m, x)
        assert cout == want_c and y.shape == (2, want_c, H // stride, H // stride), (kind, conn, stride, order, y.shape)


def test_valid_padding_size():
    for kind, k, stride in itertools.product(BODIES, (1, 3, 5, 7, 11), (1, 2)):
        h = 29
        m, _ = make_block("none", kind, CIN, WIDTH, k=k, stride=stride, padding="valid")
        if body_out(kind, h, k, stride, "valid") < 1:
            continue
        y = run(m, torch.randn(2, CIN, h, h))
        assert y.shape[-1] == body_out(kind, h, k, stride, "valid"), (kind, k, stride, y.shape)


def test_kernels_same_padding():
    x = torch.randn(2, CIN, 8, 8)
    for kind, k in itertools.product(BODIES, (1, 3, 5, 7, 11)):
        m, _ = make_block("add", kind, CIN, WIDTH, k=k)
        assert run(m, x).shape == (2, WIDTH, 8, 8)


def test_downsample_sizes():
    for kind, h in itertools.product(DOWNSAMPLES, (8, 9, 29, 32, 64)):
        y = run(make_downsample(kind, CIN, "bn", "relu", "post"), torch.randn(2, CIN, h, h))
        assert y.shape == (2, CIN, pool_out(kind, h), pool_out(kind, h)), (kind, h, y.shape)


def test_norms_and_acts():
    x = torch.randn(2, CIN, H, H)
    for norm, act in itertools.product(NORMS, ACTS):
        m, _ = make_block("add", "double_conv", CIN, WIDTH, k=3, norm=norm, act=act)
        run(m, x)


def test_residual_starts_as_shortcut():
    """The branch's last norm starts at zero: with an identity shortcut the block is act(x) (or x)."""
    x = torch.randn(2, WIDTH, H, H)
    for kind in BODIES:
        m, _ = make_block("add", kind, WIDTH, WIDTH, k=3, norm="bn", order="post")
        if m.body.out_norm is None:  # convnext ends with a conv, no norm to zero
            continue
        want = x if kind in FIXED_ORDER else torch.relu(x)
        assert torch.allclose(run(m, x), want, atol=1e-6), kind


def test_norms_run_once():
    """Every norm module runs exactly once per forward (out_norm must not be registered a second time)."""
    x = torch.randn(2, CIN, H, H)
    for kind, conn, order, norm, se in itertools.product(BODIES, ("none", "add", "concat"), ("post", "pre"),
                                                         ("bn", "ln"), (False, True)):
        m, _ = make_block(conn, kind, CIN, WIDTH, k=3, order=order, norm=norm, se=se, growth=GROWTH)
        norms = [n for n in m.modules() if isinstance(n, (nn.BatchNorm2d, LayerNorm2d))]
        calls = []
        for n in norms:
            n.register_forward_hook(lambda *a: calls.append(1))
        run(m, x)
        assert len(calls) == len(norms), (kind, conn, order, norm, se, len(calls), len(norms))


def test_layernorm2d():
    x = torch.randn(2, CIN, 5, 5)
    ln = LayerNorm2d(CIN)
    mu, var = x.mean(1, keepdim=True), x.var(1, unbiased=False, keepdim=True)
    assert torch.allclose(ln(x), (x - mu) / torch.sqrt(var + ln.eps), atol=1e-5)


def test_head():
    x = torch.randn(2, CIN, 4, 4)
    for pool, n_fc in itertools.product(HEAD_POOLS, range(4)):
        assert run(Head(pool, CIN, 4, 4, n_fc, 64, "relu"), x).shape == (2, 15)


def need_gpu():
    if not torch.cuda.is_available():
        raise SkipTest("needs a CUDA GPU")  # pytest reports it as skipped


def test_cuda_bf16_channels_last():
    need_gpu()
    x = torch.randn(4, CIN, H, H, device="cuda").to(memory_format=torch.channels_last)
    for kind, conn, ds in itertools.product(BODIES, ("none", "add", "concat"), DOWNSAMPLES):
        m, c = make_block(conn, kind, CIN, WIDTH, k=3, norm="ln", se=True, growth=GROWTH)
        net = nn.Sequential(m, make_downsample(ds, c, "ln", "gelu", "post")).cuda().to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(x)
        y.float().sum().backward()
        assert torch.isfinite(y.float()).all(), (kind, conn, ds)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"ok  {t.__name__}")
        except SkipTest as e:
            print(f"skipped  {t.__name__}: {e}")
    print(f"{len(tests)} tests run")
