"""Decoder tests. Runs with pytest or directly:

    python tests/test_decoder.py            # property test on 2000 random genotypes
    python tests/test_decoder.py 10000      # on 10000
"""

import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.models.baselines import PlainCNN, resnet18_64  # noqa: E402
from alpnet.models.blocks import round8  # noqa: E402
from alpnet.models.decoder import build, decode  # noqa: E402
from alpnet.search.space import SLOT_GENES, SLOTS, complete, sample  # noqa: E402

N_RANDOM = 2000
N_FORWARD = 200  # of those, real forward passes (NaN check) on networks within budget

B2 = complete({  # PlainCNN: stem 3×3/32, then 1 + 2 + 2 + 2 Conv3×3-BN-ReLU, maxpool between levels, GAP
    "stem.kernel": 3, "stem.width": 32,
    "s64.active": True, "s64.depth": 1, "s64.width": 32,
    "s32.active": True, "s32.depth": 2, "s32.width": 64,
    "s16.active": True, "s16.depth": 2, "s16.width": 128,
    "s8.active": True, "s8.depth": 2, "s8.width": 256,
})


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def macs(m: torch.nn.Module) -> float:
    with FlopCounterMode(display=False) as fc:
        m.eval()(torch.zeros(1, 1, 64, 64))
    return fc.get_total_flops() / 2


def same_function(ref: torch.nn.Module, net: torch.nn.Module):
    """Copy ref's weights and BN statistics into net (same module order) and compare eval outputs.
    Equal parameter counts alone miss a module that is registered, and run, twice."""
    torch.manual_seed(0)
    for m in ref.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    src, dst = ref.state_dict(), net.state_dict()
    assert [v.shape for v in src.values()] == [v.shape for v in dst.values()]
    net.load_state_dict(dict(zip(dst, src.values())))
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(ref.eval()(x), net.eval()(x), atol=1e-4)


def test_b2_reproduction():
    ph = decode(B2)
    assert ph.repairs == []
    assert ph.params == n_params(PlainCNN()) == n_params(ph.build()), ph.params
    assert ph.macs == macs(PlainCNN()), (ph.macs, macs(PlainCNN()))
    same_function(PlainCNN(), ph.build())


def test_b3_structure():
    """ResNet-18-64 is outside the gene options (width 512) and the budget: built from a canonical form."""
    stage = {"body": "double_conv", "conn": "add", "kernel": 3, "act": "relu", "norm": "bn", "se": False,
             "order": "post", "depth": 2}
    canonical = {
        "stem": {"kernel": 3, "width": 64, "norm": "bn", "act": "relu"},
        "stages": [{**stage, "slot": 64, "width": 64, "res": 64},
                   *({**stage, "slot": r, "width": w, "res": r, "downsample": {"method": "strided_conv", "count": 1}}
                     for r, w in ((32, 128), (16, 256), (8, 512)))],
        "head": {"pool": "gap", "n_fc": 0},
    }
    net = build(canonical)
    assert n_params(net) == n_params(resnet18_64()), (n_params(net), n_params(resnet18_64()))
    assert net(torch.zeros(2, 1, 64, 64)).shape == (2, 15)
    same_function(resnet18_64(), net)


def test_hash():
    rng = random.Random(0)
    g = complete({"s32.active": True, "s32.body": "conv", "s8.active": True, "head.n_fc": 0})
    h = decode(g).hash
    assert decode(dict(g)).hash == h
    same = [("s16.body", "mbconv"),       # hidden gene of a closed slot
            ("s32.expansion", 2),         # expansion has no effect on a conv body
            ("s32.growth", 48),           # growth has no effect without concat
            ("s64.downsample", "avgpool"),  # closed slot
            ("head.fc_width", 512),       # no hidden FC
            ("head.act", "tanh")]
    for k, v in same:
        assert decode({**g, k: v}).hash == h, k
    for k, v in [("s32.body", "dw_sep"), ("s8.downsample", "avgpool"), ("s16.active", True), ("head.n_fc", 1)]:
        assert decode({**g, k: v}).hash != h, k
    for _ in range(200):
        r = sample(rng)
        assert decode(r).hash == decode(dict(r)).hash
    # the same network from different genes: 1×1 kernels ignore padding, bottleneck w16 has mid 8 for e 2 / 4
    k1 = complete({"s32.active": True, "s32.body": "double_conv", "s32.kernel": 1})
    assert decode({**k1, "s32.padding": "valid"}).hash == decode(k1).hash
    assert decode({**k1, "s32.padding": "valid", "s32.kernel": 3}).hash != decode({**k1, "s32.kernel": 3}).hash
    bn = complete({"s32.active": True, "s32.body": "bottleneck", "s32.width": 16})
    assert len({decode({**bn, "s32.expansion": e}).hash for e in (2, 4)}) == 1
    assert len({decode({**bn, "s32.width": 96, "s32.expansion": e}).hash for e in (2, 4)}) == 2
    # a concat bottleneck's body outputs `growth` channels: its middle width is round8(growth / e)
    cat = {**bn, "s32.conn": "concat"}
    for w, growth, n in ((16, 32, 2), (96, 16, 1)):  # mid 16 / 8 vs 8 / 8
        nets = {e: decode({**cat, "s32.width": w, "s32.growth": growth, "s32.expansion": e}) for e in (2, 4)}
        assert len({ph.hash for ph in nets.values()}) == n and len({ph.params for ph in nets.values()}) == n, (w, growth)
        for e, ph in nets.items():  # the canonical form builds the network the genes describe
            body = ph.build().features[2].body  # stem, max pool, the dense block
            assert body[0][0].out_channels == round8(growth / e), (w, growth, e)
    # without connection and SE, double_conv depth d is conv depth 2d (the same units in the same order)
    dc = complete({"s32.active": True, "s32.body": "double_conv", "s32.depth": 2, "s32.padding": "valid"})
    assert decode(dc).hash == decode({**dc, "s32.body": "conv", "s32.depth": 4}).hash
    same_function(decode({**dc, "s32.body": "conv", "s32.depth": 4}).build(), decode(dc).build())
    for k, v in (("s32.se", True), ("s32.conn", "add")):
        assert decode({**dc, k: v}).hash != decode({**dc, "s32.body": "conv", "s32.depth": 4, k: v}).hash, k
    # neither a norm nor an activation: pre and post are the same conv
    plain = complete({"s32.active": True, "s32.body": "double_conv", "s32.norm": "none", "s32.act": "identity"})
    assert decode({**plain, "s32.order": "pre"}).hash == decode(plain).hash
    same_function(decode(plain).build(), decode({**plain, "s32.order": "pre"}).build())
    assert decode({**plain, "s32.order": "pre", "s32.act": "relu"}).hash != decode({**plain, "s32.act": "relu"}).hash
    # the narrowed space: no kernel 11, GroupNorm or expansion 6
    for k, v in (("s32.kernel", 11), ("s32.norm", "gn"), ("s32.expansion", 6)):
        try:
            complete({"s32.active": True, k: v})
            raise AssertionError(f"{k}={v} accepted")
        except ValueError:
            pass


def fingerprint(ph) -> bytes | None:
    """Output of the network built with a fixed seed on fixed CPU inputs; None when degenerate (the same
    output for every input would match unrelated networks)."""
    torch.manual_seed(0)
    net = ph.build().eval()
    x = torch.randn(4, 1, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        y = net(x)
    if not torch.isfinite(y).all() or (y - y[0]).abs().max() < 1e-6:
        return None
    return y.numpy().tobytes()


def variants(g: dict, rng: random.Random) -> list[dict]:
    """Genotypes that may build the same network as g: single slot genes changed, two random ones, and
    conv depth 2d <-> double_conv depth d, in every active slot."""
    out = []
    for r in (r for r in SLOTS if g[f"s{r}.active"]):
        p = f"s{r}."
        for k in ("order", "expansion", "padding", "growth", "se", "kernel", "downsample"):
            out.append({**g, p + k: rng.choice([v for v in SLOT_GENES[k] if v != g[p + k]])})
        if g[p + "body"] == "conv" and g[p + "depth"] % 2 == 0:
            out.append({**g, p + "body": "double_conv", p + "depth": g[p + "depth"] // 2})
        if g[p + "body"] == "double_conv" and 2 * g[p + "depth"] <= max(SLOT_GENES["depth"]):
            out.append({**g, p + "body": "conv", p + "depth": 2 * g[p + "depth"]})
        out.append({**g, **{p + k: rng.choice(SLOT_GENES[k]) for k in rng.sample(sorted(set(SLOT_GENES) - {"active"}), 2)}})
    return out


def test_same_function_same_hash(n: int = 40):
    """Functional fingerprint: a genotype that builds a network with the same output as another (same seed,
    same inputs) must have its hash. Equal params and MACs filter the candidates before building them."""
    rng, checked, compared = random.Random(1), 0, 0
    while checked < n:
        g = sample(rng)
        ph = decode(g)
        if not ph.within_budget() or ph.macs > 2e8 or (fp := fingerprint(ph)) is None:
            continue
        checked += 1
        for v in variants(g, rng):
            pv = decode(v)
            if pv.hash != ph.hash and (pv.params, pv.macs) == (ph.params, ph.macs):
                compared += 1
                assert fingerprint(pv) != fp, f"same network, two hashes:\n{ph.canonical}\n{pv.canonical}"
    print(f"    {checked} genotypes, {compared} look-alike variants built and compared")


def test_repairs():
    ph = decode(complete({}))  # every slot closed
    assert ph.canonical["stages"][0]["slot"] == 32 and ph.repairs == ["no active slot: s32 opened"]

    deep_valid = {"s64.active": True, "s64.depth": 6, "s64.body": "double_conv", "s64.kernel": 7,
                  "s64.padding": "valid", "s4.active": True}
    ph = decode(complete(deep_valid))
    assert ph.canonical["stages"][0]["padding"] == "same" and len(ph.repairs) == 1

    ph = decode(complete({"s64.active": True, "s64.padding": "valid", "s8.active": True}))
    assert ph.repairs == [] and [s["res"] for s in ph.canonical["stages"]] == [62, 7]  # 62 -> 31 -> 15 -> 7


def test_skipped_levels():
    ph = decode(complete({"s4.active": True}))
    st = ph.canonical["stages"][0]
    assert st["downsample"] == {"method": "maxpool2", "count": 4} and st["res"] == 4
    net = ph.build()
    assert sum(isinstance(m, torch.nn.MaxPool2d) for m in net.modules()) == 4

    for conn, want_units in (("add", 1), ("concat", 2)):  # add: last stride folded into the first block
        g = complete({"s64.active": True, "s16.active": True, "s16.downsample": "strided_conv", "s16.conn": conn})
        st = decode(g).canonical["stages"][1]
        assert st["res"] == 16 and st["downsample"]["count"] == 2
        assert decode(g).build()(torch.zeros(1, 1, 64, 64)).shape == (1, 15)


def test_random_genotypes(n: int = N_RANDOM, n_forward: int = N_FORWARD):
    """Every random genotype decodes and gives [2, 15]; a subset within budget also runs for real (no NaN)."""
    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0, forwarded = time.monotonic(), 0
    for _ in range(n):
        ph = decode(sample(rng))
        with torch.device("meta"):
            assert ph.build()(torch.zeros(2, 1, 64, 64)).shape == (2, 15), ph.canonical
        if forwarded < n_forward and ph.within_budget():
            torch.manual_seed(0)
            net = ph.build().to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                y = net.train()(torch.randn(2, 1, 64, 64, device=dev))
            assert y.shape == (2, 15) and torch.isfinite(y.float()).all(), ph.canonical
            forwarded += 1
    print(f"    {n} genotypes decoded, {forwarded} run for real, {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_RANDOM
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t(n) if t is test_random_genotypes else t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
