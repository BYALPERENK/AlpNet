"""Genotype -> network.

decode(genotype) never rejects: it repairs, logs the repairs, and returns a Phenotype with
  canonical  - JSON-able spec of the network that is actually built: genes without effect are
               dropped, every stage carries its real output resolution. build() reads only this,
               so the same canonical form always gives the same network.
  hash       - of the canonical form; the cache key (two genotypes giving the same network share it)
  repairs    - what had to change for the genotype to be buildable
  params, macs - cost, measured on a meta-device copy (no memory, no compute)
Activation memory is not estimated here (a leaf-output count was off by 0.4–3× against a real bf16
training step); the trainer measures it and catches OOM.

Decoding rules:
- Each active slot works at its downsampling level; the stem is at 64×64 with stride 1, so the s64
  slot never downsamples. Skipped (closed) levels are downsampled by the next active slot's own
  method, so a stage can carry several ×2 downsamplings.
- The last downsampling of a strided-conv stage is the first block's stride 2 (ResNet style);
  any extra ones, and all of a concat stage's, are standalone 3×3 stride-2 units.
- Padding only matters without a connection (add / concat use same) and with a kernel above 1. A valid
  stage that would leave too small a map for the remaining downsamplings (floor 4×4 at the end) is
  repaired to same.
- A bottleneck's expansion is kept as the smallest value giving the same middle width (round8 of the
  body's output width / e; a concat body outputs `growth` channels), so genotypes building the same
  network share the hash.
- The order gene is dropped where it changes nothing: fixed-order bodies, and stages with neither a
  norm nor an activation (pre and post are then the same conv).
- A double_conv stage of depth d with no connection and no SE is written as a conv stage of depth 2d:
  the same units in the same order (stride in the first), so the two genotypes share the hash.
- No active slot: s32 is opened.
- The stem takes its norm and activation from the first active slot.
"""

import hashlib
import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from ..search.space import SLOT_GENES, SLOTS, slot_genes, validate
from .blocks import (FIXED_ORDER, Head, body_out, make_block, make_downsample, make_stem, make_transition,
                     pool_out, round8)

DECODER_VERSION = 6  # bump when the genotype -> network mapping changes (part of the cache key)
MIN_RES = 4
MAX_PARAMS = 3e6
MAX_MACS = 1e9
LEVEL = {r: i for i, r in enumerate(SLOTS)}  # s64 -> 0 ... s4 -> 4
EXPANDING = {"bottleneck", "mbconv", "convnext"}  # bodies that use the expansion gene


class Net(nn.Module):
    def __init__(self, features: nn.Sequential, head: Head):
        super().__init__()
        self.features, self.head = features, head

    def forward(self, x):
        return self.head(self.features(x))


@dataclass
class Phenotype:
    canonical: dict
    hash: str
    repairs: list[str] = field(default_factory=list)
    params: int = 0
    macs: float = 0.0

    def build(self) -> Net:
        return build(self.canonical)

    def within_budget(self, max_params: float = MAX_PARAMS, max_macs: float = MAX_MACS) -> bool:
        return self.params <= max_params and self.macs <= max_macs


def _stage_res(st: dict, h: int, padding: str, fold: bool) -> int:
    for i in range(st["depth"]):
        h = body_out(st["body"], h, st["kernel"], 2 if i == 0 and fold else 1, padding)
    return h


def canonicalize(g: dict) -> tuple[dict, list[str]]:
    g = validate(g)
    repairs = []
    active = [r for r in SLOTS if g[f"s{r}.active"]]
    if not active:
        active = [32]
        repairs.append("no active slot: s32 opened")
    first = slot_genes(g, active[0])
    stem = {"kernel": g["stem.kernel"], "width": g["stem.width"], "norm": first["norm"], "act": first["act"]}

    stages, h, level, last = [], 64, 0, LEVEL[active[-1]]
    for r in active:
        s = slot_genes(g, r)
        st = {"slot": r, "depth": s["depth"], "width": s["width"], "body": s["body"], "conn": s["conn"],
              "kernel": s["kernel"], "act": s["act"], "norm": s["norm"], "se": s["se"]}
        if s["body"] not in FIXED_ORDER and not (s["norm"] == "none" and s["act"] == "identity"):
            st["order"] = s["order"]
        if s["body"] == "bottleneck":  # only round8(out / e) matters: the smallest e giving the same
            out = s["growth"] if s["conn"] == "concat" else s["width"]  # a concat body outputs growth channels
            mid = round8(out / s["expansion"])
            st["expansion"] = min(e for e in SLOT_GENES["expansion"] if round8(out / e) == mid)
        elif s["body"] in EXPANDING:
            st["expansion"] = s["expansion"]
        if s["conn"] == "concat":
            st["growth"] = s["growth"]
        if s["body"] == "double_conv" and s["conn"] == "none" and not s["se"]:  # = conv, twice the depth
            st["body"], st["depth"] = "conv", 2 * s["depth"]

        n = LEVEL[r] - level
        fold = n > 0 and s["downsample"] == "strided_conv" and s["conn"] != "concat"
        if n:
            st["downsample"] = {"method": s["downsample"], "count": n}
            for _ in range(n - fold):
                h = pool_out(s["downsample"], h)

        # padding only matters without a connection and with a spatial kernel (1×1: same = valid)
        padding = s["padding"] if s["conn"] == "none" and s["kernel"] > 1 else "same"
        need = MIN_RES * 2 ** (last - LEVEL[r])  # room for the downsamplings still to come
        if padding == "valid" and _stage_res(st, h, "valid", fold) < need:
            padding = "same"
            repairs.append(f"s{r}: valid -> same (map would drop below {need}×{need})")
        if s["conn"] == "none":
            st["padding"] = padding
        h = _stage_res(st, h, padding, fold)
        st["res"] = h
        stages.append(st)
        level = LEVEL[r]

    head = {"pool": g["head.pool"], "n_fc": g["head.n_fc"]}
    if g["head.n_fc"]:
        head |= {"fc_width": g["head.fc_width"], "act": g["head.act"]}
    return {"stem": stem, "stages": stages, "head": head}, repairs


def phenotype_hash(canonical: dict) -> str:
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def build(canonical: dict) -> Net:
    stem = canonical["stem"]
    layers, c = [make_stem(stem["width"], stem["kernel"], stem["norm"], stem["act"])], stem["width"]
    for st in canonical["stages"]:
        order = st.get("order", "post")
        unit = dict(norm=st["norm"], act=st["act"], order=order)
        ds = st.get("downsample")
        fold = bool(ds) and ds["method"] == "strided_conv" and st["conn"] != "concat"
        for _ in range((ds["count"] if ds else 0) - fold):
            layers.append(make_downsample(ds["method"], c, **unit))
        for i in range(st["depth"]):
            m, c = make_block(st["conn"], st["body"], c, st["width"], k=st["kernel"], stride=2 if i == 0 and fold else 1,
                              padding=st.get("padding", "same"), expansion=st.get("expansion", 4), se=st["se"],
                              growth=st.get("growth", 32), **unit)
            layers.append(m)
        if st["conn"] == "concat":
            layers.append(make_transition(c, st["width"], **unit))
            c = st["width"]
    h = canonical["stages"][-1]["res"]
    hd = canonical["head"]
    head = Head(hd["pool"], c, h, h, hd["n_fc"], hd.get("fc_width", 64), hd.get("act", "relu"))
    return Net(nn.Sequential(*layers), head)


def measure(canonical: dict) -> tuple[int, float]:
    """params and MACs per image (FlopCounterMode, like the trainer), on the meta device."""
    with torch.device("meta"):
        net = build(canonical)
        x = torch.zeros(1, 1, 64, 64)
    with FlopCounterMode(display=False) as fc:
        net.eval()(x)
    return sum(p.numel() for p in net.parameters()), fc.get_total_flops() / 2


def decode(g: dict) -> Phenotype:
    canonical, repairs = canonicalize(g)
    params, macs = measure(canonical)
    return Phenotype(canonical, phenotype_hash(canonical), repairs, params, macs)
