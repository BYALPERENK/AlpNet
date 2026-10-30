"""The search space: the single definition of every gene and its options.

A genotype is a flat dict {gene name: value} with the fixed key order of GENES:
  stem.kernel, stem.width,
  s64.active, s64.depth, ... s64.growth, s32.active, ... s4.growth   (one slot per downsampling level)
  head.pool, head.n_fc, head.fc_width, head.act
Closed slots keep their genes (hidden genes). Sampling, mutation and crossover work gene by gene on
this dict, and gene statistics read it directly. Turning a genotype into a network is decoder.py's job.

Randomness always comes from a `random.Random` passed in (the GA's own seed).
"""

import random

from ..models.blocks import ACTS, BODIES, DOWNSAMPLES, HEAD_POOLS, NORMS

SLOTS = (64, 32, 16, 8, 4)  # nominal resolution of each downsampling level

STEM_GENES = {
    "kernel": (3, 5, 7),
    "width": (16, 24, 32, 48, 64),
}
SLOT_GENES = {
    "active": (False, True),
    "depth": (1, 2, 3, 4, 5, 6),
    "width": (16, 24, 32, 48, 64, 96, 128, 192, 256, 384),
    "downsample": DOWNSAMPLES,
    "body": BODIES,
    "conn": ("none", "add", "concat"),
    "kernel": (1, 3, 5, 7),  # no 11: depthwise above 7×7 is ~10× slower in channels_last
    "padding": ("same", "valid"),
    "act": tuple(ACTS),
    "norm": NORMS,
    "order": ("post", "pre"),
    "expansion": (2, 4),  # no 6: the largest memory / time multiplier at high resolution
    "se": (False, True),
    "growth": (16, 32, 48),
}
HEAD_GENES = {
    "pool": HEAD_POOLS,
    "n_fc": (0, 1, 2, 3),
    "fc_width": (64, 128, 256, 512),
    "act": tuple(ACTS),
}

GENES: dict[str, tuple] = {
    **{f"stem.{k}": v for k, v in STEM_GENES.items()},
    **{f"s{r}.{k}": v for r in SLOTS for k, v in SLOT_GENES.items()},
    **{f"head.{k}": v for k, v in HEAD_GENES.items()},
}

# Values for genes a hand-written genotype leaves out (see complete()); not a sampling prior.
NEUTRAL = {
    "stem.kernel": 3, "stem.width": 32,
    **{f"s{r}.{k}": v for r in SLOTS for k, v in {
        "active": False, "depth": 1, "width": 64, "downsample": "maxpool2", "body": "conv", "conn": "none",
        "kernel": 3, "padding": "same", "act": "relu", "norm": "bn", "order": "post", "expansion": 4,
        "se": False, "growth": 32}.items()},
    "head.pool": "gap", "head.n_fc": 0, "head.fc_width": 128, "head.act": "relu",
}


def validate(g: dict):
    if list(g) != list(GENES):
        raise ValueError(f"genotype keys differ from GENES: {set(g) ^ set(GENES)}")
    for k, v in g.items():
        if v not in GENES[k]:
            raise ValueError(f"{k}={v!r} is not one of {GENES[k]}")


def complete(partial: dict) -> dict:
    """A full genotype from the genes given; the rest take NEUTRAL values (for hand-written genotypes)."""
    unknown = set(partial) - set(GENES)
    if unknown:
        raise ValueError(f"unknown genes {sorted(unknown)}")
    g = {k: partial.get(k, NEUTRAL[k]) for k in GENES}
    validate(g)
    return g


def sample(rng: random.Random) -> dict:
    """Every gene uniformly from its options (budget rejection is done by the caller)."""
    return {k: rng.choice(v) for k, v in GENES.items()}


def mutate(g: dict, rng: random.Random, qm: float) -> dict:
    """Each gene changes with probability qm, to a different value of its options."""
    out = dict(g)
    for k, options in GENES.items():
        if rng.random() < qm:
            out[k] = rng.choice([v for v in options if v != g[k]])
    return out


def crossover(a: dict, b: dict, rng: random.Random, p1: float, p2: float) -> dict:
    """Each gene independently: from parent a with p1, from b with p2, else a random value (pm = 1 − p1 − p2).

    Genes are aligned by name, i.e. slot by slot (same downsampling level), the active flags included.
    """
    out = {}
    for k, options in GENES.items():
        r = rng.random()
        out[k] = a[k] if r < p1 else b[k] if r < p1 + p2 else rng.choice(options)
    return out


def slot_genes(g: dict, r: int) -> dict:
    """The genes of the slot at nominal resolution r, without the slot prefix."""
    return {k: g[f"s{r}.{k}"] for k in SLOT_GENES}
