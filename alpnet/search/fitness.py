"""Search fitness: val macro-F1 in points minus a cost penalty.

    fitness = 100·F1 − a·log2(max(params, 0.1M) / 1M) − b·log2(max(MACs, 0.01G) / 0.1G),  a = b = 0.3

The penalty is known before training, so stop rules can use it (stopping.reach_elite).
"""

import math

A = B = 0.3


def cost_penalty(params: float, macs: float) -> float:
    return A * math.log2(max(params, 1e5) / 1e6) + B * math.log2(max(macs, 1e7) / 1e8)


def fitness(f1: float, params: float, macs: float) -> float:
    """f1 on the 0-1 scale of evaluate()."""
    return 100 * f1 - cost_penalty(params, macs)
