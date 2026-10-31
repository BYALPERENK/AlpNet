"""Search space tests: gene table, sampling, mutation, crossover. Runs with pytest or directly:

    python tests/test_space.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.search.space import GENES, SLOTS, complete, crossover, mutate, sample, slot_genes, validate  # noqa: E402


def n_diff(a: dict, b: dict) -> int:
    return sum(a[k] != b[k] for k in GENES)


def test_gene_table():
    assert len(GENES) == 2 + 14 * len(SLOTS) + 4
    assert all(len(v) >= 2 and len(set(v)) == len(v) for v in GENES.values())


def test_sample_valid_and_seeded():
    rng = random.Random(0)
    for _ in range(1000):
        validate(sample(rng))
    assert sample(random.Random(1)) == sample(random.Random(1))


def test_mutate():
    rng = random.Random(0)
    g = sample(rng)
    assert mutate(g, rng, 0.0) == g
    assert n_diff(mutate(g, rng, 1.0), g) == len(GENES)  # a mutated gene always takes a new value
    mean = sum(n_diff(mutate(g, rng, 0.1), g) for _ in range(2000)) / 2000
    assert abs(mean - 0.1 * len(GENES)) < 0.5, mean
    validate(mutate(g, rng, 0.5))


def test_crossover():
    rng = random.Random(0)
    a, b = sample(rng), sample(rng)
    assert crossover(a, b, rng, 1.0, 0.0) == a
    assert crossover(a, b, rng, 0.0, 1.0) == b
    for _ in range(100):
        c = crossover(a, b, rng, 0.5, 0.5)
        assert all(c[k] in (a[k], b[k]) for k in GENES)
    from_a = sum(crossover(a, b, rng, 0.45, 0.45)[k] == a[k] != b[k] for _ in range(500) for k in GENES)
    differing = 500 * n_diff(a, b)
    assert 0.42 < from_a / differing < 0.55, from_a / differing  # 0.45 + random draws that hit a's value


def test_complete():
    g = complete({"s64.active": True, "s64.width": 32, "s32.active": True, "s32.depth": 2})
    validate(g)
    assert slot_genes(g, 32)["depth"] == 2 and slot_genes(g, 16)["active"] is False
    assert list(validate(dict(sorted(g.items())))) == list(GENES)  # key order (a JSON round trip) does not matter
    try:
        complete({"s2.active": True})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown gene accepted")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
