"""Stop rule tests (CPU only, synthetic epoch rows). Runs with pytest or directly:

    python tests/test_stopping.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.train.stopping import (NotLearning, Patience, check, describe, floor, percentile,  # noqa: E402
                                   reach_elite, replay, running_best)


def rows_from(f1, loss=None, acc=None):
    return [{"epoch": i + 1, "val_macro_f1": v, "train_loss": 1.0 if loss is None else loss[i],
             "train_acc": 0.5 if acc is None else acc[i]} for i, v in enumerate(f1)]


def test_not_learning():
    rule = NotLearning()
    e, stop = replay([rule], rows_from([0.01] * 6, loss=[2.75] * 6, acc=[0.3] * 6))
    assert e == 3 and stop.prune and "not learning" in stop.reason
    assert replay([rule], rows_from([0.01] * 6, loss=[2.3] * 6, acc=[0.05] * 6))[0] == 3  # accuracy alone
    assert replay([rule], rows_from([0.3] * 6, loss=[2.3] * 6, acc=[0.3] * 6)) == (6, None)
    # only epoch 3 is checked: slow first epochs or a later collapse do not trigger it
    assert replay([rule], rows_from([0.3] * 6, loss=[2.75, 2.75, 2.3, 2.75, 2.75, 2.75]))[1] is None


def test_floor_uses_best_so_far_at_check_epochs_only():
    rule = floor({10: 0.60})
    e, stop = replay([rule], rows_from([0.5] * 9 + [0.55] + [0.1] * 5))
    assert e == 10 and stop.prune and "floor" in stop.reason
    assert replay([rule], rows_from([0.5] * 4 + [0.62] + [0.5] * 10)) == (15, None)  # best, not last
    assert replay([rule], rows_from([0.5] * 9 + [0.60]))[1] is None  # the threshold itself passes
    assert replay([rule], rows_from([0.1] * 9))[1] is None  # never reached the check epoch
    assert replay([floor({})], rows_from([0.0] * 15))[1] is None  # no thresholds: never prunes
    assert percentile([[0.5] * 4], {}, 4).thresholds == {}


def test_running_best_pads_short_runs():
    rb = running_best([[0.1, 0.3, 0.2], [0.4, 0.2, 0.5, 0.6, 0.1]], 5)
    assert np.allclose(rb, [[0.1, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.5, 0.6, 0.6]])


def test_percentile():
    curves = [[i / 10] * 40 for i in range(11)]  # best so far of run i is i/10 at every epoch
    rule = percentile(curves, {5: 0.2, 10: 0.3, 20: 0.5}, 40)
    assert np.allclose([rule.thresholds[e] for e in (5, 10, 20)], [0.2, 0.3, 0.5])
    e, stop = replay([rule], rows_from([0.25] * 40))
    assert e == 10 and stop.prune and "percentile" in stop.reason  # passes epoch 5, cut at 10
    assert replay([rule], rows_from([0.55] * 40)) == (40, None)


def test_reach_elite():
    curves = [[b + 0.01 * e for e in range(1, 41)] for b in (0.2, 0.4, 0.6)]  # gain e -> 40 is 0.01·(40 − e)
    rule = reach_elite(curves, 80.0, 1.0, (5, 10, 20, 30), 0.9, 40)  # needs 100·F1 − 1 ≥ 80 at the end
    for e in (5, 10, 20, 30):
        assert abs(rule.thresholds[e] - (0.81 - 0.01 * (40 - e))) < 1e-9
    # needs 0.46 / 0.51 / 0.61 / 0.71 at epochs 5 / 10 / 20 / 30
    assert replay([rule], rows_from([0.45 + 0.01 * e for e in range(1, 41)])) == (40, None)
    assert replay([rule], rows_from([0.52] * 40))[0] == 20  # enough early on, stalls
    e, stop = replay([rule], rows_from([0.45] * 40))
    assert e == 5 and stop.prune and "reach_elite" in stop.reason
    # a costlier candidate needs more F1
    assert reach_elite(curves, 80.0, 2.0, (5, 10, 20, 30), 0.9, 40).thresholds[10] > rule.thresholds[10]


def test_patience():
    rule = Patience()  # 5 epochs, from epoch 30
    assert replay([rule], rows_from(list(np.linspace(0.1, 0.9, 40)))) == (40, None)
    f1 = [0.01 * e for e in range(1, 41)]
    e, stop = replay([rule], rows_from(f1[:26] + [0.1] * 14))  # best at epoch 26
    assert e == 31 and not stop.prune
    assert replay([rule], rows_from(f1[:10] + [0.01] * 30))[0] == 30  # long plateau: stops at the first allowed epoch
    tie = f1[:28] + [0.1, 0.1, f1[27]] + [0.1] * 9  # equal to the best is not an improvement
    assert replay([rule], rows_from(tie))[0] == 33


def test_first_rule_wins_and_describe():
    dead = rows_from([0.01] * 40, loss=[2.75] * 40, acc=[0.06] * 40)
    floor10 = floor({10: 0.60})
    e, stop = replay([NotLearning(), floor10, Patience()], dead)
    assert e == 3 and "not learning" in stop.reason
    assert check([floor10, Patience()], dead[:10]).reason.startswith("floor")
    d = describe(floor10)
    assert d == {"rule": "MinF1", "name": "floor", "thresholds": {10: 0.6}}
    json.dumps([describe(r) for r in (NotLearning(), floor10, Patience(), lambda rows: None)])


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
