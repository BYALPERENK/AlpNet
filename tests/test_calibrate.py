"""Stop rule calibration: the setting choice (scripts/calibrate_rules.py; CPU only). Runs with pytest or directly:

    python tests/test_calibrate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.calibrate_rules import pick, settings  # noqa: E402


def test_pick_most_saved_without_wrong_cut():
    # (name, wrong cuts, seconds saved), loose to tight
    d = [("0.40", 0, 0.0), ("0.45", 0, 0.0), ("0.50", 0, 0.0), ("0.55", 1, 9.0)]
    assert pick(d) == "0.50"  # a tighter setting saving nothing more is still the tighter one
    assert pick([("0.40", 0, 2.0), ("0.45", 1, 3.0)]) == "0.40"  # no extra step looser: the loosest stays on
    assert pick([("a", 2, 5.0)]) is None
    assert pick([("a", 0, 1.0), ("b", 0, 4.0), ("c", 0, 4.0), ("d", 0, 3.0)]) == "c"


def test_settings_grid():
    for epochs in (20, 40):
        s = settings(epochs)
        for kind, options in s.items():
            names = [n for n, _ in options]
            assert len(set(names)) == len(names), (epochs, kind)
        for _, v in s["A percentile"]:
            assert all(1 <= e <= epochs for e in v), v
        assert {len(v["check"]) for _, v in s["B reach_elite"]} == {1, 2, 3, 4}  # (30) ... (5, 10, 20, 30)
        assert all(max(v["check"]) <= epochs for _, v in s["B reach_elite"])
    assert [v for n, v in settings(20)["A percentile"] if n.endswith("at 10")] == [{10: q} for q in (0.1, 0.2, 0.3, 0.5)]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
