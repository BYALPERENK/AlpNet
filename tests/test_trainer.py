"""Trainer tests (needs the GPU and the built dataset). Runs with pytest or directly:

    python tests/test_trainer.py
"""

import json
import sys
import tempfile
from functools import cache
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.data.loader import load_splits  # noqa: E402
from alpnet.train.stopping import MinF1, Stop  # noqa: E402
from alpnet.train.trainer import Recipe, seed_everything, train  # noqa: E402

splits = cache(lambda: load_splits("alpnet15_v1"))


class LinearNet(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.fc = nn.Linear(64 * 64, 15)
        self.scale = scale

    def forward(self, x):
        return self.fc(x.flatten(1)) * self.scale


def test_nan_run_fails_cleanly():
    """A NaN loss in the first epoch ends the run with a failed status instead of an exception."""
    if not torch.cuda.is_available():
        return
    with tempfile.TemporaryDirectory() as d:
        seed_everything(0)
        s = train(LinearNet(float("nan")), splits(), Recipe(epochs=2), 0, Path(d), log=lambda msg: None)
        assert s["status"].startswith("failed") and s["epochs_done"] == 0, s["status"]
        assert s["val"] is None and s["test"] is None and s["checkpoints"] == {}
        assert json.loads((Path(d) / "summary.json").read_text())["status"] == s["status"]


def test_test_at_end_matches_every_epoch():
    """Search mode (test once at the end with the checkpoint weights) reports the same scores; fused AdamW runs."""
    if not torch.cuda.is_available():
        return
    out = []
    for recipe in (Recipe(epochs=3), Recipe(epochs=3, test_every_epoch=False), Recipe(epochs=3, fused=True)):
        with tempfile.TemporaryDirectory() as d:
            seed_everything(0)
            s = train(LinearNet(), splits(), recipe, 0, Path(d), log=lambda msg: None)
            rows = [json.loads(line) for line in (Path(d) / "metrics.jsonl").read_text().splitlines()]
            assert s["status"] == "completed" and s["test_time_s"] > 0
            assert all(("test_macro_f1" in r) == recipe.test_every_epoch for r in rows)
            out.append(s)
    every, end, fused = out
    for m in every["checkpoints"]:
        a, b = every["checkpoints"][m], end["checkpoints"][m]
        assert a["epoch"] == b["epoch"] and a["val"]["macro_f1"] == b["val"]["macro_f1"]
        for k in ("acc", "macro_f1", "macro_auroc"):
            assert abs(a["test"][k] - b["test"][k]) < 1e-6, (m, k, a["test"][k], b["test"][k])
    assert 0 < fused["val"]["macro_f1"] <= 1
    # sealed (search candidates): the same test scores, in test_sealed.json only
    with tempfile.TemporaryDirectory() as d:
        seed_everything(0)
        logged = []
        s = train(LinearNet(), splits(), Recipe(epochs=3, test_every_epoch=False), 0, Path(d), log=logged.append,
                  seal_test=True)
        sealed = json.loads((Path(d) / "test_sealed.json").read_text())["checkpoints"]
        saved = json.loads((Path(d) / "summary.json").read_text())
        for summary in (s, saved):
            assert summary["test"] is None and all(c["test"] is None for c in summary["checkpoints"].values())
        assert not any("test" in line for line in logged)
        assert set(sealed) == {"macro_f1"}  # the primary checkpoint only
        for m, c in sealed.items():
            assert c["epoch"] == end["checkpoints"][m]["epoch"]
            assert abs(c["test"]["macro_f1"] - end["checkpoints"][m]["test"]["macro_f1"]) < 1e-6


def test_save_weights():
    """save_weights keeps the checkpoint epoch's weights as model.pt (scripts/inspect_errors.py)."""
    if not torch.cuda.is_available():
        return
    with tempfile.TemporaryDirectory() as d:
        seed_everything(0)
        train(LinearNet(), splits(), Recipe(epochs=2), 0, Path(d), log=lambda msg: None, save_weights=True)
        LinearNet().load_state_dict(torch.load(Path(d) / "model.pt", map_location="cpu"))
        seed_everything(0)
        train(LinearNet(), splits(), Recipe(epochs=1), 0, Path(d) / "none", log=lambda msg: None)
        assert not (Path(d) / "none" / "model.pt").exists()


def test_stop_rules():
    """A prune ends the run with status "pruned" and its partial score; a non-prune stop stays "completed"."""
    if not torch.cuda.is_available():
        return
    unreachable = MinF1("unreachable", {2: 2.0})  # F1 is at most 1: cut at epoch 2
    finished = lambda rows: Stop("done early", prune=False) if len(rows) == 2 else None  # noqa: E731
    for rule, status in ((unreachable, "pruned: unreachable at epoch 2"), (finished, "completed")):
        with tempfile.TemporaryDirectory() as d:
            seed_everything(0)
            s = train(LinearNet(), splits(), Recipe(epochs=3), 0, Path(d), log=lambda msg: None, rules=[rule])
            assert s["status"].startswith(status) and s["epochs_done"] == 2, s["status"]
            assert s["stop"] and s["val"] is not None and s["best_val_epoch"] in (1, 2)
            assert len((Path(d) / "metrics.jsonl").read_text().splitlines()) == 2
            assert json.loads((Path(d) / "config.json").read_text())["stop_rules"][0]["rule"] in ("MinF1", "function")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
