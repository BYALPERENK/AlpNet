"""Early stopping and pruning rules.

A rule looks at the epoch rows of the run so far (the metrics.jsonl lines) and returns a Stop or None.
The trainer calls the rules after every epoch and stops at the first Stop. A rule holds only numbers
fixed when it is built (at the start of a generation, from reference curves: in the GA, generation
0's curves at the phase's epoch count), so a run's decision does not depend on what other workers finish meanwhile, and `replay` applies the
same rules to a recorded metrics.jsonl (calibration on generation 0).

    NotLearning   level 1: after warmup, training loss still at chance level or accuracy below 10%
    floor         level 2, D: best val macro-F1 so far below an absolute floor at a check epoch
    percentile    level 2, A: ... below the q-quantile of the reference curves at the same epoch (median
                  stopping)
    reach_elite   level 2, B: ... too low to reach the K-th elite's fitness even with the reference curves'
                  q-quantile gain from that epoch to the end (q 1.0: the largest gain seen)
    Patience      level 3: no val improvement for `patience` epochs, from epoch `start` on

A prune (Stop.prune) keeps the partial score and the candidate cannot be an elite; a patience stop
ends a finished run. All F1 values are on the 0-1 scale of evaluate(). D, A and B take
every setting explicitly (configs/search.yaml, calibrated by scripts/calibrate_rules.py).
"""

import math
from dataclasses import asdict, dataclass, is_dataclass

import numpy as np

LN15 = math.log(15)  # training loss of a uniform prediction over 15 classes, label smoothing included


@dataclass(frozen=True)
class Stop:
    reason: str
    prune: bool  # True: cut with a partial score, cannot be an elite; False: the run counts as finished


def best_f1(rows: list[dict]) -> float:
    return max(r["val_macro_f1"] for r in rows)


@dataclass(frozen=True)
class NotLearning:
    epoch: int = 3  # the first epoch after the 2 warmup epochs
    max_loss: float = 0.98 * LN15
    min_acc: float = 0.10  # chance is 1/15

    def __call__(self, rows: list[dict]) -> Stop | None:
        if len(rows) == self.epoch:
            r = rows[-1]
            if r["train_loss"] > self.max_loss or r["train_acc"] < self.min_acc:
                return Stop(f"not learning at epoch {self.epoch}: train loss {r['train_loss']:.3f}, "
                            f"acc {r['train_acc']:.3f}", prune=True)
        return None


@dataclass(frozen=True)
class MinF1:
    """Prune at a check epoch when the best val macro-F1 so far is below that epoch's threshold."""
    name: str
    thresholds: dict[int, float]  # epoch -> best val macro-F1 needed by then

    def __call__(self, rows: list[dict]) -> Stop | None:
        need = self.thresholds.get(len(rows))
        if need is not None and (best := best_f1(rows)) < need:
            return Stop(f"{self.name} at epoch {len(rows)}: val F1 {best:.4f} < {need:.4f}", prune=True)
        return None


@dataclass(frozen=True)
class Patience:
    patience: int = 5
    start: int = 30

    def __call__(self, rows: list[dict]) -> Stop | None:
        e = len(rows)
        if e >= self.start:
            best_epoch = int(np.argmax([r["val_macro_f1"] for r in rows])) + 1
            if e - best_epoch >= self.patience:
                return Stop(f"patience at epoch {e}: best val F1 at epoch {best_epoch}", prune=False)
        return None


def running_best(curves: list[list[float]], epochs: int) -> np.ndarray:
    """(runs, epochs): best val macro-F1 up to each epoch. A run stopped early (patience) keeps its best."""
    out = np.empty((len(curves), epochs))
    for i, c in enumerate(curves):
        b = np.maximum.accumulate(np.asarray(c[:epochs], dtype=float))
        out[i, :len(b)], out[i, len(b):] = b, b[-1]
    return out


def floor(thresholds: dict[int, float]) -> MinF1:
    """D: a fixed floor at check epochs (LeNet is at ~0.65 by epoch 10). {} never prunes."""
    return MinF1("floor", thresholds)


def percentile(curves: list[list[float]], quantiles: dict[int, float], epochs: int) -> MinF1:
    """A: below the q-quantile of the reference curves' best-so-far val F1 at the same epoch.
    `curves`: val macro-F1 per epoch of fully trained candidates (patience stops included)."""
    rb = running_best(curves, epochs)
    return MinF1("percentile", {e: float(np.quantile(rb[:, e - 1], q)) for e, q in quantiles.items()})


def reach_elite(curves: list[list[float]], elite_fitness: float, penalty: float, check: tuple[int, ...], q: float,
                epochs: int) -> MinF1:
    """B: fitness = 100·F1 − penalty (the candidate's cost penalty, known before training). Pruned at
    epoch e when 100·(best so far + the reference curves' q-quantile gain from e to the end) − penalty is
    below the K-th elite's fitness (fixed at the start of the generation). Not a guarantee: the gains are
    those of the reference networks."""
    rb = running_best(curves, epochs)
    gain = {e: float(np.quantile(rb[:, -1] - rb[:, e - 1], q)) for e in check}
    return MinF1("reach_elite", {e: (elite_fitness + penalty) / 100 - g for e, g in gain.items()})


def check(rules, rows: list[dict]) -> Stop | None:
    for rule in rules:
        if stop := rule(rows):
            return stop
    return None


def replay(rules, rows: list[dict]) -> tuple[int, Stop | None]:
    """The epoch at which the rules would have stopped a recorded run; (len(rows), None) if never."""
    for e in range(1, len(rows) + 1):
        if stop := check(rules, rows[:e]):
            return e, stop
    return len(rows), None


def describe(rule) -> dict:
    """For config.json."""
    return {"rule": type(rule).__name__, **(asdict(rule) if is_dataclass(rule) else {})}
