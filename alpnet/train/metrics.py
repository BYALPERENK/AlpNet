"""Evaluation metrics: accuracy, macro-F1, macro AUROC, per-group F1, cross-group errors."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def evaluate(probs: np.ndarray, y: np.ndarray, src: np.ndarray, classes: list[str], groups: list[str]) -> dict:
    """probs [N, C] softmax, y [N] labels, src [N] group index of the true class."""
    pred = probs.argmax(1)
    n_cls = len(classes)
    class_group = np.zeros(n_cls, dtype=int)
    for c, g in zip(y, src):
        class_group[c] = g
    wrong = pred != y
    cross = wrong & (class_group[pred] != src)

    out = {
        "acc": float((~wrong).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_auroc": float(roc_auc_score(y, probs, multi_class="ovr", average="macro")),
        # share of all samples misclassified into another group (source shortcut check)
        "cross_group_err": float(cross.mean()),
        # share of the errors that cross groups
        "cross_group_share": float(cross.sum() / max(wrong.sum(), 1)),
        "group_f1": {},
        "class_f1": dict(zip(classes, f1_score(y, pred, average=None, labels=range(n_cls)).round(4).tolist())),
    }
    for gi, g in enumerate(groups):
        labels = np.flatnonzero(class_group == gi)
        out["group_f1"][g] = float(f1_score(y, pred, labels=labels, average="macro"))
    out["confusion"] = confusion_matrix(y, pred, labels=range(n_cls)).tolist()
    return out
