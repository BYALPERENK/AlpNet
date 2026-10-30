"""Aggregate runs into mean ± std tables per model, from each run's summary.json.

Checkpoint rule: best val macro-F1 by default (--rule picks another rule recorded in summary.json
"checkpoints"). Reported numbers are test-set scores at that checkpoint (AUROC as a side metric);
the epoch each run picked is listed.

    python scripts/summarize_runs.py                       # all runs
    python scripts/summarize_runs.py --prefix 20261021-1700    # only folders starting with this
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ms(vals, scale=100.0, digits=2):
    a = np.asarray(vals) * scale
    return f"{a.mean():.{digits}f} ± {a.std(ddof=1) if len(a) > 1 else 0:.{digits}f}"


def section(by_model: dict, rule: str) -> list[str]:
    first = next(iter(by_model.values()))[0]
    groups = list(first["test"]["group_f1"])
    lines = [f"## Checkpoint: best val {rule} — test scores (mean ± std over seeds, %)", "",
             f"| Model | Params | MACs | train time / run | epoch (of {first['recipe']['epochs']}) | val {rule} | test acc | "
             "test macro-F1 | test AUROC | test cross-group err | " + " | ".join(f"test {g}" for g in groups) + " |",
             "|---" * (10 + len(groups)) + "|"]
    for m, runs in by_model.items():
        cks = [r["checkpoints"][rule] for r in runs]
        t = [c["test"] for c in cks]
        lines.append(
            f"| {m} | {runs[0]['params'] / 1e6:.2f}M | {runs[0]['macs'] / 1e9:.2f}G | "
            f"{ms([r['train_time_s'] for r in runs], scale=1 / 60, digits=1)} min | "
            f"{', '.join(str(c['epoch']) for c in cks)} | {ms([c['val'][rule] for c in cks])} | "
            f"{ms([x['acc'] for x in t])} | {ms([x['macro_f1'] for x in t])} | {ms([x['macro_auroc'] for x in t])} | "
            f"{ms([x['cross_group_err'] for x in t])} | "
            + " | ".join(ms([x["group_f1"][g] for x in t]) for g in groups) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    ap.add_argument("--rule", default="macro_f1", help="checkpoint rule: macro_f1 or macro_auroc")
    ap.add_argument("--out", default=str(ROOT / "runs" / "baselines_summary.md"))
    args = ap.parse_args()

    by_model = defaultdict(list)
    for p in sorted((ROOT / "runs").glob(f"{args.prefix}*/summary.json")):
        s = json.loads(p.read_text())
        if "checkpoints" in s:  # older runs lack per-rule checkpoints and per-epoch test scores
            by_model[s["model"]].append(s)
    if not by_model:
        sys.exit("no runs found")

    runs0 = next(iter(by_model.values()))
    lines = [f"Runs: {sum(map(len, by_model.values()))} ({len(runs0)} seeds per model), "
             f"time/run: " + ", ".join(f"{m} {np.mean([r['train_time_s'] for r in rs]):.0f}s"
                                       for m, rs in by_model.items()), ""]
    lines += section(by_model, args.rule)

    text = "\n".join(lines)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
