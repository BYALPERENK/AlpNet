"""Is a shorter recipe enough to rank candidates?

The same random networks (train_random.py --like) trained with shorter recipes and with the longest one;
matched by phenotype hash. Spearman of the checkpoint val F1 and of the fitness between each short
recipe and the long one, with a bootstrap 95% interval, over all networks and over the best TOP by the
long recipe's fitness (where the GA's choices are made: similar, good networks), and how many of the
top 20% (by long-recipe fitness) the short recipe also puts in its top 20%. For comparison: the best val
F1 by epoch 5 / 10 / 20 / 30 inside the long run (what the stop rules see). Val only. Writes
epochs_report.md into the long run's folder.

    python scripts/compare_epochs.py runs/gen0_20ep runs/gen0_40ep
    python scripts/compare_epochs.py runs/random_10ep_X runs/random_20ep_Y runs/gen0_40ep
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.search.ga import read_jsonl  # noqa: E402

TOP = 30


def load(d: Path) -> dict:
    return {r["hash"]: r for r in read_jsonl(d / "results.jsonl", repair=False) if r["status"] == "completed"}


def boot(a, b, n=2000, seed=0) -> tuple[float, float, float]:
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    rs = [spearmanr(a[i], b[i])[0] for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return spearmanr(a, b)[0], *np.nanquantile(rs, (0.025, 0.975))


def top_overlap(short, full, frac=0.2) -> str:
    k = max(1, round(frac * len(full)))
    ts, tf = set(np.argsort(-np.asarray(short))[:k]), set(np.argsort(-np.asarray(full))[:k])
    return f"{len(ts & tf)} / {k}"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    *shorts, long_dir = map(Path, sys.argv[1:])
    rl = load(long_dir)
    n_long = max(len(r["curve"]) for r in rl.values())
    L = [f"# Short recipes vs {n_long} epochs ({', '.join(d.name for d in shorts)} vs {long_dir})", "",
         f"| short recipe | networks | Spearman val F1 (95% CI) | Spearman fitness (95% CI) | Spearman fitness, best {TOP} "
         "(95% CI) | top 20% shared | hours (short / long) |", "|---|---|---|---|---|---|---|"]
    for d in shorts:
        rs = load(d)
        common = [h for h in rs if h in rl]
        n_short = max(len(rs[h]["curve"]) for h in common)
        f_s = [rs[h]["val"]["macro_f1"] for h in common]
        f_l = [rl[h]["val"]["macro_f1"] for h in common]
        g_s, g_l = [rs[h]["fitness"] for h in common], [rl[h]["fitness"] for h in common]
        best = np.argsort(-np.asarray(g_l))[:TOP]
        v, w = boot(f_s, f_l), boot(g_s, g_l)
        t = boot(np.asarray(g_s)[best], np.asarray(g_l)[best]) if len(common) > TOP else None
        hours = (sum(rs[h]["total_s"] for h in common) / 3600, sum(rl[h]["total_s"] for h in common) / 3600)
        L.append(f"| {n_short} epochs | {len(common)} | {v[0]:.3f} ({v[1]:.3f}–{v[2]:.3f}) | {w[0]:.3f} ({w[1]:.3f}–{w[2]:.3f}) "
                 f"| {f'{t[0]:.3f} ({t[1]:.3f}–{t[2]:.3f})' if t else '— (all networks)'} | {top_overlap(g_s, g_l)} "
                 f"| {hours[0]:.2f} / {hours[1]:.2f} |")
    L += ["", f"Inside the {n_long}-epoch run ({len(rl)} networks): best val F1 by epoch e vs the checkpoint val F1", "",
          "| epoch | Spearman (95% CI) |", "|---|---|"]
    final = [r["val"]["macro_f1"] for r in rl.values()]
    for e in (5, 10, 20, 30):
        if e < n_long:
            v = boot([max(r["curve"][:e]) for r in rl.values()], final)
            L.append(f"| {e} | {v[0]:.3f} ({v[1]:.3f}–{v[2]:.3f}) |")
    (long_dir / "epochs_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
