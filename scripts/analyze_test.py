"""Opens the sealed test scores of a finished search: how well val predicted test.

Run only after the search: from here on no choice may look at these scores (the final networks are picked by
val, scripts/train_final.py). Per epoch count: Spearman of val and test F1 over all completed networks and
over the best TOP by val fitness, the median test − val gap, the best 20 by val fitness with their test
scores (one training each, seed 0, so about ±1 point of seed noise) and the val F1 / MACs Pareto front.
Writes test_after_search.md into --out (default <search>/analysis).

    python scripts/analyze_test.py runs/main_20261024/search --out results/search
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from alpnet.search.ga import read_jsonl  # noqa: E402
from analyze_search import f1, pareto  # noqa: E402

TOP = 30


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("search", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <search>/analysis")
    args = ap.parse_args()
    out = args.out or args.search / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    ok = []
    for r in read_jsonl(args.search / "results.jsonl", repair=False):
        sealed = ROOT / r["run_dir"] / "test_sealed.json"
        if r["status"] != "completed" or r.get("fitness") is None or not sealed.exists():
            continue
        r["t"] = json.loads(sealed.read_text(encoding="utf-8"))["checkpoints"]["macro_f1"]["test"]
        r["tf"] = 100 * r["t"]["macro_f1"]
        ok.append(r)
    groups = list(ok[0]["t"]["group_f1"])

    L = [f"# Val vs test after the search ({args.search})", "",
         "The test scores were sealed during the search (one training per network, seed 0, the search recipe; "
         "the checkpoint is the best val macro-F1 epoch). Nothing was chosen by them. Differences below ~1 point "
         "between single trainings are seed noise.", ""]
    for E in sorted({r["epochs"] for r in ok}):
        a = sorted((r for r in ok if r["epochs"] == E), key=lambda r: -r["fitness"])
        v, t = np.array([f1(r) for r in a]), np.array([r["tf"] for r in a])
        L += [f"## {E} epochs ({len(a)} networks)", "",
              "| networks | Spearman val / test F1 | median test − val |", "|---|---|---|",
              f"| all | {spearmanr(v, t)[0]:.3f} | {np.median(t - v):+.2f} |",
              f"| best {TOP} by val fitness | {spearmanr(v[:TOP], t[:TOP])[0]:.2f} | {np.median(t[:TOP] - v[:TOP]):+.2f} |",
              "", "Best 20 by val fitness:", "",
              "| # | fitness | val F1 | test F1 | test acc | " + " | ".join(f"test {g}" for g in groups)
              + " | params | MACs | gen | origin |", "|---" * (9 + len(groups)) + "|"]
        for i, r in enumerate(a[:20], 1):
            L.append(f"| {i} | {r['fitness']:.2f} | {f1(r):.2f} | {r['tf']:.2f} | {100 * r['t']['acc']:.2f} | "
                     + " | ".join(f"{100 * r['t']['group_f1'][g]:.1f}" for g in groups)
                     + f" | {r['params'] / 1e6:.2f}M | {r['macs'] / 1e9:.3f}G | {r['generation']} | {r['origin']} |")
        L += ["", "Val F1 / MACs Pareto front (chosen by val):", "",
              "| MACs | params | val F1 | test F1 | gen |", "|---|---|---|---|---|"]
        L += [f"| {r['macs'] / 1e9:.3f}G | {r['params'] / 1e6:.2f}M | {f1(r):.2f} | {r['tf']:.2f} | {r['generation']} |"
              for r in pareto(a)]
        L.append("")
    (out / "test_after_search.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
