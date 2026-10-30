"""Progress report of a GA run (run_search.py): one row per generation, the stop rules' work, the best
candidates so far. Val scores only: test scores are not looked at until the search ends.
Reads the files while the search writes them (a line being written is left out, nothing is rewritten).
Writes report.md into the run folder.

    python scripts/search_report.py runs/search_20261022-0658
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.search.ga import read_jsonl  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(sys.argv[1])
    recs = read_jsonl(out / "results.jsonl", repair=False)
    # a generation done again after a crash has a second line until the search restarts: the last one counts
    gens = list({g["generation"]: g for g in read_jsonl(out / "generations.jsonl", repair=False)}.values())
    ok = [r for r in recs if r["status"] == "completed" and r.get("fitness") is not None]

    def f(v, fmt="{:.2f}"):
        return "—" if v is None else fmt.format(v)

    L = [f"# Search progress ({out.name})", "",
         f"{len(recs)} evaluations in {len(gens)} generations, {len(ok)} completed; "
         f"training time {sum(r.get('total_s', 0) for r in recs) / 3600:.1f} h", "",
         "| gen | epochs | trained | best | elite mean | median | elite bar | epochs trained / full | "
         "completed / pruned / failed | inherit ρ crossover / mutation | gene entropy | killed by age |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in gens:
        st = g["status"]
        n_pruned = sum(v for k, v in st.items() if k.startswith("pruned"))
        n_fail = sum(v for k, v in st.items() if k != "completed" and not k.startswith("pruned"))
        inh = " / ".join(f((g.get(f"inherit_{o}") or {}).get("rho_mean")) for o in ("crossover", "mutation"))
        trained = f"{g['trained']}" + (f" ({g['reevaluated']} again)" if g.get("reevaluated") else "")
        L.append(f"| {g['generation']} | {g.get('epochs', '—')} | {trained} | {f(g['best_fitness'])} | {f(g['elite_mean_fitness'])} | "
                 f"{f(g['median_fitness'])} | {f(g['elite_bar'])} | {g['epochs_trained']} / {g['epochs_full']} | "
                 f"{st.get('completed', 0)} / {n_pruned} / {n_fail} | {inh} | {f(g.get('gene_entropy_mean'))} | "
                 f"{len(g['killed_by_age'])} |")
    later = [r for r in recs if r["generation"] > 0 and r["origin"] != "reeval"]
    reasons = Counter(r["status"].split(" at epoch")[0].split(" (")[0] for r in later)
    L += ["", f"Children ({len(later)}): " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()),
          "", "## Best 10 so far per epoch count (fitness on val)"]
    for e in sorted({r.get("epochs") for r in ok}, key=str):
        L += ["", f"{e} epochs:", "", "| fitness | val F1 | params | MACs | gen | origin | architecture |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in ok if r.get("epochs") == e), key=lambda r: -r["fitness"])[:10]:
            L.append(f"| {r['fitness']:.2f} | {100 * r['val']['macro_f1']:.2f} | {r['params'] / 1e6:.2f}M | "
                     f"{r['macs'] / 1e9:.3f}G | {r['generation']} | {r['origin']} | {r['arch']} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
