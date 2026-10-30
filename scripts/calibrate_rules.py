"""Calibrate the stop rules on fully trained candidates (generation 0).

Every completed candidate is replayed as if it were a child: its history is the other candidates, B's
elite bar is the K-th best fitness among them. A cut is wrong when the candidate ends among the best
PROTECT = 1.5·K of the others: the elite and a margin below it, the one safety margin. For D, A and B a
grid of settings is compared (epochs and time saved vs wrong cuts); the chosen setting of each is the one
that saves the most with no wrong cut (on a tie the tighter one). The protected candidates the settings
cut are listed ("vetoes"): a single slow starter can switch a rule off, and that is decided knowingly.
The combination (with the fixed not-learning and patience rules) has no wrong cut beyond not learning's
own, in the fixed rules table: a wrong cut of the combination is a wrong cut of the rule that fired. C
(successive halving at epochs 10 and 20) is simulated on the whole set. Val scores only. Writes
calibration.md next to results.jsonl.

    python scripts/calibrate_rules.py runs/gen0_20ep [--config configs/search.yaml] [--k K]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpnet.search.fitness import cost_penalty  # noqa: E402
from alpnet.search.ga import read_jsonl  # noqa: E402
from alpnet.train.stopping import NotLearning, Patience, floor, percentile, reach_elite, replay  # noqa: E402

K = 20  # the elite count (from the config)


def protect() -> int:
    return round(1.5 * K)


def scaled(e: int, epochs: int) -> int:
    """A check epoch of the 40-epoch settings at the same point of a shorter recipe."""
    return max(1, round(e * epochs / 40))


def settings(epochs: int) -> dict[str, list[tuple[str, object]]]:
    """Per rule: (name, setting), loose to tight within each family. A: checks at 20 only, at 10 / 20, at
    5 / 10 / 20 (later checks spare slow starters); B: check sets (30) ... (5, 10, 20, 30), q from 1.0 down
    (in 40-epoch terms, scaled to the recipe)."""
    def s(e):
        return scaled(e, epochs)

    d = [(f"epoch {s(10)} < {v:.2f}", {s(10): v}) for v in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)]
    a = []
    for checks, qs in (((20,), ((0.1,), (0.2,), (0.3,), (0.5,))),
                       ((10, 20), ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.4, 0.6))),
                       ((5, 10, 20), ((0.05, 0.1, 0.15), (0.1, 0.15, 0.25), (0.2, 0.3, 0.5), (0.3, 0.4, 0.6),
                                      (0.4, 0.5, 0.7)))):
        for q in qs:
            a.append((" / ".join(f"{x:.0%}" for x in q) + " at " + " / ".join(str(s(e)) for e in checks),
                      {s(e): x for e, x in zip(checks, q)}))
    b = [(f"q {q} at {'/'.join(str(s(e)) for e in ch)}", {"check": [s(e) for e in ch], "q": q})
         for ch in ((30,), (20, 30), (10, 20, 30), (5, 10, 20, 30)) for q in (1.0, 0.95, 0.9, 0.8)]
    return {"D floor": d, "A percentile": a, "B reach_elite": b}


def pick(results: list[tuple[str, int, float]]) -> str | None:
    """results: (name, wrong cuts, seconds saved) in the settings' order. The setting with no wrong cut
    that saves the most; on a tie the later (tighter) one; None when every setting cuts wrongly."""
    best = None
    for name, wrong, saved in results:
        if wrong == 0 and (best is None or saved >= best[1]):
            best = (name, saved)
    return best and best[0]


def patience(epochs: int) -> Patience:
    return Patience(patience=max(2, scaled(5, epochs)), start=scaled(30, epochs))


def build(kind, setting, others, bar, cand, epochs):
    curves = [o["curve"] for o in others]
    if kind == "D floor":
        return floor(setting)
    if kind == "A percentile":
        return percentile(curves, setting, epochs)
    if kind == "B reach_elite":
        return reach_elite(curves, bar, cost_penalty(cand["params"], cand["macs"]), tuple(setting["check"]),
                           setting["q"], epochs)
    return {"not learning": NotLearning(), "patience": patience(epochs)}[kind]


def bars(recs, i) -> tuple[float, float]:
    """B's elite bar (K-th best of the others) and the protected bar (PROTECT-th best)."""
    others = sorted((o["fitness"] for j, o in enumerate(recs) if j != i), reverse=True)
    return others[K - 1], others[protect() - 1]


def simulate(recs, epochs, kind, setting):
    """Per candidate: (stop epoch, prune?, wrong?, seconds saved)."""
    out = []
    for i, c in enumerate(recs):
        others = recs[:i] + recs[i + 1:]
        bar, keep = bars(recs, i)
        e, stop = replay([build(kind, setting, others, bar, c, epochs)], c["rows"])
        saved = (len(c["rows"]) - e) * c["epoch_s"] if stop else 0.0
        out.append((e, bool(stop and stop.prune), bool(stop and stop.prune and c["fitness"] >= keep), saved))
    return out


def summary(sim, recs) -> str:
    cut = [s for s in sim if s[1]]
    stopped = [s for s in sim if s[3] > 0]
    total = sum(c["epoch_s"] * len(c["rows"]) for c in recs)
    lost = max((recs[i]["fitness"] for i, s in enumerate(sim) if s[1]), default=float("nan"))
    return (f"{len(cut)} | {len(stopped)} | {100 * sum(len(recs[i]['rows']) - s[0] for i, s in enumerate(sim) if s[3] > 0) / sum(len(c['rows']) for c in recs):.1f}% "
            f"| {100 * sum(s[3] for s in sim) / total:.1f}% | {sum(s[2] for s in sim)} | {lost:.1f}")


def main():
    global K
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="a train_random.py run (generation 0 with one epoch count)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "search.yaml", help="K from here")
    ap.add_argument("--k", type=int, default=None, help="another elite count (small test sets)")
    args = ap.parse_args()
    src = args.src
    K = args.k or yaml.safe_load(args.config.read_text(encoding="utf-8"))["k"]
    recs = [r for r in read_jsonl(src / "results.jsonl", repair=False) if r["status"] == "completed"]
    for r in recs:
        r["rows"] = [json.loads(line) for line in open(src / f"{r['idx']:03d}_{r['hash']}" / "metrics.jsonl")]
        r["curve"] = [e["val_macro_f1"] for e in r["rows"]]
        r["epoch_s"] = np.mean([e["epoch_time"] + e["val_time"] for e in r["rows"]])
    epochs = max(len(r["rows"]) for r in recs)
    fit = np.sort([r["fitness"] for r in recs])[::-1]

    head = (f"| setting | cut | stopped | epochs saved | time saved | wrong cuts (best {protect()} at the end) "
            "| best fitness cut |")
    L = [f"# Stop rule calibration on {len(recs)} fully trained candidates ({src})", "",
         f"B's elite bar (K = {K}-th best fitness of the others) ~ {fit[K - 1]:.2f}; protected: the best "
         f"{protect()} (bar ~ {fit[protect() - 1]:.2f}, {fit[K - 1] - fit[protect() - 1]:.2f} points below the "
         "elite bar); fitness 10 / 50 / 90%: " + " / ".join(f"{v:.1f}" for v in np.quantile(fit, (0.1, 0.5, 0.9))),
         "", "Chosen setting per rule: the one saving the most with no wrong cut (ties: the tighter).", "",
         "## Fixed rules", "", head, "|---|---|---|---|---|---|---|"]
    for kind in ("not learning", "patience"):
        L.append(f"| {kind} | " + summary(simulate(recs, epochs, kind, None), recs) + " |")
    chosen, vetoes = {}, {}  # vetoes: candidate index -> {kind: [settings that cut it]}
    SETTINGS = settings(epochs)
    for kind, options in SETTINGS.items():
        L += ["", f"## {kind}", "", head, "|---|---|---|---|---|---|---|"]
        results, by_name = [], {}
        for name, setting in options:
            sim = simulate(recs, epochs, kind, setting)
            L.append(f"| {name} | " + summary(sim, recs) + " |")
            by_name[name] = setting
            results.append((name, sum(s[2] for s in sim), sum(s[3] for s in sim)))
            for i, s in enumerate(sim):
                if s[2]:
                    vetoes.setdefault(i, {}).setdefault(kind, []).append(name)
        name = pick(results)
        chosen[kind] = name and (name, by_name[name])
        L.append(f"\nChosen: **{name or 'none'}**" + ("" if name else " (every setting cuts a protected candidate)"))

    rank = {h: i + 1 for i, h in enumerate(r["hash"] for r in sorted(recs, key=lambda r: -r["fitness"]))}
    marks = [scaled(e, epochs) for e in (5, 10, 20)]
    L += ["", "## Vetoes: protected candidates the settings cut", "",
          "A rule whose every setting cuts one of these is off; decide knowingly whether that network is worth it.",
          "", f"| network | fitness (rank) | best val F1 by epoch {' / '.join(map(str, marks))} → end | cut by | "
          "architecture |", "|---|---|---|---|---|"]
    for i in sorted(vetoes, key=lambda i: -recs[i]["fitness"]):
        r = recs[i]
        f1 = " / ".join(f"{100 * max(r['curve'][:e]):.1f}" for e in marks) + f" → {100 * max(r['curve']):.1f}"
        by = "; ".join(f"{k.split()[0]} {len(v)} of {len(SETTINGS[k])} (loosest: {v[0]})" for k, v in vetoes[i].items())
        L.append(f"| {r['idx']:03d}_{r['hash']} | {r['fitness']:.2f} ({rank[r['hash']]}) | {f1} | {by} | {r.get('arch', '')} |")
    if not vetoes:
        L.append("| — | | | | |")

    # all chosen rules together, in the GA's order
    def combined(i, c):
        others = recs[:i] + recs[i + 1:]
        bar, keep = bars(recs, i)
        rules = [NotLearning()] + [build(k, chosen[k][1], others, bar, c, epochs) for k in SETTINGS if chosen[k]] \
            + [patience(epochs)]
        e, stop = replay(rules, c["rows"])
        return e, stop, keep

    res = [combined(i, c) for i, c in enumerate(recs)]
    saved = sum((len(c["rows"]) - e) * c["epoch_s"] for c, (e, stop, _) in zip(recs, res) if stop)
    total = sum(c["epoch_s"] * len(c["rows"]) for c in recs)
    wrong = sum(bool(stop and stop.prune and c["fitness"] >= keep) for c, (e, stop, keep) in zip(recs, res))
    reasons = {}
    for _, stop, _ in res:
        if stop:
            k = stop.reason.split(" at epoch")[0]
            reasons[k] = reasons.get(k, 0) + 1
    L += ["", "## All chosen rules together", "",
          f"- Stopped {sum(s is not None for _, s, _ in res)} / {len(recs)} ({reasons}); time saved "
          f"**{100 * saved / total:.1f}%**, wrong cuts {wrong} (B saves more in the GA: its bar rises)"]

    # C: successive halving over the whole set, ranked by fitness so far (best val F1 so far, cost penalty)
    def at(r, e):
        return 100 * max(r["curve"][:e]) - cost_penalty(r["params"], r["macs"])

    e1, e2 = scaled(10, epochs), scaled(20, epochs)
    half1 = sorted(recs, key=lambda r: -at(r, e1))[:len(recs) // 2]
    half2 = sorted(half1, key=lambda r: -at(r, e2))[:len(half1) // 2]
    h1, h2 = {r["hash"] for r in half1}, {r["hash"] for r in half2}
    n_ep = {r["hash"]: epochs if r["hash"] in h2 else e2 if r["hash"] in h1 else e1 for r in recs}
    t_c = sum(r["epoch_s"] * n_ep[r["hash"]] for r in recs)
    top = {r["hash"] for r in sorted(recs, key=lambda r: -r["fitness"])[:K]}
    L += ["", f"## C: successive halving (epoch {e1} → best half → epoch {e2} → best half → {epochs}), simulated", "",
          f"- Epochs {sum(n_ep.values())} of {epochs * len(recs)}, time saved {100 * (1 - t_c / total):.1f}%; of the "
          f"{K} best by final fitness, {len(top - h2)} would not reach epoch {epochs}"]
    L += ["", "## Chosen rules (configs/search.yaml)", "", "```yaml", f"calibration: {src / 'calibration.md'}",
          "rules:", "  not_learning: {epoch: 3}"]
    yk = {"D floor": "floor", "A percentile": "percentile", "B reach_elite": "reach_elite"}
    for kind, key in yk.items():
        flow = yaml.safe_dump(chosen[kind][1], default_flow_style=True).strip() if chosen[kind] else "null"
        L.append(f"  {key}: {flow}")
    p = patience(epochs)
    L += [f"  patience: {{patience: {p.patience}, start: {p.start}}}", "```"]
    (src / "calibration.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
