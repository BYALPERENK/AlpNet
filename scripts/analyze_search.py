"""Analyses of a GA search, val scores only: test scores stay sealed until the search ends.
Reads the files while the search writes them (a line being written is left out). Runs on the CPU.

1. progress per generation (best / elite mean / median fitness, gene entropy, the stop rules' work);
2. the best networks and the val F1 / MACs Pareto front;
3. crossover vs mutation: completion, reaching the elite bar, child vs parents;
4. gene drift: gene shares in the population per generation vs the random generation 0;
5. gene effects, three ways, each with its own bias:
   - generation 0 alone (random, no selection; ~100 networks: directions only);
   - all completed networks of one epoch count (many more, but the GA's children are not a random sample:
     genes ride along with good parents, and later generations are better for every reason);
   - mutation pairs: child vs its parent, a near-controlled experiment (only the changed genes differ);
     the training noise of both and the regression to the mean of the selected parent are in it;
6. per-class val F1 of the best networks, the stage motifs and stage layouts of the best.
Writes analysis.md and the charts into --out (default <search>/analysis/). The Pareto chart also draws the
front of the other epoch count, if there is one.

    python scripts/analyze_search.py runs/main_20261024/search [--epochs 20] [--top 30] [--out DIR]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from alpnet.search.ga import gene_stats, read_jsonl  # noqa: E402
from analyze_gen0 import LAMBDAS, MIN_GROUP, N_BOOT, boot_ci, features, loo_error, ridge  # noqa: E402

KERNELS = (1, 3, 5, 7)
DRIFT_GENES = ("body", "conn", "kernel", "norm", "act", "downsample", "padding", "order", "se", "expansion",
               "head.pool", "head.n_fc", "stem.kernel", "depth")
# chart tokens (dataviz reference palette, light)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def feats(r: dict) -> dict[str, float]:
    """analyze_gen0's features and the kernels (share of the active stages with each kernel, mean, stem)."""
    st = r["canonical"]["stages"]
    f = features(r)
    f |= {f"kernel {k} share": sum(s["kernel"] == k for s in st) / len(st) for k in KERNELS}
    f |= {"mean kernel": float(np.mean([s["kernel"] for s in st])), "stem kernel": r["canonical"]["stem"]["kernel"]}
    return f


def matrix(recs):
    fs = [feats(r) for r in recs]
    names = list(fs[0])
    return names, np.array([[f[c] for c in names] for f in fs], dtype=float)


def usable(x, names, n_min=MIN_GROUP):
    """Columns that vary; a binary one needs n_min networks on each side."""
    keep = []
    for j, c in enumerate(names):
        v = x[:, j]
        if np.std(v) == 0:
            continue
        if set(np.unique(v)) <= {0.0, 1.0} and not n_min <= v.sum() <= len(v) - n_min:
            continue
        keep.append(j)
    return keep


def is_binary(v):
    return set(np.unique(v)) <= {0.0, 1.0}


def one_at_a_time(x, y, names):
    """name -> (networks with it or None, effect, lo, hi): binary = y with − without, numeric = Spearman."""
    out, n = {}, len(y)
    for j in usable(x, names):
        v = x[:, j]
        if is_binary(v):
            def stat(i, v=v):
                a, b = y[i][v[i] == 1], y[i][v[i] == 0]
                return a.mean() - b.mean() if len(a) and len(b) else np.nan
            out[names[j]] = (int(v.sum()), y[v == 1].mean() - y[v == 0].mean(), *boot_ci(stat, n))
        else:
            out[names[j]] = (None, spearmanr(v, y)[0], *boot_ci(lambda i, v=v: spearmanr(v[i], y[i])[0], n))
    return out


def ridge_fit(x, y, names):
    """Ridge with the penalty of the lowest leave-one-out error: name -> (coef, lo, hi), λ, LOO R²."""
    keep = usable(x, names)
    x, names = x[:, keep], [names[j] for j in keep]
    errs = {lam: loo_error(x, y, lam) for lam in LAMBDAS}
    lam = min(errs, key=errs.get)
    beta = ridge(x, y, lam)[2]
    rng = np.random.default_rng(0)
    bs = [ridge(x[i], y[i], lam)[2] for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    lo, hi = np.quantile(bs, (0.025, 0.975), axis=0)
    return {c: (beta[j], lo[j], hi[j]) for j, c in enumerate(names)}, lam, 1 - errs[lam] / np.var(y)


def mark(lo, hi):
    return " *" if lo > 0 or hi < 0 else ""


def ci(t, fmt="{:+.2f}"):
    return "—" if t is None else f"{fmt.format(t[0])} ({fmt.format(t[1])} … {fmt.format(t[2])}){mark(t[1], t[2])}"


def f1(r):
    return 100 * r["val"]["macro_f1"]


def reason(r):
    return r["status"].split(" at epoch")[0].split(" (")[0]


def pareto(recs):
    """Val F1 up, MACs down: sorted by MACs, each one better than every cheaper one."""
    front, best = [], -1.0
    for r in sorted(recs, key=lambda r: (r["macs"], -f1(r))):
        if f1(r) > best:
            front.append(r)
            best = f1(r)
    return front


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


def gen_color(g, last):
    return MUTED if g == 0 else RAMP[round((min(g, last) - 1) / max(last - 1, 1) * (len(RAMP) - 1))]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("search", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None, help="default: <search>/analysis")
    args = ap.parse_args()
    out = args.out or args.search / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    E, TOP = args.epochs, args.top

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = read_jsonl(args.search / "results.jsonl", repair=False)
    gens = list({g["generation"]: g for g in read_jsonl(args.search / "generations.jsonl", repair=False)}.values())
    state = json.loads((args.search / "ga_state.json").read_text(encoding="utf-8"))
    at = [r for r in recs if r["epochs"] == E]
    ok = [r for r in at if r["status"] == "completed" and r.get("fitness") is not None]
    g0 = [r for r in ok if r["generation"] == 0]
    kids = [r for r in at if r["generation"] > 0 and r["origin"] != "reeval"]
    by = {r["hash"]: r for r in ok}
    last = max(g["generation"] for g in gens)
    bar = {g["generation"]: g["elite_bar"] for g in gens}

    L = [f"# Search analysis ({args.search}, {E} epochs, val only)", "",
         f"{len(recs)} records; at {E} epochs {len(at)}, completed {len(ok)} (generation 0: {len(g0)}, children: "
         f"{len(ok) - len(g0)}); finished generations 0–{last}"
         + (", the next one is still running. " if any(r["generation"] > last for r in recs) else ". ") +
         "Fitness = 100·val macro-F1 − 0.3·log2(params / 1M) − 0.3·log2(MACs / 0.1G). Test scores are not read.",
         "", "**Noise:** the training is not bit-deterministic and there is no repeated training at this epoch "
         "count, so the noise of one score is not measured here (one network differed by 2.5 points between two "
         "3-epoch runs). Differences below about 1 point are not a ranking.", ""]

    # 1. progress
    L += ["## 1. Progress per generation", "",
          "| gen | best | elite mean | median | elite bar | completed / pruned / failed | epochs trained / full | "
          "hours | gene entropy |", "|---|---|---|---|---|---|---|---|---|"]
    for g in gens:
        st = g["status"]
        pr = sum(v for k, v in st.items() if k.startswith("pruned"))
        fail = sum(v for k, v in st.items() if k != "completed" and not k.startswith("pruned"))
        fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        L.append(f"| {g['generation']} | {fmt(g['best_fitness'])} | {fmt(g['elite_mean_fitness'])} | "
                 f"{fmt(g['median_fitness'])} | {fmt(g['elite_bar'])} | {st.get('completed', 0)} / {pr} / {fail} | "
                 f"{g['epochs_trained']} / {g['epochs_full']} | {g['train_hours']:.2f} | "
                 f"{fmt(g.get('gene_entropy_mean'))} |")
    rules = Counter(reason(r) for r in kids if r["status"] != "completed")
    saved = sum((r["epochs"] - r["epochs_done"]) * r["train_s"] / max(r["epochs_done"], 1)
                for r in kids if r["status"].startswith("pruned"))
    spent = sum(r.get("total_s", 0) for r in kids)
    L += ["", f"Children not completed: " + ", ".join(f"{k} {v}" for k, v in rules.most_common())
          + f". Time saved by the stop rules ≈ {saved / 3600:.1f} h of {(spent + saved) / 3600:.1f} h "
          f"({100 * saved / (spent + saved):.0f}%; the rest of a cut training estimated at its own speed).", ""]

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for r in ok:
        ax.scatter(r["generation"] + rng.uniform(-0.25, 0.25), r["fitness"], s=10, color=gen_color(r["generation"], last),
                   alpha=0.55, linewidths=0)
    for key, col, lab in (("best_fitness", BLUE, "best"), ("elite_mean_fitness", ORANGE, "elite mean"),
                          ("median_fitness", AQUA, "population median")):
        pts = [(g["generation"], g[key]) for g in gens if g[key] is not None]
        ax.plot(*zip(*pts), color=col, linewidth=2, marker="o", markersize=4, label=lab)
    ax.set_ylim(max(60, min(r["fitness"] for r in ok) - 1), None)
    top_gen = max(r["generation"] for r in ok)  # the running generation shows its finished networks
    ax.set_xticks(range(top_gen + 1))
    ax.set_xlim(-0.5, top_gen + 0.5)
    style(ax, f"Fitness per generation ({E} epochs; dots = completed networks)", "generation", "fitness")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "progress.png", dpi=150)
    plt.close(fig)

    # 2. leaderboard, Pareto front
    best = sorted(ok, key=lambda r: -r["fitness"])
    L += [f"## 2. The best networks ({E} epochs)", "", "![progress](progress.png)", "",
          "| # | fitness | val F1 | params | MACs | gen | origin | architecture |", "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(best[:20], 1):
        L.append(f"| {i} | {r['fitness']:.2f} | {f1(r):.2f} | {r['params'] / 1e6:.2f}M | {r['macs'] / 1e9:.3f}G | "
                 f"{r['generation']} | {r['origin']} | {r['arch']} |")
    gaps = np.abs(np.diff([r["fitness"] for r in best[:TOP]]))
    L += ["", f"Spread of the best 20: {best[0]['fitness'] - best[19]['fitness']:.2f} points; neighbouring gaps "
          f"among the best {TOP}: median {np.median(gaps):.2f}.", ""]
    front = pareto(ok)
    L += [f"### Val F1 / MACs Pareto front ({len(front)} networks: none is both cheaper and better)", "",
          "| MACs | params | val F1 | fitness | gen | architecture |", "|---|---|---|---|---|---|"]
    for r in front:
        L.append(f"| {r['macs'] / 1e9:.3f}G | {r['params'] / 1e6:.2f}M | {f1(r):.2f} | {r['fitness']:.2f} | "
                 f"{r['generation']} | {r['arch']} |")
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    for r in sorted(ok, key=lambda r: r["generation"]):
        ax.scatter(r["macs"] / 1e9, f1(r), s=12, color=gen_color(r["generation"], last), alpha=0.6, linewidths=0)
    other = Counter(r["epochs"] for r in recs if r["epochs"] != E and r["status"] == "completed")
    if other:  # the front of the other epoch count, for comparison
        E2 = other.most_common(1)[0][0]
        front2 = pareto([r for r in recs if r["epochs"] == E2 and r["status"] == "completed" and r.get("fitness") is not None])
        ax.step([r["macs"] / 1e9 for r in front2], [f1(r) for r in front2], where="post", color=INK2, linewidth=1.2,
                linestyle="--", label=f"Pareto front, {E2} epochs")
    ax.step([r["macs"] / 1e9 for r in front], [f1(r) for r in front], where="post", color=ORANGE, linewidth=2)
    ax.scatter([r["macs"] / 1e9 for r in front], [f1(r) for r in front], s=36, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label=f"Pareto front, {E} epochs")
    ax.scatter([], [], s=12, color=MUTED, label="generation 0")
    first_gen = min(r["generation"] for r in ok if r["generation"] > 0)
    ax.scatter([], [], s=12, color=gen_color(first_gen, last), label=f"generation {first_gen}")
    ax.scatter([], [], s=12, color=gen_color(top_gen, last), label=f"generation {top_gen}")
    ax.set_xscale("log")
    ax.set_ylim(min(np.quantile([f1(r) for r in ok], 0.05) - 2, f1(front[0]) - 1), None)
    style(ax, f"Val F1 vs MACs ({E} epochs)", "MACs (G, log scale)", "val macro-F1")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=150)
    plt.close(fig)
    L += ["", "![pareto](pareto.png)", ""]

    # 3. crossover vs mutation
    L += ["## 3. Crossover vs mutation", "",
          "| origin | children | completed | pruned | failed | median fitness (completed) | ≥ elite bar of its generation "
          "| beats its best parent | median child − parent mean |", "|---|---|---|---|---|---|---|---|---|"]
    for o in ("crossover", "mutation"):
        rs = [r for r in kids if r["origin"] == o]
        done = [r for r in rs if r["status"] == "completed"]
        pf = [(r["fitness"], [by[p]["fitness"] for p in r["parents"]]) for r in done if all(p in by for p in r["parents"])]
        reached = [r["fitness"] >= bar[r["generation"]] for r in done if bar.get(r["generation"]) is not None]
        L.append(f"| {o} | {len(rs)} | {len(done)} | {sum(r['status'].startswith('pruned') for r in rs)} | "
                 f"{sum(r['status'] != 'completed' and not r['status'].startswith('pruned') for r in rs)} | "
                 f"{np.median([r['fitness'] for r in done]):.2f} | "
                 f"{sum(reached)} ({100 * np.mean(reached):.0f}%) | "
                 f"{sum(c > max(p) for c, p in pf)} ({100 * np.mean([c > max(p) for c, p in pf]):.0f}%) | "
                 f"{np.median([c - np.mean(p) for c, p in pf]):+.2f} |")
    pop = [by[p["hash"]] for p in state["population"] if p["hash"] in by]
    L += ["", f"Origins of the current population ({len(pop)}): "
          + ", ".join(f"{k} {v}" for k, v in Counter(r["origin"] for r in pop).most_common())
          + f"; of the best 20 so far: " + ", ".join(f"{k} {v}" for k, v in Counter(r["origin"] for r in best[:20]).most_common())
          + f"; of the best {TOP}, by generation: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["generation"] for r in best[:TOP]).items())), ""]

    # 4. gene drift
    base = gene_stats([r["genotype"] for r in g0])
    topg = gene_stats([r["genotype"] for r in best[:TOP]])
    cols = [g for g in gens if g["generation"] > 0 and g.get("gene_stats")]

    def shares(stats, gene):
        d = stats.get(gene, {})
        tot = sum(d.values())
        return {k: v / tot for k, v in d.items()} if tot else {}

    L += ["## 4. Gene drift (share in the population's genotypes, active slots)", "",
          f"Generation 0 = the {len(g0)} random networks; then the population after each generation; last column the best "
          f"{TOP} so far. A share that grows is what selection keeps, but a gene can also ride along with good "
          "parents (hitchhiking): section 5 separates that.", "",
          "| gene | value | gen 0 | " + " | ".join(str(g["generation"]) for g in cols) + f" | best {TOP} |",
          "|---|---|---|" + "---|" * len(cols) + "---|"]
    heat_rows, heat = [], []
    for gene in DRIFT_GENES:
        b = shares(base, gene)
        for v in sorted(b, key=lambda v: (len(v), v)):
            row = [shares(g["gene_stats"], gene).get(v, 0) for g in cols]
            L.append(f"| {gene} | {v} | {100 * b[v]:.0f} | " + " | ".join(f"{100 * s:.0f}" for s in row)
                     + f" | {100 * shares(topg, gene).get(v, 0):.0f} |")
            heat_rows.append(f"{gene} = {v}")
            heat.append([b[v], *row, shares(topg, gene).get(v, 0)])
    heat = np.array(heat)
    fig, ax = plt.subplots(figsize=(7.5, 0.19 * len(heat_rows) + 1.2), facecolor=SURFACE)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#c94a1d", "#f0efec", "#1c5cab"])
    delta = (heat - heat[:, :1]) * 100
    lim = max(10, np.abs(delta).max())
    ax.imshow(delta, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim))
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{100 * heat[i, j]:.0f}", ha="center", va="center", fontsize=6, color=INK)
    ax.set_yticks(range(len(heat_rows)), heat_rows, fontsize=6.5, color=INK2)
    ax.set_xticks(range(delta.shape[1]), ["gen 0"] + [str(g["generation"]) for g in cols] + [f"best {TOP}"],
                  fontsize=7, color=INK2)
    ax.set_title("Gene shares (%, numbers); color = change vs generation 0 (blue up, orange down)", loc="left",
                 fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "gene_drift.png", dpi=150)
    plt.close(fig)
    L += ["", "![gene drift](gene_drift.png)", ""]

    # 5. gene effects
    names, x_all = matrix(ok)
    y_all = np.array([f1(r) for r in ok])
    m0 = np.array([r["generation"] == 0 for r in ok])
    one_all, one_0 = one_at_a_time(x_all, y_all, names), one_at_a_time(x_all[m0], y_all[m0], names)
    rid_all, lam_a, r2_a = ridge_fit(x_all, y_all, names)
    rid_0, lam_0, r2_0 = ridge_fit(x_all[m0], y_all[m0], names)

    # mutation pairs: child − parent, both completed at E
    pairs = [(r, by[r["parents"][0]]) for r in ok if r["origin"] == "mutation" and r["parents"][0] in by]
    fc = np.array([[feats(c)[n] for n in names] for c, _ in pairs])
    fp = np.array([[feats(p)[n] for n in names] for _, p in pairs])
    dx, dy = fc - fp, np.array([f1(c) - f1(p) for c, p in pairs])
    base_dy = np.median(dy)
    n_changed = (np.abs(dx) > 1e-9).sum(1)
    flips = {}
    for j, c in enumerate(names):
        nz = np.abs(dx[:, j]) > 1e-9
        if nz.sum() < MIN_GROUP:
            continue
        if is_binary(x_all[:, j]):
            s = np.sign(dx[nz, j]) * (dy[nz] - base_dy)
            flips[c] = (int((dx[nz, j] > 0).sum()), int((dx[nz, j] < 0).sum()), s.mean(),
                        *boot_ci(lambda i, s=s: s[i].mean(), len(s)))
        else:
            a, b = dx[nz, j], dy[nz]
            flips[c] = (int(nz.sum()), None, spearmanr(a, b)[0], *boot_ci(lambda i, a=a, b=b: spearmanr(a[i], b[i])[0], len(a)))
    keep = [j for j in range(len(names)) if (np.abs(dx[:, j]) > 1e-9).sum() >= MIN_GROUP]
    rid_d, lam_d, r2_d = ridge_fit(dx[:, keep], dy, [names[j] for j in keep]) if len(pairs) > 20 else ({}, None, None)

    L += ["## 5. Gene effects (target: val macro-F1, points)", "",
          "Features of the network actually built (canonical form), as in analyze_gen0: \"body convnext\" = some stage "
          "has it, \"kernel 1 share\" = the share of the active stages with kernel 1. * = the 95% bootstrap interval "
          "excludes 0; many intervals at once, so a few * are expected by chance. None of it is causal.", "",
          f"- **gen 0**: the {m0.sum()} random networks, no selection, but few;",
          f"- **all**: the {len(ok)} completed networks at {E} epochs; most are children of elites, so a gene can look "
          "good because it sits in good networks (hitchhiking) or because later generations are better overall;",
          f"- **mutation pairs**: {len(pairs)} children vs their parent (median {np.median(n_changed):.0f} features "
          f"changed per pair). Every pair's change minus the median change of all pairs ({base_dy:+.2f}: the "
          "regression to the mean of a selected parent), signed by the direction of the flip: + = having it is better. "
          "Numeric: Spearman of the feature's change with the F1 change.", "",
          "### One feature at a time", "",
          "| feature (binary: any stage) | networks with it (all) | all: with − without | gen 0: with − without | "
          "mutation pairs: flips on / off | mutation pairs: effect |", "|---|---|---|---|---|---|"]
    for c in sorted((c for c in one_all if one_all[c][0] is not None), key=lambda c: -abs(one_all[c][1])):
        fl = flips.get(c)
        L.append(f"| {c} | {one_all[c][0]} | {ci(one_all[c][1:])} | {ci(one_0[c][1:]) if c in one_0 else '—'} | "
                 f"{f'{fl[0]} / {fl[1]}' if fl else '—'} | {ci(fl[2:]) if fl else '—'} |")
    L += ["", "| numeric feature | all: Spearman | gen 0: Spearman | mutation pairs: changed | mutation pairs: Spearman of "
          "changes |", "|---|---|---|---|---|"]
    for c in sorted((c for c in one_all if one_all[c][0] is None), key=lambda c: -abs(one_all[c][1])):
        fl = flips.get(c)
        L.append(f"| {c} | {ci(one_all[c][1:])} | {ci(one_0[c][1:]) if c in one_0 else '—'} | "
                 f"{fl[0] if fl else '—'} | {ci(fl[2:]) if fl else '—'} |")
    L += ["", f"### All together: ridge regression (log2 params and MACs among the features, so a gene's effect is "
          "beyond the size it brings)", "",
          f"λ by leave-one-out error: all λ = {lam_a} (LOO R² {r2_a:.2f}), gen 0 λ = {lam_0} (LOO R² {r2_0:.2f})"
          + (f", mutation pairs (on the changes) λ = {lam_d} (LOO R² {r2_d:.2f})" if lam_d else "")
          + ". Coefficient = points per one standard deviation of the feature.", "",
          "| feature | all | gen 0 | mutation pairs |", "|---|---|---|---|"]
    for c in sorted(rid_all, key=lambda c: -abs(rid_all[c][0])):
        L.append(f"| {c} | {ci(rid_all[c])} | {ci(rid_0.get(c))} | {ci(rid_d.get(c))} |")

    show = [c for c in sorted(rid_all, key=lambda c: -abs(rid_all[c][0]))][:24][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(show) + 1.2), facecolor=SURFACE)
    for k, (src, col, lab, off) in enumerate(((rid_all, BLUE, f"all ({len(ok)})", 0.17),
                                               (rid_0, MUTED, f"gen 0 ({m0.sum()})", -0.17))):
        ys = [i + off for i, c in enumerate(show) if c in src]
        vals = [src[c] for c in show if c in src]
        ax.errorbar([v[0] for v in vals], ys, xerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="o", color=col, markersize=4, elinewidth=1.2, capsize=0, label=lab)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_yticks(range(len(show)), show, fontsize=7.5, color=INK2)
    style(ax, "Ridge coefficients (val F1 points per SD, 95% interval)", "coefficient", "")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "gene_effects.png", dpi=150)
    plt.close(fig)
    L += ["", "![gene effects](gene_effects.png)", ""]

    # 6. per class, motifs, layouts
    classes = list(ok[0]["val"]["class_f1"])
    t10 = best[:10]
    L += [f"## 6. The best networks in detail", "", "### Per-class val F1 (the best 10 by fitness)", "",
          "| class | gen 0 median | all median | best 10 mean | best 10 min | best network |", "|---|---|---|---|---|---|"]
    rows = []
    for c in classes:
        v = [100 * r["val"]["class_f1"][c] for r in t10]
        rows.append((c, np.median([100 * r["val"]["class_f1"][c] for r in g0]),
                     np.median([100 * r["val"]["class_f1"][c] for r in ok]), np.mean(v), min(v), v[0]))
    for row in sorted(rows, key=lambda t: t[3]):
        L.append(f"| {row[0]} | " + " | ".join(f"{v:.1f}" for v in row[1:]) + " |")

    def motifs(rs):
        return Counter(f"{s['body']}/{s['conn']} k{s['kernel']} {s['norm']}" for r in rs for s in r["canonical"]["stages"])

    def layout(r):
        return " ".join(f"s{s['slot']}" for s in r["canonical"]["stages"])

    mt, ma = motifs(best[:TOP]), motifs(ok)
    nt, na = sum(mt.values()), sum(ma.values())
    L += ["", f"### Stage motifs (body/connection, kernel, norm) in the best {TOP} vs all completed", "",
          f"| motif | best {TOP} stages | share | all share | ratio |", "|---|---|---|---|---|"]
    for m, v in mt.most_common(15):
        L.append(f"| {m} | {v} | {100 * v / nt:.0f}% | {100 * ma[m] / na:.1f}% | {(v / nt) / (ma[m] / na):.1f}× |")
    lt, la = Counter(layout(r) for r in best[:TOP]), Counter(layout(r) for r in ok)
    L += ["", f"### Stage layouts (active slots) in the best {TOP}", "", f"| layout | best {TOP} | all |", "|---|---|---|"]
    for m, v in lt.most_common(8):
        L.append(f"| {m} | {v} | {la[m]} |")
    acts = Counter(s["act"] for r in best[:TOP] for s in r["canonical"]["stages"])
    L += ["", f"Activations in the best {TOP} (stages): " + ", ".join(f"{k} {v}" for k, v in acts.most_common())
          + "; kernels: " + ", ".join(f"k{k} {v}" for k, v in sorted(Counter(
              s["kernel"] for r in best[:TOP] for s in r["canonical"]["stages"]).items())), ""]

    (out / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
