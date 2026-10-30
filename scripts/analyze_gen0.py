"""Generation 0 analyses, val scores only: generation 0 is the one set without the
GA's selection, so it is where gene effects can be estimated (still conditioned on the budget, the
repairs and on completing training: only completed networks are analysed).

1. Medical val check: medical val comes from the same slides as train (optimistic). The fitness ranking
   is compared with the one computed without the 4 medical classes (11 classes): Spearman with a
   bootstrap 95% interval and the overlap of the best K, over all networks and among the best TOP (where
   the GA chooses; overall it is near 1 almost by construction). High: the fitness stays as it is.
2. Gene importance on aggregated features of the network actually built (canonical form: stages, depth,
   bodies, connections, norms, activations, ...), log2 params and log2 MACs among them so that "bigger
   networks won" is not credited to the genes:
   - one feature at a time: val F1 difference with / without (binary) or Spearman (numeric), bootstrap
     95% intervals;
   - all together: ridge regression of val F1 on the standardized features (penalty chosen by
     leave-one-out error), bootstrap 95% intervals of the coefficients.
   About 100 networks and ~40 correlated features: directions, not proof.
Writes gen0_analysis.md into the run folder.

    python scripts/analyze_gen0.py runs/random_20ep_X [K]
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.models.blocks import ACTS, BODIES, DOWNSAMPLES, HEAD_POOLS  # noqa: E402
from alpnet.search.fitness import fitness  # noqa: E402
from alpnet.search.ga import GAConfig, read_jsonl  # noqa: E402

MEDICAL = ("ADI", "LYM", "NORM", "TUM")
TOP = 30  # the best by fitness for the medical check (PROTECT of the calibration)
N_BOOT = 2000
MIN_GROUP = 5  # a binary feature needs this many networks with and without it
LAMBDAS = (1, 3, 10, 30, 100, 300)


def features(r: dict) -> dict[str, float]:
    """Aggregated features of the network built (canonical form) and its cost."""
    st, head = r["canonical"]["stages"], r["canonical"]["head"]
    f = {"stages": len(st), "total depth": sum(s["depth"] for s in st),
         "log2 max width": np.log2(max(s["width"] for s in st)), "log2 first slot": np.log2(st[0]["slot"]),
         "stem width": r["canonical"]["stem"]["width"], "head FC layers": head["n_fc"],
         "log2 params": np.log2(r["params"]), "log2 MACs": np.log2(r["macs"])}
    f |= {f"body {b}": float(any(s["body"] == b for s in st)) for b in BODIES}
    f |= {f"conn {c}": float(any(s["conn"] == c for s in st)) for c in ("add", "concat")}
    f |= {"norm bn": float(any(s["norm"] == "bn" for s in st)), "norm ln": float(any(s["norm"] == "ln" for s in st)),
          "no norm anywhere": float(all(s["norm"] == "none" for s in st))}
    f |= {f"act {a}": float(any(s["act"] == a for s in st)) for a in ACTS}
    f |= {"SE": float(any(s["se"] for s in st)), "valid padding": float(any(s.get("padding") == "valid" for s in st)),
          "pre-activation": float(any(s.get("order") == "pre" for s in st))}
    f |= {f"downsample {d}": float(any(s.get("downsample", {}).get("method") == d for s in st)) for d in DOWNSAMPLES}
    f |= {f"head {p}": float(head["pool"] == p) for p in HEAD_POOLS}
    return f


def boot_ci(stat, n: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return tuple(np.nanquantile(vals, (0.025, 0.975)))


def ridge(x: np.ndarray, y: np.ndarray, lam: float):
    """Ridge on the standardized features with a free (unpenalized) intercept: (mean, std, coef, intercept);
    coef = points of y per one standard deviation of the feature."""
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mu) / sd
    return mu, sd, np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y.mean())), y.mean()


def loo_error(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Leave-one-out mean squared error: the standardization, the intercept and the coefficients are fitted
    again without each network."""
    err = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        mu, sd, beta, b0 = ridge(x[m], y[m], lam)
        err.append((y[i] - b0 - ((x[i] - mu) / sd) @ beta) ** 2)
    return float(np.mean(err))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else GAConfig.k
    ok = [r for r in read_jsonl(src / "results.jsonl", repair=False) if r["status"] == "completed"]
    n = len(ok)
    fit = np.array([r["fitness"] for r in ok])
    f1 = np.array([100 * r["val"]["macro_f1"] for r in ok])
    nm = [100 * np.mean([v for c, v in r["val"]["class_f1"].items() if c not in MEDICAL]) for r in ok]
    fit_nm = np.array([fitness(v / 100, r["params"], r["macs"]) for v, r in zip(nm, ok)])
    med = np.array([100 * r["val"]["group_f1"]["medical"] for r in ok])

    rho = spearmanr(fit, fit_nm)[0]
    lo, hi = boot_ci(lambda i: spearmanr(fit[i], fit_nm[i])[0], n)
    top, top_nm = set(np.argsort(-fit)[:k]), set(np.argsort(-fit_nm)[:k])
    t = min(TOP, n)  # among the best (where the GA chooses: medical saturates, the differences are small)
    best = np.argsort(-fit)[:t]
    rho_t = spearmanr(fit[best], fit_nm[best])[0]
    lo_t, hi_t = boot_ci(lambda i: spearmanr(fit[best][i], fit_nm[best][i])[0], t)
    top_t, top_t_nm = set(best), set(np.argsort(-fit_nm)[:t])
    gap = np.abs(np.diff(np.sort(fit[best])))  # neighbouring fitness gaps among the best
    L = [f"# Generation 0 analyses ({src}, {n} completed networks, val only)", "",
         "## 1. Medical val check", "",
         f"- Fitness ranking vs fitness without the 4 medical classes (11 classes): Spearman **{rho:.3f}** "
         f"(95% {lo:.3f}–{hi:.3f}); best {k} shared: **{len(top & top_nm)} / {k}**",
         f"- Among the best {t} by fitness: Spearman **{rho_t:.3f}** (95% {lo_t:.3f}–{hi_t:.3f}); best {t} shared "
         f"{len(top_t & top_t_nm)} / {t}; their neighbouring fitness gaps median {np.median(gap):.2f}, "
         f"medical val F1 spread among them {np.ptp(med[best]):.1f} points",
         f"- Medical val F1 10 / 50 / 90%: " + " / ".join(f"{v:.1f}" for v in np.quantile(med, (0.1, 0.5, 0.9)))
         + "; Spearman of medical val F1 with the 11-class fitness: " + f"{spearmanr(med, fit_nm)[0]:.2f}",
         "- Decision rule: a high Spearman keeps the fitness as it is; a low one is discussed.",
         "- Limit: 11 of the 15 classes are shared and medical val saturates, so the overall Spearman is close to 1 "
         "almost by construction; the best-of rows are the informative ones. Memorizing the slides (medical val "
         "comes from the train slides) shows only on test, which this check cannot see."]

    feats = [features(r) for r in ok]
    names = list(feats[0])
    x = np.array([[f[c] for c in names] for f in feats], dtype=float)
    keep = [j for j, c in enumerate(names) if np.std(x[:, j]) > 0
            and (set(np.unique(x[:, j])) - {0.0, 1.0} or MIN_GROUP <= x[:, j].sum() <= n - MIN_GROUP)]
    names, x = [names[j] for j in keep], x[:, keep]
    binary = [set(np.unique(x[:, j])) <= {0.0, 1.0} for j in range(len(names))]

    L += ["", "## 2. Gene importance (target: val macro-F1, points)", "",
          f"Features with fewer than {MIN_GROUP} networks on one side are left out. * = the 95% interval "
          "excludes 0.", "", "### One feature at a time", ""]
    with_it, numeric = [], []
    for j, c in enumerate(names):
        v = x[:, j]
        if binary[j]:
            def stat(i, v=v):
                a, b = f1[i][v[i] == 1], f1[i][v[i] == 0]
                return a.mean() - b.mean() if len(a) and len(b) else np.nan
            with_it.append((c, int(v.sum()), f1[v == 1].mean() - f1[v == 0].mean(), *boot_ci(stat, n)))
        else:
            def stat(i, v=v):
                return spearmanr(v[i], f1[i])[0]
            numeric.append((c, spearmanr(v, f1)[0], *boot_ci(stat, n)))

    def mark(lo, hi):
        return " *" if lo > 0 or hi < 0 else ""

    L += ["| feature (any stage has it) | networks with it | val F1 with − without (95% CI) |", "|---|---|---|"]
    for c, m, eff, lo, hi in sorted(with_it, key=lambda t: -abs(t[2])):
        L.append(f"| {c} | {m} | {eff:+.2f} ({lo:+.2f} … {hi:+.2f}){mark(lo, hi)} |")
    L += ["", "| numeric feature | Spearman with val F1 (95% CI) |", "|---|---|"]
    for c, eff, lo, hi in sorted(numeric, key=lambda t: -abs(t[1])):
        L.append(f"| {c} | {eff:+.2f} ({lo:+.2f} … {hi:+.2f}){mark(lo, hi)} |")

    errs = {lam: loo_error(x, f1, lam) for lam in LAMBDAS}
    lam = min(errs, key=errs.get)
    beta = ridge(x, f1, lam)[2]
    rng = np.random.default_rng(0)
    bs = [ridge(x[i], f1[i], lam)[2] for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    lo_b, hi_b = np.quantile(bs, (0.025, 0.975), axis=0)
    r2 = 1 - errs[lam] / np.var(f1)
    L += ["", f"### All together: ridge regression (λ = {lam}, the lowest leave-one-out error; leave-one-out R² "
              f"{r2:.2f}, optimistic since λ was chosen on it)", "",
          "Coefficient = val F1 points per one standard deviation of the feature, the others held fixed. With "
          "log2 params / MACs among the features this is a gene's effect beyond the size it brings (a direct "
          "effect, not the total one), and none of it is causal. Many intervals at once: a few * are expected "
          "by chance.", "",
          "| feature | coefficient (95% CI) |", "|---|---|"]
    for j in np.argsort(-np.abs(beta)):
        L.append(f"| {names[j]} | {beta[j]:+.2f} ({lo_b[j]:+.2f} … {hi_b[j]:+.2f}){mark(lo_b[j], hi_b[j])} |")
    (src / "gen0_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
