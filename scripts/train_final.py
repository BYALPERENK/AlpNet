"""Retrain networks chosen from a search with several seeds, with the baselines' recipe, and compare them.

The networks are given by hash (chosen by val only, from the search's results.jsonl); each is decoded from
its genotype and trained like train_baseline.py: trainer.Recipe() (40 epochs, test every epoch, checkpoint =
best val macro-F1), seeds 0 1 2. A finished run of the same network, seed, dataset, decoder and recipe is
skipped, so the same command resumes; a summary.json of another run in its folder stops the script.
At the end summary.md in --out: the networks next to the baselines (summarize_runs.section, mean ± std over
seeds, test scores at the best val macro-F1 epoch), and each network's search scores (the one training it
was chosen by, its sealed test included) vs the retrained ones: the optimism of picking by val; networks.json:
each network's genotype and built form (decoder.decode(genotype).build() makes it again); test_f1_vs_cost.png
(and test_f1_vs_cost_selected.png with --chart).

    python scripts/train_final.py runs/main_20261024/search f26817817324fcc9 c00c957e623f0814 --baselines 20261021-1647         --out runs/final_X
"""

import argparse
import ctypes
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from alpnet.data.loader import load_splits  # noqa: E402
from alpnet.models.decoder import DECODER_VERSION, decode  # noqa: E402
from alpnet.search.ga import read_jsonl  # noqa: E402
from alpnet.train.trainer import Recipe, seed_everything, train  # noqa: E402
from analyze_test import sealed_tests  # noqa: E402
from summarize_runs import ms, section  # noqa: E402

DATASET = "alpnet15_v1"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("search", type=Path)
    ap.add_argument("hashes", nargs="+", help="networks to retrain, named alpnet_a, alpnet_b, ... in this order")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / f"final_{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--baselines", required=True, help="folder prefix of the baseline runs in runs/ (their date stamp)")
    ap.add_argument("--chart", nargs="+", metavar="NAME", help="also a chart with only these search networks "
                    "(e.g. alpnet_a alpnet_f): test_f1_vs_cost_selected.png")
    args = ap.parse_args()
    if os.name == "nt":  # keep the PC awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    recs = {r["hash"]: r for r in read_jsonl(args.search / "results.jsonl", repair=False) if r["status"] == "completed"}
    nets = []
    for i, h in enumerate(args.hashes):
        if h not in recs:
            raise SystemExit(f"{h}: no completed network with this hash in {args.search}")
        ph = decode(recs[h]["genotype"])
        if ph.hash != h:
            raise SystemExit(f"{h}: the genotype decodes to {ph.hash} (another decoder version?)")
        nets.append((f"alpnet_{chr(ord('a') + i)}", recs[h], ph))

    splits = load_splits(DATASET)
    recipe = Recipe()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, r, ph in nets:
        for seed in args.seeds:
            run_dir = args.out / f"{name}_s{seed}"
            if (run_dir / "summary.json").exists():
                s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                want = {"status": "completed", "hash": ph.hash, "seed": seed, "dataset": splits["train"].dataset,
                        "decoder_version": DECODER_VERSION, "recipe": dataclasses.asdict(recipe)}
                bad = {k: s.get(k) for k, v in want.items() if s.get(k) != v}
                if bad:
                    raise SystemExit(f"{run_dir}: summary.json is not this run ({name} = {ph.hash}, seed {seed}): "
                                     f"{bad}; use another --out")
                print(f"{run_dir.name}: done, skipped", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            log_f = open(run_dir / "train.log", "w", encoding="utf-8")

            def log(msg, _f=log_f, _tag=f"{name}/s{seed}"):
                line = f"{time.strftime('%H:%M:%S')} [{_tag}] {msg}"
                print(line, flush=True)
                _f.write(line + "\n")
                _f.flush()

            seed_everything(seed)  # before construction, so the init depends on the seed
            s = train(ph.build(), splits, recipe, seed, run_dir,
                      {"model": name, "hash": ph.hash, "decoder_version": DECODER_VERSION, "genotype": r["genotype"],
                       "canonical": ph.canonical}, log)
            log(f"FINAL {s['checkpoint']} epoch {s['best_val_epoch']} | test acc {s['test']['acc']:.4f} "
                f"f1 {s['test']['macro_f1']:.4f} auroc {s['test']['macro_auroc']:.5f} | {s['train_time_s']:.0f}s")
            log_f.close()

    by_model = defaultdict(list)
    for p in sorted((ROOT / "runs").glob(f"{args.baselines}*/summary.json")):
        s = json.loads(p.read_text())
        if "checkpoints" in s:
            by_model[s["model"]].append(s)
    if not by_model:
        raise SystemExit(f"no baseline runs runs/{args.baselines}*/summary.json")
    for name, _, _ in nets:
        by_model[name] = [json.loads((args.out / f"{name}_s{seed}" / "summary.json").read_text()) for seed in args.seeds]
    lines = [f"# Search networks retrained ({len(args.seeds)} seeds, the baselines' recipe) vs baselines", ""]
    lines += section(by_model, "macro_f1")
    lines += ["", "Networks (search: one training, seed 0, the search recipe; chosen by val):", ""]
    lines += [f"- {name}: {ph.hash}, generation {r['generation']}, search val F1 {100 * r['val']['macro_f1']:.2f}, "
              f"fitness {r['fitness']:.2f}: {r['arch']}" for name, r, ph in nets]
    lines += ["", "## Selection optimism (winner's curse)", "",
              "Search = the one training the network was chosen by (seed 0, the search recipe); its test score was "
              "sealed during the search. Retrained = mean ± std over the seeds above. Δ = retrained − search.", "",
              "| network | search val F1 | retrained val F1 | Δ val | search test F1 | retrained test F1 | Δ test |",
              "|---|---|---|---|---|---|---|"]
    tests = sealed_tests(args.search, [r for _, r, _ in nets])
    for name, r, _ in nets:
        sv, st = 100 * r["val"]["macro_f1"], 100 * tests[r["run_dir"]]["macro_f1"]
        runs = by_model[name]
        rv = [100 * s["checkpoints"]["macro_f1"]["val"]["macro_f1"] for s in runs]
        rt = [100 * s["checkpoints"]["macro_f1"]["test"]["macro_f1"] for s in runs]
        lines.append(f"| {name} | {sv:.2f} | {ms(rv, 1)} | {sum(rv) / len(rv) - sv:+.2f} | {st:.2f} | {ms(rt, 1)} | "
                     f"{sum(rt) / len(rt) - st:+.2f} |")
    text = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    nets_json = [{"name": name, "hash": ph.hash, "decoder_version": DECODER_VERSION, "params": ph.params, "macs": ph.macs,
                  "search": {k: r[k] for k in ("generation", "origin", "fitness")} | {"val_macro_f1": r["val"]["macro_f1"]},
                  "arch": r["arch"], "genotype": r["genotype"], "canonical": ph.canonical} for name, r, ph in nets]
    (args.out / "networks.json").write_text(json.dumps(nets_json, indent=1), encoding="utf-8")
    chart(by_model, args.out / "test_f1_vs_cost.png")
    if args.chart:
        chart(by_model, args.out / "test_f1_vs_cost_selected.png", only=args.chart)
    print(text)


def chart(by_model: dict, path: Path, only: list | None = None):
    """Test macro-F1 (mean ± std over seeds) vs params and vs MACs; LeNet left out (far below the rest).
    `only`: the search networks to draw (full names on the labels); default all, labelled a, b, ..."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from analyze_search import BLUE, MUTED, SURFACE, style

    names = {"b2_plain": "PlainCNN", "b3_resnet18": "ResNet-18", "b4_mobilevitv2": "MobileViTv2"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, key, unit, title, xlabel in ((axes[0], "params", 1e6, "Test macro-F1 vs parameters", "parameters (M, log scale)"),
                                         (axes[1], "macs", 1e9, "Test macro-F1 vs MACs", "MACs (G, log scale)")):
        for m, runs in by_model.items():
            if m not in names and not (m.startswith("alpnet_") and (only is None or m in only)):
                continue
            t = 100 * np.array([r["checkpoints"]["macro_f1"]["test"]["macro_f1"] for r in runs])
            x, col = runs[0][key] / unit, BLUE if m.startswith("alpnet_") else MUTED
            ax.errorbar(x, t.mean(), yerr=t.std(ddof=1), fmt="o", color=col, markersize=6, capsize=3, linewidth=1.2)
            twin = any(o != m and o.startswith("alpnet_") and (only is None or o in only) and o < m
                       and rs[0][key] == runs[0][key] for o, rs in by_model.items())
            ax.annotate(names.get(m, m if only else m.removeprefix("alpnet_")), (x, t.mean()), xytext=(-12, -10) if twin else (6, 4),
                        textcoords="offset points", fontsize=8, color=col)  # same cost as an earlier one: label below left
        ax.set_xscale("log")
        style(ax, title, xlabel, "test macro-F1 (%, mean ± std)")
    axes[0].errorbar([], [], fmt="o", color=BLUE, label=f"search networks ({len(next(r for m, r in by_model.items() if m.startswith('alpnet_')))} seeds)")
    n_base = sorted({len(r) for m, r in by_model.items() if m in names})
    axes[0].errorbar([], [], fmt="o", color=MUTED, label=f"baselines ({'/'.join(map(str, n_base))} seeds)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
