"""Run the GA: one candidate at a time on the GPU, PyTorch capped at 20 GB.

Candidates train with trainer.SEARCH_RECIPE (epochs from the config's phases) and seed 0, like
train_random.py, whose runs are imported as generation 0: one run per phase epoch count, of the same
networks (--like). The same command continues a stopped / crashed search; with more generations in the
config's last phase it extends a finished one. The memory cap is part of the cache key.

    python scripts/train_random.py --n 100 --epochs 20 --out runs/gen0_20ep
    python scripts/train_random.py --epochs 40 --like runs/gen0_20ep --out runs/gen0_40ep
    python scripts/run_search.py --out runs/search_X --gen0 runs/gen0_20ep runs/gen0_40ep
    python scripts/run_search.py --out runs/search_X                         # continue
    python scripts/run_search.py --out runs/search_trial --config configs/search_trial.yaml
"""

import argparse
import dataclasses
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.data.loader import load_splits  # noqa: E402
from alpnet.search.ga import GAConfig, Search  # noqa: E402
from alpnet.train.trainer import SEARCH_RECIPE  # noqa: E402
from scripts.train_random import DATASET, MEM_CAP, run_one, search_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> GAConfig:
    """A phase's rules are given in full (a rule left out is off)."""
    return GAConfig(**(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "search.yaml")
    ap.add_argument("--gen0", type=Path, nargs="+", default=None,
                    help="import generation 0: train_random.py runs, one per phase epoch count")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this many new evaluations (test)")
    ap.add_argument("--mem-cap", type=float, default=MEM_CAP / 1e9, help="GB (a trial next to a real run: less)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, args.out / f"config_{time.strftime('%Y%m%d-%H%M%S')}.yaml")  # every start
    log_f = open(args.out / "search.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    torch.cuda.set_per_process_memory_fraction(args.mem_cap * 1e9 / torch.cuda.get_device_properties(0).total_memory)
    splits = load_splits(DATASET)
    def evaluate(idx, g, ph, rules, gen_dir, epochs):
        return run_one(idx, g, ph, splits, dataclasses.replace(SEARCH_RECIPE, epochs=epochs), gen_dir, rules,
                       args.mem_cap * 1e9)

    search = Search(cfg, args.out, evaluate, args.stop_after, log, key=search_key(SEARCH_RECIPE, args.mem_cap * 1e9))
    if args.gen0 and search.generation is None:
        search.import_generation0(args.gen0)
    log(f"config {args.config.name}, recipe {SEARCH_RECIPE} (epochs: generation 0 "
        + " and ".join(map(str, cfg.epoch_counts)) + ", "
        + ", ".join(f"generations {a}-{b}: {e}" for a, b, e in cfg.ranges()) + ")")
    for p in cfg.phases:
        if not p.get("calibration"):
            log(f"WARNING: the {p['epochs']}-epoch phase's stop rules are not calibrated (no `calibration` report "
                "in the config): scripts/calibrate_rules.py on generation 0 first")
    search.run()
    log(f"done: {len(search.history)} evaluations, generation {search.generation}")


if __name__ == "__main__":
    main()
