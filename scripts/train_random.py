"""Train random architectures within budget with the shared recipe (generation 0 of the search).

Genotypes are drawn uniformly (space.sample) and kept if within budget, as the GA's generation 0 is.
Each network: seed_everything(0), build, trainer.SEARCH_RECIPE (--epochs, test once at the end), one
after another on the GPU alone. PyTorch is capped at 20 GB. OOM and NaN runs are recorded as eliminated and
more genotypes are drawn until --n networks complete all epochs. results.jsonl gets one line per candidate
(genotype, canonical form, cost, status, checkpoint val scores, train / val / test time, peak memory). The
test scores are sealed: only in each candidate's test_sealed.json, never printed or logged (no test score
of a candidate is looked at during the search). --like trains exactly the networks of another run with
another --epochs (generation 0 is trained with every phase's epoch count; the same --seed alone may
differ: a network that completes with one epoch count and fails with the other changes how many are
drawn). A rerun with the same --out skips the candidates already in results.jsonl; one that failed
with an error is trained again. A CUDA error stops the run (the GPU state may be broken), and so do
MAX_ERRORS errors in a row.

    python scripts/train_random.py --n 100 --epochs 20 --out runs/gen0_20ep
    python scripts/train_random.py --epochs 40 --like runs/gen0_20ep --out runs/gen0_40ep
    python scripts/train_random.py --n 100 --epochs 20 --out runs/gen0_20ep     # continue
"""

import argparse
import dataclasses
import gc
import json
import random
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.data.loader import dataset_id, load_splits  # noqa: E402
from alpnet.models.decoder import DECODER_VERSION, decode  # noqa: E402
from alpnet.search.fitness import fitness  # noqa: E402
from alpnet.search.ga import MAX_ERRORS, append_jsonl, cache_key, is_error, read_jsonl, write_atomic  # noqa: E402
from alpnet.search.space import sample  # noqa: E402
from alpnet.train.trainer import SEARCH_RECIPE, Recipe, seed_everything, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEM_CAP = 20e9
SEED = 0  # every candidate trains with the same seed
DATASET = "alpnet15_v1"


def search_key(recipe: Recipe, mem_cap: float = MEM_CAP) -> dict:
    """The cache key of a training by run_one (ga.cache_key)."""
    return cache_key(dataclasses.asdict(recipe), dataset_id(DATASET), SEED, DECODER_VERSION, mem_cap / 1e9)


def device_error(e: Exception) -> bool:
    """A CUDA / driver failure: the process's GPU state may be broken and every later candidate would fail
    too, so the first one is raised, not recorded (a restart, with a fresh CUDA context, trains the
    candidate again). A second one of the same candidate (device_error.txt left by the first) is that
    network's own: recorded as failed, never retried, so a restart cannot stop at it forever."""
    return isinstance(e, torch.AcceleratorError) or (
        isinstance(e, RuntimeError) and any(s in str(e) for s in ("CUDA", "cuDNN", "CUBLAS", "cublas")))


def draws(seed: int):
    """(idx, genotype, phenotype) of uniform random genotypes within budget, numbered from 1."""
    rng, idx = random.Random(seed), 0
    while True:
        g = sample(rng)
        ph = decode(g)
        if ph.within_budget():
            idx += 1
            yield idx, g, ph


def describe(ph) -> str:
    return " | ".join(f"s{s['slot']} {s['body']}/{s['conn']} d{s['depth']} w{s['width']} k{s['kernel']} {s['norm']}"
                      for s in ph.canonical["stages"])


def scores(ev: dict | None) -> dict | None:
    return {k: v for k, v in ev.items() if k != "confusion"} if ev else None


def run_one(idx: int, g: dict, ph, splits, recipe: Recipe, out: Path, rules=(), mem_cap: float = MEM_CAP) -> dict:
    """Train one candidate into out/<idx>_<hash>/ and return its results line (the GA's record).
    mem_cap: the PyTorch memory cap the process runs with (part of the cache key)."""
    run_dir = out / f"{idx:03d}_{ph.hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "device_error.txt"  # left by an earlier try of this candidate that hit a device error
    log_f = open(run_dir / "train.log", "w", encoding="utf-8")

    def log(msg):
        log_f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log_f.flush()

    row = {"idx": idx, "hash": ph.hash, "params": ph.params, "macs": ph.macs, "repairs": ph.repairs,
           "arch": describe(ph), "genotype": g, "canonical": ph.canonical, "epochs": recipe.epochs,
           "key": search_key(recipe, mem_cap)}
    t0 = time.monotonic()
    error = None
    try:
        seed_everything(SEED)  # right before construction: the init depends on the seed
        s = train(ph.build(), splits, recipe, SEED, run_dir,
                  {"hash": ph.hash, "decoder_version": DECODER_VERSION, "genotype": g, "canonical": ph.canonical}, log,
                  rules, seal_test=True)
    except torch.OutOfMemoryError:
        error = f"oom (>= {mem_cap / 1e9:g} GB)"
    except Exception as e:  # noqa: BLE001  (an unattended run records the failure and goes on)
        log(traceback.format_exc())
        if device_error(e) and not marker.exists():
            marker.write_text(traceback.format_exc(), encoding="utf-8")
            log_f.close()
            raise
        error = (f"failed: device error twice: {type(e).__name__}: {e}" if device_error(e)
                 else f"error: {type(e).__name__}: {e}")
    log_f.close()
    if not error:
        marker.unlink(missing_ok=True)  # it failed only because an earlier candidate broke the GPU state
    if error:
        row |= {"status": error, "total_s": time.monotonic() - t0}
    else:
        epochs = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
        row |= {"status": s["status"], "stop": s["stop"], "epochs_done": s["epochs_done"],
                "best_val_epoch": s["best_val_epoch"], "curve": [e["val_macro_f1"] for e in epochs],
                "val": scores(s["val"]), "val_final_epoch": s["val_final_epoch"],
                "train_s": sum(e["epoch_time"] for e in epochs), "val_s": sum(e["val_time"] for e in epochs),
                "test_s": s["test_time_s"], "total_s": s["train_time_s"],
                "peak_mem_gb": s["peak_mem_gb"]}
        if s["val"]:
            row["fitness"] = fitness(s["val"]["macro_f1"], ph.params, ph.macs)
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="networks to train to the end (OOM / NaN ones do not count)")
    ap.add_argument("--seed", type=int, default=2, help="genotype sampling seed")
    ap.add_argument("--epochs", type=int, default=SEARCH_RECIPE.epochs)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--like", type=Path, default=None,
                    help="train exactly the networks of another train_random.py run (its idx), instead of new draws")
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(MEM_CAP / torch.cuda.get_device_properties(0).total_memory)
    splits = load_splits(DATASET)
    recipe = dataclasses.replace(SEARCH_RECIPE, epochs=args.epochs)
    if args.like:  # the same networks with another epoch count (generation 0 of every phase)
        like = [r for r in read_jsonl(args.like / "results.jsonl", repair=False) if not is_error(r)]
        if bad := [r for r in like if r["key"] != search_key(recipe)]:
            raise SystemExit(f"{args.like}: {len(bad)} candidates were trained with another decoder / recipe / dataset")
        todo, n = ((r["idx"], r["genotype"], decode(r["genotype"])) for r in like), len(like)
    else:
        todo, n = draws(args.seed), args.n
    out = args.out or ROOT / "runs" / f"random_{args.epochs}ep_{time.strftime('%Y%m%d-%H%M')}"
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    rows = read_jsonl(results)  # a last line cut by a crash is dropped
    if bad := [r for r in rows if r.get("key") != search_key(recipe) or r.get("epochs") != args.epochs]:
        raise SystemExit(f"{out}: {len(bad)} candidates were trained with another decoder / recipe / dataset / "
                         "epoch count; use a new --out")
    if retry := [r for r in rows if is_error(r)]:  # failed with an error, not by the network: trained again
        rows = [r for r in rows if not is_error(r)]
        write_atomic(results, "".join(json.dumps(r) + "\n" for r in rows))
        print(f"{len(retry)} candidates that failed with an error are trained again", flush=True)
    done = {r["hash"]: r for r in rows}
    trained = sum(r["status"] == "completed" for r in done.values())
    print(f"-> {out}  ({len(done)} candidates already done, {trained} trained)", flush=True)

    errors, t_start = 0, time.monotonic()
    for idx, g, ph in todo:
        if not args.like and trained >= n:
            break
        if ph.hash in done:
            continue
        r = run_one(idx, g, ph, splits, recipe, out)
        append_jsonl(results, r)
        done[ph.hash] = r  # a network drawn again later in this run is not trained twice
        trained += r["status"] == "completed"
        errors = errors + 1 if is_error(r) else 0
        if errors >= MAX_ERRORS:
            raise SystemExit(f"{errors} candidates in a row failed with an error (last: {r['status']}); stopped. "
                             "The same command trains them again.")
        if "val_s" in r and r["epochs_done"]:
            ep = (r["train_s"] + r["val_s"] + r["test_s"]) / r["epochs_done"]
            print(f"{idx:3d} [{trained:3d}/{n}] {r['status']:9s} {ep:5.1f} s/ep  {r['total_s'] / 60:5.1f} min  "
                  f"val F1 {100 * r['val']['macro_f1']:5.1f}  "
                  f"{r['peak_mem_gb']:4.1f} GB  {r['params'] / 1e6:.2f}M {r['macs'] / 1e9:.3f}G  {r['arch']}", flush=True)
        else:
            print(f"{idx:3d} [{trained:3d}/{n}] {r['status']}  {r['arch']}", flush=True)
    print(f"done in {(time.monotonic() - t_start) / 3600:.1f} h -> {out}", flush=True)


if __name__ == "__main__":
    main()
