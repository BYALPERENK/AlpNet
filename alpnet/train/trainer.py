"""The shared training recipe: AdamW + warmup/cosine, bf16, label smoothing.

Checkpoint rule: the epoch with the best val `recipe.checkpoint_metric` (macro-F1 by default).
Its val/test scores are reported as `val` / `test`. Weights are not saved (`save_weights`: the
checkpoint's, as model.pt, for scripts/inspect_errors.py). Every rule in
CHECKPOINT_METRICS is also recorded under `checkpoints`, and metrics.jsonl holds val and test scores
for every epoch, so another selection rule can be applied later without retraining. With
`test_every_epoch=False` (search mode) metrics.jsonl has val only: the weights of each checkpoint
epoch are kept in GPU memory and the test set is evaluated once at the end, with the same result;
`seal_test` then keeps those scores out of the log and the summary (test_sealed.json only).

Every run writes to runs/<stamp>_<name>/: config.json, metrics.jsonl (one line per epoch),
summary.json. A run whose training loss becomes NaN / inf stops and gets status "failed: ..." with
the epochs it completed; it does not raise. Stop rules (stopping.py) are checked after every epoch:
a prune gives status "pruned: ...", a patience stop keeps "completed"; `stop` holds the reason.
"""

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from ..data.loader import GpuSplit
from ..data.sources import ROOT
from .metrics import evaluate
from .stopping import check, describe


@dataclass
class Recipe:
    epochs: int = 40
    batch_size: int = 256
    eval_batch_size: int = 256  # 1024 needed 1.5-2.6× the training memory
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    checkpoint_metric: str = "macro_f1"  # key of evaluate(); higher is better
    test_every_epoch: bool = True  # False in the search: test once at the end (~8.5% of a run saved)
    fused: bool = False  # fused AdamW kernel


# The search (generation 0 and the GA): test once at the end (same scores) and fused AdamW (the same
# algorithm; training ~35% faster on most networks, runs/ab_fused_20261022-0645). Baselines: Recipe().
SEARCH_RECIPE = Recipe(test_every_epoch=False, fused=True)

CHECKPOINT_METRICS = ("macro_f1", "macro_auroc")  # selection rules recorded in summary.json


def write_synced(path: Path, text: str):
    """Written and flushed to disk (a power cut right after a run must not lose its files)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator(device="cuda").manual_seed(seed)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases and norm parameters (1-D tensors)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def count_cost(model: nn.Module) -> tuple[int, float]:
    params = sum(p.numel() for p in model.parameters())
    x = torch.zeros(1, 1, 64, 64, device="cuda").to(memory_format=torch.channels_last)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return params, fc.get_total_flops() / 2  # MACs per image


@torch.no_grad()
def predict(model: nn.Module, split: GpuSplit, batch_size: int = Recipe.eval_batch_size):
    model.eval()
    probs, ys, srcs = [], [], []
    for x, y, src in split.batches(batch_size, shuffle=False, drop_last=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        probs.append(logits.float().softmax(1))
        ys.append(y)
        srcs.append(src)
    return (torch.cat(probs).cpu().numpy(), torch.cat(ys).cpu().numpy(), torch.cat(srcs).cpu().numpy())


def train(model: nn.Module, splits: dict[str, GpuSplit], recipe: Recipe, seed: int, run_dir: Path,
          extra_config: dict | None = None, log=print, rules=(), seal_test: bool = False,
          save_weights: bool = False) -> dict:
    """The caller builds `model` right after seed_everything(seed), so the init depends on the seed.
    Data order and augmentation use their own generator: the same stream for every architecture.
    `rules`: stop rules, checked in order after every epoch. `seal_test` (search candidates, test at the
    end only): only the primary checkpoint is tested, its scores go to test_sealed.json alone, not to the
    log or the returned summary. Every file is flushed to disk (metrics.jsonl every epoch)."""
    assert not (seal_test and recipe.test_every_epoch), "a sealed test runs once, at the end"
    run_dir.mkdir(parents=True, exist_ok=True)
    gen = seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    model = model.cuda().to(memory_format=torch.channels_last)
    params, macs = count_cost(model)

    tr, va = splits["train"], splits["val"]
    steps_per_epoch = len(tr) // recipe.batch_size
    total, warmup = recipe.epochs * steps_per_epoch, recipe.warmup_epochs * steps_per_epoch
    opt = torch.optim.AdamW(param_groups(model, recipe.weight_decay), lr=recipe.lr, fused=recipe.fused)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))

    config = {"seed": seed, "recipe": asdict(recipe), "params": params, "macs": macs,
              "dataset": splits["train"].dataset, "stop_rules": [describe(r) for r in rules], **(extra_config or {})}
    write_synced(run_dir / "config.json", json.dumps(config, indent=2))
    log(f"params {params / 1e6:.2f}M  MACs {macs / 1e9:.3f}G  steps/epoch {steps_per_epoch}")

    primary = recipe.checkpoint_metric
    # every selection rule we may want later; test is only logged, never used for selection
    tracked = list(dict.fromkeys((primary, *CHECKPOINT_METRICS)))
    best = {m: {"epoch": 0, "score": -1.0} for m in tracked}
    te = splits["test"]
    status, ev, done, stop, rows = "completed", None, 0, None, []
    states, test_s, best_state = {}, 0.0, None  # checkpoint epoch -> weights (test at the end only)
    t_start = time.monotonic()
    with open(run_dir / "metrics.jsonl", "w") as metrics_f:
        for epoch in range(1, recipe.epochs + 1):
            model.train()
            t0 = time.monotonic()
            loss_sum = torch.zeros((), device="cuda")
            correct = torch.zeros((), device="cuda")
            seen = 0
            for x, y, _ in tr.batches(recipe.batch_size, generator=gen):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=recipe.label_smoothing)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.detach() * len(y)
                correct += (logits.argmax(1) == y).sum()
                seen += len(y)
            train_loss = (loss_sum / seen).item()
            train_time = time.monotonic() - t0
            if not math.isfinite(train_loss):
                status = f"failed: loss {train_loss} at epoch {epoch}"
                log(status)
                break

            t0 = time.monotonic()
            ev = evaluate(*predict(model, va, recipe.eval_batch_size), va.classes, va.groups)
            val_time = time.monotonic() - t0
            et, test_time = None, 0.0
            if recipe.test_every_epoch:
                t0 = time.monotonic()
                et = evaluate(*predict(model, te, recipe.eval_batch_size), va.classes, va.groups)
                test_time = time.monotonic() - t0
                test_s += test_time
            done = epoch
            row = {"epoch": epoch, "train_loss": train_loss, "train_acc": (correct / seen).item(),
                   "lr": sched.get_last_lr()[0], "epoch_time": train_time, "val_time": val_time,
                   "test_time": test_time,
                   **{f"val_{k}": v for k, v in ev.items() if k != "confusion"},
                   **{f"test_{k}": v for k, v in (et or {}).items() if k != "confusion"}}
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            os.fsync(metrics_f.fileno())
            rows.append(row)
            for m in tracked:
                if ev[m] > best[m]["score"]:
                    best[m] = {"epoch": epoch, "score": ev[m], "val": ev, "test": et}
            if save_weights and best[primary]["epoch"] == epoch:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if not recipe.test_every_epoch:  # sealed: the other checkpoints' test scores would be unused
                keep = {best[primary]["epoch"]} if seal_test else {b["epoch"] for b in best.values()}
                if epoch in keep:
                    states[epoch] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                states = {e: s for e, s in states.items() if e in keep}
            log(f"ep {epoch:2d}  loss {train_loss:.3f}  train_acc {row['train_acc']:.3f}  "
                f"val_acc {ev['acc']:.4f}  val_f1 {ev['macro_f1']:.4f}  auroc {ev['macro_auroc']:.4f}  "
                f"xgrp {ev['cross_group_err']:.4f}  {train_time:.1f}s")
            if stop := check(rules, rows):
                log(f"stop: {stop.reason}")
                if stop.prune:
                    status = f"pruned: {stop.reason}"
                break

    if states:  # test once, with the weights of each checkpoint epoch (the model ends with the last loaded)
        t0 = time.monotonic()
        tests = {}
        for e, state in sorted(states.items()):
            model.load_state_dict(state)
            tests[e] = evaluate(*predict(model, te, recipe.eval_batch_size), va.classes, va.groups)
        for b in best.values():
            if b["epoch"]:
                b["test"] = tests.get(b["epoch"])
        test_s = time.monotonic() - t0
    if seal_test:
        b = best[primary]
        sealed = {primary: {"epoch": b["epoch"], "test": b["test"]}} if b.get("test") else {}
        write_synced(run_dir / "test_sealed.json", json.dumps({"checkpoint": f"best_val_{primary}", "checkpoints": sealed}))
        for b in best.values():
            b["test"] = None
    for m in tracked:
        if best[m]["epoch"]:
            log(f"best val {m} epoch {best[m]['epoch']}: val {best[m]['score']:.5f}" + ("" if seal_test else
                f"  test acc {best[m]['test']['acc']:.4f}  f1 {best[m]['test']['macro_f1']:.4f}  "
                f"auroc {best[m]['test']['macro_auroc']:.5f}"))
    summary = {"status": status, "stop": stop.reason if stop else None, "epochs_done": done,
               "checkpoint": f"best_val_{primary}",
               "val": best[primary].get("val"), "test": best[primary].get("test"),
               "best_val_epoch": best[primary]["epoch"],
               "checkpoints": {m: {"epoch": b["epoch"], "val": b["val"], "test": b["test"]}
                               for m, b in best.items() if b["epoch"]},
               "val_final_epoch": {k: v for k, v in ev.items() if k != "confusion"} if ev else None,
               "train_time_s": time.monotonic() - t_start, "test_time_s": test_s,
               "peak_mem_gb": torch.cuda.max_memory_reserved() / 1e9,  # reserved, data on the GPU included
               **config}
    write_synced(run_dir / "summary.json", json.dumps(summary, indent=2))
    if best_state is not None:
        torch.save(best_state, run_dir / "model.pt")
    return summary
