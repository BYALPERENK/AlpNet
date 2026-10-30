"""Train baselines B1-B4 with the shared recipe.

    python scripts/train_baseline.py                          # all models x seeds 0 1 2
    python scripts/train_baseline.py --models b2_plain --seeds 0 --epochs 5
    python scripts/train_baseline.py --sanity                 # one-batch overfit check
    python scripts/train_baseline.py --models b2_plain --seeds 0 --save-weights   # for inspect_errors.py
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.data.loader import load_splits  # noqa: E402
from alpnet.data.sources import ROOT  # noqa: E402
from alpnet.models.baselines import MODELS  # noqa: E402
from alpnet.train.trainer import Recipe, seed_everything, train  # noqa: E402


def sanity(splits, names):
    """Each model must be able to memorize 64 fixed training images (no aug, no smoothing). The images are
    taken class by class in turn (the arrays are stored class by class: the first 64 are all ADI, and
    predicting one class would pass)."""
    tr = splits["train"]
    per = -(-64 // len(tr.classes))
    idx = torch.stack([torch.nonzero(tr.y == c).flatten()[:per] for c in range(len(tr.classes))], 1).flatten()[:64]
    tr.train_aug = False
    x, y, _ = tr.take(idx)
    tr.train_aug = True
    assert len(set(y.tolist())) == len(tr.classes), "every class in the sanity batch"
    for name in names:
        seed_everything(0)
        model = MODELS[name]().cuda().to(memory_format=torch.channels_last).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(1, 301):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        acc = (logits.argmax(1) == y).float().mean().item()
        ok = "OK" if loss.item() < 0.05 and acc == 1.0 else "FAIL"
        print(f"sanity {name:16s} loss {loss.item():.4f}  acc {acc:.3f}  {ok}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=Recipe.epochs)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--save-weights", action="store_true", help="keep the checkpoint's weights (inspect_errors.py)")
    args = ap.parse_args()

    splits = load_splits("alpnet15_v1")
    if args.sanity:
        sanity(splits, args.models)
        return

    recipe = Recipe(epochs=args.epochs)
    stamp = time.strftime("%Y%m%d-%H%M")
    for name in args.models:
        for seed in args.seeds:
            run_dir = ROOT / "runs" / f"{stamp}_{name}_s{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_f = open(run_dir / "train.log", "w", encoding="utf-8")

            def log(msg, _f=log_f, _tag=f"{name}/s{seed}"):
                line = f"{time.strftime('%H:%M:%S')} [{_tag}] {msg}"
                print(line, flush=True)
                _f.write(line + "\n")
                _f.flush()

            seed_everything(seed)  # before construction, so the init depends on the seed
            s = train(MODELS[name](), splits, recipe, seed, run_dir, {"model": name}, log,
                      save_weights=args.save_weights)
            log(f"FINAL {s['checkpoint']} epoch {s['best_val_epoch']} | test acc {s['test']['acc']:.4f} "
                f"f1 {s['test']['macro_f1']:.4f} auroc {s['test']['macro_auroc']:.5f} | {s['train_time_s']:.0f}s")
            log_f.close()


if __name__ == "__main__":
    main()
