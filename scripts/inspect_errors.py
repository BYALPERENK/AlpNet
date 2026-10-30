"""Dataset health check from a trained baseline run: top confusions and the highest-loss val images.
Needs the checkpoint's weights: train it with `train_baseline.py --save-weights` (weights are not kept
otherwise).

    python scripts/inspect_errors.py runs/<run_dir> [--top 60]

Writes <run_dir>/confusions.txt and <run_dir>/top_loss.png (each tile: true -> predicted).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.data.loader import GpuSplit  # noqa: E402
from alpnet.models.baselines import MODELS  # noqa: E402
from alpnet.train.trainer import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = json.loads((run / "config.json").read_text())

    val = GpuSplit(cfg["dataset"].split("@")[0], "val")  # the name, without the content hash (loader.dataset_id)
    model = MODELS[cfg["model"]]().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cuda"))
    probs, y, src = predict(model, val)
    pred = probs.argmax(1)
    cls, groups = val.classes, val.groups
    class_group = {c: g for c, g in zip(y, src)}

    # top off-diagonal confusions
    cm = np.zeros((len(cls), len(cls)), dtype=int)
    np.add.at(cm, (y, pred), 1)
    pairs = [(cm[i, j], i, j) for i in range(len(cls)) for j in range(len(cls)) if i != j and cm[i, j]]
    pairs.sort(reverse=True)
    lines = [f"{n:4d}  {cls[i]:>16s} -> {cls[j]:<16s} {'CROSS-GROUP' if class_group[i] != class_group[j] else ''}"
             for n, i, j in pairs[:25]]
    cross = [(n, i, j) for n, i, j in pairs if class_group[i] != class_group[j]]
    lines += ["", f"cross-group errors: {sum(n for n, _, _ in cross)} / {len(y)}"]
    for n, i, j in cross[:10]:
        lines.append(f"  {n:4d}  {groups[class_group[i]]}/{cls[i]} -> {groups[class_group[j]]}/{cls[j]}")
    text = "\n".join(lines)
    (run / "confusions.txt").write_text(text + "\n")
    print(text)

    # highest-loss images (center crop as the model saw them)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12)
    worst = np.argsort(-loss)[:args.top]
    x = val.x.cpu().numpy()[:, 8:72, 8:72]
    cols, tile, pad = 10, 64, 14
    rows = (len(worst) + cols - 1) // cols
    canvas = Image.new("L", (cols * tile, rows * (tile + pad)), 255)
    draw = ImageDraw.Draw(canvas)
    for k, i in enumerate(worst):
        cx, cy = (k % cols) * tile, (k // cols) * (tile + pad)
        canvas.paste(Image.fromarray(x[i]), (cx, cy + pad))
        draw.text((cx + 1, cy), f"{cls[y[i]][:5]}>{cls[pred[i]][:5]}", fill=0)
    canvas.save(run / "top_loss.png")
    print(f"saved {run / 'top_loss.png'}")


if __name__ == "__main__":
    main()
