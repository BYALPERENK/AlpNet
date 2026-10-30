"""Sample grids from alpnet15: 5 random images per class from each split, at the 64x64 the model sees.

    python scripts/show_samples.py [--n 5] [--seed 0] [--scale 2]

Writes to data/processed/<name>_preview/:
  samples_train_aug.png    train | val | test, all with the training crop (random 64 offset + hflip p=0.5)
  samples_center_crop.png  the same val | test images with the evaluation crop (center 64)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.data.loader import CROP, PROCESSED  # noqa: E402

LABEL_W, HEAD_H, GAP, BLOCK_GAP = 150, 28, 4, 16


def train_crop(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    margin = img.shape[0] - CROP
    oy, ox = rng.integers(0, margin + 1, size=2)
    out = img[oy:oy + CROP, ox:ox + CROP]
    return out[:, ::-1] if rng.random() < 0.5 else out


def center_crop(img: np.ndarray) -> np.ndarray:
    o = (img.shape[0] - CROP) // 2
    return img[o:o + CROP, o:o + CROP]


def grid(blocks: dict[str, np.ndarray], classes: list[str], groups: list[str], class_group: list[int],
         scale: int, title: str) -> Image.Image:
    """blocks: split -> uint8 [n_classes, n, 64, 64]."""
    font = ImageFont.load_default(size=16)
    tile = CROP * scale
    n = next(iter(blocks.values())).shape[1]
    block_w = n * tile + (n - 1) * GAP
    w = LABEL_W + len(blocks) * block_w + (len(blocks) - 1) * BLOCK_GAP
    h = 2 * HEAD_H + len(classes) * (tile + GAP)
    canvas = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=0, font=font)
    for b, split in enumerate(blocks):
        x0 = LABEL_W + b * (block_w + BLOCK_GAP)
        draw.text((x0 + block_w // 2 - 20, HEAD_H + 4), split, fill=0, font=font)
    for c, name in enumerate(classes):
        y0 = 2 * HEAD_H + c * (tile + GAP)
        draw.text((4, y0 + tile // 2 - 18), name, fill=0, font=font)
        draw.text((4, y0 + tile // 2 + 2), groups[class_group[c]], fill=110, font=font)
        for b, imgs in enumerate(blocks.values()):
            x0 = LABEL_W + b * (block_w + BLOCK_GAP)
            for k in range(n):
                im = Image.fromarray(np.ascontiguousarray(imgs[c, k])).resize((tile, tile), Image.NEAREST)
                canvas.paste(im, (x0 + k * (tile + GAP), y0))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="alpnet15_v1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=2, help="nearest-neighbour upscale for viewing only")
    args = ap.parse_args()

    meta = json.loads((PROCESSED / f"{args.name}_metadata.json").read_text(encoding="utf-8"))
    data = np.load(PROCESSED / f"{args.name}.npz")
    classes, groups = meta["classes"], meta["groups"]
    rng = np.random.default_rng(args.seed)
    out = PROCESSED / f"{args.name}_preview"
    out.mkdir(exist_ok=True)

    raw, class_group = {}, [0] * len(classes)
    for split in ("train", "val", "test"):
        x, y, src = data[f"x_{split}"], data[f"y_{split}"], data[f"src_{split}"]
        counts = np.bincount(y, minlength=len(classes))
        print(f"{split:5s} {len(y):6d}  per class: {dict(zip(classes, counts.tolist()))}")
        picks = []
        for c in range(len(classes)):
            idx = np.flatnonzero(y == c)
            class_group[c] = int(src[idx[0]])
            picks.append(x[np.sort(rng.choice(idx, args.n, replace=False))])
        raw[split] = np.stack(picks)  # [C, n, 80, 80]

    aug = {s: np.stack([[train_crop(im, rng) for im in row] for row in imgs]) for s, imgs in raw.items()}
    grid(aug, classes, groups, class_group, args.scale,
         f"{args.name} - training crop (random 64x64 offset + hflip), {args.n} per class per split"
         ).save(out / "samples_train_aug.png")

    cen = {s: np.stack([[center_crop(im) for im in row] for row in raw[s]]) for s in ("val", "test")}
    grid(cen, classes, groups, class_group, args.scale,
         f"{args.name} - evaluation crop (center 64x64), same val/test images"
         ).save(out / "samples_center_crop.png")
    print(f"saved {out / 'samples_train_aug.png'}\nsaved {out / 'samples_center_crop.png'}")


if __name__ == "__main__":
    main()
