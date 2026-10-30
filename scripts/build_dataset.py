"""Build alpnet15 from the raw sources.

    python scripts/build_dataset.py                     # all four groups -> alpnet15_v1
    python scripts/build_dataset.py --groups medical animal --name test_build

Per group: walk each split's candidates in their seeded order, preprocess (gray, square,
80x80, no upscaling), drop rejects and near-duplicates (pHash, Hamming <= DUP_DIST, checked
against everything already kept in the group, train first), keep the first N per class.
Outputs in data/processed/: <name>.npz, <name>_manifest.csv, <name>_metadata.json,
<name>_preview/ (per-class grids, class means).
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpnet.data.preprocess import SIZE, Reject, hamming, phash, to_gray_square  # noqa: E402
from alpnet.data.sources import ADAPTERS, ROOT, SPLITS, Candidate  # noqa: E402

NEED = {"train": 2000, "val": 500, "test": 500}
DUP_DIST = 4
BATCH = 256
SEED = 0


def process(c: Candidate):
    try:
        arr = to_gray_square(c.load())
        return arr, phash(arr), None
    except Reject as e:
        return None, None, str(e)
    except Exception as e:  # corrupt file etc.
        return None, None, f"error:{type(e).__name__}"


def build_group(name: str, pool: ThreadPoolExecutor, log) -> tuple[list[dict], list[np.ndarray], dict]:
    t0 = time.monotonic()
    src = ADAPTERS[name](SEED)
    kept_hashes = np.zeros(0, dtype=np.uint64)
    kept_cls: list[str] = []
    rows, images = [], []
    report = {"notes": src.notes, "classes": src.classes, "rejects": {}, "shortfall": {}}

    for split in SPLITS:  # train first: a val/test image that duplicates a train image is dropped
        for cls in src.classes:
            cands = src.splits[split][cls]
            got, pos, reasons = 0, 0, Counter()
            while got < NEED[split] and pos < len(cands):
                batch = cands[pos:pos + BATCH]
                pos += len(batch)
                for c, (arr, h, why) in zip(batch, pool.map(process, batch)):
                    if got >= NEED[split]:
                        break
                    if arr is None:
                        reasons[why] += 1
                        continue
                    if len(kept_hashes):
                        d = hamming(h, kept_hashes)
                        j = int(d.argmin())
                        if d[j] <= DUP_DIST:
                            reasons["dup" if kept_cls[j] == cls else "dup_other_class"] += 1
                            continue
                    kept_hashes = np.append(kept_hashes, h)
                    kept_cls.append(cls)
                    images.append(arr)
                    rows.append({"split": split, "group": name, "cls": cls, "ref": c.ref,
                                 "leak_group": c.group, "phash": f"{int(h):016x}"})
                    got += 1
            report["rejects"][f"{split}/{cls}"] = dict(reasons)
            if got < NEED[split]:
                report["shortfall"][f"{split}/{cls}"] = f"{got}/{NEED[split]} (pool {len(cands)})"
            log(f"  {name:8s} {split:5s} {cls:17s} kept {got:4d}/{NEED[split]}  "
                f"used {pos:5d}/{len(cands):5d}  rejects {dict(reasons)}")
    log(f"  {name} done in {time.monotonic() - t0:.0f}s")
    return rows, images, report


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def save_previews(out_dir: Path, x: np.ndarray, man: pd.DataFrame, classes: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tr = man[man.split == "train"]
    # overview: one row per class, 12 random train images + class mean at the end
    cols = 13
    canvas = Image.new("L", (cols * SIZE, len(classes) * SIZE), 255)
    for r, cls in enumerate(classes):
        idx = tr.index[tr.cls == cls].to_numpy()
        pick = rng.choice(idx, size=min(12, len(idx)), replace=False)
        for k, i in enumerate(pick):
            canvas.paste(Image.fromarray(x[i]), (k * SIZE, r * SIZE))
        mean = x[idx].mean(axis=0).astype(np.uint8)
        canvas.paste(Image.fromarray(mean), (12 * SIZE, r * SIZE))
    canvas.save(out_dir / "overview.png")
    # per-class 8x8 grids
    for cls in classes:
        idx = tr.index[tr.cls == cls].to_numpy()
        pick = rng.choice(idx, size=min(64, len(idx)), replace=False)
        grid = Image.new("L", (8 * SIZE, 8 * SIZE), 255)
        for k, i in enumerate(pick):
            grid.paste(Image.fromarray(x[i]), ((k % 8) * SIZE, (k // 8) * SIZE))
        grid.save(out_dir / f"{cls}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(ADAPTERS), choices=list(ADAPTERS))
    ap.add_argument("--name", default="alpnet15_v1")
    args = ap.parse_args()

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{args.name}_build.log"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    rows, images, reports, classes = [], [], {}, []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for g in args.groups:
            log(f"group {g}")
            r, im, rep = build_group(g, pool, log)
            rows += r
            images += im
            reports[g] = rep
            classes += rep["classes"]

    man = pd.DataFrame(rows)
    man["label"] = man.cls.map({c: i for i, c in enumerate(classes)})
    man["source"] = man.group.map({g: i for i, g in enumerate(args.groups)})
    x_all = np.stack(images)

    arrays = {}
    for split in SPLITS:
        m = man[man.split == split]
        arrays[f"x_{split}"] = x_all[m.index.to_numpy()]
        arrays[f"y_{split}"] = m.label.to_numpy(np.int64)
        arrays[f"src_{split}"] = m.source.to_numpy(np.int64)
    npz_path = out / f"{args.name}.npz"
    np.savez(npz_path, **arrays)
    man.to_csv(out / f"{args.name}_manifest.csv", index=False)

    tr = arrays["x_train"].astype(np.float64) / 255.0
    meta = {
        "name": args.name,
        "size": SIZE,
        "seed": SEED,
        "dup_hamming_max": DUP_DIST,
        "per_class": NEED,
        "classes": classes,
        "groups": args.groups,
        "counts": {s: man[man.split == s].cls.value_counts().reindex(classes).fillna(0).astype(int).to_dict()
                   for s in SPLITS},
        "train_mean": float(tr.mean()),
        "train_std": float(tr.std()),
        "class_pixel_mean": {c: float(tr[arrays["y_train"] == i].mean()) for i, c in enumerate(classes)},
        "sources": reports,
        # over the array bytes, not the .npz file: the zip container stores write timestamps
        "content_sha256": content_hash(arrays),
    }
    (out / f"{args.name}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_previews(out / f"{args.name}_preview", x_all, man, classes)

    short = {g: r["shortfall"] for g, r in reports.items() if r["shortfall"]}
    log(f"saved {npz_path.name}: " + ", ".join(f"{s}={len(arrays[f'x_{s}'])}" for s in SPLITS))
    log(f"content_sha256 {meta['content_sha256']}")
    log("SHORTFALL: " + json.dumps(short) if short else "all classes full")


if __name__ == "__main__":
    main()
