"""Select Open Images V7 boxes for the vehicle classes and download only the images they live in.

Needs the CSVs from `download_raw.py openimages`. Box filters:
  - IsDepiction == 0  (no drawings, toys, paintings)
  - IsGroupOf == 0    (no "crowd of cars" boxes)
  - IsInside == 0     (no photos taken from inside the vehicle)
  - normalized short side >= MIN_FRAC (pre-filter; the exact >= 80 px check runs after
    download, in the dataset build, because the CSV has no pixel sizes)

Images are picked in a seeded random order until every class has TARGET_BOXES candidate
boxes, so we download ~as few images as needed.

    python scripts/download_openimages.py
"""

import csv
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OI = ROOT / "data" / "raw" / "openimages"
IMG_DIR = OI / "images" / "train"
# our class name -> Open Images display name ("Aircraft" would also include helicopters)
OI_NAMES = {"Airplane": "Fixed-wing aircraft", "Boat": "Boat", "Car": "Car", "Truck": "Truck"}
CLASSES = list(OI_NAMES)
MIN_FRAC = 0.12
TARGET_BOXES = 6000  # 3000 needed; margin for the exact pixel filter and cleaning
SEED = 0
S3_URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"


def class_mids() -> dict[str, str]:
    path = OI / "oidv7-class-descriptions-boxable.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    by_name = {r[1]: r[0] for r in rows if len(r) >= 2}
    missing = [n for n in OI_NAMES.values() if n not in by_name]
    if missing:
        sys.exit(f"classes not found in {path.name}: {missing}")
    return {by_name[oi]: ours for ours, oi in OI_NAMES.items()}


def candidate_boxes(mid_to_name: dict[str, str]) -> pd.DataFrame:
    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax",
            "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
    parts = []
    reader = pd.read_csv(OI / "oidv6-train-annotations-bbox.csv", usecols=cols, chunksize=5_000_000)
    for i, chunk in enumerate(reader):
        c = chunk[chunk["LabelName"].isin(mid_to_name.keys())]
        c = c[(c["IsDepiction"] == 0) & (c["IsGroupOf"] == 0) & (c["IsInside"] == 0)]
        short = (c["XMax"] - c["XMin"]).where(lambda w: w < c["YMax"] - c["YMin"], c["YMax"] - c["YMin"])
        parts.append(c[short >= MIN_FRAC])
        print(f"  scanned chunk {i}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["cls"] = df["LabelName"].map(mid_to_name)
    return df


def pick_images(df: pd.DataFrame) -> list[str]:
    per_image = defaultdict(Counter)
    for img, cls in zip(df["ImageID"], df["cls"]):
        per_image[img][cls] += 1
    order = sorted(per_image)
    random.Random(SEED).shuffle(order)
    have = Counter()
    picked = []
    for img in order:
        need = [c for c in CLASSES if have[c] < TARGET_BOXES]
        if not need:
            break
        if any(per_image[img][c] for c in need):
            picked.append(img)
            have.update(per_image[img])
    print("candidate boxes in picked images:", dict(have), flush=True)
    return picked


def fetch(image_id: str) -> str:
    dest = IMG_DIR / f"{image_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    tmp = dest.with_suffix(".part")
    for attempt in range(2):  # one retry for transient network errors
        try:
            with urllib.request.urlopen(S3_URL.format(image_id), timeout=60) as r:
                tmp.write_bytes(r.read())
            tmp.replace(dest)
            return "ok"
        except Exception:
            if attempt:
                return "fail"
            time.sleep(2)
    return "fail"


def main() -> None:
    mid_to_name = class_mids()
    print("class MIDs:", mid_to_name, flush=True)
    df = candidate_boxes(mid_to_name)
    print("candidate boxes per class:", df["cls"].value_counts().to_dict(), flush=True)

    picked = pick_images(df)
    sel = df[df["ImageID"].isin(set(picked))]
    sel.to_csv(OI / "selected_boxes.csv", index=False)
    print(f"{len(picked)} images, {len(sel)} boxes -> selected_boxes.csv", flush=True)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = [pool.submit(fetch, i) for i in picked]
        for n, fut in enumerate(as_completed(futs), 1):
            stats[fut.result()] += 1
            if n % 1000 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} {dict(stats)} {time.monotonic() - t0:.0f}s", flush=True)
    failed = [i for i in picked if not (IMG_DIR / f"{i}.jpg").exists()]
    if failed:
        (OI / "failed_images.txt").write_text("\n".join(failed))
        print(f"{len(failed)} images failed -> failed_images.txt", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
