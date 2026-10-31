"""Source adapters for alpnet15 (Config A).

Each adapter returns, for every split, a per-class list of candidates in preference order.
The builder walks those lists, drops rejects and duplicates, and keeps the first N.
Split assignment happens here, before any sampling, so that leakage groups
(Open Images photos) never straddle two splits.
"""

import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from .preprocess import SIZE, Reject

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SPLITS = ("train", "val", "test")


@dataclass
class Candidate:
    cls: str
    ref: str          # traceable id: file path inside the archive, npz index, box id...
    group: str        # leakage group; candidates sharing it always share a split
    load: Callable[[], Image.Image] = field(repr=False)


@dataclass
class Source:
    name: str
    classes: list[str]
    splits: dict[str, dict[str, list[Candidate]]]  # split -> class -> ordered candidates
    notes: str


def _shuffled(items: list, seed: int) -> list:
    items = sorted(items, key=lambda c: c.ref)
    random.Random(seed).shuffle(items)
    return items


def _by_class(cands: list[Candidate], classes: list[str], seed: int) -> dict[str, list[Candidate]]:
    return {c: _shuffled([x for x in cands if x.cls == c], seed) for c in classes}


# --------------------------------------------------------------------------- PathMNIST

PATH_CLASSES = {0: "ADI", 3: "LYM", 6: "NORM", 8: "TUM"}  # MedMNIST label index -> name


def pathmnist(seed: int) -> Source:
    data = np.load(RAW / "pathmnist" / "pathmnist_128.npz")
    splits = {}
    for split in SPLITS:
        imgs, labels = data[f"{split}_images"], data[f"{split}_labels"].ravel()
        cands = [
            Candidate(PATH_CLASSES[int(y)], f"{split}:{i}", f"{split}:{i}",
                      lambda a=imgs[i]: Image.fromarray(a))
            for i, y in enumerate(labels) if int(y) in PATH_CLASSES
        ]
        splits[split] = _by_class(cands, list(PATH_CLASSES.values()), seed)
    return Source(
        "medical", list(PATH_CLASSES.values()), splits,
        "PathMNIST (MedMNIST+ 128px). train/val = MedMNIST train/val (NCT-CRC-HE-100K, patch-level "
        "random split: same slides on both sides, val is optimistic). test = CRC-VAL-HE-7K "
        "(50 different patients, no overlap).",
    )


# --------------------------------------------------------------------------- AFHQ v2

def extract_afhq() -> Path:
    """Unzip once; reading 15k members straight from the zip is GIL-bound and very slow."""
    import zipfile

    out = RAW / "afhq" / "extracted"
    done = out / ".done"
    if not done.exists():
        with zipfile.ZipFile(RAW / "afhq" / "afhq_v2.zip") as z:
            z.extractall(out)
        done.write_text("ok")
    return out


def afhq(seed: int) -> Source:
    out = extract_afhq()
    classes = ["cat", "dog", "wild"]

    def cands(split: str) -> list[Candidate]:
        return [Candidate(p.parent.name, f"{split}/{p.parent.name}/{p.name}", f"{split}/{p.parent.name}/{p.name}",
                          lambda p=p: Image.open(p))
                for c in classes for p in (out / split / c).glob("*.png")]

    official_test = _by_class(cands("test"), classes, seed)
    trainpool = _by_class(cands("train"), classes, seed)
    splits = {s: {} for s in SPLITS}
    for c in classes:
        pool = trainpool[c]
        # official test has only ~485/class: top it up from a slice of train held out for that
        splits["val"][c] = pool[:650]
        splits["test"][c] = official_test[c] + pool[650:800]
        splits["train"][c] = pool[800:]
    return Source(
        "animal", classes, splits,
        "AFHQ v2 (512px PNG). test = official test (~485/class) topped up from held-out train "
        "images; train/val = random split of official train. No group info in the source.",
    )


# --------------------------------------------------------------------------- Open Images V7

OI_MIN_SIDE = 32     # box short side in px; very thin boxes are mostly noise
OI_MAX_PER_IMG = 2   # per class per photo, so one parking lot does not dominate a class


def openimages(seed: int) -> Source:
    oi = RAW / "openimages"
    boxes = pd.read_csv(oi / "selected_boxes.csv")
    boxes["area"] = (boxes.XMax - boxes.XMin) * (boxes.YMax - boxes.YMin)
    boxes = (boxes.sort_values(["ImageID", "cls", "area"], ascending=[True, True, False])
                  .groupby(["ImageID", "cls"]).head(OI_MAX_PER_IMG))
    classes = ["Airplane", "Boat", "Car", "Truck"]

    def loader(img_id: str, x0: float, x1: float, y0: float, y1: float) -> Callable[[], Image.Image]:
        def load() -> Image.Image:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(Image.open(oi / "images" / "train" / f"{img_id}.jpg"))
            w, h = img.size
            bw, bh = (x1 - x0) * w, (y1 - y0) * h
            if min(bw, bh) < OI_MIN_SIDE:
                raise Reject("box_thin")
            side = min(max(bw, bh) * 1.1, w, h)  # square around the box, 10% context
            if side < SIZE:
                raise Reject("too_small")
            cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
            left = min(max(cx - side / 2, 0), w - side)
            top = min(max(cy - side / 2, 0), h - side)
            return img.crop((round(left), round(top), round(left + side), round(top + side)))
        return load

    # split by photo: shuffle image ids, then 60/20/20
    ids = sorted(boxes.ImageID.unique())
    random.Random(seed).shuffle(ids)
    n = len(ids)
    split_of = {i: ("train" if k < 0.6 * n else "val" if k < 0.8 * n else "test") for k, i in enumerate(ids)}

    per_split = {s: [] for s in SPLITS}
    for r in boxes.itertuples(index=False):
        ref = f"{r.ImageID}:{r.cls}:{r.XMin:.4f},{r.YMin:.4f},{r.XMax:.4f},{r.YMax:.4f}"
        per_split[split_of[r.ImageID]].append(
            Candidate(r.cls, ref, r.ImageID, loader(r.ImageID, r.XMin, r.XMax, r.YMin, r.YMax)))
    splits = {s: _by_class(per_split[s], classes, seed) for s in SPLITS}
    return Source(
        "vehicle", classes, splits,
        "Open Images V7 train-split boxes (IsDepiction=0, IsGroupOf=0, IsInside=0), square crop "
        "around the box with 10% context, crop side >= 80 px, <= 2 boxes per class per photo. "
        "Split by photo (ImageID), 60/20/20, so crops from one photo never cross splits.",
    )


# --------------------------------------------------------------------------- Places365

PLACES_CLASSES = {"desert/sand": "desert_sand", "forest/broadleaf": "forest_broadleaf",
                  "mountain": "mountain", "ocean": "ocean"}


def extract_places() -> Path:
    """Stream the 26 GB train tar once and keep only our classes."""
    out = RAW / "places365" / "extracted"
    done = out / ".done"
    if done.exists():
        return out
    tar_path = RAW / "places365" / "train_256_places365standard.tar"
    if not tar_path.exists():
        raise FileNotFoundError(f"{tar_path} not downloaded yet")
    prefixes = {f"data_256/{k[0]}/{k}/": v for k, v in PLACES_CLASSES.items()}
    with open(tar_path, "rb") as f, tarfile.open(fileobj=f, mode="r|") as t:
        for m in t:
            for p, name in prefixes.items():
                if m.isfile() and m.name.startswith(p):
                    dest = out / name / m.name[len(p):]
                    if not dest.resolve().is_relative_to((out / name).resolve()):  # "../" in a member name
                        raise ValueError(f"{tar_path.name}: member {m.name!r} points outside {out / name}")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(t.extractfile(m).read())
    done.write_text("ok")
    return out


def places365(seed: int) -> Source:
    out = extract_places()
    classes = list(PLACES_CLASSES.values())
    cands = [Candidate(c, f"{c}/{p.name}", f"{c}/{p.name}", lambda p=p: Image.open(p))
             for c in classes for p in (out / c).glob("*.jpg")]
    by_cls = _by_class(cands, classes, seed)
    splits = {s: {} for s in SPLITS}
    for c in classes:
        pool = by_cls[c]
        splits["test"][c], splits["val"][c], splits["train"][c] = pool[:650], pool[650:1300], pool[1300:]
    return Source(
        "scene", classes, splits,
        "Places365-Standard train (256px). Random per-class split; the source has no group info, "
        "near-duplicates are removed by perceptual hash.",
    )


ADAPTERS = {"medical": pathmnist, "animal": afhq, "vehicle": openimages, "scene": places365}
