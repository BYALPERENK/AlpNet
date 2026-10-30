"""Download the raw source files for alpnet15 (Config A) into data/raw/<source>/.

Resumable (HTTP Range) and parallel. Re-running skips files that are already complete.

    python scripts/download_raw.py            # everything
    python scripts/download_raw.py pathmnist  # only some sources
"""

import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FILES = {
    "pathmnist": [
        ("https://zenodo.org/records/10519652/files/pathmnist_128.npz?download=1", "pathmnist_128.npz"),
    ],
    "afhq": [
        ("https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1", "afhq_v2.zip"),
    ],
    "places365": [
        ("http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar", "filelist_places365-standard.tar"),
        ("http://data.csail.mit.edu/places/places365/val_256.tar", "val_256.tar"),
        ("http://data.csail.mit.edu/places/places365/train_256_places365standard.tar", "train_256_places365standard.tar"),
    ],
    "openimages": [
        ("https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv", "oidv7-class-descriptions-boxable.csv"),
        ("https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv", "validation-annotations-bbox.csv"),
        ("https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv", "test-annotations-bbox.csv"),
        ("https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv", "oidv6-train-annotations-bbox.csv"),
    ],
}

CHUNK = 1 << 20
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    part = dest.with_name(dest.name + ".part")
    if dest.exists() and (total is None or dest.stat().st_size == total):
        log(f"skip  {dest.name} (complete)")
        return

    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:  # server ignored Range: start over
            have = 0
        mode = "ab" if have else "wb"
        log(f"start {dest.name} ({(total or 0) / 1e9:.2f} GB, resume at {have / 1e9:.2f} GB)")
        next_report = time.monotonic() + 60
        with open(part, mode) as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                have += len(chunk)
                if time.monotonic() >= next_report:
                    pct = f"{100 * have / total:.1f}%" if total else "?"
                    log(f"      {dest.name}: {have / 1e9:.2f} GB ({pct})")
                    next_report += 60

    if total is not None and have != total:
        raise RuntimeError(f"{dest.name}: got {have} bytes, expected {total}")
    part.replace(dest)
    log(f"done  {dest.name}")


def main() -> None:
    sources = sys.argv[1:] or list(FILES)
    jobs = [(url, RAW / src / name) for src in sources for url, name in FILES[src]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, url, dest): dest for url, dest in jobs}
        failed = []
        for fut, dest in futures.items():
            try:
                fut.result()
            except Exception as e:  # report and keep the other downloads going
                failed.append(dest.name)
                log(f"FAIL  {dest.name}: {type(e).__name__}: {e}")
    log("all done" if not failed else f"finished with failures: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
