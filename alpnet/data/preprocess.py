"""Shared image preprocessing: grayscale, square crop, downscale-only resize, perceptual hash."""

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

SIZE = 80  # stored resolution; training takes random 64x64 crops from it


class Reject(Exception):
    """Raised when a sample fails a quality rule; the message is the reason code."""


def to_gray_square(img: Image.Image, size: int = SIZE) -> np.ndarray:
    """Grayscale (ITU-R 601 luma), center square crop, Lanczos downscale to size x size uint8."""
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    img = img.convert("L")
    w, h = img.size
    side = min(w, h)
    if side < size:
        raise Reject("too_small")  # no upscaling, ever
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if side != size:
        img = img.resize((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.std() < 3.0:
        raise Reject("blank")
    return arr


def phash(arr: np.ndarray) -> np.uint64:
    """64-bit DCT perceptual hash of a grayscale image."""
    small = np.asarray(Image.fromarray(arr).resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = dctn(small, norm="ortho")[:8, :8].ravel()
    bits = low > np.median(low[1:])  # skip the DC term when picking the threshold
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def hamming(h: np.uint64, others: np.ndarray) -> np.ndarray:
    return np.bitwise_count(others ^ h)
