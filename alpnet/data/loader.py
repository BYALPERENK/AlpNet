"""GPU-resident data loading for alpnet15.

Each split lives on the GPU as uint8 [N, 80, 80]. Batches are cropped to 64x64 on the GPU
(random offsets + horizontal flip for training, center crop for evaluation), converted to float and normalized
with the train mean/std. No DataLoader, no host<->device copies per step.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .sources import ROOT

PROCESSED = ROOT / "data" / "processed"
CROP = 64


def dataset_id(name: str) -> str:
    """Name and content hash (build_dataset.py, over the arrays): a rebuild with other data under the same
    name gives another id, so the search cache does not mix the two."""
    meta = json.loads((PROCESSED / f"{name}_metadata.json").read_text(encoding="utf-8"))
    return f"{name}@{meta['content_sha256'][:16]}"


class GpuSplit:
    def __init__(self, name: str, split: str, device: str = "cuda", train_aug: bool | None = None):
        meta = json.loads((PROCESSED / f"{name}_metadata.json").read_text(encoding="utf-8"))
        self.dataset = dataset_id(name)
        data = np.load(PROCESSED / f"{name}.npz")
        self.x = torch.from_numpy(data[f"x_{split}"]).to(device)          # uint8 [N, S, S]
        self.y = torch.from_numpy(data[f"y_{split}"]).to(device)          # int64 [N]
        self.src = torch.from_numpy(data[f"src_{split}"]).to(device)      # int64 [N], group index
        self.classes: list[str] = meta["classes"]
        self.groups: list[str] = meta["groups"]
        self.mean, self.std = meta["train_mean"], meta["train_std"]
        self.train_aug = split == "train" if train_aug is None else train_aug
        self.device = device
        self.size = self.x.shape[-1]
        self._ar = torch.arange(CROP, device=device)

    def __len__(self) -> int:
        return len(self.y)

    def _crop(self, idx: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        b, margin = len(idx), self.size - CROP
        if self.train_aug:
            oy = torch.randint(0, margin + 1, (b,), device=self.device, generator=generator)
            ox = torch.randint(0, margin + 1, (b,), device=self.device, generator=generator)
        else:
            oy = ox = torch.full((b,), margin // 2, device=self.device)
        rows = (oy[:, None] + self._ar)[:, :, None]    # [B, 64, 1]
        cols = (ox[:, None] + self._ar)[:, None, :]    # [B, 1, 64]
        if self.train_aug:  # horizontal flip: every Config A class is mirror-invariant
            flip = torch.rand(b, device=self.device, generator=generator) < 0.5
            cols = torch.where(flip[:, None, None], cols.flip(-1), cols)
        return self.x[idx[:, None, None], rows, cols]  # [B, 64, 64] uint8

    def batches(self, batch_size: int, shuffle: bool | None = None, generator: torch.Generator | None = None,
                drop_last: bool | None = None):
        """Yield (x float [B,1,64,64] channels_last, y, src). Shuffles and drops the last partial
        batch for training by default."""
        shuffle = self.train_aug if shuffle is None else shuffle
        drop_last = self.train_aug if drop_last is None else drop_last
        n = len(self)
        order = (torch.randperm(n, device=self.device, generator=generator) if shuffle
                 else torch.arange(n, device=self.device))
        stop = n - n % batch_size if drop_last else n
        for i in range(0, stop, batch_size):
            yield self.take(order[i:i + batch_size], generator)

    def take(self, idx: torch.Tensor, generator: torch.Generator | None = None):
        """(x float [B,1,64,64] channels_last, y, src) of the samples `idx`."""
        x = self._crop(idx, generator).unsqueeze(1).float().div_(255.0)
        x = x.sub_(self.mean).div_(self.std).contiguous(memory_format=torch.channels_last)
        return x, self.y[idx], self.src[idx]


def load_splits(name: str = "alpnet15_v1", device: str = "cuda") -> dict[str, GpuSplit]:
    return {s: GpuSplit(name, s, device) for s in ("train", "val", "test")}
