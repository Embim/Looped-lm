"""Flat-memmap token streams with an exact, auditable token budget.

Training windows are non-overlapping (stride == seq_len) and are visited in a
seeded permutation, so a run of `total_tokens` consumes exactly that many
*distinct* target tokens -- no repeats, no epoch bookkeeping.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch


class TokenStream:
    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int,
        batch_size: int,
        device: str = "cuda",
        seed: int = 0,
        shuffle: bool = True,
        max_windows: Optional[int] = None,
    ):
        self.path = Path(bin_path)
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        n = len(self.data)
        starts = np.arange(0, n - seq_len - 1, seq_len, dtype=np.int64)
        if shuffle:
            starts = starts[np.random.default_rng(seed).permutation(len(starts))]
        if max_windows is not None:
            starts = starts[:max_windows]
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts) // self.batch_size

    @property
    def n_tokens(self) -> int:
        return len(self) * self.batch_size * self.seq_len

    def batches(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        S, B = self.seq_len, self.batch_size
        for i in range(len(self)):
            sl = self.starts[i * B:(i + 1) * B]
            buf = np.stack([self.data[s:s + S + 1] for s in sl]).astype(np.int64)
            t = torch.from_numpy(buf)
            t = t.pin_memory() if self.device.startswith("cuda") else t
            t = t.to(self.device, non_blocking=True)
            yield t[:, :-1], t[:, 1:]


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text())


def bpb_from_nll(nll_nats: float, bytes_per_token: float) -> float:
    """Bits per byte -- comparable across tokenizers, unlike token perplexity."""
    return nll_nats / (math.log(2.0) * bytes_per_token)
