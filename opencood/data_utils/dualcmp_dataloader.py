"""DDP-aware DataLoader for independent DualCMP clips."""

from __future__ import annotations

import random
from functools import partial
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int) -> None:
    worker_seed = int(seed) + int(rank) * int(num_workers) + int(worker_id)
    _seed_rng(worker_seed)


def _seed_rng(seed: int) -> None:
    """Seed all random sources used by DualCMP data loading."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataloader(dataset, batch_size: int, distributed: bool, *, workers: int = 4,
                     rank: int = 0, world_size: int = 1, seed: int = 0,
                     training: bool = True, collate_fn=None, drop_last: Optional[bool] = None,
                     pin_memory: bool = True, prefetch_factor: Optional[int] = 2):
    """Build a loader where ``batch_size`` means independent clips per GPU.

    ``drop_last`` defaults to ``training`` so distributed training drops both
    sampler padding and incomplete loader batches. Evaluation defaults to
    padding in ``DistributedSampler``; a later metric adapter must deduplicate
    scene/receiver/frame records, and this helper does not claim otherwise.
    With ``workers == 0``, the calling process's torch, NumPy, and Python RNGs
    are seeded with ``seed + rank`` as a deterministic contract.
    """
    if batch_size < 1 or workers < 0: raise ValueError("batch_size/workers are invalid")
    if distributed and (world_size < 1 or not 0 <= rank < world_size): raise ValueError("invalid distributed rank/world_size")
    resolved_drop_last = bool(training) if drop_last is None else bool(drop_last)
    main_seed = int(seed) + int(rank)
    if workers == 0:
        _seed_rng(main_seed)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=training, seed=seed, drop_last=resolved_drop_last) if distributed else None
    generator = torch.Generator().manual_seed(main_seed)
    kwargs = {"dataset": dataset, "batch_size": batch_size, "shuffle": sampler is None and training,
              "sampler": sampler, "drop_last": resolved_drop_last, "num_workers": workers,
              "pin_memory": pin_memory, "persistent_workers": workers > 0, "collate_fn": collate_fn,
              "worker_init_fn": partial(worker_init_fn, num_workers=workers, rank=rank, seed=seed) if workers else None,
              "generator": generator}
    if workers and prefetch_factor is not None: kwargs["prefetch_factor"] = int(prefetch_factor)
    return dataset, DataLoader(**kwargs), sampler
