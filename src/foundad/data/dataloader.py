import multiprocessing as mp
import os
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

from .dataset import EvaluateDataset, TrainDataset


def _get_mp_context(num_workers: int) -> Any:
    if num_workers > 0 and os.name == "posix" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return None


def get_train_loader(
    *,
    dataset: TrainDataset,
    batch_size: int,
    batch_per_epoch: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int | None = None,
) -> DataLoader[torch.Tensor]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    num_samples = batch_size * batch_per_epoch
    sampler = RandomSampler(
        dataset,
        replacement=num_samples > len(dataset),
        num_samples=num_samples,
        generator=generator,
    )

    ctx = _get_mp_context(num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        generator=generator,
        multiprocessing_context=ctx,
    )


def get_inference_loader(
    *,
    dataset: EvaluateDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader[tuple[str, torch.Tensor]]:
    ctx = _get_mp_context(num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        multiprocessing_context=ctx,
    )
