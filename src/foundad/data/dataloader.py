import torch
from torch.utils.data import DataLoader, RandomSampler

from .dataset import TrainDataset


def get_train_loader(
    *,
    dataset: TrainDataset,
    batch_size: int,
    batch_per_epoch: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int | None = None
) -> DataLoader[torch.Tensor]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    num_samples = batch_size*batch_per_epoch
    sampler = RandomSampler(
        dataset,
        replacement=num_samples > len(dataset), # Only replacement (i.e. repeated samples) if the dataset is smaller
        num_samples=num_samples,
        generator=generator
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        generator=generator
    )
