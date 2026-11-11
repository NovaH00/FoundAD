from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

"""
The reason why we split TrainDataset and EvaluateDataset is because the model
will be trained using CutPaste implementation, and we can't reliable evaluate
metrics like accuracy, recall, ... with CutPaste since the anomaly images are synthetized.

So to evaluate we need a real anomaly dataset.
"""

class TrainDataset(Dataset):
    def __init__(
        self,
        images: Iterable[Image.Image],
        transform: Callable[[Image.Image], torch.Tensor],
    ):
        """
        Since FoundAD will be trained with a relatively small "clean" (i.e. no anomalies) images,
        we can load all the images into memory and use it as the dataset, no need to read from disk
        lazily.
        """
        self.images = list(images)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.images[index]

        return self.transform(image).squeeze(0)


class EvaluateDataset(Dataset):
    def __init__(
        self,
        image_paths: Iterable[Path],
        transform: Callable[[Image.Image], torch.Tensor]
    ):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")

        return self.transform(image)


