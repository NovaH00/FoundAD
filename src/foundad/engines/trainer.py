from collections.abc import Callable, Iterable, Iterator

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from torch import Tensor, nn
from torch.optim import Optimizer

from foundad.modules import FoundAD
from foundad.modules.cutpaste import CutPasteModel


class FoundADTrainer:
    def __init__(
        self,
        model: FoundAD,
        optimizer_factory: Callable[[Iterator[nn.Parameter]], Optimizer],
        criterion: nn.Module,
        cutpaste: CutPasteModel,
        device: torch.device
    ):
        self.model     = model
        self.model.to(device)
        self.optimizer = optimizer_factory(model.parameters())
        self.criterion = criterion
        self.device    = device
        self.cutpaste  = cutpaste
        self.cutpaste.to(device)

        self.engine = Engine(self.train_step)

        pbar = ProgressBar(persist=True)
        pbar.attach(self.engine, output_transform=lambda loss: {"loss": loss})

    def train_step(
        self,
        _engine: Engine,
        batch: Tensor
    ):
        self.model.train()

        self.optimizer.zero_grad()

        X = batch.to(self.device)

        Z = self.cutpaste(X)
        X_en, Z_proj = self.model.forward(X, Z)

        loss = self.criterion(Z_proj, X_en)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def run(
        self,
        train_loader: Iterable[Tensor],
        max_epochs: int
    ):
        self.engine.run(train_loader, max_epochs)
