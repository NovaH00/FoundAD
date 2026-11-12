import json
import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from foundad.modules import FoundAD
from foundad.modules.cutpaste import CutPasteModel


class FoundADTrainer:
    def __init__(
        self,
        model: FoundAD,
        optimizer_factory: Callable[[Iterator[nn.Parameter]], Optimizer],
        criterion: nn.Module,
        cutpaste: CutPasteModel,
        device: torch.device,
        output_dir: str | Path | None = None,
        config: dict | None = None,
    ):
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer_factory(model.parameters())
        self.criterion = criterion
        self.device = device
        self.cutpaste = cutpaste
        self.cutpaste.to(device)

        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.checkpoints_dir: Path | None = None
        self.logs_dir: Path | None = None
        self.tb_writer: SummaryWriter | None = None
        self.logger = logging.getLogger("foundad.trainer")

        self.best_loss = float("inf")
        self._epoch_loss_sum = 0.0
        self._epoch_steps = 0

        if self.output_dir is not None:
            self._setup_output_dirs()

        self.engine = Engine(self.train_step)

        pbar = ProgressBar(persist=True)
        pbar.attach(self.engine, output_transform=lambda loss: {"loss": loss})

        if self.output_dir is not None:
            self._setup_events()

    def _setup_output_dirs(self) -> None:
        assert self.output_dir is not None
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Save config.json
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        # Setup file logger
        file_handler = logging.FileHandler(self.logs_dir / "train.log")
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        # Setup TensorBoard
        tb_dir = self.logs_dir / "tensorboard"
        self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

    def _setup_events(self) -> None:
        @self.engine.on(Events.EPOCH_STARTED)
        def _reset_epoch_metrics(_engine: Engine) -> None:
            self._epoch_loss_sum = 0.0
            self._epoch_steps = 0

        @self.engine.on(Events.ITERATION_COMPLETED)
        def _accumulate_loss(engine: Engine) -> None:
            loss = engine.state.output
            if isinstance(loss, (int, float)):
                self._epoch_loss_sum += loss
                self._epoch_steps += 1

        @self.engine.on(Events.EPOCH_COMPLETED)
        def _on_epoch_completed(engine: Engine) -> None:
            epoch = engine.state.epoch
            avg_loss = (
                self._epoch_loss_sum / self._epoch_steps
                if self._epoch_steps > 0
                else 0.0
            )

            # Log to file and console
            msg = f"Epoch {epoch:4d} | Average Loss: {avg_loss:.6f}"
            self.logger.info(msg)

            # Log to TensorBoard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss/train_epoch", avg_loss, epoch)

            # Checkpoint payload
            checkpoint_payload = {
                "epoch": epoch,
                "loss": avg_loss,
                "projector_type": self.model.projector.__class__.__name__,
                "projector_config": self.model.projector.get_config(),
                "projector_state_dict": self.model.projector.state_dict(),
                "encoder_type": self.model.encoder.__class__.__name__,
                "encoder_config": self.model.encoder.get_config(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            }

            assert self.checkpoints_dir is not None
            # Always save last.pt
            last_path = self.checkpoints_dir / "last.pt"
            torch.save(checkpoint_payload, last_path)

            # Save best.pt when loss improves
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                best_path = self.checkpoints_dir / "best.pt"
                torch.save(checkpoint_payload, best_path)
                self.logger.info(
                    f"New best model saved to {best_path} (Loss: {avg_loss:.6f})"
                )

        @self.engine.on(Events.COMPLETED)
        def _on_completed(_engine: Engine) -> None:
            if self.tb_writer is not None:
                self.tb_writer.close()
            self.logger.info("Training completed.")

    def train_step(
        self,
        _engine: Engine,
        batch: Tensor,
    ) -> float:
        self.model.train()

        self.optimizer.zero_grad()

        X = batch.to(self.device)

        Z = self.cutpaste(X)
        X_en, Z_proj = self.model.forward(X, Z)

        loss = self.criterion(Z_proj, X_en)
        loss.backward()

        self.optimizer.step()

        return float(loss.item())

    def run(
        self,
        train_loader: Iterable[Tensor],
        max_epochs: int,
    ) -> None:
        self.engine.run(train_loader, max_epochs)
