from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW

from foundad.data.dataloader import get_train_loader
from foundad.data.dataset import TrainDataset
from foundad.modules.cutpaste import CutPasteModel, CutPasteRectBatch
from foundad.types import InferenceResult

from .encoder import Encoder
from .projector import Projector


class FoundAD(nn.Module):
    """FoundAD foundation-model anomaly detection model."""

    def __init__(
        self,
        encoder: Encoder,
        projector: Projector,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def process_image(
        self,
        image_input: Image.Image | list[Image.Image] | Tensor | list[Tensor],
    ) -> Tensor:
        return self.encoder.process_image(image_input)

    def forward(self, X: Tensor, Z: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            X_en = self.encoder(X)
            Z_en = self.encoder(Z)

        Z_proj = self.projector(Z_en)

        return X_en, Z_proj

    def train(self, mode: bool = True) -> FoundAD:
        """Sets the model in training or evaluation mode.

        Overrides `nn.Module.train`. The encoder always remains in eval mode
        with gradients disabled. The projector is toggled according to `mode`.
        """
        super().train(mode)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        if mode:
            self.projector.train()
        else:
            self.projector.eval()
        return self

    def train_model(
        self,
        dataset_dir: str | Path | None = None,
        *,
        images: Iterable[Image.Image] | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        batches: int = 16,
        learning_rate: float = 1e-4,
        device: str | torch.device | None = None,
        output_dir: str | Path | None = None,
        cutpaste: CutPasteModel | None = None,
        criterion: nn.Module | None = None,
    ) -> FoundAD:
        """Trains the projector using CutPaste self-supervision on clean images.

        Args:
            dataset_dir: Directory containing clean training images.
            images: Alternatively, an iterable of pre-loaded PIL Images.
            epochs: Number of training epochs.
            batch_size: Batch size per iteration.
            batches: Number of batches per epoch.
            learning_rate: Learning rate for AdamW optimizer.
            device: Training device ('cuda' or 'cpu'). Auto-detected if None.
            output_dir: Directory to save checkpoints, logs, and config.
            cutpaste: CutPaste augmentation model. Defaults to CutPasteRectBatch.
            criterion: Loss criterion. Defaults to nn.MSELoss().

        Returns:
            Trained FoundAD model instance.
        """
        from foundad.engines.trainer import FoundADTrainer

        # Resolve device
        if device is None:
            train_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            train_device = torch.device(device)

        self.to(train_device)

        # Load images
        loaded_images: list[Image.Image]
        image_paths: list[str] = []
        if dataset_dir is not None:
            directory = Path(dataset_dir)
            if not directory.is_dir():
                raise NotADirectoryError(f"Directory not found: {directory}")

            supported_extensions = list(Image.registered_extensions().keys())
            matched_files = [
                f
                for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
            if not matched_files:
                raise FileNotFoundError(
                    f"No supported image files found in {directory}."
                )
            image_paths = [str(f.resolve()) for f in matched_files]
            loaded_images = [Image.open(p) for p in matched_files]
        elif images is not None:
            loaded_images = list(images)
        else:
            raise ValueError("Either `dataset_dir` or `images` must be provided for training.")

        dataset = TrainDataset(loaded_images, self.process_image)
        train_loader = get_train_loader(
            dataset=dataset,
            batch_size=batch_size,
            batch_per_epoch=batches,
        )

        # Build config dictionary for tracking
        config = {
            "model": {
                "encoder": self.encoder.get_config(),
                "projector": self.projector.get_config(),
            },
            "data": {
                "path": str(Path(dataset_dir).resolve()) if dataset_dir else None,
                "total_samples": len(loaded_images),
                "image_paths": image_paths,
                "batch_size": batch_size,
                "batch_per_epoch": batches,
            },
            "training": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "loss": criterion.__class__.__name__ if criterion else "MSELoss",
                "optimizer": "AdamW",
                "device": str(train_device),
            },
        }

        trainer = FoundADTrainer(
            model=self,
            optimizer_factory=lambda params: AdamW(params, lr=learning_rate),
            criterion=criterion if criterion is not None else nn.MSELoss(),
            cutpaste=cutpaste if cutpaste is not None else CutPasteRectBatch(),
            device=train_device,
            output_dir=output_dir,
            config=config,
        )

        trainer.run(train_loader, max_epochs=epochs)
        return self

    def save(self, path: str | Path) -> None:
        """Saves only the projector weights and architecture metadata.

        Encoder weights are not saved here; they are loaded dynamically via transformers.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        projector_cfg = self.projector.get_config()
        encoder_cfg = self.encoder.get_config()

        checkpoint_payload = {
            "projector_type": self.projector.__class__.__name__,
            "projector_config": projector_cfg,
            "projector_state_dict": self.projector.state_dict(),
            "encoder_type": self.encoder.__class__.__name__,
            "encoder_config": encoder_cfg,
            "config": {
                "projector": projector_cfg,
                "encoder": encoder_cfg,
            },
        }
        torch.save(checkpoint_payload, save_path)

    def load(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> FoundAD:
        """Loads trained projector weights from a checkpoint.

        Raises:
            ValueError: If the checkpoint was saved with a different projector architecture
                or conflicting hyperparameters.
            RuntimeError: If layer shapes or tensor weights mismatch.
        """
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        if device is None:
            # Match current device of model
            try:
                load_device = next(self.projector.parameters()).device
            except StopIteration:
                load_device = torch.device("cpu")
        else:
            load_device = torch.device(device)

        checkpoint = torch.load(checkpoint_file, map_location=load_device, weights_only=False)

        # 1. Architecture type validation
        saved_projector_config = checkpoint.get("projector_config", {})
        saved_projector_type = (
            checkpoint.get("projector_type")
            or saved_projector_config.get("type")
        )
        current_projector_type = self.projector.__class__.__name__

        if saved_projector_type is not None and saved_projector_type != current_projector_type:
            raise ValueError(
                f"Architecture mismatch: Checkpoint projector was trained with '{saved_projector_type}', "
                f"but current model uses '{current_projector_type}'."
            )

        # 2. Hyperparameter validation (e.g. embed_dim, num_heads, kernel_size)
        current_config = self.projector.get_config()
        for key, saved_value in saved_projector_config.items():
            if key in current_config and current_config[key] != saved_value:
                raise ValueError(
                    f"Configuration mismatch for '{key}': checkpoint has {saved_value}, "
                    f"but current projector has {current_config[key]}."
                )

        # 3. Extract projector state dict
        if "projector_state_dict" in checkpoint:
            state_dict = checkpoint["projector_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Strict weight loading will fail on mismatched layers/dimensions
        self.projector.load_state_dict(state_dict, strict=True)
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> FoundAD:
        """Reconstructs encoder, projector, and weights directly from a checkpoint or run directory."""
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.is_dir():
            best_pt = checkpoint_file / "checkpoints" / "best.pt"
            last_pt = checkpoint_file / "checkpoints" / "last.pt"
            if best_pt.is_file():
                checkpoint_file = best_pt
            elif last_pt.is_file():
                checkpoint_file = last_pt
            else:
                raise FileNotFoundError(f"No checkpoint file found in directory: {checkpoint_path}")

        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        proj_cfg = (
            checkpoint.get("projector_config")
            or cfg.get("projector")
            or cfg.get("model", {}).get("projector")
        )
        enc_cfg = (
            checkpoint.get("encoder_config")
            or cfg.get("encoder")
            or cfg.get("model", {}).get("encoder")
        )

        if proj_cfg is None:
            raise ValueError(
                f"No projector configuration found in checkpoint: {checkpoint_file}"
            )

        from foundad.modules.encoder.impl import DinoV2Encoder
        from foundad.modules.projector.impl import TransformerProjector

        projector_type = proj_cfg.get("type", "TransformerProjector")
        if projector_type == "TransformerProjector":
            projector = TransformerProjector.from_config(proj_cfg)
        else:
            raise ValueError(f"Unsupported projector type '{projector_type}' in checkpoint.")

        if enc_cfg is not None and enc_cfg.get("type") == "DinoV2Encoder":
            encoder = DinoV2Encoder.from_config(enc_cfg)
        else:
            encoder = DinoV2Encoder("small")

        model = cls(encoder=encoder, projector=projector)
        model.load(checkpoint_file, device=device)
        return model

    @torch.no_grad()
    def predict(
        self,
        images: Image.Image | list[Image.Image] | str | Path | list[str | Path] | Tensor,
        metric: Literal["cosine", "l2"] = "cosine",
        device: str | torch.device | None = None,
    ) -> InferenceResult:
        """Runs anomaly inference and generates pixel anomaly heatmaps and image scores.

        Args:
            images: Image path, list of image paths, PIL Image, list of PIL Images, or preprocessed Tensor.
            metric: Anomaly metric ('cosine' or 'l2'). Defaults to 'cosine'.
            device: Evaluation device. Defaults to current model device.

        Returns:
            InferenceResult with anomaly_map, anomaly_score, and patch_anomaly.
        """
        if device is not None:
            self.to(torch.device(device))

        infer_device = next(self.projector.parameters()).device
        self.eval()

        # Handle file paths and image formats
        if isinstance(images, (str, Path)):
            pil_images = [Image.open(images).convert("RGB")]
            x = self.process_image(pil_images).to(infer_device)
        elif isinstance(images, list):
            pil_images: list[Image.Image] = []
            for item in images:
                if isinstance(item, (str, Path)):
                    pil_images.append(Image.open(item).convert("RGB"))
                elif isinstance(item, Image.Image):
                    pil_images.append(item)
                else:
                    raise TypeError(f"Unsupported item in images list: {type(item)}")
            x = self.process_image(pil_images).to(infer_device)
        elif isinstance(images, Image.Image):
            x = self.process_image(images).to(infer_device)
        elif isinstance(images, Tensor):
            x = images.to(infer_device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image input type: {type(images)}")

        B, _, H, W = x.shape

        # DinoV2 patch embeddings
        x_en = self.encoder(x)
        z_proj = self.projector(x_en)

        # Distance metric calculation
        if metric == "cosine":
            similarity = F.cosine_similarity(z_proj, x_en, dim=-1)
            patch_anomaly = 1.0 - similarity
        elif metric == "l2":
            patch_anomaly = torch.norm(z_proj - x_en, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'cosine' or 'l2'.")

        L = patch_anomaly.shape[1]
        grid_size = int(L**0.5)
        if grid_size * grid_size != L:
            raise ValueError(f"Token length {L} cannot form a square spatial grid.")

        patch_grid = patch_anomaly.view(B, 1, grid_size, grid_size)

        # Upsample anomaly heatmap to input image resolution
        anomaly_map = F.interpolate(
            patch_grid,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        anomaly_score = patch_anomaly.amax(dim=-1)

        return InferenceResult(
            anomaly_map=anomaly_map,
            anomaly_score=anomaly_score,
            patch_anomaly=patch_anomaly,
        )
