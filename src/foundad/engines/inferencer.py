import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from foundad.modules import FoundAD
from foundad.modules.encoder.impl import DinoV2Encoder
from foundad.modules.projector.impl import TransformerProjector
from foundad.types import InferenceResult


class FoundADInferencer:
    """General interface for FoundAD inference and anomaly evaluation."""

    def __init__(
        self,
        model: FoundAD,
        device: torch.device,
        config: dict | None = None,
    ):
        self.model = model
        self.device = device
        self.config = config or {}

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        checkpoint_name: str = "best.pt",
        device: torch.device | str | None = None,
    ) -> FoundADInferencer:
        """Loads inferencer from an experiment run directory containing config and checkpoints."""
        run_path = Path(run_dir)
        config_path = run_path / "config.json"
        checkpoint_path = run_path / "checkpoints" / checkpoint_name

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: dict | None = None,
        device: torch.device | str | None = None,
    ) -> FoundADInferencer:
        """Loads inferencer directly from a checkpoint file, using either provided or embedded config."""
        if device is None:
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            selected_device = torch.device(device)

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=selected_device, weights_only=False)

        cfg = config or checkpoint.get("config")
        if cfg is None:
            raise ValueError(
                f"No configuration provided and none embedded in checkpoint: {checkpoint_path}"
            )

        # Build projector from config
        proj_cfg = cfg.get("model", {}).get("projector", {})
        projector = TransformerProjector(
            embed_dim=proj_cfg.get("embed_dim", 384),
            num_heads=proj_cfg.get("num_heads", 12),
            num_blocks=proj_cfg.get("num_blocks", 6),
            mlp_ratio=proj_cfg.get("mlp_ratio", 4),
        )

        # Build encoder from config
        encoder = DinoV2Encoder()
        model = FoundAD(encoder=encoder, projector=projector)

        # Load weights
        if "projector_state_dict" in checkpoint:
            model.projector.load_state_dict(checkpoint["projector_state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.projector.load_state_dict(checkpoint)

        return cls(model=model, device=selected_device, config=cfg)

    @torch.no_grad()
    def predict(
        self,
        images: Image.Image | list[Image.Image] | Tensor,
        metric: Literal["cosine", "l2"] = "cosine",
    ) -> InferenceResult:
        """Runs anomaly detection on input image(s).

        Args:
            images: Single PIL image, list of PIL images, or preprocessed (B, C, H, W) Tensor.
            metric: Distance metric to compare projected features with encoded features
                ('cosine' or 'l2'). Defaults to 'cosine'.

        Returns:
            InferenceResult containing anomaly map, anomaly score, and patch scores.
        """
        if isinstance(images, (Image.Image, list)):
            x = self.model.process_image(images).to(self.device)
        elif isinstance(images, Tensor):
            x = images.to(self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image input type: {type(images)}")

        B, _, H, W = x.shape

        # Extract features and project
        x_en = self.model.encoder(x)
        z_proj = self.model.projector(x_en)

        # Compute token-level anomaly score
        if metric == "cosine":
            similarity = F.cosine_similarity(z_proj, x_en, dim=-1)  # (B, L)
            patch_anomaly = 1.0 - similarity                        # (B, L)
        elif metric == "l2":
            patch_anomaly = torch.norm(z_proj - x_en, p=2, dim=-1)  # (B, L)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'cosine' or 'l2'.")

        # Reshape tokens into spatial feature grid
        L = patch_anomaly.shape[1]
        grid_size = int(L**0.5)
        if grid_size * grid_size != L:
            raise ValueError(f"Token length {L} cannot form a square spatial grid.")

        patch_grid = patch_anomaly.view(B, 1, grid_size, grid_size)

        # Upsample anomaly map to input resolution
        anomaly_map = F.interpolate(
            patch_grid,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Image-level score is the maximum patch score
        anomaly_score = patch_anomaly.amax(dim=-1)

        return InferenceResult(
            anomaly_map=anomaly_map,
            anomaly_score=anomaly_score,
            patch_anomaly=patch_anomaly,
        )
