import math

import torch
from torch import nn
from torchvision.transforms import v2

from foundad.types import ValueRange


class CutPasteModel(nn.Module):
    pass

class CutPasteRectBatch(CutPasteModel):
    """Applies CutPaste augmentation to batched image tensors.

    Cuts a rectangular patch from an image and pastes it at a random coordinate
    within the same image (or across different images in the batch). This
    simulates local structural defects for self-supervised anomaly detection.

    Args:
        area_ratio: Range defining lower and upper bounds for the patch area
            relative to the total image area. Defaults to
            `ValueRange(0.02, 0.15)`.
        aspect_ratio: Range defining lower and upper bounds for the patch aspect
            ratio (width / height). Defaults to `ValueRange(0.3, 3.3)`.
        cross_batch: If True, pastes patches across different samples in the
            batch (CutMix-style). If False, pastes the patch back onto the same
            source image. Defaults to False.
        color_jitter: If True, applies random brightness, contrast, and
            saturation jitter to the extracted patch before pasting. Defaults to
            True.
    """

    def __init__(
        self,
        area_ratio: ValueRange[float] | None = None,
        aspect_ratio: ValueRange[float] | None = None,
        cross_batch: bool = False,
        color_jitter: bool = True,
    ):
        super().__init__()

        if area_ratio is None:
            self.area_ratio = ValueRange(0.02, 0.15)
        else:
            self.area_ratio = area_ratio

        if aspect_ratio is None:
            self.aspect_ratio = ValueRange(0.3, 3.3)
        else:
            self.aspect_ratio = aspect_ratio

        self.cross_batch = cross_batch

        self.color_jitter = (
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            if color_jitter
            else None
        )

    def _sample_patch_size(
        self, H: int, W: int, device: torch.device
    ) -> tuple[float, float]:
        """Samples random patch dimensions constrained by area and aspect ratio."""
        img_area = H * W
        target_area = (
            torch.empty(1, device=device)
            .uniform_(self.area_ratio.lower, self.area_ratio.upper)
            .item()
        ) * img_area

        aspect = (
            torch.empty(1, device=device)
            .uniform_(self.aspect_ratio.lower, self.aspect_ratio.upper)
            .item()
        )

        patch_w = round(math.sqrt(target_area * aspect))
        patch_h = round(math.sqrt(target_area / aspect))

        patch_w = max(1, min(patch_w, W - 1))
        patch_h = max(1, min(patch_h, H - 1))
        return patch_h, patch_w

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies CutPaste to an input image batch.

        Args:
            x: Input tensor of shape `(B, C, H, W)`.

        Returns:
            Augmented tensor of shape `(B, C, H, W)`.
        """
        B, _, H, W = x.shape
        out = x.clone()

        if self.cross_batch and B > 1:
            dest_indices = torch.randperm(B, device=x.device)
        else:
            dest_indices = torch.arange(B, device=x.device)

        for src_idx, dst_idx in enumerate(dest_indices):
            ph, pw = self._sample_patch_size(H, W, x.device)

            from_y = torch.randint(0, int(H - ph + 1), (1,), device=x.device).item()
            from_x = torch.randint(0, int(W - pw + 1), (1,), device=x.device).item()

            patch = x[src_idx, :, from_y : from_y + ph, from_x : from_x + pw]

            if self.color_jitter is not None:
                patch = self.color_jitter(patch)

            to_y = torch.randint(0, int(H - ph + 1), (1,), device=x.device).item()
            to_x = torch.randint(0, int(W - pw + 1), (1,), device=x.device).item()

            out[dst_idx, :, to_y : to_y + ph, to_x : to_x + pw] = patch

        return out


class CutPasteScarBatch(CutPasteRectBatch):
    """Applies the CutPaste-Scar augmentation variant to batched image tensors.

    Specialized for simulating thin, elongated defects such as scratches or cracks
    by cutting narrow strips and pasting them back onto the image.

    Args:
        width_range: Inclusive pixel bounds for the scar width. Defaults to
            `ValueRange(2, 16)`.
        length_range: Inclusive pixel bounds for the scar length. Defaults to
            `ValueRange(10, 40)`.
        rotation_range: Range of rotation angles (in degrees) to apply to the
            scar patch. Defaults to `ValueRange(-45.0, 45.0)`.
        cross_batch: If True, pastes scars across different samples in the
            batch. If False, pastes onto the same image. Defaults to False.
    """

    def __init__(
        self,
        width_range: ValueRange[int] | None = None,
        length_range: ValueRange[int] | None = None,
        rotation_range: ValueRange[float] | None = None,
        cross_batch: bool = False,
    ):
        super().__init__(cross_batch=cross_batch, color_jitter=False)
        if width_range is None:
            self.width_range = ValueRange(2, 16)
        else:
            self.width_range = width_range

        if length_range is None:
            self.length_range = ValueRange(10, 40)
        else:
            self.length_range = length_range

        if rotation_range is None:
            self.rotation_range = ValueRange(-45.0, 45.0)
        else:
            self.rotation_range = rotation_range

    def _sample_patch_size(
        self, H: int, W: int, device: torch.device
    ) -> tuple[float, float]:
        """Samples random scar dimensions in pixels."""
        pw = torch.randint(
            self.width_range.lower,
            self.width_range.upper + 1,
            (1,),
            device=device,
        ).item()

        ph = torch.randint(
            self.length_range.lower,
            self.length_range.upper + 1,
            (1,),
            device=device,
        ).item()

        return min(ph, H - 1), min(pw, W - 1)
