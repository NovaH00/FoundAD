from dataclasses import dataclass

from torch import Tensor


class ValueRange[T: int | float]:
    """Represents a numeric range with lower and upper bounds.

    The range is used to constrain numeric values within specified
    boundaries.

    Attributes:
        lower: The inclusive lower bound of the range.
        upper: The inclusive upper bound of the range.
    """

    def __init__(self, lower_bound: T, upper_bound: T):
        """Initialize a numeric range.

        Args:
            lower_bound: The lower boundary of the range.
            upper_bound: The upper boundary of the range.
        """
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower(self) -> T:
        """Return the lower bound of the range."""
        return self._lower_bound

    @property
    def upper(self) -> T:
        """Return the upper bound of the range."""
        return self._upper_bound

    def clamp(self, value: T) -> T:
        """Constrain a value within the range boundaries.

        Values below the lower bound are replaced with the lower bound.
        Values above the upper bound are replaced with the upper bound.

        Args:
            value: The numeric value to constrain.

        Returns:
            The value constrained within the range.
        """
        if value <= self._lower_bound:
            return self._lower_bound
        elif value >= self._upper_bound:
            return self._upper_bound
        else:
            return value


@dataclass
class InferenceResult:
    """Container for anomaly inference results."""

    anomaly_map: Tensor
    """Pixel-level anomaly heatmap interpolated to input image resolution. Shape: (B, 1, H, W)."""

    anomaly_score: Tensor
    """Image-level anomaly score per sample. Shape: (B,)."""

    patch_anomaly: Tensor
    """Token-level anomaly score before spatial interpolation. Shape: (B, L)."""

