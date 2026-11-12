from abc import abstractmethod
from typing import Any, Self

from torch import Tensor, nn

from foundad.utils import assert_shape


class Projector(nn.Module):
    @abstractmethod
    def project(self, x: Tensor) -> Tensor: ...

    def forward(self, x: Tensor) -> Tensor:
        assert_shape(x, "(_, _, _)")

        B, L, D = x.shape
        out = self.project(x)
        assert_shape(out, f"({B}, {L}, {D})")

        return out

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Returns initialization arguments for serialization."""
        ...

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiates the projector from a configuration dictionary."""
        kwargs = {k: v for k, v in config.items() if k != "type"}
        return cls(**kwargs)
