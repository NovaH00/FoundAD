from abc import abstractmethod
from typing import Any, Self

from PIL.Image import Image
from torch import Tensor, nn

from foundad.utils import assert_shape


class Encoder(nn.Module):
    @abstractmethod
    def process_image(
        self,
        image_input: Image | list[Image] | Tensor | list[Tensor]
    ) -> Tensor: ...

    @abstractmethod
    def get_embed_dim(self) -> int: ...

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor: ...

    def forward(self, x: Tensor) -> Tensor:
        assert_shape(x, "(_, _, _, _)")  # (B, C, W, H)
        B, *_ = x.shape
        out = self.encode(x)
        assert_shape(out, f"({B}, _, _)")  # (B, L, D)
        return out

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Returns initialization arguments for serialization."""
        ...

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiates the encoder from a configuration dictionary."""
        kwargs = {k: v for k, v in config.items() if k != "type"}
        return cls(**kwargs)
