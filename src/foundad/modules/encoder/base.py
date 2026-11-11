from abc import abstractmethod

from PIL.Image import Image
from torch import Tensor, nn

from foundad.utils import assert_shape


class Encoder(nn.Module):
    @staticmethod
    @abstractmethod
    def process_image(
        image_input: Image | list[Image] | Tensor | list[Tensor]
    ) -> Tensor: ...

    @abstractmethod
    def get_embed_dim(self) -> int: ...

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor: ...

    def forward(self, x: Tensor) -> Tensor:
        assert_shape(x, "(_, _, _, _)") # (B, C, W, H)
        B, *_ = x.shape
        out = self.encode(x)
        assert_shape(out, f"({B}, _, _)") # (B, L, D)
        return out
