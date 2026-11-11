from abc import abstractmethod

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
