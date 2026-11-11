import torch
from PIL.Image import Image
from torch import Tensor, nn

from .encoder import Encoder
from .projector import Projector


class FoundAD(nn.Module):
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
        image_input: Image | list[Image] | Tensor | list[Tensor]
    ):
        return self.encoder.process_image(image_input)

    def train(self, mode: bool = True):
        # encoder should be frozen and in eval mode at all times
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        if mode:
            self.projector.train()
        else:
            self.projector.eval()

    def forward(self, X: Tensor, Z: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            X_en = self.encoder(X)
            Z_en = self.encoder(Z)

        Z_proj = self.projector(Z_en)

        return X_en, Z_proj
