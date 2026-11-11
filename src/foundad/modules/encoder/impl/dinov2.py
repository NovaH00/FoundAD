
import torch
from PIL.Image import Image
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

from ..base import Encoder


class DinoV2Encoder(Encoder):
    MODEL_NAME = "facebook/dinov2-with-registers-small"
    PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)

    @staticmethod
    def process_image(
        image_input: Image | list[Image] | Tensor | list[Tensor]
    ) -> Tensor:
        out = DinoV2Encoder.PROCESSOR(
            images=image_input,
            return_tensors="pt"
        )

        return out.pixel_values

    def encode(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            out = self.model(pixel_values=x)

        # The final output is in the following structure
        # [CLS, REG1, REG2, REG3, REG4, EMB1, EMB2 ...]
        # Where:
        #    - CLS: Classification token
        #    - REG: Register tokens (temporary tokens used to store memory)
        #    - EMB: Embedding tokens
        # So we need remove the CLS and REG tokens to get the feature embeddings
        patches = out.last_hidden_state[:, 5:, :]

        return patches
