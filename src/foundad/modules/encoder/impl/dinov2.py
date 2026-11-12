from typing import Any, Literal

import torch
from PIL.Image import Image
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

from ..base import Encoder


class DinoV2Encoder(Encoder):
    def __init__(
        self,
        size: Literal["small", "base", "large"] = "small"
    ):
        super().__init__()
        self.size = size
        self.model_name = f"facebook/dinov2-with-registers-{size}"
        self.model = AutoModel.from_pretrained(self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)

    def process_image(
        self,
        image_input: Image | list[Image] | Tensor | list[Tensor]
    ) -> Tensor:
        out = self.processor(
            images=image_input,
            return_tensors="pt"
        )

        return out.pixel_values

    def get_embed_dim(self) -> int:
        return self.model.config.hidden_size

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

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "size": self.size,
        }
