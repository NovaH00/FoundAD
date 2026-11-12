from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from foundad.utils import assert_shape

from ..base import Projector


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of attention heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        assert_shape(x, f"(_, _, {self.embed_dim})")
        B, L, D = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = k.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).flatten(2)

        out = self.proj(x)
        assert_shape(out, f"({B}, {L}, {D})")

        return out

class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim*mlp_ratio, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerProjector(Projector):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        num_blocks: int = 6,
        mlp_ratio: int = 4
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ])

    def project(self, x: Tensor) -> Tensor:
        for block in self.transformer_blocks:
            x = x + block(x)

        return x

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "mlp_ratio": self.mlp_ratio,
        }


