"""Sinusoidal embeddings for integer diffusion timesteps."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        if dimension < 4:
            raise ValueError("Timestep embedding dimension must be at least four.")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("Timesteps must have shape [B].")
        half = self.dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32) / (half - 1))
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
