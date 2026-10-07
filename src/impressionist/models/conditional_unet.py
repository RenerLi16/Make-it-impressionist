"""Readable multiscale CNN-conditioned U-Net; all parameters start random."""

import torch
from torch import nn
from torch.nn import functional as F
from impressionist.models.blocks import ConditionEncoder, ResidualBlock, normalization
from impressionist.models.embeddings import SinusoidalTimeEmbedding


class ConditionalUNet(nn.Module):
    """Predict RGB noise from [B,3,H,W] noisy targets and aligned conditions.

    The independent condition encoder exposes a feature-pyramid boundary for
    future RGB/edge/depth experiments. This baseline uses only RGB.
    """

    def __init__(self, base_channels: int = 32, channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 8),
                 condition_channels: int = 16, input_channels: int = 3):
        super().__init__()
        if base_channels < 4 or condition_channels < 2 or input_channels < 1:
            raise ValueError("Require base_channels >= 4, condition_channels >= 2, input_channels >= 1.")
        if not channel_multipliers or any(not isinstance(m, int) or m < 1 for m in channel_multipliers):
            raise ValueError("channel_multipliers must be a nonempty list of positive integers.")
        widths = [base_channels * m for m in channel_multipliers]
        condition_widths = [condition_channels * m for m in channel_multipliers]
        self.input_channels = input_channels
        self.factor = 2 ** (len(widths) - 1)
        time_width = base_channels * 4
        self.time_embedding = nn.Sequential(SinusoidalTimeEmbedding(base_channels),
                                             nn.Linear(base_channels, time_width), nn.SiLU(),
                                             nn.Linear(time_width, time_width))
        self.condition_encoder = ConditionEncoder(input_channels, condition_widths)
        self.input_projection = nn.Conv2d(3, widths[0], 3, padding=1)
        self.down_blocks, self.downsample = nn.ModuleList(), nn.ModuleList()
        current = widths[0]
        for level, width in enumerate(widths):
            self.down_blocks.append(ResidualBlock(current + condition_widths[level], width, time_width))
            if level < len(widths) - 1:
                self.downsample.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
            current = width
        self.middle = nn.ModuleList([ResidualBlock(current, current, time_width) for _ in range(2)])
        self.up_blocks = nn.ModuleList()
        for width in reversed(widths):
            self.up_blocks.append(ResidualBlock(current + width, width, time_width))
            current = width
        self.output = nn.Sequential(normalization(current), nn.SiLU(), nn.Conv2d(current, 3, 3, padding=1))

    def encode_condition(self, condition: torch.Tensor) -> list[torch.Tensor]:
        if condition.ndim != 4 or condition.shape[1] != self.input_channels:
            raise ValueError(f"Condition must be [B,{self.input_channels},H,W].")
        if min(condition.shape[-2:]) < self.factor or any(s % self.factor for s in condition.shape[-2:]):
            raise ValueError(f"Spatial dimensions must be positive multiples of {self.factor}.")
        return self.condition_encoder(condition)

    def forward(self, noisy_target: torch.Tensor, timesteps: torch.Tensor,
                condition: torch.Tensor | None = None,
                condition_features: list[torch.Tensor] | None = None) -> torch.Tensor:
        if noisy_target.ndim != 4 or noisy_target.shape[1] != 3:
            raise ValueError("noisy_target must have shape [B,3,H,W].")
        if timesteps.shape != (noisy_target.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be int64 with shape [B].")
        if min(noisy_target.shape[-2:]) < self.factor or any(s % self.factor for s in noisy_target.shape[-2:]):
            raise ValueError(f"Spatial dimensions must be positive multiples of {self.factor}.")
        if condition_features is None:
            if condition is None or condition.shape[0] != noisy_target.shape[0] or condition.shape[-2:] != noisy_target.shape[-2:]:
                raise ValueError("Condition must match target batch size and spatial dimensions.")
            condition_features = self.encode_condition(condition)
        if len(condition_features) != len(self.down_blocks):
            raise ValueError("Condition feature pyramid has the wrong number of levels.")
        time = self.time_embedding(timesteps)  # [B, time_width]
        h, skips = self.input_projection(noisy_target), []
        for level, block in enumerate(self.down_blocks):
            feature = condition_features[level]
            if feature.shape[0] != h.shape[0] or feature.shape[-2:] != h.shape[-2:]:
                raise ValueError(f"Condition feature shape mismatch at level {level}.")
            h = block(torch.cat([h, feature], dim=1), time)
            skips.append(h)
            if level < len(self.downsample):
                h = self.downsample[level](h)
        for block in self.middle:
            h = block(h, time)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), time)
        return self.output(h)  # [B, 3, H, W], predicted epsilon
