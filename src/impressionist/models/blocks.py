"""Small residual CNN blocks with additive timestep conditioning."""

import torch
from torch import nn


def normalization(channels: int) -> nn.GroupNorm:
    # At least two channels/group also works for B=1 and 1x1 feature maps.
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.first = nn.Sequential(normalization(in_channels), nn.SiLU(),
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))
        self.second = nn.Sequential(normalization(out_channels), nn.SiLU(),
                                    nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.first(x) + self.time(time)[:, :, None, None]
        return self.second(h) + self.skip(x)


class ConditionEncoder(nn.Module):
    """Return an RGB feature pyramid with one feature map per U-Net scale."""

    def __init__(self, input_channels: int, widths: list[int]):
        super().__init__()
        blocks = []
        current = input_channels
        for level, width in enumerate(widths):
            blocks.append(nn.Sequential(
                nn.Conv2d(current, width, 3, stride=1 if level == 0 else 2, padding=1),
                normalization(width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU()))
            current = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, conditioning: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.blocks:
            conditioning = block(conditioning)
            features.append(conditioning)
        return features
