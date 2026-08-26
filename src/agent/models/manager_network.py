"""Manager feature extractor network (macro map + vector → latent)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ManagerNetwork(nn.Module):
    def __init__(
        self,
        map_channels: int = 4,
        macro_dim: int = 12,
        hidden_dim: int = 128,
        map_resolution: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map_cnn = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat = 32 * (map_resolution // 2) * (map_resolution // 2)
        self.map_fc = nn.Linear(flat, 128)
        self.macro_fc = nn.Linear(macro_dim, 64)
        self.trunk = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, map_obs: torch.Tensor, macro_vector: torch.Tensor) -> torch.Tensor:
        m = self.map_fc(self.map_cnn(map_obs))
        v = self.macro_fc(macro_vector)
        return self.trunk(torch.cat([m, v], dim=-1))
