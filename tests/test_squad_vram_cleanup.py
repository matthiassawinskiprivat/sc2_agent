"""Tests for squad VRAM cleanup helpers."""
import torch
from gymnasium import spaces

from src.agent.models.squad_network import SquadNetwork
from src.agent.squad_constants import SQUAD_UNIT_FEAT_DIM


def _obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(6, 128, 128), dtype=float),
            "command": spaces.Box(low=-1, high=1, shape=(10,), dtype=float),
            "anchor_tactical": spaces.Box(
                low=0, high=1, shape=(4, 32, 32), dtype=float
            ),
            "unit_scalars": spaces.Box(
                low=-1, high=1, shape=(24, SQUAD_UNIT_FEAT_DIM), dtype=float
            ),
            "unit_mask": spaces.Box(low=0, high=1, shape=(24,), dtype=float),
        }
    )


def test_clear_forward_cache_drops_stash():
    net = SquadNetwork(_obs_space(), features_dim=64, thinking_steps=1)
    obs = {
        "map": torch.randn(2, 6, 128, 128),
        "command": torch.randn(2, 10),
        "anchor_tactical": torch.randn(2, 4, 32, 32),
        "unit_scalars": torch.randn(2, 24, SQUAD_UNIT_FEAT_DIM),
        "unit_mask": torch.ones(2, 24),
    }
    net(obs)
    assert net._pointer_logits is not None
    assert net._spatial_fused is not None
    net.clear_forward_cache()
    assert net._pointer_logits is None
    assert net._spatial_fused is None
