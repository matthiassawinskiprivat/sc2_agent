"""Tests for grafting universal weights onto squad policy."""

import numpy as np
import torch
from gymnasium import spaces

from src.agent.models.squad_network import SquadNetwork
from src.agent.squad_constants import SQUAD_UNIT_FEAT_DIM
from src.utils.model_utils import graft_squad_policy_from_universal


def test_graft_squad_remmaps_cnn_keys():
    universal_sd = {
        "features_extractor.cnn_trunk.0.weight": torch.randn(32, 4, 3, 3),
        "features_extractor.tactical_cnn_to16.0.weight": torch.randn(32, 4, 3, 3),
        "features_extractor.linear.0.weight": torch.randn(512, 138),
        "features_extractor.linear.0.bias": torch.randn(512),
        "spatial_decoder.decoder.0.weight": torch.randn(64, 160, 3, 3),
        "command_head.weight": torch.randn(64, 512),
    }
    squad_sd = {
        "features_extractor.global_cnn.0.weight": torch.randn(32, 4, 3, 3),
        "features_extractor.tactical_cnn.0.weight": torch.randn(32, 4, 3, 3),
        "features_extractor.query_linear.0.weight": torch.randn(512, 138),
        "features_extractor.query_linear.0.bias": torch.randn(512),
        "features_extractor.entity_encoder.encoder.0.weight": torch.randn(64, 68),
        "spatial_decoder.decoder.0.weight": torch.randn(64, 160, 3, 3),
        "command_head.weight": torch.randn(64, 512),
    }
    grafted = graft_squad_policy_from_universal(universal_sd, squad_sd, new_in_channels=4)
    assert torch.allclose(
        grafted["features_extractor.global_cnn.0.weight"],
        universal_sd["features_extractor.cnn_trunk.0.weight"],
    )
    assert torch.allclose(
        grafted["features_extractor.tactical_cnn.0.weight"],
        universal_sd["features_extractor.tactical_cnn_to16.0.weight"],
    )
    assert torch.allclose(
        grafted["features_extractor.query_linear.0.weight"],
        universal_sd["features_extractor.linear.0.weight"],
    )
    assert torch.allclose(
        grafted["spatial_decoder.decoder.0.weight"],
        universal_sd["spatial_decoder.decoder.0.weight"],
    )


def test_squad_query_linear_matches_universal_fusion_width():
    """Critic/LSTM thought must see global+tactical GAP+command (same as universal)."""
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), np.float32),
            "command": spaces.Box(0, 1, (10,), np.float32),
            "anchor_tactical": spaces.Box(0, 1, (4, 32, 32), np.float32),
            "unit_scalars": spaces.Box(0, 1, (24, SQUAD_UNIT_FEAT_DIM), np.float32),
            "unit_mask": spaces.Box(0, 1, (24,), np.float32),
        }
    )
    net = SquadNetwork(obs_space, features_dim=64, thinking_steps=1)
    # Linear in: 64 global GAP + 64 tactical GAP + 10 command
    assert net.query_linear[0].in_features == 138
