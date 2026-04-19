import torch
import gymnasium as gym
from gymnasium import spaces

from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.models.spatial_decoder import SpatialDecoder, tactical_center_spatial_bias
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy


def test_unit_control_network_universal_spatial_fused_shape():
    obs_space = spaces.Dict({
        "map": spaces.Box(0.0, 1.0, (4, 128, 128)),
        "tactical": spaces.Box(0.0, 1.0, (4, 32, 32)),
        "command": spaces.Box(0.0, 1.0, (10,)),
    })
    net = UnitControlNetwork(obs_space, features_dim=128, agent_architecture="universal")
    batch = {
        "map": torch.randn(3, 4, 128, 128),
        "tactical": torch.randn(3, 4, 32, 32),
        "command": torch.randn(3, 10),
    }
    out = net(batch)
    assert out.shape == (3, 128)
    assert net._spatial_fused is not None
    assert net._spatial_fused.shape == (3, 128, 16, 16)


def test_spatial_decoder_output_dim():
    dec = SpatialDecoder(512, fused_channels=128, spatial_resolution=32)
    latent = torch.randn(2, 512)
    fused = torch.randn(2, 128, 16, 16)
    logits = dec(latent, fused)
    assert logits.shape == (2, 1024)


def test_universal_recurrent_policy_builds():
    obs_space = spaces.Dict({
        "map": spaces.Box(0.0, 1.0, (4, 128, 128)),
        "tactical": spaces.Box(0.0, 1.0, (4, 32, 32)),
        "command": spaces.Box(0.0, 1.0, (10,)),
    })
    action_space = gym.spaces.MultiDiscrete([2, 1024])
    policy = UniversalRecurrentPolicy(obs_space, action_space, lr_schedule=lambda _: 0.001)
    assert hasattr(policy, "command_head")
    assert hasattr(policy, "spatial_decoder")
    assert not hasattr(policy, "action_net")


def test_disable_spatial_mask_preserves_command_columns():
    """_disable_spatial_mask must force spatial columns True while keeping command columns unchanged."""
    obs_space = spaces.Dict({
        "map": spaces.Box(0.0, 1.0, (4, 128, 128)),
        "tactical": spaces.Box(0.0, 1.0, (4, 32, 32)),
        "command": spaces.Box(0.0, 1.0, (10,)),
    })
    action_space = gym.spaces.MultiDiscrete([2, 1024])
    policy = UniversalRecurrentPolicy(obs_space, action_space, lr_schedule=lambda _: 0.001)

    mask = torch.zeros(4, 1026, dtype=torch.bool)
    mask[:, 0] = True  # only command 0 allowed

    result = policy._disable_spatial_mask(mask)

    assert result[:, 0].all(), "Command column 0 should stay True"
    assert not result[:, 1].any(), "Command column 1 should stay False"
    assert result[:, 2:].all(), "All spatial columns should be forced True"


def test_disable_spatial_mask_returns_none_for_none():
    obs_space = spaces.Dict({
        "map": spaces.Box(0.0, 1.0, (4, 128, 128)),
        "tactical": spaces.Box(0.0, 1.0, (4, 32, 32)),
        "command": spaces.Box(0.0, 1.0, (10,)),
    })
    action_space = gym.spaces.MultiDiscrete([2, 1024])
    policy = UniversalRecurrentPolicy(obs_space, action_space, lr_schedule=lambda _: 0.001)

    assert policy._disable_spatial_mask(None) is None


def test_get_action_dist_uses_zero_fallback_when_no_spatial_fused():
    """When features_extractor._spatial_fused is absent, a zero tensor should be used."""
    obs_space = spaces.Dict({
        "map": spaces.Box(0.0, 1.0, (4, 128, 128)),
        "tactical": spaces.Box(0.0, 1.0, (4, 32, 32)),
        "command": spaces.Box(0.0, 1.0, (10,)),
    })
    action_space = gym.spaces.MultiDiscrete([2, 1024])
    policy = UniversalRecurrentPolicy(obs_space, action_space, lr_schedule=lambda _: 0.001)

    if hasattr(policy.features_extractor, "_spatial_fused"):
        delattr(policy.features_extractor, "_spatial_fused")

    latent_pi = torch.randn(2, policy.mlp_extractor.latent_dim_pi)
    dist = policy._get_action_dist_from_latent(latent_pi)

    assert dist is not None


def test_tactical_center_spatial_bias_argmax_matches_transformer_unit_cell():
    ref = torch.zeros((), dtype=torch.float32)
    b = tactical_center_spatial_bias(32, ref)
    assert b.shape == (1024,)
    assert int(b.argmax().item()) == 15 * 32 + 16
