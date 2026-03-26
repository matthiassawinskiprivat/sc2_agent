import torch
import gymnasium as gym
from gymnasium import spaces

from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.models.spatial_decoder import SpatialDecoder
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
