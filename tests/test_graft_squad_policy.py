"""Tests for grafting universal weights onto squad policy."""

import tempfile

import gymnasium as gym
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
    assert net.race_dim == 0


def test_squad_query_linear_grows_with_include_race():
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), np.float32),
            "command": spaces.Box(0, 1, (10,), np.float32),
            "race": spaces.Box(0, 1, (6,), np.float32),
            "anchor_tactical": spaces.Box(0, 1, (4, 32, 32), np.float32),
            "unit_scalars": spaces.Box(0, 1, (24, SQUAD_UNIT_FEAT_DIM), np.float32),
            "unit_mask": spaces.Box(0, 1, (24,), np.float32),
        }
    )
    net = SquadNetwork(obs_space, features_dim=64, thinking_steps=1)
    assert net.race_dim == 6
    assert net.query_linear[0].in_features == 144
    batch = 2
    obs = {
        "map": torch.randn(batch, 4, 128, 128),
        "command": torch.randn(batch, 10),
        "race": torch.zeros(batch, 6),
        "anchor_tactical": torch.randn(batch, 4, 32, 32),
        "unit_scalars": torch.randn(batch, 24, SQUAD_UNIT_FEAT_DIM),
        "unit_mask": torch.ones(batch, 24),
    }
    out = net(obs)
    assert out.shape == (batch, 64)


def test_graft_query_linear_preserves_prefix_when_race_appended():
    """CTF 11→12: copy 138-wide fusion into leading cols; race cols stay fresh."""
    from src.utils.model_utils import graft_model_weights

    old_w = torch.randn(512, 138)
    new_w = torch.randn(512, 144)
    state = {"features_extractor.query_linear.0.weight": old_w.clone()}
    target = {"features_extractor.query_linear.0.weight": new_w.clone()}
    grafted = graft_model_weights(state, new_in_channels=4, target_state_dict=target)
    out = grafted["features_extractor.query_linear.0.weight"]
    assert out.shape == (512, 144)
    assert torch.allclose(out[:, :138], old_w)
    # Trailing race columns must not equal the old command slice (would be the bug).
    assert not torch.allclose(out[:, -10:], old_w[:, -10:])


def test_graft_squad_to_squad_expands_query_linear_for_race():
    """Squad checkpoint 138 → squad target 144 keeps prior fusion weights."""
    old_w = torch.randn(512, 138)
    source_sd = {
        "features_extractor.global_cnn.0.weight": torch.randn(32, 4, 3, 3),
        "features_extractor.query_linear.0.weight": old_w.clone(),
        "features_extractor.query_linear.0.bias": torch.randn(512),
    }
    target_sd = {
        "features_extractor.global_cnn.0.weight": torch.randn(32, 4, 3, 3),
        "features_extractor.query_linear.0.weight": torch.randn(512, 144),
        "features_extractor.query_linear.0.bias": torch.randn(512),
    }
    grafted = graft_squad_policy_from_universal(source_sd, target_sd, new_in_channels=4)
    out = grafted["features_extractor.query_linear.0.weight"]
    assert out.shape == (512, 144)
    assert torch.allclose(out[:, :138], old_w)


def test_load_policy_weights_grafts_query_linear_race_mismatch():
    """Learner/collector sync must not crash when query_linear differs by race dims."""
    from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
    from src.agent.models.squad_policy import SquadRecurrentPolicy
    from src.agent.squad_constants import SQUAD_RACE_DIM, SQUAD_UNIT_FEAT_DIM, squad_action_space_nvec
    from src.utils.model_utils import load_policy_weights_from_zip

    def _minimal_env(obs_space, action_space):
        class _Env(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = obs_space
                self.action_space = action_space

            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        return _Env()

    obs_no_race = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), np.float32),
            "command": spaces.Box(0, 1, (10,), np.float32),
            "anchor_tactical": spaces.Box(0, 1, (4, 32, 32), np.float32),
            "unit_scalars": spaces.Box(0, 1, (24, SQUAD_UNIT_FEAT_DIM), np.float32),
            "unit_mask": spaces.Box(0, 1, (24,), np.float32),
        }
    )
    obs_with_race = spaces.Dict(
        {
            **obs_no_race.spaces,
            "race": spaces.Box(0, 1, (SQUAD_RACE_DIM,), np.float32),
        }
    )
    action_space = spaces.MultiDiscrete(squad_action_space_nvec())
    policy_kwargs = {
        "features_extractor_class": SquadNetwork,
        "features_extractor_kwargs": {"features_dim": 64, "thinking_steps": 1},
        "lstm_hidden_size": 32,
    }
    src = MaskableRecurrentPPO(
        SquadRecurrentPolicy,
        _minimal_env(obs_with_race, action_space),
        n_steps=64,
        batch_size=32,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )
    w144 = src.policy.features_extractor.query_linear[0].weight.detach().clone()
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/snap.zip"
        src.save(path)
        dst = MaskableRecurrentPPO(
            SquadRecurrentPolicy,
            _minimal_env(obs_no_race, action_space),
            n_steps=64,
            batch_size=32,
            policy_kwargs=policy_kwargs,
            device="cpu",
        )
        ok, missing, _unexpected = load_policy_weights_from_zip(
            dst.policy, path, "cpu", strict=False
        )
        assert ok
        assert not missing
        loaded = dst.policy.features_extractor.query_linear[0].weight.detach()
        assert loaded.shape == (64, 138)
        assert torch.allclose(loaded, w144[:, :138])
