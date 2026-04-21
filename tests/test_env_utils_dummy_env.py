import numpy as np

from train.env_utils import create_dummy_env_from_config
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM


def test_dummy_env_respects_independent_map_and_tactical_channels():
    config = {
        "active_layers": ["a", "b", "c", "d"],
        "active_tactical_layers": ["t1", "t2"],
        "action_heads": ["move_local", "attack_local"],
        "env": {},
    }

    env = create_dummy_env_from_config(config)
    try:
        obs_space = env.observation_space
        assert obs_space["map"].shape == (4, 128, 128)
        assert obs_space["tactical"].shape == (2, 32, 32)
        assert env.action_space.nvec.tolist() == [FIXED_UNIVERSAL_COMMAND_DIM, 1024]
    finally:
        env.close()


def test_dummy_env_uses_nested_env_fallback_for_heads_and_tactical():
    config = {
        "env": {
            "active_layers": ["g1", "g2", "g3"],
            "active_tactical_layers": ["t1"],
            "action_heads": ["attack_local"],
        }
    }

    env = create_dummy_env_from_config(config)
    try:
        obs_space = env.observation_space
        assert obs_space["map"].shape == (3, 128, 128)
        assert obs_space["tactical"].shape == (1, 32, 32)
        assert env.action_space.nvec.tolist() == [FIXED_UNIVERSAL_COMMAND_DIM, 1024]
    finally:
        env.close()
