import numpy as np

from src.utils.config_manager import ConfigManager
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


def test_dummy_env_level_name_matches_config_manager():
    cm = ConfigManager("config_universal")
    ng = len(cm.get_cumulative_active_layers("ctf_level_01"))
    nt = len(cm.get_cumulative_active_tactical_layers("ctf_level_01"))
    env = create_dummy_env_from_config(
        {"config_dir": "config_universal"}, level_name="ctf_level_01"
    )
    try:
        assert env.observation_space["map"].shape[0] == ng
        assert env.observation_space["tactical"].shape[0] == nt
    finally:
        env.close()


def test_dummy_squad_env_includes_race_when_configured():
    from src.agent.squad_constants import SQUAD_RACE_DIM
    from src.utils.config_manager import ConfigManager

    cm = ConfigManager("config_universal")
    cfg = cm.get_level_config("ctf_level_12")
    cfg["config_dir"] = "config_universal"
    cfg["agent_architecture"] = cm.get_agent_architecture("ctf_level_12")
    cfg["active_layers"] = cm.get_cumulative_active_layers("ctf_level_12")
    cfg["active_tactical_layers"] = cm.get_cumulative_active_tactical_layers("ctf_level_12")
    cfg["active_unit_scalars"] = cm.get_cumulative_active_unit_scalars("ctf_level_12")

    env = create_dummy_env_from_config(cfg, level_name="ctf_level_12")
    try:
        assert "race" in env.observation_space.spaces
        assert env.observation_space["race"].shape == (SQUAD_RACE_DIM,)
    finally:
        env.close()


def test_dummy_squad_env_omits_race_before_level_12():
    from src.utils.config_manager import ConfigManager

    cm = ConfigManager("config_universal")
    cfg = cm.get_level_config("ctf_level_11")
    cfg["config_dir"] = "config_universal"
    cfg["agent_architecture"] = cm.get_agent_architecture("ctf_level_11")
    cfg["active_layers"] = cm.get_cumulative_active_layers("ctf_level_11")
    cfg["active_tactical_layers"] = cm.get_cumulative_active_tactical_layers("ctf_level_11")
    cfg["active_unit_scalars"] = cm.get_cumulative_active_unit_scalars("ctf_level_11")

    env = create_dummy_env_from_config(cfg, level_name="ctf_level_11")
    try:
        assert "race" not in env.observation_space.spaces
    finally:
        env.close()
