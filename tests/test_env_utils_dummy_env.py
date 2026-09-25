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
        # ctf_level_01 is squad: tactical crop is ``anchor_tactical``, not ``tactical``.
        assert "anchor_tactical" in env.observation_space.spaces
        assert env.observation_space["anchor_tactical"].shape[0] == nt
        assert "tactical" not in env.observation_space.spaces
        # AR pointer keys — must match CaptureTheFlagSingleEnv or the learner
        # builds SquadNetwork without enemy_encoder and hot-swap zips break.
        for key in (
            "enemy_scalars",
            "enemy_mask",
            "tactical_enemy_mask",
            "engage_enemy_mask",
            "command_modes",
            "heal_actor_mask",
            "attack_actor_mask",
            "aoe_actor_mask",
            "unload_actor_mask",
            "cloak_actor_mask",
            "uncloak_actor_mask",
            "move_actor_mask",
        ):
            assert key in env.observation_space.spaces, key
        assert float(env.observation_space["command_modes"].high.flat[0]) >= 9.0
    finally:
        env.close()


def test_is_squad_pointer_action_space_matches_autoregressive_nvec():
    """Collector must detect the command→actors→target→spatial MultiDiscrete.

    A stale ``N_max + 2`` check would skip SquadPointerRolloutHelper and feed
    universal ``tactical`` obs into the squad policy.
    """
    from gymnasium import spaces

    from src.agent.squad_constants import (
        MAX_SQUAD_UNITS,
        SQUAD_ACTION_DIM,
        squad_action_space_nvec,
    )
    from train.env_utils import is_squad_pointer_action_space

    squad = spaces.MultiDiscrete(squad_action_space_nvec())
    assert len(squad.nvec) == SQUAD_ACTION_DIM
    assert is_squad_pointer_action_space(squad) is True

    # Legacy who→command→spatial (no terminal target slot) must not match.
    legacy = spaces.MultiDiscrete(
        [2] * MAX_SQUAD_UNITS + [FIXED_UNIVERSAL_COMMAND_DIM, 1024]
    )
    assert is_squad_pointer_action_space(legacy) is False

    universal = spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 1024])
    assert is_squad_pointer_action_space(universal) is False


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


def test_dummy_squad_env_omits_race_on_ctf_level_01():
    """Only ctf_level_01 opts out of race so the attack→squad graft stays 138-wide."""
    from src.utils.config_manager import ConfigManager

    cm = ConfigManager("config_universal")
    cfg = cm.get_level_config("ctf_level_01")
    cfg["config_dir"] = "config_universal"
    cfg["agent_architecture"] = cm.get_agent_architecture("ctf_level_01")
    cfg["active_layers"] = cm.get_cumulative_active_layers("ctf_level_01")
    cfg["active_tactical_layers"] = cm.get_cumulative_active_tactical_layers("ctf_level_01")
    cfg["active_unit_scalars"] = cm.get_cumulative_active_unit_scalars("ctf_level_01")

    env = create_dummy_env_from_config(cfg, level_name="ctf_level_01")
    try:
        assert "race" not in env.observation_space.spaces
        assert "enemy_scalars" in env.observation_space.spaces
    finally:
        env.close()
