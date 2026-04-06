"""Regression: Level3Env and Level0Env must agree on Computer player for movement levels."""

from train.envs.sc2_session_config import use_computer_player


def test_move_multi_level_defaults_without_computer():
    assert use_computer_player("level_move_multi_3", {}) is False


def test_legacy_level_3_prefix_defaults_without_computer():
    assert use_computer_player("level_3", {}) is False


def test_attack_level_defaults_with_computer():
    assert use_computer_player("level_attack_0", {}) is True


def test_env_use_computer_overrides_movement_default():
    conf = {"env": {"use_computer": True}}
    assert use_computer_player("level_move_multi_0", conf) is True
