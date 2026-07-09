"""Regression: Level3Env and Level0Env must agree on Computer player for movement levels."""

from train.envs.sc2_session_config import session_realtime, use_computer_player


def test_move_multi_level_defaults_without_computer():
    assert use_computer_player("level_move_multi_3", {}) is False


def test_legacy_level_3_prefix_defaults_without_computer():
    assert use_computer_player("level_3", {}) is False


def test_attack_level_defaults_with_computer():
    assert use_computer_player("level_attack_0", {}) is True


def test_env_use_computer_overrides_movement_default():
    conf = {"env": {"use_computer": True}}
    assert use_computer_player("level_move_multi_0", conf) is True


def test_session_realtime_defaults_false_even_for_preview():
    """Visualize must not imply realtime — breaks debug spawn on multi-unit levels."""
    assert session_realtime({"preview_mode": True}) is False
    assert session_realtime({}) is False


def test_session_realtime_honors_env_override():
    assert session_realtime({"env": {"realtime": True}}) is True
    assert session_realtime({"env": {"realtime": False}}) is False


def test_session_realtime_honors_flat_training_conf():
    assert session_realtime({"realtime": True}) is True
    assert session_realtime({"realtime": False}) is False
