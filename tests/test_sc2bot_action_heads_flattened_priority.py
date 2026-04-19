"""Regression: SC2Bot must use cumulative action_heads from training_conf, not env-only."""

from unittest.mock import MagicMock, patch

from src.sc2_bot import SC2Bot


def _noop_init_vars(self):
    """BotAI / sc2 version in CI may not expose _initialize_variables."""
    pass


def test_sc2_bot_prefers_flattened_action_heads_over_env_partial():
    """Matches Level0Env merge: root has cumulative heads; env lists only level deltas."""
    training_conf = {
        "action_heads": ["move_global", "move_local", "attack_local"],
        "env": {"action_heads": ["attack_local"]},
    }
    with patch.object(SC2Bot, "_initialize_variables", _noop_init_vars, create=True):
        bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    assert bot.action_heads == ["move_global", "move_local", "attack_local"]


def test_sc2_bot_categorical_heads_prefers_flattened():
    training_conf = {
        "categorical_heads": ["hold", "stop"],
        "env": {"categorical_heads": ["stop"]},
    }
    with patch.object(SC2Bot, "_initialize_variables", _noop_init_vars, create=True):
        bot = SC2Bot(conn=MagicMock(), training_conf=training_conf)
    assert bot.categorical_heads == ["hold", "stop"]
