"""Tests for SC2BotUniversal — action masks, apply_action dispatch, reward calculation."""

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM, build_action_head_manifest


def _create_universal_bot(training_conf=None):
    """Instantiate SC2BotUniversal without triggering BotAI.__init__."""
    from src.sc2_bot_universal import SC2BotUniversal

    if training_conf is None:
        training_conf = {
            "agent_architecture": "universal",
            "action_heads": ["move_global", "move_local"],
            "categorical_heads": [],
            "active_layers": ["allied_unit_density", "enemy_unit_density",
                              "terrain_pathable", "command_objective"],
            "rewards": {
                "step_penalty": -0.05,
                "success": 100.0,
                "distance_multiplier": 0.1,
            },
        }

    with patch.object(SC2BotUniversal, "__init__", lambda self, *a, **kw: None):
        bot = SC2BotUniversal.__new__(SC2BotUniversal)

    bot.conn = MagicMock()
    bot.training_conf = training_conf
    bot.agent_system = None
    bot.zergling = None
    bot.beacon = None
    bot.previous_distance = None
    bot.last_target_pos = None
    bot._game_state_initialized = False
    bot._debug_show_map_enabled = False
    bot.action_heads = training_conf.get("action_heads", ["move_global"])
    bot.categorical_heads = training_conf.get("categorical_heads", [])
    bot.spatial_res = 32
    bot.distance_calculation_method = 2
    bot.unit_command_uses_self_do = False
    bot.actions = []
    bot.unit_tags_received_action = set()
    bot.action_head_manifest = build_action_head_manifest(
        bot.training_conf, list(bot.action_heads)
    )
    bot.training_conf["action_head_manifest"] = bot.action_head_manifest
    return bot


class TestActionMasks:
    def test_fallback_mask_shape(self):
        """When no agent_system exists, mask is 64 + 1024; only defined command prefix is legal."""
        bot = _create_universal_bot()
        mask = bot.action_masks()
        assert mask.shape == (FIXED_UNIVERSAL_COMMAND_DIM + 1024,)
        assert mask.dtype == bool
        n_def = len(bot.action_heads)
        assert mask[:n_def].all()
        assert not mask[n_def:FIXED_UNIVERSAL_COMMAND_DIM].any()
        assert mask[FIXED_UNIVERSAL_COMMAND_DIM:].all()

    def test_fallback_spatial_all_true(self):
        bot = _create_universal_bot()
        mask = bot.action_masks()
        assert mask[FIXED_UNIVERSAL_COMMAND_DIM:].all()

    def test_mask_with_agent_system(self):
        bot = _create_universal_bot()
        mock_agent = MagicMock()
        mock_agent.get_action_mask.return_value = np.ones(1024, dtype=bool)

        bot.agent_system = MagicMock()
        bot.agent_system.unit_agents = {"agent_0": mock_agent}
        bot.zergling = None

        mask = bot.action_masks()
        assert mask.shape == (FIXED_UNIVERSAL_COMMAND_DIM + 1024,)

    def test_mask_shape_with_three_heads(self):
        conf = {
            "agent_architecture": "universal",
            "action_heads": ["move_global", "move_local", "attack_global"],
            "categorical_heads": [],
            "active_layers": ["allied_unit_density"],
            "rewards": {},
        }
        bot = _create_universal_bot(training_conf=conf)
        mask = bot.action_masks()
        assert mask.shape == (FIXED_UNIVERSAL_COMMAND_DIM + 1024,)
        assert mask[:3].all()
        assert not mask[3:FIXED_UNIVERSAL_COMMAND_DIM].any()


class TestCalculateReward:
    def test_zero_when_zergling_missing(self):
        bot = _create_universal_bot()
        bot.zergling = None
        bot.beacon = MagicMock()
        bot.previous_distance = None
        reward, done = bot._calculate_reward()
        assert reward == 0
        assert done is False

    def test_zero_when_beacon_missing(self):
        bot = _create_universal_bot()
        bot.zergling = MagicMock()
        bot.beacon = None
        bot.previous_distance = None
        reward, done = bot._calculate_reward()
        assert reward == 0
        assert done is False

    def test_success_when_close(self):
        bot = _create_universal_bot()
        bot.zergling = MagicMock()
        bot.beacon = MagicMock()
        bot.zergling.distance_to.return_value = 1.5
        bot.previous_distance = 3.0
        reward, done = bot._calculate_reward()
        assert reward == 100.0
        assert done is True

    def test_distance_shaping(self):
        bot = _create_universal_bot()
        bot.zergling = MagicMock()
        bot.beacon = MagicMock()
        bot.zergling.distance_to.return_value = 8.0
        bot.previous_distance = 10.0

        reward, done = bot._calculate_reward()
        assert done is False
        expected = -0.05 + (10.0 - 8.0) * 0.1
        assert abs(reward - expected) < 1e-6

    def test_inferred_success_when_both_missing_and_was_close(self):
        """When zergling AND beacon disappear (map trigger deletion) and last distance < 4."""
        bot = _create_universal_bot()
        bot.zergling = None
        bot.beacon = None
        bot.previous_distance = 3.0
        reward, done = bot._calculate_reward()
        assert reward == 100.0
        assert done is True


class TestApplyActionDispatch:
    def test_no_crash_without_agent_system(self):
        bot = _create_universal_bot()
        bot.agent_system = None
        asyncio.run(bot.apply_action(123, [0, 0]))

    def test_dispatches_move_for_head_0(self):
        bot = _create_universal_bot()
        mock_agent = MagicMock()
        mock_agent.convert_spatial_action_to_world.return_value = (50.0, 50.0)

        bot.agent_system = MagicMock()
        bot.agent_system.unit_agents = {"agent_0": mock_agent}

        mock_unit = MagicMock()
        mock_unit.move.return_value = MagicMock()

        mock_units = MagicMock()
        mock_units.find_by_tag.return_value = mock_unit
        bot.units = mock_units
        bot.do = MagicMock()

        asyncio.run(bot.apply_action(mock_unit.tag, [0, 512]))

        mock_agent.convert_spatial_action_to_world.assert_called_once_with(
            "move_global", 512, active_unit=mock_unit,
        )
        bot.do.assert_called_once()
