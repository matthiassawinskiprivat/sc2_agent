"""
Tests for timeout stall penalty — loss when agent attack units survive to max_steps.
"""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from sc2.ids.unit_typeid import UnitTypeId

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


def _make_env_for_step():
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.max_steps = 10
    env.steps_count = 9
    env.active_player_id = 1
    env.current_unit_index = 0
    env.squad_unit_tags = []
    env.unit_pools_config = [
        {"high_level_command": "attack", "agent": True, "region": "A"},
        {"high_level_command": "defend", "region": "A"},
        {"high_level_command": "defend", "agent": True, "region": "B"},
        {"high_level_command": "attack", "region": "B"},
    ]
    env.unit_pool_index = {101: 0, 102: 2}
    env.player_units = {1: [101, 102], 2: [201]}
    env._promoted_pool_index = None
    env.label_to_defender_pos = {}

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.unit_command_map = {101: "attack", 102: "defend"}
    env.bot.unit_target_map = {}
    env.bot._unit_last_order = {}
    env.bot.calculate_reward = MagicMock(return_value=0.0)
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.apply_action = AsyncMock()
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.client.actions = AsyncMock()
    env.bot.structures = MagicMock(return_value=MagicMock(amount=1))
    env.bot.enemy_structures = MagicMock(return_value=MagicMock(amount=1))
    env.bot.all_units = MagicMock()
    env.bot.all_units.filter.return_value.amount = 5
    env.bot.all_units.find_by_tag.return_value = None
    env.bot.do = MagicMock()

    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value={"map": MagicMock()})
    env._maybe_promote_defender_to_attack = MagicMock()
    env._has_combat_units = MagicMock(return_value=True)
    env._apply_action = AsyncMock()
    env._current_squad = MagicMock(return_value=(1, [101]))

    return env


def test_agent_attack_units_alive_uses_command_map():
    env = _make_env_for_step()
    assert env._agent_attack_units_alive() == 1


def test_timeout_with_attack_units_applies_loss_penalty():
    env = _make_env_for_step()
    env.training_conf = {
        "rewards": {"loss_penalty": -5.0, "draw_penalty": -2.0},
        "env": {},
    }

    _, reward, done, truncated, _ = asyncio.run(env._async_step(0))

    assert truncated is True
    assert done is False
    assert reward == -5.0


def test_timeout_with_only_defenders_applies_draw_penalty():
    env = _make_env_for_step()
    env.bot.unit_command_map = {101: "defend", 102: "defend"}
    env.unit_pool_index = {101: 0, 102: 2}
    env.training_conf = {
        "rewards": {"loss_penalty": -5.0, "draw_penalty": -2.0},
        "env": {},
    }

    _, reward, done, truncated, _ = asyncio.run(env._async_step(0))

    assert truncated is True
    assert done is False
    assert reward == -2.0
