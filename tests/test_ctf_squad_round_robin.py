"""
Tests for squad-only round-robin: all units in the active squad act before one physics step.
"""
import asyncio

import pytest
import numpy as np
from gymnasium import spaces
from unittest.mock import MagicMock, AsyncMock, patch

UNIT_POOLS_CONFIG = [
    {"high_level_command": "attack", "agent": True, "region": "A"},
    {"high_level_command": "defend", "agent": False, "region": "A"},
]


def _make_env(agent_squad_tags=None, opponent_squad_tags=None):
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    with patch("train.envs.capture_the_flag_single_env.SingleClientEnv.__init__",
               lambda self, *a, **kw: None):
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)

    env.unit_pools_config = UNIT_POOLS_CONFIG
    env.squad_unit_tags = [
        (1, agent_squad_tags or [10, 11, 12]),
        (2, opponent_squad_tags or [20, 21]),
    ]
    env.current_squad_index = 0
    env.current_unit_index = 0
    env.active_player_id = 1
    env.player_units = {1: agent_squad_tags or [10, 11, 12], 2: opponent_squad_tags or [20, 21]}
    env.unit_pool_index = {}
    env.steps_count = 0
    env.max_steps = 100
    env.training_conf = {"rewards": {}, "env": {"step_mul_per_unit": 2}}
    env.action_space = spaces.MultiDiscrete([4, 16])

    env.bot = MagicMock()
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.client.actions = AsyncMock()
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.calculate_squad_step_reward = MagicMock(return_value=1.0)
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    env.bot.all_units = MagicMock()
    env.bot.all_units.find_by_tag = MagicMock(return_value=None)
    env.bot.structures = MagicMock()
    env.bot.enemy_structures = MagicMock()
    env.bot.structures.return_value.amount = 1
    env.bot.enemy_structures.return_value.amount = 1

    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value={"map": np.zeros((1, 4, 4))})
    env._maybe_promote_defender_to_attack = MagicMock()
    env._has_combat_units = MagicMock(return_value=True)
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env.opponent_model = None
    env._opponent_lstm_state = None
    env._opponent_last_focus_tag = None

    return env


def test_agent_squad_batch_actions_single_physics_step():
    env = _make_env()
    env.set_squad_batch_actions(
        np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    )
    obs, reward, done, truncated, info = asyncio.run(
        env._async_step(np.array([0, 0]))
    )
    assert info.get("squad_physics_step") is True
    assert reward == 1.0
    env.bot.client.step.assert_called_once()
    assert env.bot.client.step.await_args.kwargs.get("step_size") == 6
    assert env.bot.apply_action.await_count == 3


def test_agent_squad_partial_steps_no_physics_until_complete():
    env = _make_env()
    action = np.array([0, 0], dtype=np.int64)

    for _i in range(2):
        obs, reward, done, truncated, info = asyncio.run(env._async_step(action))
        assert reward == 0.0
        assert info.get("squad_physics_step") is False
        assert done is False
        env.bot.client.step.assert_not_called()

    obs, reward, done, truncated, info = asyncio.run(env._async_step(action))
    assert info.get("squad_physics_step") is True
    assert reward == 1.0
    env.bot.client.step.assert_called_once()
    assert env.bot.apply_action.await_count == 3


def test_opponent_squad_batch_single_physics_step():
    env = _make_env()
    env.current_squad_index = 1
    env.active_player_id = 2
    env.current_unit_index = 0
    env.opponent_model = MagicMock()

    with patch(
        "train.squad_batch_inference.opponent_squad_batch_predict",
        return_value=(np.array([[0, 0], [0, 0]]), None, 20),
    ):
        assert env.requires_policy_action() is False

        obs, reward, done, truncated, info = asyncio.run(
            env._async_step(np.array([0, 0]))
        )

    assert info.get("squad_physics_step") is True
    assert env.bot.apply_action.await_count == 2
    env.bot.client.step.assert_called_once()


def test_calculate_squad_step_reward_sums_per_unit_penalties():
    from src.agent.ctf_bot import CaptureTheFlagBot

    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={"rewards": {"step_penalty": -0.1, "defender_step_reward": 0.0}},
    )
    bot.prev_pylon_health = {1: 100.0, 2: 100.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=False)
    bot.unit_command_map = {101: "attack", 102: "attack"}
    bot.prev_dist_per_squad = {}

    def _struct_col(pylon=None):
        col = MagicMock()
        col.exists = pylon is not None
        col.first = pylon
        col.amount = 1 if pylon is not None else 0
        return MagicMock(return_value=col)

    bot.structures = _struct_col(MagicMock(health=100, shield=0))
    bot.enemy_structures = _struct_col(MagicMock(health=100, shield=0))
    bot.units = []
    bot.enemy_units = []
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag = MagicMock(return_value=None)

    reward = bot.calculate_squad_step_reward([101, 102], perspective_player_id=1)
    assert reward == pytest.approx(-0.2)
