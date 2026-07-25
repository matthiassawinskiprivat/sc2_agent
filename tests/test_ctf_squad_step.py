"""Regression tests for CTF squad pointer env stepping."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from gymnasium import spaces

from src.agent.squad_constants import SQUAD_UNIT_FEAT_DIM, squad_action_space_nvec


@pytest.fixture
def squad_env():
    with patch("train.envs.capture_the_flag_single_env.ConfigManager") as mock_cm, patch(
        "train.envs.capture_the_flag_single_env.SingleClientEnv.__init__", return_value=None
    ):
        mock_cm.return_value.get_agent_architecture.return_value = "squad"
        mock_cm.return_value.get_cumulative_active_layers.return_value = ["layer_a"]
        mock_cm.return_value.get_cumulative_active_tactical_layers.return_value = ["layer_a"]
        mock_cm.return_value.get_cumulative_action_heads.return_value = ["move_global", "attack_global"]
        mock_cm.return_value.get_cumulative_categorical_heads.return_value = []

        from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
        env.training_conf = {
            "agent_architecture": "squad",
            "active_layers": ["layer_a"],
            "active_tactical_layers": ["layer_a"],
            "action_heads": ["move_global", "attack_global"],
            "categorical_heads": [],
            "env": {"squad_selection": {"score_threshold": 0.5, "min_selected_units": 1}},
            "rewards": {},
        }
        env.level_name = "test_ctf"
        env.config_manager = mock_cm.return_value
        env.agent_architecture = "squad"
        env.unit_pools_config = [
            {"high_level_command": "attack", "region": "A", "race": "terran"},
            {"high_level_command": "defend", "region": "A", "race": "zerg"},
            {"high_level_command": "defend", "region": "B", "race": "terran"},
            {"high_level_command": "attack", "region": "B", "race": "zerg"},
        ]
        env._alternate_trainee_race = False
        env.squad_unit_tags = [
            (1, [101, 102]),
            (2, [201]),
            (1, [301]),
            (2, [401]),
        ]
        env.current_trainee_pool_index = 0
        env.current_squad_index = 0
        env.current_unit_index = 0
        env.steps_count = 0
        env.max_steps = 200
        env.player_units = {1: [101, 102, 301], 2: [201, 401]}
        env._deferred_done = False
        env._deferred_truncated = False
        env._preview_computer_opponent = MagicMock(return_value=True)
        env._physics_step_size = MagicMock(return_value=1)
        env._maybe_promote_defender_to_attack = MagicMock()
        env._has_combat_units = MagicMock(return_value=True)
        env._agent_attack_units_alive = MagicMock(return_value=1)
        env._refresh_bot_state = AsyncMock()
        env._refresh_unit_lists = AsyncMock()
        env._flush_bot_actions = AsyncMock()
        env._reissue_non_squad_orders = AsyncMock()
        env._advance_trainee_pool_index = AsyncMock()
        env._get_squad_observation_for_pool = MagicMock(
            return_value={
                "map": np.zeros((1, 128, 128), dtype=np.float32),
                "command": np.zeros(10, dtype=np.float32),
                "anchor_tactical": np.zeros((1, 32, 32), dtype=np.float32),
                "unit_scalars": np.zeros((24, SQUAD_UNIT_FEAT_DIM), dtype=np.float32),
                "unit_mask": np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32),
            }
        )
        env._apply_squad_action_for_pool = AsyncMock()
        env._get_joint_observation = AsyncMock(
            return_value={
                "map": np.zeros((1, 128, 128), dtype=np.float32),
                "command": np.zeros(10, dtype=np.float32),
                "anchor_tactical": np.zeros((1, 32, 32), dtype=np.float32),
                "unit_scalars": np.zeros((24, SQUAD_UNIT_FEAT_DIM), dtype=np.float32),
                "unit_mask": np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32),
            }
        )
        env._pool_player_id = lambda pool: 1 if pool["race"] == "terran" else 2
        env._is_pool_trainee = lambda pool: pool["race"] == "terran"
        env._paired_opponent_pool_index = MagicMock(return_value=1)
        env._pool_alive_tags = lambda idx: env.squad_unit_tags[idx][1]
        env.bot = MagicMock()
        env.bot.client = MagicMock()
        env.bot.client.step = AsyncMock()
        env.bot.structures = MagicMock(return_value=MagicMock(amount=1))
        env.bot.enemy_structures = MagicMock(return_value=MagicMock(amount=1))
        env.bot.calculate_squad_step_reward = MagicMock(return_value=0.5)
        env.bot.squad_action_masks = MagicMock(return_value=np.ones(1136, dtype=bool))
        yield env


def test_requires_policy_action_true_when_trainee_pool_alive(squad_env):
    assert squad_env.requires_policy_action() is True


def test_requires_policy_action_false_when_trainee_pool_empty(squad_env):
    squad_env.squad_unit_tags[0] = (1, [])
    assert squad_env.requires_policy_action() is False


def test_get_obs_focus_unit_tag_returns_minus_one(squad_env):
    assert squad_env.get_obs_focus_unit_tag() == -1


def test_squad_pointer_step_applies_trainee_and_advances(squad_env):
    action = np.zeros(len(squad_action_space_nvec()), dtype=np.int64)
    action[0] = 1
    obs, reward, done, truncated, info = asyncio.run(
        squad_env._async_step_squad_pointer(action)
    )
    squad_env._apply_squad_action_for_pool.assert_awaited_once()
    squad_env.bot.client.step.assert_awaited()
    squad_env._advance_trainee_pool_index.assert_awaited()
    assert reward == 0.5
    assert info.get("squad_pointer") is True


def test_paired_opponent_pool_index_integration(squad_env):
    from train.envs.ctf.squad_pool_pairing import paired_opponent_pool_index

    squad_env._is_pool_trainee = squad_env._is_pool_trainee
    idx = paired_opponent_pool_index(
        0, squad_env.unit_pools_config, is_pool_trainee=squad_env._is_pool_trainee
    )
    assert idx == 1


def test_squad_action_masks_reuses_cache_without_rebuilding_obs(squad_env):
    obs = {
        "unit_mask": np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32),
    }
    mask = np.ones(26, dtype=bool)
    squad_env.steps_count = 3
    squad_env.current_trainee_pool_index = 0
    squad_env.bot.squad_action_masks = MagicMock(return_value=mask)
    squad_env._cache_squad_action_mask(obs, 0)

    with patch.object(
        squad_env, "_get_squad_observation_for_pool", side_effect=AssertionError("should not rebuild obs")
    ), patch.object(
        squad_env, "_squad_unit_mask_for_pool", side_effect=AssertionError("should not rebuild unit mask")
    ):
        assert np.array_equal(squad_env.action_masks(), mask)
        assert np.array_equal(squad_env.action_masks(), mask)
    assert squad_env.bot.squad_action_masks.call_count == 1


def test_squad_action_masks_uses_alive_tags_not_full_obs(squad_env):
    unit_mask = np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32)
    flat_mask = np.ones(1136, dtype=bool)
    squad_env.steps_count = 2
    squad_env.current_trainee_pool_index = 0
    squad_env._squad_unit_mask_for_pool = MagicMock(return_value=unit_mask)
    squad_env.bot.squad_action_masks = MagicMock(return_value=flat_mask)

    with patch.object(
        squad_env, "_get_squad_observation_for_pool", side_effect=AssertionError("should not rebuild obs")
    ):
        out = squad_env.action_masks()

    squad_env._squad_unit_mask_for_pool.assert_called_once_with(0)
    squad_env.bot.squad_action_masks.assert_called_once()
    assert np.array_equal(out, flat_mask)


def test_hard_anchor_tag_remembered_and_resolved(squad_env):
    """Option A: previous-step hard selection drives the next obs crop tag."""
    squad_env._squad_anchor_tag_by_pool = {}
    squad_env._remember_squad_anchor_tag(0, 102)
    assert squad_env._resolve_squad_anchor_tag(0, [101, 102]) == 102
    # Dead preferred tag falls back to first sorted alive.
    assert squad_env._resolve_squad_anchor_tag(0, [101]) == 101
    # No memory → provisional first sorted alive.
    squad_env._squad_anchor_tag_by_pool = {}
    assert squad_env._resolve_squad_anchor_tag(0, [102, 101]) == 101
