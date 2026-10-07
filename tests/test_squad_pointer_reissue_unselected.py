"""Unselected units in an acting squad must keep receiving their last orders."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


def test_finalize_reissues_unselected_pool_mates():
    """
    Regression: acting_set used to be all pool tags, so pointer-deselected
    attackers never got _reissue_non_squad_orders and stood idle.
    """
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.steps_count = 0
    env.bot = MagicMock()
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.training_conf = {"rewards": {}}
    env.max_steps = 999
    env.player_units = {1: [1, 2], 2: [10]}
    env.unit_pools_config = [{"agent": True}, {"agent": False}]
    env.squad_unit_tags = [(1, [1, 2]), (2, [10])]
    env._physics_step_size = lambda n: 8
    env._refresh_bot_state = AsyncMock()
    env._maybe_promote_defender_to_attack = MagicMock()
    env._pylon_loss_flags = MagicMock(return_value=(False, False))
    env._has_combat_units = MagicMock(return_value=True)
    env._refresh_unit_lists = AsyncMock()
    env._advance_trainee_pool_index = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value={"map": np.zeros(1)})
    env._cache_squad_action_mask = MagicMock()
    env._attach_action_masks = lambda info: info
    env.current_trainee_pool_index = 0
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env._p1_defended_on_timeout = MagicMock(return_value=False)
    env._flush_bot_actions = AsyncMock()
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)

    reissued = []

    async def _capture(acting_set, skip_player_ids=None):
        reissued.append(set(acting_set))

    env._reissue_non_squad_orders = _capture

    # Only tag 1 was pointer-selected; tag 2 must NOT be in the skip set.
    asyncio.run(
        env._finalize_squad_pointer_physics_step(
            trainee_tags=[1, 2],
            opponent_tags=[10],
            trainee_player_id=1,
            ordered_tags={1, 10},
        )
    )

    assert len(reissued) == 1
    assert 1 in reissued[0]
    assert 10 in reissued[0]
    assert 2 not in reissued[0], (
        "Unselected pool-mate must be eligible for last-order reissue"
    )


def test_null_target_noop_does_not_block_opponent_reissue():
    """
    Regression: attack_unit/heal_unit with a null target used to return the
    pointer-selected tags as ``ordered_tags``. With opponent_decision_interval>1
    the next hold steps then skipped reissue while ``_unit_last_order`` was empty,
    so both attack- and defend-role opponents stood still forever.
    """
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.steps_count = 0
    env.bot = MagicMock()
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.training_conf = {"rewards": {}}
    env.max_steps = 999
    env.player_units = {1: [1], 2: [201, 202]}
    env.unit_pools_config = [{"agent": True}, {"agent": False}]
    env.squad_unit_tags = [(1, [1]), (2, [201, 202])]
    env._physics_step_size = lambda n: 8
    env._refresh_bot_state = AsyncMock()
    env._maybe_promote_defender_to_attack = MagicMock()
    env._pylon_loss_flags = MagicMock(return_value=(False, False))
    env._has_combat_units = MagicMock(return_value=True)
    env._refresh_unit_lists = AsyncMock()
    env._advance_trainee_pool_index = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value={"map": np.zeros(1)})
    env._cache_squad_action_mask = MagicMock()
    env._attach_action_masks = lambda info: info
    env.current_trainee_pool_index = 0
    env._agent_attack_units_alive = MagicMock(return_value=1)
    env._p1_defended_on_timeout = MagicMock(return_value=False)
    env._flush_bot_actions = AsyncMock()
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)
    env.bot._unit_last_order = {
        201: ("move_global", Point2((40.0, 40.0))),
        202: ("attack_local", Point2((55.0, 55.0))),
    }

    reissued = []

    async def _capture(acting_set, skip_player_ids=None):
        reissued.append(set(acting_set))

    env._reissue_non_squad_orders = _capture

    # Trainee ordered tag 1; opponent unit-target no-op returned no ordered tags.
    asyncio.run(
        env._finalize_squad_pointer_physics_step(
            trainee_tags=[1],
            opponent_tags=[201, 202],
            trainee_player_id=1,
            ordered_tags={1},
        )
    )

    assert len(reissued) == 1
    assert 201 not in reissued[0]
    assert 202 not in reissued[0], (
        "Opponent no-op must leave tags eligible for last-order reissue"
    )


def test_reissue_skips_snipe_last_orders():
    """Reissuing snipe as move(target) cancels Yamato / Steady Targeting prep."""
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.player_units = {1: [20], 2: []}
    env.bot = MagicMock()
    env.bot._unit_last_order = {20: ("snipe", Point2((12.0, 34.0)))}
    env.bot.do = MagicMock()
    env._all_units_by_tag = MagicMock(return_value={20: MagicMock(owner_id=1)})
    env._collector_profile_enabled = MagicMock(return_value=False)

    with patch(
        "train.envs.capture_the_flag_single_env.issue_attack_at_point"
    ) as mock_atk:
        asyncio.run(env._reissue_non_squad_orders(set()))
        mock_atk.assert_not_called()
    env.bot.do.assert_not_called()


def test_reissue_skips_aoe_last_orders():
    """Reissuing aoe_local cancels Corrosive Bile / Psi Storm wind-up."""
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.player_units = {1: [21], 2: []}
    env.bot = MagicMock()
    env.bot._unit_last_order = {21: ("aoe_local", Point2((12.0, 34.0)))}
    env.bot.do = MagicMock()
    env._all_units_by_tag = MagicMock(return_value={21: MagicMock(owner_id=1)})
    env._collector_profile_enabled = MagicMock(return_value=False)

    with patch(
        "train.envs.capture_the_flag_single_env.issue_attack_at_point"
    ) as mock_atk:
        asyncio.run(env._reissue_non_squad_orders(set()))
        mock_atk.assert_not_called()
    env.bot.do.assert_not_called()
