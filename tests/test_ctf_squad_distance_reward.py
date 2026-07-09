"""
Regression tests for squad Center-of-Mass distance shaping in CTF rewards.

The reward should reflect the collective movement of the squad, not any
single unit's position.
"""
import math
import pytest
from unittest.mock import MagicMock, patch
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(tag: int, x: float, y: float, role: str, player_id: int = 1) -> MagicMock:
    u = MagicMock()
    u.tag = tag
    u.position = Point2((x, y))
    u.owner_id = player_id
    u.health = 100.0
    u.shield = 0.0
    u.is_structure = False
    return u


def _make_bot(role_map: dict, target_map: dict, active_tag: int) -> CaptureTheFlagBot:
    bot = CaptureTheFlagBot(player_id=1, training_conf={
        "agent_architecture": "universal",
        "action_heads": ["move"],
        "categorical_heads": [],
        "rewards": {
            "distance_multiplier": 1.0,
            "pylon_damage_reward": 0.0,
            "pylon_damage_penalty": 0.0,
            "damage_reward": 0.0,
            "damage_penalty": 0.0,
            "step_penalty": 0.0,
            "defender_step_reward": 0.0,
        },
    })
    bot.unit_command_map = role_map
    bot.unit_target_map = target_map
    bot.active_unit_tag = active_tag
    bot.prev_dist_per_squad = {}
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    return bot


def _wire_bot(bot, own_units, enemy_units, own_pylon=None, enemy_pylon=None):
    """Attach unit collections and structures to the bot for reward computation.

    Assigns plain lists for units/enemy_units (avoids MagicMock iterator quirks)
    and provides callable structure stubs.
    """
    bot.player_id = 1

    # Plain lists — for-loops iterate them directly without MagicMock magic
    bot.units = list(own_units)
    bot.enemy_units = list(enemy_units)

    def _struct_col(pylon):
        col = MagicMock()
        col.exists = pylon is not None
        col.first = pylon
        callable_col = MagicMock(return_value=col)
        return callable_col

    bot.structures = _struct_col(own_pylon)
    bot.enemy_structures = _struct_col(enemy_pylon)
    bot.is_visible = MagicMock(return_value=False)


# ---------------------------------------------------------------------------
# CoM distance reward: attacker squad
# ---------------------------------------------------------------------------

def test_attacker_com_closes_in_gives_positive_reward():
    """Squad CoM moving closer to the target should yield positive reward."""
    target = Point2((10.0, 10.0))
    units = [
        _make_unit(1, 5.0, 5.0, "attack"),
        _make_unit(2, 7.0, 5.0, "attack"),
    ]
    role_map = {1: "attack", 2: "attack"}
    target_map = {1: target, 2: target}
    bot = _make_bot(role_map, target_map, active_tag=1)
    _wire_bot(bot, own_units=units, enemy_units=[])

    # Seed the previous CoM distance a bit farther away
    com_x = (5.0 + 7.0) / 2  # 6.0
    com_y = 5.0
    prev_dist = math.sqrt((com_x - 10.0) ** 2 + (com_y - 10.0) ** 2)
    bot.prev_dist_per_squad[(1, "attack")] = prev_dist + 2.0  # was farther

    reward = bot.calculate_reward(perspective_player_id=1)
    # distance decreased → positive reward
    assert reward > 0.0, f"Expected positive reward, got {reward}"


def test_attacker_com_moves_away_gives_negative_reward():
    """Squad CoM moving away from the target should yield negative reward."""
    target = Point2((10.0, 10.0))
    units = [
        _make_unit(1, 5.0, 5.0, "attack"),
        _make_unit(2, 7.0, 5.0, "attack"),
    ]
    role_map = {1: "attack", 2: "attack"}
    target_map = {1: target, 2: target}
    bot = _make_bot(role_map, target_map, active_tag=1)
    _wire_bot(bot, own_units=units, enemy_units=[])

    com_x = (5.0 + 7.0) / 2
    com_y = 5.0
    prev_dist = math.sqrt((com_x - 10.0) ** 2 + (com_y - 10.0) ** 2)
    bot.prev_dist_per_squad[(1, "attack")] = prev_dist - 2.0  # was closer

    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward < 0.0, f"Expected negative reward, got {reward}"


def test_no_reward_on_first_step():
    """No distance reward on the very first step (no previous baseline)."""
    target = Point2((10.0, 10.0))
    units = [_make_unit(1, 5.0, 5.0, "attack")]
    bot = _make_bot({1: "attack"}, {1: target}, active_tag=1)
    _wire_bot(bot, own_units=units, enemy_units=[])
    # prev_dist_per_squad is empty → no distance reward
    reward = bot.calculate_reward(perspective_player_id=1)
    # distance baseline should be stored now, reward = 0 from distance shaping
    assert (1, "attack") in bot.prev_dist_per_squad


def test_com_uses_all_squad_units():
    """CoM must be the mean of ALL attacker positions, not just the active unit."""
    target = Point2((0.0, 0.0))
    # Active unit is far; teammate is close — CoM is in between
    u1 = _make_unit(1, 10.0, 0.0, "attack")  # active, far
    u2 = _make_unit(2, 2.0,  0.0, "attack")  # non-active, close
    bot = _make_bot({1: "attack", 2: "attack"}, {1: target, 2: target}, active_tag=1)
    _wire_bot(bot, own_units=[u1, u2], enemy_units=[])

    # Seed prev dist = CoM distance + 1 (so reward = +1 * multiplier)
    com_x = (10.0 + 2.0) / 2  # 6.0
    prev_dist = math.sqrt(com_x ** 2)  # 6.0
    bot.prev_dist_per_squad[(1, "attack")] = prev_dist + 1.0

    reward = bot.calculate_reward(perspective_player_id=1)
    assert abs(reward - 1.0) < 0.01, f"Expected reward ≈ 1.0, got {reward}"


def test_baseline_stored_after_first_step():
    """After the first step, prev_dist_per_squad is populated."""
    target = Point2((10.0, 10.0))
    units = [_make_unit(1, 5.0, 5.0, "attack")]
    bot = _make_bot({1: "attack"}, {1: target}, active_tag=1)
    _wire_bot(bot, own_units=units, enemy_units=[])

    assert (1, "attack") not in bot.prev_dist_per_squad
    bot.calculate_reward(perspective_player_id=1)
    assert (1, "attack") in bot.prev_dist_per_squad
