"""
Regression tests for CTF cooldown-retreat shaping (tactical fovea enemy CoM).
"""
import math
import pytest
from unittest.mock import MagicMock
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(
    tag: int,
    x: float,
    y: float,
    owner_id: int = 1,
    weapon_cd: float = 0.0,
    ground_range: float = 5.0,
):
    u = MagicMock()
    u.tag = tag
    u.position = Point2((x, y))
    u.owner_id = owner_id
    u.is_structure = False
    u.health = 100.0
    u.shield = 0.0
    u.weapon_cooldown = weapon_cd
    u.ground_range = ground_range
    u.air_range = 0.0
    u.is_flying = False
    return u


def _make_bot(scale: float = 0.8) -> CaptureTheFlagBot:
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={"rewards": {"cooldown_retreat_scale": scale}},
    )
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=False)

    def _struct_col(pylon=None):
        col = MagicMock()
        col.exists = pylon is not None
        col.first = pylon
        return MagicMock(return_value=col)

    bot.structures = _struct_col()
    bot.enemy_structures = _struct_col()
    return bot


def test_cooldown_retreat_fires_when_moving_away_from_enemy_com():
    bot = _make_bot(scale=0.8)
    ally = _make_unit(101, 5.0, 0.0, owner_id=1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = ally
    bot.units = [ally]
    bot.enemy_units = [enemy]
    bot.active_unit_tag = 101

    bot._pre_action_weapon_cd_by_unit[101] = 7.0
    bot._pre_action_dist_to_enemy_com[101] = 4.0
    bot.last_action_was_move_by_unit[101] = True

    reward = bot.calculate_cooldown_retreat_reward(101, [enemy])
    assert reward == pytest.approx(0.8 * (5.0 - 4.0))


def test_no_reward_for_melee_unit():
    bot = _make_bot(scale=0.8)
    ally = _make_unit(101, 6.0, 0.0, ground_range=0.1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2, ground_range=0.1)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    bot._pre_action_weapon_cd_by_unit[101] = 5.0
    bot._pre_action_dist_to_enemy_com[101] = 4.0
    bot.last_action_was_move_by_unit[101] = True

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_when_weapon_ready():
    bot = _make_bot()
    ally = _make_unit(101, 6.0, 0.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    bot._pre_action_weapon_cd_by_unit[101] = 0.0
    bot._pre_action_dist_to_enemy_com[101] = 4.0
    bot.last_action_was_move_by_unit[101] = True

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_when_attack_command_issued():
    bot = _make_bot()
    ally = _make_unit(101, 6.0, 0.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    bot._pre_action_weapon_cd_by_unit[101] = 5.0
    bot._pre_action_dist_to_enemy_com[101] = 4.0
    bot.last_action_was_move_by_unit[101] = False

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_when_moving_toward_enemy_com():
    bot = _make_bot()
    ally = _make_unit(101, 3.0, 0.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    bot._pre_action_weapon_cd_by_unit[101] = 5.0
    bot._pre_action_dist_to_enemy_com[101] = 5.0
    bot.last_action_was_move_by_unit[101] = True

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_no_reward_when_enemy_outside_tactical_fovea():
    bot = _make_bot()
    ally = _make_unit(101, 50.0, 50.0)
    # Enemy 20 units away — outside 17-unit fovea half-extent
    enemy = _make_unit(201, 70.0, 50.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    bot._pre_action_weapon_cd_by_unit[101] = 5.0
    bot._pre_action_dist_to_enemy_com[101] = 20.0
    bot.last_action_was_move_by_unit[101] = True

    assert bot.calculate_cooldown_retreat_reward(101, [enemy]) == 0.0


def test_enemy_com_uses_all_fovea_enemies():
    bot = _make_bot(scale=1.0)
    ally = _make_unit(101, 10.0, 0.0)
    e1 = _make_unit(201, 0.0, 0.0, owner_id=2)
    e2 = _make_unit(202, 0.0, 4.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)

    com_x = (0.0 + 0.0) / 2
    com_y = (0.0 + 4.0) / 2
    com_dist = math.hypot(10.0 - com_x, 0.0 - com_y)

    bot._pre_action_weapon_cd_by_unit[101] = 3.0
    bot._pre_action_dist_to_enemy_com[101] = com_dist - 1.0
    bot.last_action_was_move_by_unit[101] = True

    reward = bot.calculate_cooldown_retreat_reward(101, [e1, e2])
    assert reward == pytest.approx(1.0)


def test_record_pre_action_retreat_state_snapshots_baseline():
    bot = _make_bot()
    ally = _make_unit(101, 8.0, 0.0, weapon_cd=6.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)
    bot.enemy_units = [enemy]
    bot.units = [ally]

    bot.record_pre_action_retreat_state(101)

    assert bot._pre_action_weapon_cd_by_unit[101] == 6.0
    assert bot._pre_action_dist_to_enemy_com[101] == pytest.approx(8.0)


def test_record_pre_action_clears_when_no_fovea_enemies():
    bot = _make_bot()
    ally = _make_unit(101, 50.0, 50.0)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    bot.all_units.find_by_tag = MagicMock(return_value=ally)
    bot.enemy_units = [enemy]
    bot.units = [ally]

    bot.record_pre_action_retreat_state(101)

    assert 101 not in bot._pre_action_dist_to_enemy_com


def test_integrated_in_calculate_reward():
    bot = _make_bot(scale=0.5)
    ally = _make_unit(101, 6.0, 0.0, owner_id=1)
    enemy = _make_unit(201, 0.0, 0.0, owner_id=2)
    own_pylon = _make_unit(888, 0.0, 0.0)
    own_pylon.is_structure = True
    own_pylon.health = 20.0
    own_pylon.shield = 0.0
    enemy_pylon = _make_unit(999, 40.0, 0.0, owner_id=2)
    enemy_pylon.is_structure = True
    enemy_pylon.health = 20.0
    enemy_pylon.shield = 0.0

    bot.all_units.find_by_tag = MagicMock(return_value=ally)
    bot.units = [ally]
    bot.enemy_units = [enemy]
    bot.active_unit_tag = 101
    bot.unit_command_map = {101: "attack"}
    bot.prev_dist_per_squad = {}

    def _struct_col(pylon):
        col = MagicMock()
        col.exists = pylon is not None
        col.first = pylon
        return MagicMock(return_value=col)

    bot.structures = _struct_col(own_pylon)
    bot.enemy_structures = _struct_col(enemy_pylon)

    bot._pre_action_weapon_cd_by_unit[101] = 4.0
    bot._pre_action_dist_to_enemy_com[101] = 5.0
    bot.last_action_was_move_by_unit[101] = True

    reward = bot.calculate_reward(perspective_player_id=1)
    expected_retreat = 0.5 * (6.0 - 5.0)
    step_penalty = -0.05
    assert reward == pytest.approx(expected_retreat + step_penalty)
