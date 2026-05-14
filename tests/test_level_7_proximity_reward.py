import pytest
from unittest.mock import MagicMock
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2


def _bare_level7_bot(training_conf: dict) -> Level7Bot:
    """Level7Bot without full BotAI init (see tests/test_level_7_greedy_logic.py)."""
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    env_conf = training_conf.get("env", {})
    allied_name = training_conf.get("unit_type", env_conf.get("unit_type", "MARINE")).upper()
    enemy_name = training_conf.get("enemy_type", env_conf.get("enemy_type", "ZERGLING")).upper()
    bot.allied_unit_type = getattr(UnitTypeId, allied_name, UnitTypeId.MARINE)
    bot.enemy_unit_type = getattr(UnitTypeId, enemy_name, UnitTypeId.ZERGLING)
    bot.previous_distances = {}
    bot.previous_danger_counts = {}
    bot.current_assignments = {}
    bot._kite_damage_credit_ok = {}
    bot._micro_chain_stage = {}
    bot._micro_chain_attack_danger = {}
    bot._micro_under_fire_min_dist_prev = {}
    bot.last_action_was_move_by_unit = {}
    bot._allied_took_damage_this_step = {}
    return bot


def _make_bot_with_pair(
    *,
    ally_pos,
    enemy_pos,
    ally_ground_range,
    enemy_ground_range,
    dist_mult=0.05,
    micro_bonus=2.0,
    unit_type="ROACH",
    enemy_type="ZERGLING",
):
    training_conf = {
        "unit_type": unit_type,
        "enemy_type": enemy_type,
        "rewards": {
            "distance_multiplier": dist_mult,
            "micro_bonus": micro_bonus,
        },
    }
    bot = _bare_level7_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2(ally_pos)
    ally.ground_range = ally_ground_range
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    enemy = MagicMock()
    enemy.tag = 201
    enemy.position = Point2(enemy_pos)
    enemy.ground_range = enemy_ground_range
    enemy.air_range = 0.0
    enemy.is_flying = False

    mock_allied = MagicMock()
    mock_allied.__iter__.return_value = [ally]
    mock_allied.amount = 1

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.__iter__.return_value = [enemy]
    mock_enemy_grp.amount = 1
    mock_enemy_grp.exists = True

    def units_side_effect(unit_type_id):
        if unit_type_id == bot.allied_unit_type:
            return mock_allied
        return MagicMock()

    def enemy_units_side_effect(unit_type_id):
        if unit_type_id == bot.enemy_unit_type:
            return mock_enemy_grp
        return MagicMock()

    bot.units = MagicMock(side_effect=units_side_effect)
    bot.units.find_by_tag.side_effect = lambda tag: ally if tag == 101 else None

    bot.enemy_units = MagicMock(side_effect=enemy_units_side_effect)
    # Code paths use self.enemy_units(type).find_by_tag(...) on the collection, not self.enemy_units.find_by_tag
    mock_enemy_grp.find_by_tag.side_effect = lambda tag: enemy if tag == 201 else None

    bot.get_active_units = MagicMock(return_value=[ally])
    return bot, ally, enemy


def test_level_7_distance_shaping_when_out_of_ally_range():
    bot, ally, _enemy = _make_bot_with_pair(
        ally_pos=(10, 10),
        enemy_pos=(20, 10),
        ally_ground_range=4.0,
        enemy_ground_range=1.0,
        dist_mult=0.05,
    )
    bot.previous_distances = {101: 10.0}
    ally.position = Point2((15, 10))
    r = bot.calculate_distance_reward(unit_tag=101)
    assert pytest.approx(r) == (10.0 - 5.0) * 0.05


def test_level_7_no_distance_shaping_when_assigned_enemy_in_ally_weapon_range():
    bot, ally, enemy = _make_bot_with_pair(
        ally_pos=(10, 10),
        enemy_pos=(12, 10),
        ally_ground_range=5.0,
        enemy_ground_range=1.0,
        dist_mult=0.1,
    )
    bot.previous_distances = {101: 8.0}
    ally.position = Point2((10, 10))
    enemy.position = Point2((12, 10))
    r = bot.calculate_distance_reward(unit_tag=101)
    dist = ally.position.distance_to(enemy.position)
    assert dist == pytest.approx(2.0)
    assert dist <= ally.ground_range
    assert r == pytest.approx(0.0)


def test_level_7_threat_penalty_uses_half_ally_range_for_long_range_unit():
    """Long-range ally vs short-range enemy: distance shaping is gated in range."""
    bot, ally, enemy = _make_bot_with_pair(
        ally_pos=(0, 0),
        enemy_pos=(6, 0),
        ally_ground_range=14.0,
        enemy_ground_range=1.0,
        dist_mult=0.2,
        unit_type="STALKER",
        enemy_type="ZERGLING",
    )
    bot.previous_distances = {}
    r = bot.calculate_distance_reward(unit_tag=101)
    assert ally.position.distance_to(enemy) == pytest.approx(6.0)
    assert r == pytest.approx(0.0)


def test_level_7_no_threat_penalty_beyond_max_of_ranges():
    """Far from enemy: only approach shaping is applied."""
    bot, ally, enemy = _make_bot_with_pair(
        ally_pos=(4, 0),
        enemy_pos=(20, 0),
        ally_ground_range=4.0,
        enemy_ground_range=1.0,
        dist_mult=0.2,
    )
    bot.previous_distances = {101: 16.0}
    ally.position = Point2((6, 0))
    enemy.position = Point2((20, 0))
    r = bot.calculate_distance_reward(unit_tag=101)
    assert ally.position.distance_to(enemy) == pytest.approx(14.0)
    assert r == pytest.approx((16.0 - 14.0) * 0.2)


def test_ranged_enemy_danger_zone_has_no_extra_expansion():
    """Ranged enemies use the same threat radius as melee (no +10% expansion)."""
    bot, ally, enemy = _make_bot_with_pair(
        ally_pos=(0, 0),
        enemy_pos=(6.2, 0),
        ally_ground_range=8.0,
        enemy_ground_range=6.0,
        micro_bonus=2.0,
        enemy_type="ROACH",
    )
    # threat = max(6, 4) = 6; dist 6.2 > 6 -> outside.
    assert bot._is_in_enemy_danger_zone(ally, enemy) is False

    enemy.ground_range = 2.0
    # max(2, 4) = 4; dist 6.2 > 4 -> outside.
    assert bot._is_in_enemy_danger_zone(ally, enemy) is False


def test_danger_zone_count_ignores_enemies_outside_tactical_fovea():
    """Far enemies in weapon range must not inflate danger count (matches 32×32 tactical crop)."""
    training_conf = {
        "unit_type": "ROACH",
        "enemy_type": "ZERGLING",
        "rewards": {"micro_bonus": 2.0},
    }
    bot = _bare_level7_bot(training_conf)

    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0, 0))
    ally.ground_range = 8.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    e_close = MagicMock()
    e_close.tag = 201
    e_close.position = Point2((3, 0))
    e_close.ground_range = 2.0
    e_close.air_range = 0.0
    e_close.is_flying = False

    e_far = MagicMock()
    e_far.tag = 202
    e_far.position = Point2((50, 0))
    e_far.ground_range = 100.0
    e_far.air_range = 0.0
    e_far.is_flying = False

    mock_allied = MagicMock()
    mock_allied.__iter__.return_value = [ally]
    mock_allied.amount = 1

    mock_enemy_grp = MagicMock()
    mock_enemy_grp.__iter__.return_value = [e_close, e_far]
    mock_enemy_grp.amount = 2
    mock_enemy_grp.exists = True

    def units_side_effect(unit_type_id):
        if unit_type_id == bot.allied_unit_type:
            return mock_allied
        return MagicMock()

    def enemy_units_side_effect(unit_type_id):
        if unit_type_id == bot.enemy_unit_type:
            return mock_enemy_grp
        return MagicMock()

    bot.units = MagicMock(side_effect=units_side_effect)
    bot.units.find_by_tag.side_effect = lambda tag: ally if tag == 101 else None
    bot.enemy_units = MagicMock(side_effect=enemy_units_side_effect)

    assert bot._is_in_enemy_danger_zone(ally, e_close) is True
    assert bot._is_in_enemy_danger_zone(ally, e_far) is True
    assert bot._count_enemy_danger_zones_for_unit(ally) == 1


