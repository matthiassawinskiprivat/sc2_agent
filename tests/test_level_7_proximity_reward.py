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
    bot._kite_phase = {}
    bot._kite_streak = {}
    bot._kiting_prev_dist = {}
    bot._kiting_chain_bonus_pending = {}
    bot._combat_reward_suppressed = {}
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


def test_micro_bonus_penalizes_enter_and_stay_in_danger_zone():
    bot, ally, enemy = _make_bot_with_pair(
        ally_pos=(0, 0),
        enemy_pos=(6, 0),
        ally_ground_range=8.0,
        enemy_ground_range=2.0,
        micro_bonus=2.0,
    )
    # threat radius = max(2, 4) = 4
    bot.previous_danger_counts = {101: 0}
    ally.position = Point2((3, 0))
    enemy.position = Point2((6, 0))
    enter_reward = bot.calculate_micro_bonus(101)
    assert enter_reward == pytest.approx(-2.0)

    ally.position = Point2((2.5, 0))
    stay_reward = bot.calculate_micro_bonus(101)
    assert stay_reward == pytest.approx(-2.0)


def test_micro_bonus_rewards_kiting_sweet_spot():
    bot, ally, enemy = _make_bot_with_pair(
        ally_pos=(0, 0),
        enemy_pos=(6, 0),
        ally_ground_range=8.0,
        enemy_ground_range=2.0,
        micro_bonus=2.0,
    )
    # threat radius = max(2, 4) = 4; sweet spot is (4, 8]
    bot.previous_danger_counts = {101: 0}
    ally.position = Point2((0, 0))
    enemy.position = Point2((5.0, 0))
    r = bot.calculate_micro_bonus(101)
    assert r == pytest.approx(2.0)


def test_ranged_enemy_danger_zone_is_wider_than_melee_geometry():
    """Ranged enemies (range > cutoff) apply RANGED_ENEMY_DANGER_ZONE_MULT to threat radius."""
    bot, ally, enemy = _make_bot_with_pair(
        ally_pos=(0, 0),
        enemy_pos=(6.2, 0),
        ally_ground_range=8.0,
        enemy_ground_range=6.0,
        micro_bonus=2.0,
        enemy_type="ROACH",
    )
    assert Level7Bot.RANGED_ENEMY_MICRO_RANGE_CUTOFF == 2.5
    assert Level7Bot.RANGED_ENEMY_DANGER_ZONE_MULT == pytest.approx(1.1)
    # Base threat = max(6, 4) = 6; dist 6.2 > 6 -> outside without buff.
    # Buffed threat = 6 * 1.1 = 6.6; dist 6.2 <= 6.6 -> inside.
    assert bot._is_in_enemy_danger_zone(ally, enemy) is True

    enemy.ground_range = 2.0
    # max(2, 4) = 4; no ranged buff; dist 6.2 > 4 -> outside.
    assert bot._is_in_enemy_danger_zone(ally, enemy) is False


def test_micro_bonus_rewards_exit_scaled_by_previous_enemy_count():
    bot = _bare_level7_bot(
        {
            "unit_type": "MARINE",
            "enemy_type": "ZERGLING",
            "rewards": {"distance_multiplier": 0.1, "micro_bonus": 2.0},
        }
    )
    ally = MagicMock()
    ally.tag = 101
    ally.position = Point2((0, 0))
    ally.ground_range = 8.0
    ally.air_range = 0.0
    ally.is_flying = False
    ally.distance_to.side_effect = lambda other: ally.position.distance_to(other.position)

    e1 = MagicMock()
    e1.tag = 201
    e1.position = Point2((1.0, 0))
    e1.ground_range = 2.0
    e1.air_range = 0.0
    e1.is_flying = False
    e2 = MagicMock()
    e2.tag = 202
    e2.position = Point2((1.5, 0))
    e2.ground_range = 2.0
    e2.air_range = 0.0
    e2.is_flying = False

    allies = MagicMock()
    allies.__iter__.return_value = [ally]
    enemies = MagicMock()
    enemies.__iter__.return_value = [e1, e2]
    enemies.exists = True

    bot.units = MagicMock(side_effect=lambda unit_type_id: allies if unit_type_id == bot.allied_unit_type else MagicMock())
    bot.units.find_by_tag.side_effect = lambda tag: ally if tag == 101 else None
    bot.enemy_units = MagicMock(side_effect=lambda unit_type_id: enemies if unit_type_id == bot.enemy_unit_type else MagicMock())

    # Transition from inside 2 zones to outside all zones -> +1.5 * 2 * micro_bonus
    bot.previous_danger_counts = {101: 2}
    ally.position = Point2((20, 20))
    r = bot.calculate_micro_bonus(101)
    assert r == pytest.approx(6.0)
