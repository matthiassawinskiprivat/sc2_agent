import asyncio

import pytest
from unittest.mock import MagicMock

from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

from src.level_7_bot import Level7Bot


def _bare_bot(training_conf: dict) -> Level7Bot:
    bot = object.__new__(Level7Bot)
    bot.training_conf = training_conf
    bot.allied_unit_type = UnitTypeId.MARINE
    bot.enemy_unit_type = UnitTypeId.ZERGLING
    bot.previous_health = {}
    bot.previous_allied_health = {}
    bot.previous_danger_counts = {}
    bot.last_attack_target_tag_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot._allied_took_damage_this_step = {}
    bot._kiting_simple_prev_dist = {}
    bot._prev_enemy_in_attack_range = {}
    bot._attack_order_weapon_cd_before = {}
    bot._pending_rewards = {}
    bot._enemy_last_known_pos = {}
    return bot


def test_unit_centric_combat_delta_attributes_all_damage_to_actor():
    """All enemy damage in a step is attributed to the acting unit; pooled is always 0."""
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})

    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    e = MagicMock()
    e.tag = 201
    e.health = 8
    e.shield = 0
    e.health_max = 10
    e.shield_max = 0

    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e]
    bot.enemy_units = MagicMock(return_value=enemies)

    bot.previous_health = {201: 10.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(2.0)
    assert pooled == pytest.approx(0.0)


def test_unit_centric_combat_delta_attributes_all_multi_enemy_damage():
    """Damage to multiple enemies in one step is summed and fully attributed; pooled is 0."""
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    e1 = MagicMock(tag=201, health=8, shield=0, health_max=10, shield_max=0)
    e2 = MagicMock(tag=202, health=9, shield=0, health_max=10, shield_max=0)

    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]
    bot.enemy_units = MagicMock(return_value=enemies)

    bot.previous_health = {201: 10.0, 202: 10.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    # 2 HP from e1 + 1 HP from e2, all attributed
    assert attributed == pytest.approx(3.0)
    assert pooled == pytest.approx(0.0)


def test_unit_centric_combat_delta_attributes_kill():
    """Kill reward goes to attributed; pooled is 0."""
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    enemies = MagicMock()
    enemies.exists = False
    enemies.__iter__.return_value = []
    bot.enemy_units = MagicMock(return_value=enemies)

    bot.previous_health = {201: 3.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(10.0)
    assert pooled == pytest.approx(0.0)


def test_collateral_damage_attributed_to_closest_ally_not_acting_unit():
    """
    Damage on an enemy the acting unit did NOT target goes to the closest active ally,
    which may be a different unit than the current actor.
    """
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 10.0}})

    # Acting unit 101 targeted enemy 201 only.
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    # Enemy 201: acted-upon (2 HP dmg). Enemy 202: collateral (3 HP dmg).
    e1 = MagicMock(tag=201, health=8, shield=0, health_max=10, shield_max=0)
    e1.position = MagicMock()
    e2 = MagicMock(tag=202, health=7, shield=0, health_max=10, shield_max=0)
    e2.position = MagicMock()

    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]
    bot.enemy_units = MagicMock(return_value=enemies)
    bot.previous_health = {201: 10.0, 202: 10.0}

    # Ally 102 is the closest unit to enemy 202.
    ally_101 = MagicMock()
    ally_101.tag = 101
    ally_102 = MagicMock()
    ally_102.tag = 102
    # distance_to: ally_101 is far from e2, ally_102 is close.
    ally_101.distance_to.side_effect = lambda pos: 10.0
    ally_102.distance_to.side_effect = lambda pos: 2.0
    bot.get_active_units = MagicMock(return_value=[ally_101, ally_102])

    # Step 1: acting unit 101. Gets: direct-target damage on 201 (2.0).
    # Collateral damage on 202 (3.0) goes to ally 102 (closest), buffered.
    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(2.0)
    assert pooled == pytest.approx(0.0)
    assert bot._pending_rewards.get(102, 0.0) == pytest.approx(3.0)

    # Step 2: acting unit 102. Its pending buffer (3.0) is flushed.
    bot.last_action_was_attack_by_unit[102] = False
    enemies2 = MagicMock()
    enemies2.exists = True
    enemies2.__iter__.return_value = [e1, e2]
    bot.enemy_units = MagicMock(return_value=enemies2)
    # No new damage this step.
    bot.previous_health[201] = float(e1.health + e1.shield)
    bot.previous_health[202] = float(e2.health + e2.shield)

    attributed2, pooled2 = bot.calculate_unit_centric_combat_delta(102)
    assert attributed2 == pytest.approx(3.0)   # buffered from step 1
    assert pooled2 == pytest.approx(0.0)


def test_kill_of_direct_target_attributed_to_acting_unit():
    """Kill of the acting unit's targeted enemy → full kill reward to acting unit."""
    bot = _bare_bot({"rewards": {"damage_reward": 1.0, "kill_reward": 5.0}})
    bot.last_action_was_attack_by_unit[101] = True
    bot.last_attack_target_tag_by_unit[101] = 201

    enemies = MagicMock()
    enemies.exists = False
    enemies.__iter__.return_value = []
    bot.enemy_units = MagicMock(return_value=enemies)
    bot.previous_health = {201: 3.0}

    attributed, pooled = bot.calculate_unit_centric_combat_delta(101)
    assert attributed == pytest.approx(5.0)
    assert pooled == pytest.approx(0.0)


def test_apply_action_attack_records_nearest_enemy_tag_for_attribution():
    bot = _bare_bot(
        {
            "agent_architecture": "universal",
            "action_heads": ["move_local", "attack_local"],
        }
    )
    bot.action_heads = ["move_local", "attack_local"]
    bot.agent_system = MagicMock()
    bot.agent_system.unit_agents = {"agent_0": MagicMock()}
    bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (10.0, 10.0)
    bot.last_action_was_move_by_unit = {}
    bot.last_action_was_attack_by_unit = {}
    bot.last_attack_target_tag_by_unit = {}
    bot._attack_order_weapon_cd_before = {}
    bot.do = MagicMock()

    unit = MagicMock()
    unit.tag = 101
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = unit

    e1 = MagicMock()
    e1.tag = 201
    e1.position = Point2((8.0, 8.0))
    e2 = MagicMock()
    e2.tag = 202
    e2.position = Point2((40.0, 40.0))
    enemies = MagicMock()
    enemies.exists = True
    enemies.__iter__.return_value = [e1, e2]
    bot.enemy_units = MagicMock(return_value=enemies)

    asyncio.run(bot.apply_action(101, [1, 0]))
    assert bot.last_attack_target_tag_by_unit[101] == 201
    unit.attack.assert_called_once()
