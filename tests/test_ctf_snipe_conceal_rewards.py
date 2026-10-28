"""Snipe target value, low-energy caster spacing, and conceal / ambush shaping."""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.ctf_economic_rewards import unit_economic_value


def _unit(
    tag,
    type_id,
    x,
    y,
    owner_id,
    *,
    health=100.0,
    health_max=100.0,
    energy=0.0,
    is_flying=False,
    is_structure=False,
    is_cloaked=False,
    is_burrowed=False,
    detect_range=0.0,
    ground_range=0.0,
):
    unit = MagicMock()
    unit.tag = tag
    unit.type_id = type_id
    unit.owner_id = owner_id
    unit.position = Point2((float(x), float(y)))
    unit.health = health
    unit.health_max = health_max
    unit.shield = 0.0
    unit.shield_max = 0.0
    unit.energy = energy
    unit.is_flying = is_flying
    unit.is_structure = is_structure
    unit.is_cloaked = is_cloaked
    unit.is_burrowed = is_burrowed
    unit.detect_range = detect_range
    unit.ground_range = ground_range
    unit.air_range = 0.0
    unit.weapon_cooldown = 0.0
    unit.passengers = []
    unit.is_alive = True
    return unit


def _bot(**reward_overrides):
    rewards = {
        "attacker_enemy_value_reward": 0.0,
        "attacker_own_value_penalty": 0.0,
        "defender_enemy_value_reward": 0.0,
        "defender_own_value_penalty": 0.0,
        "heal_value_reward": 0.0,
        "pylon_value_scale": 0.0,
        "distance_multiplier": 0.0,
        "step_penalty": 0.0,
        "cooldown_retreat_scale": 0.0,
        "detection_reward": 0.0,
        "defender_zone_leave_penalty": 0.0,
        "aoe_land_reward": 0.0,
        "snipe_value_reward": 1.0,
        "spellcaster_retreat_scale": 0.6,
        "low_hp_conceal_reward": 1.0,
        "low_hp_reveal_reward": 0.75,
        "ambush_burrow_reward": 0.25,
        "ambush_unburrow_reward": 0.5,
    }
    rewards.update(reward_overrides)
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={"agent_architecture": "squad", "rewards": rewards},
    )
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=False)
    bot.structures = []
    bot.enemy_structures = []
    bot.do = MagicMock()
    return bot


def _wire(bot, own, enemies):
    bot.units = list(own) + list(enemies)
    bot.enemy_units = list(enemies)
    by_tag = {u.tag: u for u in list(own) + list(enemies)}
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = by_tag.get
    bot.unit_command_map = {u.tag: "attack" for u in own}


def _reward(bot, tags):
    bot.prime_combat_reward_baselines()
    return bot.calculate_squad_step_reward(list(tags), perspective_player_id=1)


def test_snipe_pays_more_for_a_costlier_target():
    bot = _bot()
    infestor = _unit(2, UnitTypeId.INFESTOR, 10.0, 0.0, 2)
    queen = _unit(3, UnitTypeId.QUEEN, 12.0, 0.0, 2)
    _wire(bot, [_unit(1, UnitTypeId.GHOST, 0.0, 0.0, 1)], [infestor, queen])

    bot._credit_snipe_target_value(queen)
    queen_value = bot._step_snipe_target_value
    bot._step_snipe_target_value = 0.0
    bot._snipe_value_paid_tags.clear()
    bot._credit_snipe_target_value(infestor)
    infestor_value = bot._step_snipe_target_value

    assert queen_value == pytest.approx(unit_economic_value(queen))
    assert infestor_value == pytest.approx(unit_economic_value(infestor))
    assert infestor_value > queen_value

    reward = _reward(bot, [1])
    assert reward == pytest.approx(infestor_value)
    assert bot.pop_economic_metrics()["snipe_target_value"] == pytest.approx(infestor_value)


def test_snipe_value_is_once_per_enemy_per_episode(monkeypatch):
    bot = _bot(snipe_value_reward=2.0)
    ghost = _unit(1, UnitTypeId.GHOST, 0.0, 0.0, 1, energy=75.0)
    target = _unit(2, UnitTypeId.MUTALISK, 8.0, 0.0, 2, is_flying=True)
    _wire(bot, [ghost], [target])
    monkeypatch.setattr("src.agent.ctf_bot.issue_snipe_on_unit", lambda *args, **kwargs: True)

    assert bot._issue_snipe_with_lock(ghost, target) is True
    first = bot._step_snipe_target_value
    assert first == pytest.approx(unit_economic_value(target))

    # Reaffirming the in-flight cast must not pay again.
    assert bot._issue_snipe_with_lock(ghost, target) is True
    assert bot._step_snipe_target_value == pytest.approx(first)

    # A later cast on the same body, after the lock, still pays once.
    bot._snipe_lock = {}
    assert bot._issue_snipe_with_lock(ghost, target) is True
    assert bot._step_snipe_target_value == pytest.approx(first)

    reward = _reward(bot, [1])
    assert reward == pytest.approx(2.0 * first)


def test_snipe_coefficient_zero_credits_nothing():
    bot = _bot(snipe_value_reward=0.0)
    target = _unit(2, UnitTypeId.BATTLECRUISER, 5.0, 0.0, 2)
    bot._credit_snipe_target_value(target)
    assert bot._step_snipe_target_value == pytest.approx(0.0)


def test_low_energy_spellcaster_is_paid_to_open_space_up_to_10():
    bot = _bot()
    ghost = _unit(1, UnitTypeId.GHOST, 4.0, 0.0, 1, energy=10.0)
    enemy = _unit(2, UnitTypeId.MARINE, 0.0, 0.0, 2)
    _wire(bot, [ghost], [enemy])

    bot.record_pre_action_retreat_state(1)
    assert bot._pre_action_dist_to_nearest_enemy[1] == pytest.approx(4.0)
    # Away from the frozen marine, still inside the 10-unit cap.
    ghost.position = Point2((7.0, 0.0))

    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.6 * 3.0)


def test_retreat_credit_stops_at_10_and_when_energy_can_still_cast():
    bot = _bot()
    ghost = _unit(1, UnitTypeId.GHOST, 13.0, 0.0, 1, energy=10.0)
    _wire(bot, [ghost], [_unit(2, UnitTypeId.MARINE, 0.0, 0.0, 2)])
    bot._pre_action_spell_energy[1] = 10.0
    bot._pre_action_nearest_enemy[1] = Point2((0.0, 0.0))
    bot._pre_action_dist_to_nearest_enemy[1] = 8.0
    # 8 → 13 only counts the slice that reaches 10.
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.6 * 2.0)

    bot._pre_action_dist_to_nearest_enemy[1] = 11.0
    ghost.position = Point2((14.0, 0.0))
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.0)

    # Ghost Snipe costs 50, which is cheaper than EMP. 50 energy can still cast.
    bot._pre_action_spell_energy[1] = 50.0
    bot._pre_action_dist_to_nearest_enemy[1] = 4.0
    ghost.position = Point2((7.0, 0.0))
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.0)


def test_infestor_retreats_only_below_fungal_not_neural():
    """Fungal is 75 and Neural is 100. 80 energy can still cast, so no retreat."""
    bot = _bot()
    infestor = _unit(1, UnitTypeId.INFESTOR, 6.0, 0.0, 1, energy=80.0)
    _wire(bot, [infestor], [_unit(2, UnitTypeId.MARINE, 0.0, 0.0, 2)])
    bot._pre_action_nearest_enemy[1] = Point2((0.0, 0.0))
    bot._pre_action_dist_to_nearest_enemy[1] = 2.0

    bot._pre_action_spell_energy[1] = 80.0
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.0)

    bot._pre_action_spell_energy[1] = 70.0
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.6 * 4.0)


def test_viper_retreats_below_parasitic_bomb_energy():
    """Parasitic Bomb costs 125. A Viper is a backline caster like a Ghost."""
    bot = _bot()
    viper = _unit(1, UnitTypeId.VIPER, 6.0, 0.0, 1, energy=100.0, is_flying=True)
    _wire(bot, [viper], [_unit(2, UnitTypeId.MARINE, 0.0, 0.0, 2)])
    bot._pre_action_nearest_enemy[1] = Point2((0.0, 0.0))
    bot._pre_action_dist_to_nearest_enemy[1] = 2.0

    bot._pre_action_spell_energy[1] = 100.0
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.6 * 4.0)

    bot._pre_action_spell_energy[1] = 125.0
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.0)


def test_swarm_host_on_cooldown_is_paid_to_open_space():
    """Locust cooldown plays the same role as a caster with no energy."""
    bot = _bot()
    host = _unit(1, UnitTypeId.SWARMHOSTMP, 4.0, 0.0, 1)
    host.weapon_cooldown = 40.0
    _wire(bot, [host], [_unit(2, UnitTypeId.MARINE, 0.0, 0.0, 2)])

    bot.record_pre_action_retreat_state(1)
    host.position = Point2((7.0, 0.0))
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.6 * 3.0)

    # Already 8 away; only the slice that reaches 10 is paid.
    bot._pre_action_dist_to_nearest_enemy[1] = 8.0
    host.position = Point2((13.0, 0.0))
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.6 * 2.0)


def test_ready_swarm_host_is_not_paid_to_retreat():
    bot = _bot()
    host = _unit(1, UnitTypeId.SWARMHOSTMP, 4.0, 0.0, 1)
    host.weapon_cooldown = 0.0
    _wire(bot, [host], [_unit(2, UnitTypeId.MARINE, 0.0, 0.0, 2)])
    bot.record_pre_action_retreat_state(1)
    host.position = Point2((7.0, 0.0))
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.0)


def test_burrowed_swarm_host_on_cooldown_retreats_the_same_way():
    bot = _bot()
    host = _unit(1, UnitTypeId.SWARMHOSTBURROWEDMP, 4.0, 0.0, 1, is_burrowed=True)
    host.weapon_cooldown = 25.0
    _wire(bot, [host], [_unit(2, UnitTypeId.MARINE, 0.0, 0.0, 2)])
    bot.record_pre_action_retreat_state(1)
    host.position = Point2((7.0, 0.0))
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.6 * 3.0)


def test_marine_is_not_paid_the_spellcaster_retreat():
    bot = _bot()
    marine = _unit(1, UnitTypeId.MARINE, 6.0, 0.0, 1, energy=0.0)
    _wire(bot, [marine], [_unit(2, UnitTypeId.ZERGLING, 0.0, 0.0, 2)])
    bot._pre_action_spell_energy[1] = 0.0
    bot._pre_action_nearest_enemy[1] = Point2((0.0, 0.0))
    bot._pre_action_dist_to_nearest_enemy[1] = 2.0
    assert bot.calculate_spellcaster_retreat_reward(1) == pytest.approx(0.0)


def test_low_hp_conceal_outside_detectors_and_reveal_inside():
    bot = _bot()
    roach = _unit(1, UnitTypeId.ROACH, 0.0, 0.0, 1, health=10.0, health_max=100.0)
    _wire(bot, [roach], [])

    bot._credit_conceal_command(roach, conceal=True)
    assert bot._step_conceal_shaping == pytest.approx(1.0)

    bot._step_conceal_shaping = 0.0
    roach.health = 11.0
    bot._credit_conceal_command(roach, conceal=True)
    assert bot._step_conceal_shaping == pytest.approx(0.0)

    roach.health = 8.0
    observer = _unit(
        2, UnitTypeId.OBSERVER, 5.0, 0.0, 2, is_flying=True, detect_range=11.0
    )
    _wire(bot, [roach], [observer])
    bot._credit_conceal_command(roach, conceal=True)
    assert bot._step_conceal_shaping == pytest.approx(0.0)

    roach.is_burrowed = True
    bot._credit_conceal_command(roach, conceal=False)
    assert bot._step_conceal_shaping == pytest.approx(0.75)
    reward = _reward(bot, [1])
    assert reward == pytest.approx(0.75)
    assert bot.pop_economic_metrics()["conceal_shaping"] == pytest.approx(0.75)


def test_low_hp_reveal_requires_a_detector():
    bot = _bot()
    roach = _unit(
        1,
        UnitTypeId.ROACHBURROWED,
        0.0,
        0.0,
        1,
        health=5.0,
        health_max=100.0,
        is_burrowed=True,
    )
    _wire(bot, [roach], [_unit(2, UnitTypeId.MARINE, 3.0, 0.0, 2)])
    bot._credit_conceal_command(roach, conceal=False)
    assert bot._step_conceal_shaping == pytest.approx(0.0)


def test_widow_mine_burrow_scales_with_enemies_in_range_and_ignores_low_hp_cloak():
    bot = _bot()
    mine = _unit(1, UnitTypeId.WIDOWMINE, 0.0, 0.0, 1, health=1.0, health_max=100.0)
    close = [
        _unit(10 + i, UnitTypeId.MARINE, 3.0, float(i), 2) for i in range(3)
    ]
    far = _unit(20, UnitTypeId.MARINE, 5.2, 0.0, 2)
    _wire(bot, [mine], close + [far])

    bot._credit_conceal_command(mine, conceal=True)
    # Three marines inside range 5. The far marine and the 1% health do not add
    # the generic low-HP conceal bonus.
    assert bot._step_conceal_shaping == pytest.approx(0.25 * 3)


def test_lurker_burrow_ignores_air_and_uses_reported_range():
    bot = _bot()
    lurker = _unit(1, UnitTypeId.LURKERMP, 0.0, 0.0, 1, ground_range=10.0)
    ground = _unit(2, UnitTypeId.MARINE, 9.0, 0.0, 2)
    air = _unit(3, UnitTypeId.MUTALISK, 2.0, 0.0, 2, is_flying=True)
    _wire(bot, [lurker], [ground, air])

    bot._credit_conceal_command(lurker, conceal=True)
    assert bot._step_conceal_shaping == pytest.approx(0.25)


def test_ambush_unburrow_pays_once_until_the_unit_plants_again():
    bot = _bot()
    mine = _unit(
        1,
        UnitTypeId.WIDOWMINEBURROWED,
        0.0,
        0.0,
        1,
        is_burrowed=True,
    )
    _wire(bot, [mine], [_unit(2, UnitTypeId.MARINE, 30.0, 0.0, 2)])

    bot._credit_conceal_command(mine, conceal=False)
    assert bot._step_conceal_shaping == pytest.approx(0.5)
    bot._credit_conceal_command(mine, conceal=False)
    assert bot._step_conceal_shaping == pytest.approx(0.5)

    # Planting on an enemy rearms the empty-field unburrow.
    mine.type_id = UnitTypeId.WIDOWMINE
    mine.is_burrowed = False
    enemy = _unit(3, UnitTypeId.MARINE, 2.0, 0.0, 2)
    _wire(bot, [mine], [enemy])
    bot._credit_conceal_command(mine, conceal=True)
    assert bot._step_conceal_shaping == pytest.approx(0.5 + 0.25)

    mine.type_id = UnitTypeId.WIDOWMINEBURROWED
    mine.is_burrowed = True
    _wire(bot, [mine], [_unit(4, UnitTypeId.MARINE, 40.0, 0.0, 2)])
    bot._credit_conceal_command(mine, conceal=False)
    assert bot._step_conceal_shaping == pytest.approx(0.5 + 0.25 + 0.5)


def test_ambush_unburrow_pays_nothing_while_an_enemy_is_in_range():
    bot = _bot()
    lurker = _unit(
        1,
        UnitTypeId.LURKERMPBURROWED,
        0.0,
        0.0,
        1,
        is_burrowed=True,
    )
    _wire(bot, [lurker], [_unit(2, UnitTypeId.MARINE, 6.0, 0.0, 2)])
    bot._credit_conceal_command(lurker, conceal=False)
    assert bot._step_conceal_shaping == pytest.approx(0.0)


def test_issue_cloak_burrow_credits_a_dying_roach_outside_detection():
    bot = _bot()
    roach = _unit(1, UnitTypeId.ROACH, 0.0, 0.0, 1, health=9.0, health_max=100.0)
    _wire(bot, [roach], [])
    assert bot._issue_cloak_burrow(roach, conceal=True) is True
    assert bot._step_conceal_shaping == pytest.approx(1.0)
    bot.do.assert_called_once()
