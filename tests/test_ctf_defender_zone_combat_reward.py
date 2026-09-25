"""Defend-role economic rewards are gated by ``defender_zone_radius``.

Enemy-value credit only counts enemies inside the zone around the defend pylon,
so a defender cannot farm reward by abandoning the pylon to chase a fight. Own
value loss is never zone-gated. Attack-role credit stays ungated.

Also covers the two defender-specific signals that replaced the old per-step
survival payout: the heavier pylon-damage penalty, and the terminal
retained-army-value bonus.
"""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.ctf_economic_rewards import (
    DEFAULT_VALUE_NORM,
    DEFAULT_VESPENE_WEIGHT,
)


def econ_value(minerals: float, vespene: float = 0.0) -> float:
    return (minerals + DEFAULT_VESPENE_WEIGHT * vespene) / DEFAULT_VALUE_NORM


MARINE = econ_value(50, 0)
HYDRALISK = econ_value(100, 50)


def _make_unit(tag, x, y, hp=100.0, shield=0.0, owner_id=1, *, max_hp=None,
               type_id=UnitTypeId.MARINE):
    u = MagicMock()
    u.tag = tag
    u.position = Point2((x, y))
    u.owner_id = owner_id
    u.health = hp
    u.health_max = max_hp if max_hp is not None else hp
    u.shield = shield
    u.shield_max = shield
    u.is_structure = False
    u.type_id = type_id
    u.passengers = []
    u.weapon_cooldown = 0.0
    u.ground_range = 5.0
    u.air_range = 0.0
    u.is_flying = False
    return u


def _make_pylon(tag, x, y, owner_id):
    p = _make_unit(tag, x, y, hp=20.0, owner_id=owner_id, type_id=UnitTypeId.PYLON)
    p.is_structure = True
    return p


def _make_bot(**reward_overrides) -> CaptureTheFlagBot:
    rewards = {
        "pylon_damage_reward": 0.0,
        "pylon_damage_penalty": 0.0,
        "attacker_enemy_value_reward": 0.0,
        "attacker_own_value_penalty": 0.0,
        "defender_enemy_value_reward": 0.0,
        "defender_own_value_penalty": 0.0,
        "defender_retained_value_bonus": 0.0,
        "melee_multiplier": 0.0,
        "heal_value_reward": 0.0,
        "distance_multiplier": 0.0,
        "defender_zone_radius": 10.0,
        "step_penalty": 0.0,
        "cooldown_retreat_scale": 0.0,
    }
    rewards.update(reward_overrides)
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={"agent_architecture": "squad", "rewards": rewards},
    )
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.prev_enemy_unit_count = {1: 0, 2: 0}
    bot.prev_own_unit_count = {1: 0, 2: 0}
    bot.prev_enemy_combat_snapshot = {1: {}, 2: {}}
    bot.prev_enemy_value_snapshot = {1: {}, 2: {}}
    bot.prev_own_value_snapshot = {1: {}, 2: {}}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=False)
    return bot


def _wire(bot, own_units, enemy_units, pylon_pos: Point2, *, trainee_pid: int = 1):
    enemy_pid = 2 if trainee_pid == 1 else 1
    own_pylon = _make_pylon(888, pylon_pos.x, pylon_pos.y, owner_id=trainee_pid)
    for u in own_units:
        u.owner_id = trainee_pid
    for u in enemy_units:
        u.owner_id = enemy_pid
    mobiles = list(own_units) + list(enemy_units)
    bot.units = mobiles
    bot.enemy_units = []
    bot.structures = [own_pylon]
    bot.enemy_structures = []
    by_tag = {u.tag: u for u in mobiles + [own_pylon]}
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = by_tag.get


def _seed(bot, own_units, enemy_units, pylon_pos, *, trainee_pid: int = 1):
    """Wire the field and take the pre-step value snapshot."""
    _wire(bot, own_units, enemy_units, pylon_pos, trainee_pid=trainee_pid)
    bot.prime_combat_reward_baselines()


# ---------------------------------------------------------------------------
# Enemy-value credit gated by enemy position
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_credits_enemy_value_inside_zone(trainee_pid):
    bot = _make_bot(defender_enemy_value_reward=2.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 5.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [enemy], pylon, trainee_pid=trainee_pid)

    enemy.health = 70.0
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(2.0 * 0.3 * MARINE)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_ignores_enemy_value_outside_zone(trainee_pid):
    bot = _make_bot(defender_enemy_value_reward=2.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 20.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [enemy], pylon, trainee_pid=trainee_pid)

    enemy.health = 70.0
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(0.0)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_only_credits_the_in_zone_enemy_when_both_are_damaged(trainee_pid):
    bot = _make_bot(defender_enemy_value_reward=1.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 1.0, 0.0)
    inside = _make_unit(201, 4.0, 0.0, hp=100.0)
    outside = _make_unit(202, 25.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [inside, outside], pylon, trainee_pid=trainee_pid)

    inside.health = 80.0
    outside.health = 40.0
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(0.2 * MARINE)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_credits_a_death_inside_the_zone(trainee_pid):
    bot = _make_bot(defender_enemy_value_reward=2.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 6.0, 0.0, hp=40.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [enemy], pylon, trainee_pid=trainee_pid)

    _wire(bot, [defender], [], pylon, trainee_pid=trainee_pid)
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(2.0 * MARINE)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_ignores_a_death_outside_the_zone(trainee_pid):
    bot = _make_bot(defender_enemy_value_reward=2.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 18.0, 0.0, hp=40.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [enemy], pylon, trainee_pid=trainee_pid)

    _wire(bot, [defender], [], pylon, trainee_pid=trainee_pid)
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(0.0)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_attack_still_credits_enemy_value_outside_the_defender_zone(trainee_pid):
    bot = _make_bot(attacker_enemy_value_reward=2.0)
    pylon = Point2((0.0, 0.0))
    attacker = _make_unit(101, 20.0, 0.0)
    enemy = _make_unit(201, 25.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "attack"}
    bot.unit_target_map = {101: Point2((40.0, 0.0))}
    bot.active_unit_tag = 101
    _seed(bot, [attacker], [enemy], pylon, trainee_pid=trainee_pid)

    enemy.health = 80.0
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(2.0 * 0.2 * MARINE)


# ---------------------------------------------------------------------------
# Own value loss is never zone-gated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defender_own_damage_costs_value_outside_the_zone(trainee_pid):
    bot = _make_bot(defender_own_value_penalty=1.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 16.0, 0.0, hp=100.0, type_id=UnitTypeId.HYDRALISK)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [], pylon, trainee_pid=trainee_pid)

    defender.health = 70.0
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(-0.3 * HYDRALISK)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defender_own_death_costs_value_outside_the_zone(trainee_pid):
    bot = _make_bot(defender_own_value_penalty=1.0)
    pylon = Point2((0.0, 0.0))
    survivor = _make_unit(101, 18.0, 0.0, type_id=UnitTypeId.HYDRALISK)
    doomed = _make_unit(102, 18.0, 0.0, type_id=UnitTypeId.HYDRALISK)
    bot.unit_command_map = {101: "defend", 102: "defend"}
    bot.unit_target_map = {101: pylon, 102: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [survivor, doomed], [], pylon, trainee_pid=trainee_pid)
    bot.unit_command_map = {101: "defend", 102: "defend"}

    _wire(bot, [survivor], [], pylon, trainee_pid=trainee_pid)
    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(-HYDRALISK)


# ---------------------------------------------------------------------------
# Defender-specific signals replacing the old per-step survival payout
# ---------------------------------------------------------------------------

def test_defender_pays_a_heavier_pylon_damage_penalty_than_an_attacker():
    """Protecting the pylon is the defender's job, so letting it be hit costs more."""
    def _pylon_hit(role: str) -> float:
        bot = _make_bot(pylon_damage_penalty=1.0, defender_pylon_penalty_scale=1.5)
        pylon = Point2((0.0, 0.0))
        unit = _make_unit(101, 2.0, 0.0)
        bot.unit_command_map = {101: role}
        bot.unit_target_map = {101: pylon}
        bot.active_unit_tag = 101
        _seed(bot, [unit], [], pylon, trainee_pid=1)
        bot.prev_pylon_health[1] = 20.0
        bot.structures[0].health = 12.0  # own pylon took 8 damage
        return bot.calculate_squad_step_reward([101], perspective_player_id=1)

    assert _pylon_hit("attack") == pytest.approx(-8.0)
    assert _pylon_hit("defend") == pytest.approx(-12.0)


def test_terminal_retained_value_bonus_scales_with_army_value_preserved():
    bot = _make_bot(defender_retained_value_bonus=4.0)
    pylon = Point2((0.0, 0.0))
    full = _make_unit(101, 2.0, 0.0, hp=100.0, type_id=UnitTypeId.HYDRALISK)
    half = _make_unit(102, 2.0, 0.0, hp=100.0, type_id=UnitTypeId.HYDRALISK)
    bot.unit_command_map = {101: "defend", 102: "defend"}
    bot.unit_target_map = {101: pylon, 102: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [full, half], [], pylon, trainee_pid=1)

    half.health = 50.0
    bot.calculate_squad_step_reward([101, 102], perspective_player_id=1)

    # Two equally priced defenders, one at full and one at half vitality.
    assert bot.retained_defender_value_fraction(1) == pytest.approx(0.75)
    assert bot.terminal_retained_value_bonus([101, 102], 1) == pytest.approx(3.0)


def test_terminal_retained_value_bonus_is_zero_for_attackers():
    """Attackers are paid for pressure, not for staying alive."""
    bot = _make_bot(defender_retained_value_bonus=4.0)
    pylon = Point2((0.0, 0.0))
    attacker = _make_unit(101, 2.0, 0.0, type_id=UnitTypeId.HYDRALISK)
    bot.unit_command_map = {101: "attack"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [attacker], [], pylon, trainee_pid=1)
    bot.calculate_squad_step_reward([101], perspective_player_id=1)

    assert bot.terminal_retained_value_bonus([101], 1) == pytest.approx(0.0)


def test_terminal_retained_value_bonus_is_zero_for_a_wiped_defender():
    bot = _make_bot(defender_retained_value_bonus=4.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0, type_id=UnitTypeId.HYDRALISK)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [], pylon, trainee_pid=1)

    _wire(bot, [], [], pylon, trainee_pid=1)
    bot.calculate_squad_step_reward([101], perspective_player_id=1)

    assert bot.retained_defender_value_fraction(1) == pytest.approx(0.0)
    assert bot.terminal_retained_value_bonus([101], 1) == pytest.approx(0.0)


def test_no_per_step_survival_payout_for_an_idle_defender():
    """The removed ``defender_step_reward`` was farmable by hiding all episode."""
    bot = _make_bot(step_penalty=-0.03)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0, type_id=UnitTypeId.HYDRALISK)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed(bot, [defender], [], pylon, trainee_pid=1)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=1)

    assert reward == pytest.approx(-0.03)
