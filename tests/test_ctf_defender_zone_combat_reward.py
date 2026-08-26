"""
Defend-role combat rewards are gated by defender_zone_radius.

Damage/kill rewards only count enemies inside the zone around the defend pylon.
Own damage_penalty / kill_penalty still apply when defenders are outside the zone.
Attack-role combat rewards stay ungated.
"""
from unittest.mock import MagicMock

import pytest
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(
    tag: int,
    x: float,
    y: float,
    hp: float = 100.0,
    shield: float = 0.0,
    owner_id: int = 1,
) -> MagicMock:
    u = MagicMock()
    u.tag = tag
    u.position = Point2((x, y))
    u.owner_id = owner_id
    u.health = hp
    u.shield = shield
    u.is_structure = False
    u.passengers = []
    u.weapon_cooldown = 0.0
    u.ground_range = 5.0
    u.air_range = 0.0
    u.is_flying = False
    return u


def _make_pylon(tag: int, x: float, y: float, owner_id: int) -> MagicMock:
    p = _make_unit(tag, x, y, hp=20.0, owner_id=owner_id)
    p.is_structure = True
    return p


def _make_bot(**reward_overrides) -> CaptureTheFlagBot:
    rewards = {
        "pylon_damage_reward": 0.0,
        "pylon_damage_penalty": 0.0,
        "damage_reward": 0.0,
        "damage_penalty": 0.0,
        "heal_reward": 0.0,
        "kill_reward": 0.0,
        "kill_penalty": 0.0,
        "distance_multiplier": 0.0,
        "defender_zone_radius": 10.0,
        "step_penalty": 0.0,
        "defender_step_reward": 0.0,
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
    bot.all_units.find_by_tag.side_effect = lambda tag: by_tag.get(tag)


def _seed_enemy_snapshot(bot, enemies, *, trainee_pid: int = 1):
    snap = {}
    total_hp = 0.0
    for e in enemies:
        hp = float(e.health) + float(e.shield)
        snap[e.tag] = (hp, e.position)
        total_hp += hp
    bot.prev_enemy_combat_snapshot[trainee_pid] = snap
    bot.prev_enemy_units_health[trainee_pid] = total_hp
    bot.prev_enemy_unit_count[trainee_pid] = len(enemies)


# ---------------------------------------------------------------------------
# Damage / kill rewards gated by enemy position
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_damage_reward_inside_zone(trainee_pid):
    bot = _make_bot(damage_reward=0.2)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 5.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed_enemy_snapshot(bot, [enemy], trainee_pid=trainee_pid)
    enemy.health = 70.0
    _wire(bot, [defender], [enemy], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    assert reward == pytest.approx(30.0 * 0.2)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_no_damage_reward_outside_zone(trainee_pid):
    bot = _make_bot(damage_reward=0.2)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 20.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed_enemy_snapshot(bot, [enemy], trainee_pid=trainee_pid)
    enemy.health = 70.0
    _wire(bot, [defender], [enemy], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    assert reward == pytest.approx(0.0)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_only_credits_in_zone_enemy_when_both_damaged(trainee_pid):
    bot = _make_bot(damage_reward=1.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 1.0, 0.0)
    inside = _make_unit(201, 4.0, 0.0, hp=100.0)
    outside = _make_unit(202, 25.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed_enemy_snapshot(bot, [inside, outside], trainee_pid=trainee_pid)
    inside.health = 80.0
    outside.health = 40.0
    _wire(bot, [defender], [inside, outside], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    assert reward == pytest.approx(20.0)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_kill_reward_inside_zone(trainee_pid):
    bot = _make_bot(kill_reward=0.5)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 6.0, 0.0, hp=40.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed_enemy_snapshot(bot, [enemy], trainee_pid=trainee_pid)
    _wire(bot, [defender], [], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    assert reward == pytest.approx(0.5)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_no_kill_reward_outside_zone(trainee_pid):
    bot = _make_bot(kill_reward=0.5, damage_reward=1.0)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 2.0, 0.0)
    enemy = _make_unit(201, 18.0, 0.0, hp=40.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    _seed_enemy_snapshot(bot, [enemy], trainee_pid=trainee_pid)
    _wire(bot, [defender], [], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    assert reward == pytest.approx(0.0)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_attack_still_rewards_damage_outside_defender_zone(trainee_pid):
    bot = _make_bot(damage_reward=0.2)
    pylon = Point2((0.0, 0.0))
    attacker = _make_unit(101, 20.0, 0.0)
    enemy = _make_unit(201, 25.0, 0.0, hp=80.0)
    bot.unit_command_map = {101: "attack"}
    bot.unit_target_map = {101: Point2((40.0, 0.0))}
    bot.active_unit_tag = 101
    bot.prev_enemy_units_health[trainee_pid] = 100.0
    _wire(bot, [attacker], [enemy], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    assert reward == pytest.approx(20.0 * 0.2)


# ---------------------------------------------------------------------------
# Own attrition still penalized when defenders are outside the zone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_damage_penalty_applies_outside_zone(trainee_pid):
    bot = _make_bot(damage_penalty=0.1)
    pylon = Point2((0.0, 0.0))
    defender = _make_unit(101, 16.0, 0.0, hp=70.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    bot.prev_own_units_health[trainee_pid] = 100.0
    bot.prev_own_unit_count[trainee_pid] = 1
    _wire(bot, [defender], [], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    assert reward == pytest.approx(-30.0 * 0.1)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_defend_kill_penalty_applies_outside_zone(trainee_pid):
    bot = _make_bot(kill_penalty=-0.4)
    pylon = Point2((0.0, 0.0))
    survivor = _make_unit(101, 18.0, 0.0, hp=100.0)
    bot.unit_command_map = {101: "defend"}
    bot.unit_target_map = {101: pylon}
    bot.active_unit_tag = 101
    bot.prev_own_units_health[trainee_pid] = 200.0
    bot.prev_own_unit_count[trainee_pid] = 2
    _wire(bot, [survivor], [], pylon, trainee_pid=trainee_pid)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=trainee_pid)
    # remaining HP drop (200 → 100) is not credited as damage_penalty here (coeff 0);
    # kill_penalty still fires for the missing unit.
    assert reward == pytest.approx(-0.4)
