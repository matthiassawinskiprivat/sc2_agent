"""Pylon damage pays fraction × spawn-army value, not raw HP × a flat coeff.

With ``pylon_value_scale: 1.0``, destroying the enemy pylon pays the same as
wiping that owner's spawn army once (under ``attacker_enemy_value_reward``).
Early cheap armies and late expensive ones keep the same pylon∶army ratio.
"""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.ctf_economic_rewards import (
    DEFAULT_VALUE_NORM,
    DEFAULT_VESPENE_WEIGHT,
    EconomicRewardConfig,
    pylon_fraction_credit,
    snapshot_total_value,
)


def econ_value(minerals: float, vespene: float = 0.0) -> float:
    return (minerals + DEFAULT_VESPENE_WEIGHT * vespene) / DEFAULT_VALUE_NORM


ZERGLING = econ_value(25, 0)
REAPER = econ_value(50, 50)
THOR = econ_value(300, 200)
ENEMY_COEFF = 3.0
OWN_COEFF = 1.0
PYLON_MAX = 40.0


def _make_unit(tag, hp=100.0, *, type_id, owner_id=1, max_hp=None):
    u = MagicMock()
    u.tag = tag
    u.position = Point2((0.0, 0.0))
    u.owner_id = owner_id
    u.health = hp
    u.health_max = max_hp if max_hp is not None else hp
    u.shield = 0.0
    u.shield_max = 0.0
    u.is_structure = False
    u.type_id = type_id
    u.passengers = []
    return u


def _make_pylon(tag, owner_id, *, hp=None, max_hp=None):
    max_vit = max_hp if max_hp is not None else PYLON_MAX
    cur = hp if hp is not None else max_vit
    p = MagicMock()
    p.tag = tag
    p.position = Point2((10.0, 10.0))
    p.owner_id = owner_id
    p.health = cur
    p.health_max = max_vit
    p.shield = 0.0
    p.shield_max = 0.0
    p.is_structure = True
    p.type_id = UnitTypeId.PYLON
    p.passengers = []
    return p


def _make_bot(**reward_overrides) -> CaptureTheFlagBot:
    rewards = {
        "attacker_enemy_value_reward": ENEMY_COEFF,
        "attacker_own_value_penalty": 0.0,
        "defender_enemy_value_reward": 0.0,
        "defender_own_value_penalty": 0.0,
        "pylon_value_scale": 1.0,
        "pylon_own_value_penalty": OWN_COEFF,
        "defender_pylon_penalty_scale": 1.5,
        "melee_multiplier": 0.0,
        "heal_value_reward": 0.0,
        "defender_retained_value_bonus": 0.0,
        "distance_multiplier": 0.0,
        "step_penalty": 0.0,
        "cooldown_retreat_scale": 0.0,
        "objective_credit_min_scale": 1.0,
        "objective_credit_falloff": 0.0,
        "attacker_objective_radius": 1000.0,
        "defender_zone_radius": 1000.0,
        "defender_zone_leave_penalty": 0.0,
        "detection_reward": 0.0,
        "aoe_land_reward": 0.0,
    }
    rewards.update(reward_overrides)
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "agent_architecture": "squad",
            "action_heads": ["move"],
            "categorical_heads": [],
            "rewards": rewards,
        },
    )
    bot.unit_command_map = {}
    bot.unit_target_map = {}
    bot.active_unit_tag = 1
    bot.prev_dist_per_squad = {}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.is_visible = MagicMock(return_value=True)
    return bot


def _wire(bot, own_units, enemy_units, own_pylon, enemy_pylon, *, trainee_pid=1):
    enemy_pid = 2 if trainee_pid == 1 else 1
    for u in own_units:
        u.owner_id = trainee_pid
    for u in enemy_units:
        u.owner_id = enemy_pid
    own_pylon.owner_id = trainee_pid
    enemy_pylon.owner_id = enemy_pid
    bot.player_id = 1
    bot.units = list(own_units) + list(enemy_units)
    bot.enemy_units = []
    bot.structures = [own_pylon]
    bot.enemy_structures = [enemy_pylon]


def _seed(bot, own_units, enemy_units, own_pylon, enemy_pylon, *, trainee_pid=1):
    _wire(bot, own_units, enemy_units, own_pylon, enemy_pylon, trainee_pid=trainee_pid)
    bot.prime_combat_reward_baselines()


def test_pylon_fraction_credit_helper():
    assert pylon_fraction_credit(
        10.0,
        max_vitality=40.0,
        reference_army_value=1.5,
        value_scale=1.0,
        coefficient=3.0,
    ) == pytest.approx(0.25 * 1.5 * 3.0)
    assert pylon_fraction_credit(
        10.0,
        max_vitality=40.0,
        reference_army_value=1.5,
        value_scale=0.0,
        coefficient=3.0,
    ) == 0.0


def test_full_enemy_pylon_destroy_equals_wiping_enemy_spawn_army():
    """``pylon_value_scale: 1`` → full pylon = one full army wipe under the same coeff."""
    bot = _make_bot()
    own = [_make_unit(1, type_id=UnitTypeId.REAPER)]
    # Six zerglings: same normalised value as a 150/0 ling budget on ctf_level_01.
    enemies = [_make_unit(10 + i, type_id=UnitTypeId.ZERGLING) for i in range(6)]
    own_pylon = _make_pylon(888, 1)
    enemy_pylon = _make_pylon(999, 2)
    _seed(bot, own, enemies, own_pylon, enemy_pylon)

    enemy_army = 6 * ZERGLING
    assert bot._episode_army_value[2] == pytest.approx(enemy_army)

    enemy_pylon.health = 0.0
    _wire(bot, own, enemies, own_pylon, enemy_pylon)
    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    assert reward == pytest.approx(ENEMY_COEFF * enemy_army)


def test_early_and_late_armies_keep_the_same_pylon_to_army_ratio():
    """Cheap vs expensive spawn: full pylon / full army wipe stays 1.0 under scale=1."""
    def _full_pylon_over_full_army(enemy_type, n_enemies) -> float:
        bot = _make_bot()
        own = [_make_unit(1, type_id=UnitTypeId.MARINE)]
        enemies = [
            _make_unit(10 + i, type_id=enemy_type) for i in range(n_enemies)
        ]
        own_pylon = _make_pylon(888, 1)
        enemy_pylon = _make_pylon(999, 2)
        _seed(bot, own, enemies, own_pylon, enemy_pylon)
        army = bot._episode_army_value[2]
        enemy_pylon.health = 0.0
        _wire(bot, own, enemies, own_pylon, enemy_pylon)
        # Disable army-death credit so only pylon remains: keep enemies alive.
        _wire(bot, own, enemies, own_pylon, enemy_pylon)
        pylon_reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
        return pylon_reward / (ENEMY_COEFF * army)

    early = _full_pylon_over_full_army(UnitTypeId.ZERGLING, 6)
    late = _full_pylon_over_full_army(UnitTypeId.THOR, 2)
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(1.0)
    assert early == pytest.approx(late)


def test_partial_pylon_damage_scales_with_fraction_of_max_vitality():
    bot = _make_bot()
    own = [_make_unit(1, type_id=UnitTypeId.REAPER)]
    enemies = [_make_unit(10 + i, type_id=UnitTypeId.ZERGLING) for i in range(4)]
    own_pylon = _make_pylon(888, 1)
    enemy_pylon = _make_pylon(999, 2, hp=PYLON_MAX)
    _seed(bot, own, enemies, own_pylon, enemy_pylon)

    enemy_pylon.health = PYLON_MAX * 0.75  # 25% removed
    _wire(bot, own, enemies, own_pylon, enemy_pylon)
    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    assert reward == pytest.approx(0.25 * ENEMY_COEFF * 4 * ZERGLING)


def test_pylon_reference_army_value_is_frozen_at_prime():
    """Killing the enemy army mid-episode must not shrink later pylon credit."""
    bot = _make_bot()
    own = [_make_unit(1, type_id=UnitTypeId.REAPER)]
    enemies = [_make_unit(10 + i, type_id=UnitTypeId.ZERGLING) for i in range(4)]
    own_pylon = _make_pylon(888, 1)
    enemy_pylon = _make_pylon(999, 2)
    _seed(bot, own, enemies, own_pylon, enemy_pylon)
    frozen = bot._episode_army_value[2]

    # Wipe enemies first (army credit), then finish the pylon.
    _wire(bot, own, [], own_pylon, enemy_pylon)
    bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert bot._episode_army_value[2] == pytest.approx(frozen)

    enemy_pylon.health = 0.0
    _wire(bot, own, [], own_pylon, enemy_pylon)
    pylon_only = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert pylon_only == pytest.approx(ENEMY_COEFF * frozen)


def test_defender_pays_heavier_own_pylon_penalty_than_attacker():
    def _own_hit(role: str) -> float:
        bot = _make_bot()
        unit = _make_unit(101, type_id=UnitTypeId.HYDRALISK)
        own_pylon = _make_pylon(888, 1, hp=PYLON_MAX)
        enemy_pylon = _make_pylon(999, 2)
        enemy = _make_unit(50, type_id=UnitTypeId.MARINE)
        bot.unit_command_map = {101: role}
        bot.active_unit_tag = 101
        _seed(bot, [unit], [enemy], own_pylon, enemy_pylon)
        own_army = bot._episode_army_value[1]
        own_pylon.health = PYLON_MAX * 0.5  # 50% removed
        _wire(bot, [unit], [enemy], own_pylon, enemy_pylon)
        return bot.calculate_squad_step_reward([101], perspective_player_id=1), own_army

    attack_r, army = _own_hit("attack")
    defend_r, _ = _own_hit("defend")
    assert attack_r == pytest.approx(-0.5 * OWN_COEFF * army)
    assert defend_r == pytest.approx(-0.5 * OWN_COEFF * 1.5 * army)


def test_zero_pylon_value_scale_disables_pylon_credit():
    bot = _make_bot(pylon_value_scale=0.0)
    own = [_make_unit(1, type_id=UnitTypeId.REAPER)]
    enemies = [_make_unit(10, type_id=UnitTypeId.ZERGLING)]
    own_pylon = _make_pylon(888, 1)
    enemy_pylon = _make_pylon(999, 2)
    _seed(bot, own, enemies, own_pylon, enemy_pylon)
    enemy_pylon.health = 0.0
    _wire(bot, own, enemies, own_pylon, enemy_pylon)
    assert bot.calculate_squad_step_reward([1], perspective_player_id=1) == pytest.approx(
        0.0
    )


def test_economic_config_reads_pylon_value_keys():
    cfg = EconomicRewardConfig.from_rewards_conf(
        {"pylon_value_scale": 0.5, "pylon_own_value_penalty": 2.0}
    )
    assert cfg.pylon_value_scale == 0.5
    assert cfg.pylon_own_value_penalty == 2.0


def test_snapshot_total_value_sums_purchase_prices():
    from src.agent.ctf_economic_rewards import UnitValue

    snap = {
        1: UnitValue(
            type_id=1, value=1.25, max_vitality=60, vitality=30, position=None, role="attack"
        ),
        2: UnitValue(
            type_id=2, value=0.25, max_vitality=35, vitality=35, position=None, role="attack"
        ),
    }
    assert snapshot_total_value(snap) == pytest.approx(1.5)
