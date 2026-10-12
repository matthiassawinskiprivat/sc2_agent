"""Fog-of-war behaviour for CTF pylon HP and distance shaping."""
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


ZERGLING = econ_value(25, 0)
ENEMY_COEFF = 3.0
PYLON_MAX = 40.0


def _make_unit(tag, pos, *, type_id=UnitTypeId.ZERGLING, owner_id=1, is_structure=False):
    u = MagicMock()
    u.tag = tag
    u.position = pos
    u.owner_id = owner_id
    u.is_structure = is_structure
    u.health = PYLON_MAX if is_structure else 35.0
    u.health_max = PYLON_MAX if is_structure else 35.0
    u.shield = 0.0
    u.shield_max = 0.0
    u.type_id = UnitTypeId.PYLON if is_structure else type_id
    u.passengers = []
    return u


def _make_bot(**reward_overrides):
    rewards = {
        "distance_multiplier": 1.0,
        "pylon_value_scale": 1.0,
        "pylon_own_value_penalty": 0.0,
        "attacker_enemy_value_reward": ENEMY_COEFF,
        "attacker_own_value_penalty": 0.0,
        "defender_enemy_value_reward": 0.0,
        "defender_own_value_penalty": 0.0,
        "melee_multiplier": 0.0,
        "heal_value_reward": 0.0,
        "step_penalty": 0.0,
        "cooldown_retreat_scale": 0.0,
        "defender_zone_leave_penalty": 0.0,
        "detection_reward": 0.0,
        "aoe_land_reward": 0.0,
        "objective_credit_min_scale": 1.0,
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
    bot.prev_dist_per_squad = {}
    bot.enemy_pylon_pos = {1: None, 2: None}
    return bot


def test_distance_reward_fog_of_war():
    """Attacker CoM distance shaping uses the persistent enemy-pylon position in fog."""
    bot = _make_bot(pylon_value_scale=0.0, attacker_enemy_value_reward=0.0)
    pylon_pos = Point2((50.0, 50.0))
    start_pos = Point2((10.0, 10.0))
    closer_pos = Point2((20.0, 20.0))

    own = _make_unit(101, closer_pos, type_id=UnitTypeId.REAPER, owner_id=1)
    own_pylon = _make_unit(888, Point2((0.0, 0.0)), owner_id=1, is_structure=True)
    enemy_pylon = _make_unit(999, pylon_pos, owner_id=2, is_structure=True)

    bot.units = [own]
    bot.enemy_units = []
    bot.structures = [own_pylon]
    bot.enemy_structures = [enemy_pylon]
    bot.unit_command_map = {101: "attack"}
    bot.active_unit_tag = 101
    bot.prime_combat_reward_baselines()

    # Hide the enemy pylon but keep the remembered objective.
    bot.enemy_structures = []
    bot.enemy_pylon_pos[1] = pylon_pos
    bot.is_visible = MagicMock(return_value=False)
    bot.prev_dist_per_squad[(1, "attack")] = start_pos.distance_to(pylon_pos)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=1)
    dist_change = start_pos.distance_to(pylon_pos) - closer_pos.distance_to(pylon_pos)
    assert reward == pytest.approx(dist_change * 1.0)


def test_hp_reward_fog_of_war():
    """Missing enemy pylon in fog must not count as destroyed."""
    bot = _make_bot()
    pylon_pos = Point2((50.0, 50.0))
    own = _make_unit(101, Point2((0.0, 0.0)), type_id=UnitTypeId.REAPER, owner_id=1)
    enemy = _make_unit(50, Point2((40.0, 40.0)), type_id=UnitTypeId.ZERGLING, owner_id=2)
    own_pylon = _make_unit(888, Point2((0.0, 0.0)), owner_id=1, is_structure=True)
    enemy_pylon = _make_unit(999, pylon_pos, owner_id=2, is_structure=True)

    bot.units = [own, enemy]
    bot.enemy_units = []
    bot.structures = [own_pylon]
    bot.enemy_structures = [enemy_pylon]
    bot.unit_command_map = {101: "attack"}
    bot.active_unit_tag = 101
    bot.prime_combat_reward_baselines()

    bot.enemy_structures = []
    bot.enemy_pylon_pos[1] = pylon_pos
    bot.is_visible = MagicMock(side_effect=lambda p: False if p == pylon_pos else True)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=1)
    assert reward == pytest.approx(0.0)
    assert bot.prev_pylon_health[2] == pytest.approx(PYLON_MAX)


def test_hp_reward_pylon_dead():
    """Visible empty cell at the remembered pylon position = confirmed destroy."""
    bot = _make_bot()
    pylon_pos = Point2((50.0, 50.0))
    own = _make_unit(101, Point2((0.0, 0.0)), type_id=UnitTypeId.REAPER, owner_id=1)
    enemy = _make_unit(50, Point2((40.0, 40.0)), type_id=UnitTypeId.ZERGLING, owner_id=2)
    own_pylon = _make_unit(888, Point2((0.0, 0.0)), owner_id=1, is_structure=True)
    enemy_pylon = _make_unit(999, pylon_pos, owner_id=2, is_structure=True)

    bot.units = [own, enemy]
    bot.enemy_units = []
    bot.structures = [own_pylon]
    bot.enemy_structures = [enemy_pylon]
    bot.unit_command_map = {101: "attack"}
    bot.active_unit_tag = 101
    bot.prime_combat_reward_baselines()
    frozen = bot._episode_army_value[2]

    bot.enemy_structures = []
    bot.enemy_pylon_pos[1] = pylon_pos
    bot.is_visible = MagicMock(return_value=True)

    reward = bot.calculate_squad_step_reward([101], perspective_player_id=1)
    assert reward == pytest.approx(ENEMY_COEFF * frozen)
    assert bot.prev_pylon_health[2] == pytest.approx(0.0)
