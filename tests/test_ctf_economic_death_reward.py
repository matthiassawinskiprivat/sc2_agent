"""Economic death/damage credit in ``CaptureTheFlagBot.calculate_squad_step_reward``.

Replaces the old flat ``kill_reward`` / ``kill_penalty`` signals. A unit's death
now pays only the economic value that was **still standing**, so per-step damage
credit plus the death credit sum to exactly the unit's purchase price and nothing
is double-counted.

Invariants covered here:

* death pays ``remaining_fraction × value``, scaled by the unit's real price
* nothing is credited when the field did not change, or on the first step
  (no snapshot baseline yet)
* structures never count as army value
* temporary bodies (Broodlings, Locusts, Interceptors) are excluded entirely,
  while damage those bodies *deal* to permanent units still counts
* transformed/burrowed forms are priced as the same purchased unit
"""
from unittest.mock import MagicMock

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.ctf_economic_rewards import (
    DEFAULT_VALUE_NORM,
    DEFAULT_VESPENE_WEIGHT,
    FALLBACK_MINERALS,
)

ENEMY_COEFF = 2.0
OWN_COEFF = 1.0


def econ_value(minerals: float, vespene: float = 0.0) -> float:
    return (minerals + DEFAULT_VESPENE_WEIGHT * vespene) / DEFAULT_VALUE_NORM


HYDRALISK = econ_value(100, 50)
CORRUPTOR = econ_value(150, 100)
CARRIER = econ_value(350, 250)
BROODLORD = econ_value(300, 250)
SWARMHOST = econ_value(100, 75)
SIEGETANK = econ_value(150, 125)


def _make_unit(tag, hp=100.0, shield=0.0, *, max_hp=None, max_shield=None,
               is_structure=False, type_id=None):
    u = MagicMock()
    u.tag = tag
    u.position = Point2((0.0, 0.0))
    u.owner_id = 1
    u.health = hp
    u.health_max = max_hp if max_hp is not None else hp
    u.shield = shield
    u.shield_max = max_shield if max_shield is not None else shield
    u.is_structure = is_structure
    u.type_id = type_id
    u.passengers = []
    return u


def _make_bot(**reward_overrides) -> CaptureTheFlagBot:
    """Bot with only the economic combat coefficients active."""
    rewards = {
        "attacker_enemy_value_reward": ENEMY_COEFF,
        "attacker_own_value_penalty": OWN_COEFF,
        "defender_enemy_value_reward": ENEMY_COEFF,
        "defender_own_value_penalty": OWN_COEFF,
        "melee_multiplier": 0.0,
        "pylon_damage_reward": 0.0,
        "pylon_damage_penalty": 0.0,
        "heal_value_reward": 0.0,
        "defender_retained_value_bonus": 0.0,
        "distance_multiplier": 0.0,
        "step_penalty": 0.0,
        "cooldown_retreat_scale": 0.0,
    }
    rewards.update(reward_overrides)
    bot = CaptureTheFlagBot(player_id=1, training_conf={
        "agent_architecture": "universal",
        "action_heads": ["move"],
        "categorical_heads": [],
        "rewards": rewards,
    })
    bot.unit_command_map = {}
    bot.unit_target_map = {}
    bot.active_unit_tag = 1
    bot.prev_dist_per_squad = {}
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.prev_enemy_unit_count = {1: 0, 2: 0}
    bot.prev_own_unit_count = {1: 0, 2: 0}
    bot.prev_enemy_value_snapshot = {1: {}, 2: {}}
    bot.prev_own_value_snapshot = {1: {}, 2: {}}
    bot.enemy_pylon_pos = {1: None, 2: None}
    return bot


def _wire(bot, own_units, enemy_units, *, trainee_pid: int = 1):
    bot.player_id = 1
    enemy_pid = 2 if trainee_pid == 1 else 1
    for u in own_units:
        u.owner_id = trainee_pid
    for u in enemy_units:
        u.owner_id = enemy_pid
    bot.units = list(own_units) + list(enemy_units)
    bot.enemy_units = []

    def _struct_col():
        col = MagicMock()
        col.exists = False
        col.first = None
        return MagicMock(return_value=col)

    bot.structures = _struct_col()
    bot.enemy_structures = _struct_col()
    bot.is_visible = MagicMock(return_value=False)


def _seed(bot, own_units, enemy_units, *, trainee_pid: int = 1):
    """Wire the field and take the pre-step value snapshot."""
    _wire(bot, own_units, enemy_units, trainee_pid=trainee_pid)
    bot.prime_combat_reward_baselines()


# ---------------------------------------------------------------------------
# Enemy deaths credit the value still standing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_enemy_death_credits_full_value_at_full_health(trainee_pid):
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    enemy = _make_unit(99, type_id=UnitTypeId.CORRUPTOR)
    bot = _make_bot()
    _seed(bot, [own], [enemy], trainee_pid=trainee_pid)

    _wire(bot, [own], [], trainee_pid=trainee_pid)
    reward = bot.calculate_reward(perspective_player_id=trainee_pid)

    assert reward == pytest.approx(ENEMY_COEFF * CORRUPTOR)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_enemy_death_credit_scales_with_number_of_kills(trainee_pid):
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    e1 = _make_unit(98, type_id=UnitTypeId.CORRUPTOR)
    e2 = _make_unit(99, type_id=UnitTypeId.CORRUPTOR)
    bot = _make_bot()
    _seed(bot, [own], [e1, e2], trainee_pid=trainee_pid)

    _wire(bot, [own], [], trainee_pid=trainee_pid)
    reward = bot.calculate_reward(perspective_player_id=trainee_pid)

    assert reward == pytest.approx(2 * ENEMY_COEFF * CORRUPTOR)


def test_wounded_enemy_death_pays_only_remaining_value():
    """The already-damaged share was paid on earlier steps; do not pay twice."""
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    enemy = _make_unit(99, hp=200.0, type_id=UnitTypeId.CORRUPTOR)
    bot = _make_bot()
    _seed(bot, [own], [enemy], trainee_pid=1)

    enemy.health = 50.0  # 75% removed by damage
    damage = bot.calculate_reward(perspective_player_id=1)

    _wire(bot, [own], [], trainee_pid=1)
    death = bot.calculate_reward(perspective_player_id=1)

    assert damage == pytest.approx(ENEMY_COEFF * 0.75 * CORRUPTOR)
    assert death == pytest.approx(ENEMY_COEFF * 0.25 * CORRUPTOR)
    assert damage + death == pytest.approx(ENEMY_COEFF * CORRUPTOR)


def test_no_credit_when_field_unchanged():
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    enemy = _make_unit(99, type_id=UnitTypeId.CORRUPTOR)
    bot = _make_bot()
    _seed(bot, [own], [enemy], trainee_pid=1)

    assert bot.calculate_reward(perspective_player_id=1) == pytest.approx(0.0)


def test_no_credit_on_first_step_without_a_baseline():
    """An empty snapshot means "no baseline yet", not "the enemy army just died"."""
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    enemy = _make_unit(99, type_id=UnitTypeId.CORRUPTOR)
    bot = _make_bot()
    _wire(bot, [own], [enemy], trainee_pid=1)  # deliberately no prime

    assert bot.calculate_reward(perspective_player_id=1) == pytest.approx(0.0)


def test_zero_coefficients_disable_economic_shaping():
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    enemy = _make_unit(99, type_id=UnitTypeId.CORRUPTOR)
    bot = _make_bot(
        attacker_enemy_value_reward=0.0,
        attacker_own_value_penalty=0.0,
        defender_enemy_value_reward=0.0,
        defender_own_value_penalty=0.0,
    )
    _seed(bot, [own], [enemy], trainee_pid=1)

    _wire(bot, [own], [], trainee_pid=1)
    assert bot.calculate_reward(perspective_player_id=1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Own losses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_own_death_penalizes_its_value(trainee_pid):
    survivor = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    doomed = _make_unit(2, type_id=UnitTypeId.HYDRALISK)
    bot = _make_bot()
    _seed(bot, [survivor, doomed], [], trainee_pid=trainee_pid)

    _wire(bot, [survivor], [], trainee_pid=trainee_pid)
    reward = bot.calculate_reward(perspective_player_id=trainee_pid)

    assert reward == pytest.approx(-OWN_COEFF * HYDRALISK)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_own_loss_penalty_scales_with_losses(trainee_pid):
    units = [_make_unit(i, type_id=UnitTypeId.HYDRALISK) for i in (1, 2, 3)]
    bot = _make_bot()
    _seed(bot, units, [], trainee_pid=trainee_pid)

    _wire(bot, [], [], trainee_pid=trainee_pid)
    reward = bot.calculate_reward(perspective_player_id=trainee_pid)

    assert reward == pytest.approx(-3 * OWN_COEFF * HYDRALISK)


def test_attacker_loss_penalty_is_smaller_than_defender_loss_penalty():
    """Attackers must be able to learn a costly-but-necessary trade."""
    def _loss(role: str) -> float:
        survivor = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
        doomed = _make_unit(2, type_id=UnitTypeId.HYDRALISK)
        bot = _make_bot(
            attacker_own_value_penalty=0.35, defender_own_value_penalty=1.2
        )
        bot.unit_command_map = {1: role, 2: role}
        _seed(bot, [survivor, doomed], [], trainee_pid=1)
        bot.unit_command_map = {1: role, 2: role}
        _wire(bot, [survivor], [], trainee_pid=1)
        return bot.calculate_squad_step_reward([1], perspective_player_id=1)

    attacker_loss = _loss("attack")
    defender_loss = _loss("defend")

    assert attacker_loss == pytest.approx(-0.35 * HYDRALISK)
    assert defender_loss == pytest.approx(-1.2 * HYDRALISK)
    assert attacker_loss > defender_loss  # less negative


def test_value_snapshot_refreshed_each_step():
    own = [_make_unit(1, type_id=UnitTypeId.HYDRALISK),
           _make_unit(2, type_id=UnitTypeId.HYDRALISK)]
    enemies = [_make_unit(10 + i, type_id=UnitTypeId.CORRUPTOR) for i in range(3)]
    bot = _make_bot()
    _wire(bot, own, enemies, trainee_pid=1)

    bot.calculate_reward(perspective_player_id=1)

    assert set(bot.prev_own_value_snapshot[1]) == {1, 2}
    assert set(bot.prev_enemy_value_snapshot[1]) == {10, 11, 12}


# ---------------------------------------------------------------------------
# Structures are not army value
# ---------------------------------------------------------------------------

def test_structure_death_is_not_economic_army_credit():
    """Pylon loss is handled by the pylon reward, not by army value."""
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    pylon = _make_unit(200, is_structure=True, type_id=UnitTypeId.PYLON)
    bot = _make_bot()
    _seed(bot, [own], [pylon], trainee_pid=1)

    _wire(bot, [own], [], trainee_pid=1)
    assert bot.calculate_reward(perspective_player_id=1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Temporary summons carry no purchased value
# ---------------------------------------------------------------------------

def test_temporary_units_stay_out_of_the_value_snapshot():
    bot = _make_bot()
    permanent = _make_unit(1, type_id=UnitTypeId.BROODLORD)
    broodling = _make_unit(2, hp=30.0, type_id=UnitTypeId.BROODLING)
    locust = _make_unit(3, hp=50.0, type_id=UnitTypeId.LOCUSTMP)
    _wire(bot, [permanent, broodling, locust], [], trainee_pid=1)

    bot.calculate_reward(perspective_player_id=1)

    assert set(bot.prev_own_value_snapshot[1]) == {1}


@pytest.mark.parametrize(
    "temp_type",
    [UnitTypeId.BROODLING, UnitTypeId.LOCUSTMP, UnitTypeId.LOCUSTMPFLYING],
)
def test_temporary_unit_expiration_is_not_an_own_loss(temp_type):
    bot = _make_bot()
    permanent = _make_unit(1, type_id=UnitTypeId.SWARMHOSTMP)
    temp = _make_unit(2, type_id=temp_type)
    _seed(bot, [permanent, temp], [], trainee_pid=1)

    _wire(bot, [permanent], [], trainee_pid=1)
    assert bot.calculate_reward(perspective_player_id=1) == pytest.approx(0.0)


def test_temporary_unit_damage_is_not_an_own_loss():
    bot = _make_bot()
    permanent = _make_unit(1, hp=100.0, type_id=UnitTypeId.BROODLORD)
    broodling = _make_unit(2, hp=30.0, type_id=UnitTypeId.BROODLING)
    _seed(bot, [permanent, broodling], [], trainee_pid=1)

    broodling.health = 5.0
    assert bot.calculate_reward(perspective_player_id=1) == pytest.approx(0.0)


def test_permanent_death_still_costs_value_with_temps_on_the_field():
    bot = _make_bot()
    survivor = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    doomed = _make_unit(2, type_id=UnitTypeId.HYDRALISK)
    broodling = _make_unit(99, type_id=UnitTypeId.BROODLING)
    _seed(bot, [survivor, doomed, broodling], [], trainee_pid=1)

    _wire(bot, [survivor, broodling], [], trainee_pid=1)
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(-OWN_COEFF * HYDRALISK)


def test_enemy_interceptor_despawn_is_not_economic_credit():
    """Interceptor recall must not read as the Carrier's army losing value."""
    bot = _make_bot()
    own = _make_unit(1, type_id=UnitTypeId.CORRUPTOR)
    carrier = _make_unit(10, hp=300.0, max_shield=150.0, shield=150.0,
                         type_id=UnitTypeId.CARRIER)
    interceptor = _make_unit(20, hp=40.0, shield=40.0, type_id=UnitTypeId.INTERCEPTOR)
    _seed(bot, [own], [carrier, interceptor], trainee_pid=1)

    _wire(bot, [own], [carrier], trainee_pid=1)
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(0.0)
    assert set(bot.prev_enemy_value_snapshot[1]) == {10}


def test_interceptor_shots_on_a_permanent_still_cost_own_value():
    """Only the Interceptor *body* is ignored; the damage it deals is real."""
    bot = _make_bot()
    own = _make_unit(1, hp=200.0, type_id=UnitTypeId.CORRUPTOR)
    carrier = _make_unit(10, hp=300.0, type_id=UnitTypeId.CARRIER)
    interceptor = _make_unit(20, hp=40.0, shield=40.0, type_id=UnitTypeId.INTERCEPTOR)
    _seed(bot, [own], [carrier, interceptor], trainee_pid=1)

    own.health = 50.0  # shot by the Interceptor
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(-OWN_COEFF * 0.75 * CORRUPTOR)


def test_own_interceptor_damage_to_enemy_permanent_still_credits_value():
    bot = _make_bot()
    carrier = _make_unit(1, hp=300.0, type_id=UnitTypeId.CARRIER)
    interceptor = _make_unit(2, hp=40.0, shield=40.0, type_id=UnitTypeId.INTERCEPTOR)
    enemy = _make_unit(99, hp=200.0, type_id=UnitTypeId.CORRUPTOR)
    _seed(bot, [carrier, interceptor], [enemy], trainee_pid=1)

    enemy.health = 150.0
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(ENEMY_COEFF * 0.25 * CORRUPTOR)


# ---------------------------------------------------------------------------
# Transformed bodies are the same purchased unit
# ---------------------------------------------------------------------------

def test_sieging_a_tank_is_not_a_death_and_a_respawn():
    """A Sieged Tank must be priced as a Siege Tank, or sieging reads as a kill."""
    bot = _make_bot()
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    tank = _make_unit(99, hp=175.0, type_id=UnitTypeId.SIEGETANK)
    _seed(bot, [own], [tank], trainee_pid=1)

    tank.type_id = UnitTypeId.SIEGETANKSIEGED  # same tag, transformed body
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(0.0)


def test_sieged_tank_damage_is_priced_like_a_siege_tank():
    bot = _make_bot()
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    tank = _make_unit(99, hp=175.0, type_id=UnitTypeId.SIEGETANKSIEGED)
    _seed(bot, [own], [tank], trainee_pid=1)

    tank.health = 87.5
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(ENEMY_COEFF * 0.5 * SIEGETANK)


def test_unpriced_type_falls_back_to_a_nonzero_value():
    bot = _make_bot()
    own = _make_unit(1, type_id=UnitTypeId.HYDRALISK)
    unknown = _make_unit(99, type_id=None)
    _seed(bot, [own], [unknown], trainee_pid=1)

    _wire(bot, [own], [], trainee_pid=1)
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(ENEMY_COEFF * econ_value(FALLBACK_MINERALS))
