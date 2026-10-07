"""CTF healing reward: economic value restored, gated on a real heal order.

Healing pays ``heal_value_reward × restored_fraction × unit_value``, so topping up
a Battlecruiser is worth far more than topping up an SCV — the same trade a human
player makes. Credit is double-gated: this squad must have issued a heal order
*and* vitality must actually have been restored.
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

HEAL_COEFF = 4.0


def econ_value(minerals: float, vespene: float = 0.0) -> float:
    return (minerals + DEFAULT_VESPENE_WEIGHT * vespene) / DEFAULT_VALUE_NORM


MARINE = econ_value(50, 0)
SCV = econ_value(50, 0)
BATTLECRUISER = econ_value(400, 300)


def _make_unit(tag: int, health: float, shield: float = 0.0, owner_id: int = 1,
               *, max_health: float = 100.0, type_id=UnitTypeId.MARINE) -> MagicMock:
    u = MagicMock()
    u.tag = tag
    u.position = Point2((5.0, 5.0))
    u.owner_id = owner_id
    u.health = health
    u.health_max = max_health
    u.shield = shield
    u.shield_max = shield
    u.is_structure = False
    u.type_id = type_id
    u.passengers = []
    return u


def _make_bot(rewards: dict, trainee_pid: int = 1) -> CaptureTheFlagBot:
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "agent_architecture": "squad",
            "action_heads": ["move_local", "heal_unit"],
            "rewards": rewards,
        },
    )
    bot.unit_command_map = {1: "attack"}
    bot.unit_target_map = {1: Point2((10.0, 10.0))}
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


def _wire(bot, own_units):
    bot.units = list(own_units)
    bot.enemy_units = []
    empty = MagicMock()
    empty.exists = False
    empty.first = None
    bot.structures = MagicMock(return_value=empty)
    bot.enemy_structures = MagicMock(return_value=empty)
    bot.is_visible = MagicMock(return_value=False)
    by_tag = {u.tag: u for u in own_units}
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.side_effect = by_tag.get


def _seed_wounded(bot, units):
    """Wire the field and snapshot it as the pre-heal baseline."""
    _wire(bot, units)
    bot.prime_combat_reward_baselines()


_HEAL_ZERO = {
    "heal_value_reward": HEAL_COEFF,
    "attacker_enemy_value_reward": 0.0,
    "attacker_own_value_penalty": 0.0,
    "defender_enemy_value_reward": 0.0,
    "defender_own_value_penalty": 0.0,
    "defender_retained_value_bonus": 0.0,
    "melee_multiplier": 0.0,
    "pylon_damage_reward": 0.0,
    "pylon_damage_penalty": 0.0,
    "distance_multiplier": 0.0,
    "step_penalty": 0.0,
}


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_credits_restored_value(trainee_pid):
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=80.0, owner_id=trainee_pid)
    _seed_wounded(bot, [unit])
    unit.health = 100.0  # +20% of vitality restored
    # A fresh cast populates both sets (see _issue_heal_at_point).
    bot._heal_cast_tags.add(1)
    bot._heal_order_tags.add(1)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(HEAL_COEFF * 0.2 * MARINE)
    assert bot._heal_cast_tags == set()
    assert bot._heal_order_tags == set()


def test_healing_an_expensive_unit_pays_more_than_a_cheap_one():
    """Same vitality fraction restored, very different economic worth."""
    def _heal(type_id, value) -> float:
        bot = _make_bot(_HEAL_ZERO)
        unit = _make_unit(1, health=50.0, type_id=type_id)
        _seed_wounded(bot, [unit])
        unit.health = 100.0  # half its vitality restored
        bot._heal_order_tags.add(1)
        return bot.calculate_squad_step_reward([1], perspective_player_id=1)

    bc_reward = _heal(UnitTypeId.BATTLECRUISER, BATTLECRUISER)
    scv_reward = _heal(UnitTypeId.SCV, SCV)

    assert bc_reward == pytest.approx(HEAL_COEFF * 0.5 * BATTLECRUISER)
    assert scv_reward == pytest.approx(HEAL_COEFF * 0.5 * SCV)
    assert bc_reward > 15 * scv_reward


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_ignores_vitality_gain_without_an_order(trainee_pid):
    """Battery ticks, Heal autocast, or Medivac attack-move HP are not heal_unit."""
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=80.0, owner_id=trainee_pid)
    _seed_wounded(bot, [unit])
    unit.health = 100.0

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(0.0)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_credits_a_continuing_step_not_just_the_fresh_cast(trainee_pid):
    """A Medivac beam / SCV repair ticks vitality over many steps after the *one*
    step that starts the walk-in/lock. heal_cast_reward is fresh-cast-only
    (``_heal_cast_tags``), but the economic heal credit must key off the broader
    ``_heal_order_tags`` (fresh OR continuing) or a channeled heal would earn
    credit on essentially none of its steps."""
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=80.0, owner_id=trainee_pid)
    _seed_wounded(bot, [unit])
    unit.health = 100.0
    # Continuing step: NOT in _heal_cast_tags (no fresh-cast bonus), but the
    # healer did issue a real order this step, so it is in _heal_order_tags.
    assert 1 not in bot._heal_cast_tags
    bot._heal_order_tags.add(1)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(HEAL_COEFF * 0.2 * MARINE)
    assert bot._heal_order_tags == set()


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_skips_uninitialized_episode_baseline(trainee_pid):
    """Spawn vitality vs an empty snapshot must not look like healing the whole army."""
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=400.0, max_health=400.0, owner_id=trainee_pid,
                      type_id=UnitTypeId.BATTLECRUISER)
    _wire(bot, [unit])  # deliberately no baseline snapshot
    bot._heal_order_tags.add(1)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(0.0)
    assert set(bot.prev_own_value_snapshot[trainee_pid]) == {1}


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_absent_does_not_count(trainee_pid):
    """Missing heal_value_reward key → default 0; restored value adds nothing."""
    rewards = {k: v for k, v in _HEAL_ZERO.items() if k != "heal_value_reward"}
    rewards["heal_value_reward"] = 0.0
    bot = _make_bot(rewards, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=80.0, owner_id=trainee_pid)
    _seed_wounded(bot, [unit])
    unit.health = 100.0
    bot._heal_order_tags.add(1)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)

    assert reward == pytest.approx(0.0)


def test_heal_order_without_restored_vitality_pays_nothing():
    """An order that never ticks vitality must not be rewarded as healing."""
    bot = _make_bot(_HEAL_ZERO)
    unit = _make_unit(1, health=80.0)
    _seed_wounded(bot, [unit])
    bot._heal_order_tags.add(1)  # order issued, but no vitality change

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)

    assert reward == pytest.approx(0.0)


def test_count_heal_casts_for_tags_only_selected():
    bot = _make_bot(_HEAL_ZERO)
    bot._heal_cast_tags = {1, 99}
    bot._heal_attempt_tags = {1, 2, 99}
    assert bot.count_heal_casts_for_tags([1, 2]) == 1
    assert bot.count_heal_casts_for_tags([2]) == 0
    assert bot.count_heal_casts_for_tags([]) == 0
    assert bot.count_heal_attempts_for_tags([1, 2]) == 2
    assert bot.count_heal_attempts_for_tags([3]) == 0


def test_heal_cast_reward_pays_without_hp_gain():
    """Incoming damage can wipe Transfuse HP; the cast itself must still credit."""
    rewards = dict(_HEAL_ZERO)
    rewards["heal_cast_reward"] = 2.0
    bot = _make_bot(rewards)
    unit = _make_unit(1, health=80.0)  # same as prev_own 80
    _wire(bot, [unit])
    bot._heal_cast_tags.add(1)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert abs(reward - 2.0) < 1e-6
    assert bot._heal_cast_tags == set()
    assert bot._heal_attempt_tags == set()


def test_heal_cast_info_accumulates_episode_and_emits_on_done():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env._episode_heal_casts = 0
    env._episode_heal_attempts = 0
    env._episode_move_commands = 0
    env._episode_attack_commands = 0
    env._episode_cloak_commands = 0
    env._episode_uncloak_commands = 0
    env._episode_cooldown_retreat = 0.0
    env.bot = _make_bot(_HEAL_ZERO)
    env.bot._heal_cast_tags = {1, 2}
    env.bot._heal_attempt_tags = {1, 2, 3}

    casts, attempts = env._record_heal_metrics([1, 3])
    assert casts == 1
    assert attempts == 2
    assert env._episode_heal_casts == 1
    assert env._episode_heal_attempts == 2
    mid = env._heal_cast_info(casts, done=False, truncated=False, step_attempts=attempts)
    assert mid == {
        "heal_casts": 1,
        "heal_attempts": 2,
        "move_commands": 0,
        "attack_commands": 0,
        "cloak_commands": 0,
        "uncloak_commands": 0,
        "aoe_commands": 0,
        "snipe_commands": 0,
        "load_commands": 0,
        "unload_commands": 0,
        "cooldown_retreat": 0.0,
    }
    env.bot._heal_cast_tags = {3}
    env.bot._heal_attempt_tags = {3}
    casts2, attempts2 = env._record_heal_metrics([3])
    assert env._episode_heal_casts == 2
    assert env._episode_heal_attempts == 3
    terminal = env._heal_cast_info(
        casts2, done=True, truncated=False, step_attempts=attempts2
    )
    assert terminal == {
        "heal_casts": 1,
        "heal_attempts": 1,
        "move_commands": 0,
        "attack_commands": 0,
        "cloak_commands": 0,
        "uncloak_commands": 0,
        "aoe_commands": 0,
        "snipe_commands": 0,
        "load_commands": 0,
        "unload_commands": 0,
        "cooldown_retreat": 0.0,
        "heal_casts_episode": 2,
        "heal_attempts_episode": 3,
        "move_commands_episode": 0,
        "attack_commands_episode": 0,
        "cloak_commands_episode": 0,
        "uncloak_commands_episode": 0,
        "aoe_commands_episode": 0,
        "snipe_commands_episode": 0,
        "load_commands_episode": 0,
        "unload_commands_episode": 0,
        "cooldown_retreat_episode": 0.0,
    }


def test_command_metric_info_accumulates_episode_and_emits_on_done():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env._episode_heal_casts = 0
    env._episode_heal_attempts = 0
    env._episode_move_commands = 0
    env._episode_attack_commands = 0
    env._episode_cloak_commands = 0
    env._episode_uncloak_commands = 0
    env._episode_cooldown_retreat = 0.0
    env.bot = _make_bot(_HEAL_ZERO)
    env.bot._command_attempt_tags = {
        "move": {1, 99},
        "attack": {2},
        "cloak": {1, 2},
        "uncloak": {3},
        "aoe": {1},
        "load": {2, 3},
        "unload": {1},
    }

    step = env._record_command_metrics([1, 2, 3])
    assert step == {
        "move": 1,
        "attack": 1,
        "cloak": 2,
        "uncloak": 1,
        "aoe": 1,
        "snipe": 0,
        "load": 2,
        "unload": 1,
    }
    assert env._episode_move_commands == 1
    assert env._episode_attack_commands == 1
    assert env._episode_cloak_commands == 2
    assert env._episode_uncloak_commands == 1
    assert env._episode_aoe_commands == 1
    assert env._episode_load_commands == 2
    assert env._episode_unload_commands == 1
    mid = env._heal_cast_info(0, done=False, truncated=False, step_commands=step)
    assert mid["move_commands"] == 1
    assert mid["attack_commands"] == 1
    assert mid["cloak_commands"] == 2
    assert mid["uncloak_commands"] == 1
    assert mid["aoe_commands"] == 1
    assert mid["load_commands"] == 2
    assert mid["unload_commands"] == 1
    assert "move_commands_episode" not in mid

    env.bot._command_attempt_tags = {
        "move": {2},
        "attack": set(),
        "cloak": set(),
        "uncloak": {2},
        "aoe": set(),
        "load": {2},
        "unload": set(),
    }
    step2 = env._record_command_metrics([2])
    terminal = env._heal_cast_info(
        0, done=True, truncated=False, step_commands=step2
    )
    assert terminal["move_commands"] == 1
    assert terminal["uncloak_commands"] == 1
    assert terminal["move_commands_episode"] == 2
    assert terminal["attack_commands_episode"] == 1
    assert terminal["cloak_commands_episode"] == 2
    assert terminal["uncloak_commands_episode"] == 2
    assert terminal["aoe_commands_episode"] == 1
    assert terminal["load_commands_episode"] == 3
    assert terminal["unload_commands_episode"] == 1


def test_count_command_attempts_for_tags_only_selected():
    bot = _make_bot(_HEAL_ZERO)
    bot._command_attempt_tags = {
        "move": {1, 99},
        "attack": {2},
        "cloak": {1},
        "uncloak": set(),
    }
    assert bot.count_command_attempts_for_tags("move", [1, 2]) == 1
    assert bot.count_command_attempts_for_tags("attack", [1, 2]) == 1
    assert bot.count_command_attempts_for_tags("cloak", [2]) == 0
    assert bot.count_command_attempts_for_tags("uncloak", [1]) == 0
    assert bot.count_command_attempts_for_tags("move", []) == 0


def test_command_attempt_tags_cleared_with_heal_tags():
    bot = _make_bot(_HEAL_ZERO)
    unit = _make_unit(1, health=80.0)
    _wire(bot, [unit])
    bot._command_attempt_tags["move"].add(1)
    bot._command_attempt_tags["attack"].add(1)
    bot.calculate_squad_step_reward([1], perspective_player_id=1)
    assert bot._command_attempt_tags["move"] == set()
    assert bot._command_attempt_tags["attack"] == set()
