"""CTF heal_reward: credit own HP gains only when configured."""
from unittest.mock import MagicMock

import pytest
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_unit(tag: int, health: float, shield: float = 0.0, owner_id: int = 1) -> MagicMock:
    u = MagicMock()
    u.tag = tag
    u.position = Point2((5.0, 5.0))
    u.owner_id = owner_id
    u.health = health
    u.shield = shield
    u.is_structure = False
    return u


def _make_bot(rewards: dict, trainee_pid: int = 1) -> CaptureTheFlagBot:
    bot = CaptureTheFlagBot(
        player_id=1,
        training_conf={
            "agent_architecture": "squad",
            "action_heads": ["move_local", "heal_local"],
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
    bot.prev_own_units_health[trainee_pid] = 80.0
    bot.prev_enemy_unit_count = {1: 0, 2: 0}
    bot.prev_own_unit_count = {1: 0, 2: 0}
    bot.prev_own_unit_count[trainee_pid] = 1
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


_HEAL_ZERO = {
    "heal_reward": 0.1,
    "damage_reward": 0.0,
    "damage_penalty": 0.0,
    "pylon_damage_reward": 0.0,
    "pylon_damage_penalty": 0.0,
    "distance_multiplier": 0.0,
    "step_penalty": 0.0,
    "defender_step_reward": 0.0,
    "kill_reward": 0.0,
    "kill_penalty": 0.0,
}


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_credits_own_hp_gain(trainee_pid):
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=100.0, owner_id=trainee_pid)  # was 80 → +20 HP
    _wire(bot, [unit])
    # A fresh cast populates both sets (see _issue_heal_at_point).
    bot._heal_cast_tags.add(1)
    bot._heal_order_tags.add(1)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)
    assert abs(reward - 2.0) < 1e-6  # 20 * 0.1
    assert bot._heal_cast_tags == set()
    assert bot._heal_order_tags == set()


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_ignores_hp_gain_without_cast(trainee_pid):
    """Battery ticks, Heal autocast, or Medivac attack-move HP are not heal_local."""
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=100.0, owner_id=trainee_pid)
    _wire(bot, [unit])

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)
    assert abs(reward) < 1e-6


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_credits_hp_gain_on_continuing_step_not_just_fresh_cast(trainee_pid):
    """A Medivac beam / SCV repair ticks HP over many steps after the *one*
    step that starts the walk-in/lock. heal_cast_reward is fresh-cast-only
    (``_heal_cast_tags``), but heal_reward's per-HP shaping must key off the
    broader ``_heal_order_tags`` (fresh OR continuing) or a channeled heal
    would earn HP credit on essentially none of its steps."""
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    unit = _make_unit(1, health=100.0, owner_id=trainee_pid)  # was 80 → +20 HP
    _wire(bot, [unit])
    # Continuing step: NOT in _heal_cast_tags (no fresh-cast bonus), but the
    # healer did issue a real order this step, so it is in _heal_order_tags.
    assert 1 not in bot._heal_cast_tags
    bot._heal_order_tags.add(1)

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)
    assert abs(reward - 2.0) < 1e-6  # 20 * 0.1
    assert bot._heal_order_tags == set()


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_skips_uninitialized_episode_baseline(trainee_pid):
    """Spawn HP vs prev_own=0 must not look like a 0.1×army heal."""
    bot = _make_bot(_HEAL_ZERO, trainee_pid=trainee_pid)
    bot.prev_own_units_health[trainee_pid] = 0.0
    bot.prev_own_unit_count[trainee_pid] = 0
    unit = _make_unit(1, health=400.0, owner_id=trainee_pid)
    _wire(bot, [unit])

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)
    assert abs(reward) < 1e-6
    assert bot.prev_own_units_health[trainee_pid] == pytest.approx(400.0)


@pytest.mark.parametrize("trainee_pid", [1, 2])
def test_heal_reward_absent_does_not_count(trainee_pid):
    """Missing heal_reward key → default 0; HP gain adds nothing."""
    rewards = {k: v for k, v in _HEAL_ZERO.items() if k != "heal_reward"}
    bot = _make_bot(rewards, trainee_pid=trainee_pid)
    assert "heal_reward" not in bot.training_conf["rewards"]
    unit = _make_unit(1, health=100.0, owner_id=trainee_pid)
    _wire(bot, [unit])

    reward = bot.calculate_squad_step_reward([1], perspective_player_id=trainee_pid)
    assert reward == 0.0


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
        "cooldown_retreat": 0.0,
        "heal_casts_episode": 2,
        "heal_attempts_episode": 3,
        "move_commands_episode": 0,
        "attack_commands_episode": 0,
        "cloak_commands_episode": 0,
        "uncloak_commands_episode": 0,
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
    }

    step = env._record_command_metrics([1, 2, 3])
    assert step == {"move": 1, "attack": 1, "cloak": 2, "uncloak": 1}
    assert env._episode_move_commands == 1
    assert env._episode_attack_commands == 1
    assert env._episode_cloak_commands == 2
    assert env._episode_uncloak_commands == 1
    mid = env._heal_cast_info(0, done=False, truncated=False, step_commands=step)
    assert mid["move_commands"] == 1
    assert mid["attack_commands"] == 1
    assert mid["cloak_commands"] == 2
    assert mid["uncloak_commands"] == 1
    assert "move_commands_episode" not in mid

    env.bot._command_attempt_tags = {
        "move": {2},
        "attack": set(),
        "cloak": set(),
        "uncloak": {2},
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
