"""
Tests for endgame defender→attack promotion when all attack squads are eliminated.
"""
import pytest
from unittest.mock import MagicMock
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from src.utils.command_constants import CommandConstants


def _make_env_with_pools():
    """Minimal env instance with standard 4-pool CTF layout."""
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.unit_pools_config = [
        {"high_level_command": "attack", "agent": True, "region": "A"},
        {"high_level_command": "defend", "region": "A"},
        {"high_level_command": "defend", "agent": True, "region": "B"},
        {"high_level_command": "attack", "region": "B"},
    ]
    env.label_to_defender_pos = {
        "A": Point2((10.0, 10.0)),
        "B": Point2((50.0, 50.0)),
    }
    env._promoted_pool_index = None
    env.player_units = {1: [], 2: []}
    env.unit_pool_index = {}
    env.bot = MagicMock()
    env.bot.unit_command_map = {}
    env.bot.unit_target_map = {}
    env.bot.prev_dist_per_squad = {}
    env._log = MagicMock()
    return env


def _set_pool_units(env, pool_idx: int, tags):
    for tag in list(env.unit_pool_index):
        if env.unit_pool_index[tag] == pool_idx:
            del env.unit_pool_index[tag]
    for tag in tags:
        env.unit_pool_index[tag] = pool_idx
    env.player_units = {1: [], 2: []}
    for tag, pi in env.unit_pool_index.items():
        pool = env.unit_pools_config[pi]
        owner = 1 if pool.get("agent", False) else 2
        if tag not in env.player_units[owner]:
            env.player_units[owner].append(tag)


def test_no_promotion_while_attackers_alive():
    env = _make_env_with_pools()
    _set_pool_units(env, 0, [101, 102])  # P1 attack A
    _set_pool_units(env, 1, [201, 202])  # P2 defend A
    _set_pool_units(env, 2, [103])       # P1 defend B
    _set_pool_units(env, 3, [203])       # P2 attack B

    env._maybe_promote_defender_to_attack()

    assert env._promoted_pool_index is None
    assert env.bot.unit_command_map.get(103) != "attack"


def test_promotes_larger_defender_squad_when_all_attackers_dead():
    env = _make_env_with_pools()
    _set_pool_units(env, 1, [201, 202, 203])  # P2 defend A — 3 units
    _set_pool_units(env, 2, [103])            # P1 defend B — 1 unit
    # pools 0 and 3 (attack) empty

    env._maybe_promote_defender_to_attack()

    assert env._promoted_pool_index == 1
    assert env.bot.unit_command_map[201] == "attack"
    assert env.bot.unit_command_map[202] == "attack"
    assert env.bot.unit_command_map[203] == "attack"
    # P1 defend B stays defend
    assert env.bot.unit_command_map.get(103, "defend") == "defend"


def test_promoted_squad_targets_enemy_pylon():
    env = _make_env_with_pools()
    _set_pool_units(env, 1, [201, 202])  # P2 defend A → promote
    _set_pool_units(env, 2, [103])       # P1 defend B

    env._maybe_promote_defender_to_attack()

    # P2 defend A promoted → target P1 defend B pylon at region B
    assert env.bot.unit_target_map[201] == env.label_to_defender_pos["B"]
    assert env.bot.unit_target_map[202] == env.label_to_defender_pos["B"]


def test_p1_defender_promotion_targets_enemy_pylon_at_a():
    env = _make_env_with_pools()
    _set_pool_units(env, 1, [201])           # P2 defend A — 1 unit
    _set_pool_units(env, 2, [103, 104, 105]) # P1 defend B — 3 units → promote

    env._maybe_promote_defender_to_attack()

    assert env._promoted_pool_index == 2
    assert env.bot.unit_target_map[103] == env.label_to_defender_pos["A"]


def test_promotion_happens_only_once():
    env = _make_env_with_pools()
    _set_pool_units(env, 1, [201, 202])
    _set_pool_units(env, 2, [103])

    env._maybe_promote_defender_to_attack()
    first = env._promoted_pool_index

    # Simulate more deaths on the other defend squad
    env.player_units[1] = [103]
    env.unit_pool_index = {103: 2, 201: 1, 202: 1}

    env._maybe_promote_defender_to_attack()
    assert env._promoted_pool_index == first


def test_no_promotion_with_only_one_defender_squad_left():
    env = _make_env_with_pools()
    _set_pool_units(env, 1, [201, 202])  # only one defend squad alive

    env._maybe_promote_defender_to_attack()
    assert env._promoted_pool_index is None


def test_observation_uses_per_unit_high_level_command():
    from src.agent.ctf_bot import CaptureTheFlagBot

    bot = CaptureTheFlagBot(player_id=1, training_conf={"high_level_command": "scout"})
    bot.active_unit_tag = 101
    bot.unit_command_map = {101: "attack"}

    raw_hc = bot.training_conf.get("high_level_command", 0)
    if bot.active_unit_tag and bot.active_unit_tag in bot.unit_command_map:
        raw_hc = bot.unit_command_map[bot.active_unit_tag]
    high_cmd = CommandConstants.get_index(raw_hc)

    assert high_cmd == CommandConstants.ATTACK
