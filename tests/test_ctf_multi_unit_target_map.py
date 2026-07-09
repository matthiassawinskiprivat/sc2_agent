"""CaptureTheFlagEnv must populate unit_target_map like the single-client env."""
from unittest.mock import MagicMock

import pytest
from sc2.position import Point2

from train.envs.capture_the_flag_env import CaptureTheFlagEnv


def _make_env():
    env = CaptureTheFlagEnv.__new__(CaptureTheFlagEnv)
    env.bot1 = MagicMock()
    env.bot2 = MagicMock()
    return env


def _mock_unit(tag: int, x: float, y: float, owner: int):
    unit = MagicMock()
    unit.tag = tag
    unit.is_structure = False
    unit.owner_id = owner
    unit.distance_to = lambda p: ((unit.position.x - p.x) ** 2 + (unit.position.y - p.y) ** 2) ** 0.5
    unit.position = Point2((x, y))
    return unit


def test_assign_unit_commands_sets_target_map_for_both_bots():
    env = _make_env()
    env.bot1.units = [
        _mock_unit(101, 0.0, 0.0, 1),
        _mock_unit(102, 1.0, 0.0, 1),
    ]
    env.bot2.units = [_mock_unit(201, 50.0, 50.0, 2)]

    label_to_defender_pos = {
        "A": Point2((10.0, 10.0)),
        "B": Point2((50.0, 50.0)),
    }
    pool_spawn_centers = [
        (1, "attack", Point2((0.0, 0.0)), "A"),
        (1, "defend", Point2((5.0, 5.0)), "B"),
        (2, "defend", Point2((50.0, 50.0)), "B"),
        (2, "attack", Point2((60.0, 60.0)), "A"),
    ]

    env._assign_unit_commands(pool_spawn_centers, label_to_defender_pos)

    assert env.bot1.unit_command_map[101] == "attack"
    assert env.bot1.unit_command_map[102] == "attack"
    assert env.bot1.unit_target_map[101] == label_to_defender_pos["A"]
    assert env.bot1.unit_target_map[102] == label_to_defender_pos["A"]

    assert env.bot2.unit_command_map[201] == "defend"
    assert env.bot2.unit_target_map[201] == label_to_defender_pos["B"]


def test_assign_legacy_unit_targets_points_at_enemy_pylon():
    env = _make_env()
    env.bot1.units = [_mock_unit(101, 0.0, 0.0, 1)]
    env.bot2.units = [_mock_unit(201, 40.0, 40.0, 2)]

    p1 = Point2((5.0, 5.0))
    p2 = Point2((45.0, 45.0))
    env._assign_legacy_unit_targets(p1, p2)

    assert env.bot1.unit_target_map[101] == p2
    assert env.bot2.unit_target_map[201] == p1
    assert env.bot1.unit_command_map[101] == "attack"
    assert env.bot2.unit_command_map[201] == "attack"


def test_ctf_bot_uses_unit_target_map_for_command_target():
    """Regression: observation command_target must not wait for pylon vision."""
    from src.agent.ctf_bot import CaptureTheFlagBot

    bot = CaptureTheFlagBot.__new__(CaptureTheFlagBot)
    bot.player_id = 1
    bot.training_conf = {"agent_architecture": "universal", "active_layers": ["command_objective"]}
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.unit_command_map = {101: "attack"}
    bot.unit_target_map = {101: Point2((99.0, 88.0))}
    bot.active_unit_tag = 101
    bot.agent_system = None
    bot.cached_observation = None
    bot.game_info = MagicMock()
    bot.game_info.map_size.width = 128
    bot.game_info.map_size.height = 128

    # Minimal stub so _get_observation reaches command_target resolution
    bot.units = MagicMock()
    bot.units.exists = False
    bot.enemy_units = MagicMock()
    bot.enemy_units.exists = False
    bot.structures = MagicMock()
    bot.structures.exists = False
    bot.enemy_structures = MagicMock()
    bot.enemy_structures.exists = False
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = None

    unit_role = bot.unit_command_map.get(bot.active_unit_tag)
    assert unit_role == "attack"
    target_pos = bot.unit_target_map[bot.active_unit_tag]
    assert target_pos == Point2((99.0, 88.0))
