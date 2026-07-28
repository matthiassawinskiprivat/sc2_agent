"""SHIELDBATTERY spawns from the pool but is not agent-selectable (like pylons/tumors)."""
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv


def _unit(tag, type_id, owner_id, x, y, is_structure=False):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.owner_id = owner_id
    u.position = Point2((x, y))
    u.is_structure = is_structure
    u.distance_to = lambda p: abs(u.position.x - p.x) + abs(u.position.y - p.y)
    return u


def test_structures_not_squad_eligible():
    assert CaptureTheFlagSingleEnv._is_squad_eligible_unit(
        _unit(1, UnitTypeId.ZEALOT, 1, 0, 0)
    )
    battery = _unit(2, UnitTypeId.SHIELDBATTERY, 1, 0, 0, is_structure=True)
    assert not CaptureTheFlagSingleEnv._is_squad_eligible_unit(battery)
    pylon = _unit(3, UnitTypeId.PYLON, 1, 0, 0, is_structure=True)
    assert not CaptureTheFlagSingleEnv._is_squad_eligible_unit(pylon)
    tumor = _unit(4, UnitTypeId.CREEPTUMOR, 1, 0, 0, is_structure=True)
    assert not CaptureTheFlagSingleEnv._is_squad_eligible_unit(tumor)


def test_assign_unit_commands_excludes_battery_pylon_tumor():
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.unit_pools_config = [
        {
            "high_level_command": "defend",
            "race": "protoss",
            "region": "A",
            "agent": True,
        }
    ]
    env._pool_player_id = lambda pool: 1
    env.player_units = {1: [], 2: []}
    env.unit_pool_index = {}
    env.bot = MagicMock()
    env.bot.unit_command_map = {}
    env.bot.unit_target_map = {}

    zeal = _unit(10, UnitTypeId.ZEALOT, 1, 10.0, 10.0)
    battery = _unit(11, UnitTypeId.SHIELDBATTERY, 1, 10.5, 10.5, is_structure=True)
    pylon = _unit(12, UnitTypeId.PYLON, 1, 10.0, 10.0, is_structure=True)
    tumor = _unit(13, UnitTypeId.CREEPTUMOR, 1, 11.0, 11.0, is_structure=True)
    units = [zeal, battery, pylon, tumor]

    env.bot.all_units = MagicMock()
    env.bot.all_units.filter = lambda pred: [u for u in units if pred(u)]

    pool_spawn_centers = [
        (1, "defend", Point2((10.0, 10.0)), "A"),
    ]
    label_to_defender_pos = {"A": Point2((20.0, 20.0))}

    env._assign_unit_commands(pool_spawn_centers, label_to_defender_pos)

    assert 10 in env.unit_pool_index
    assert 11 not in env.unit_pool_index
    assert 12 not in env.unit_pool_index
    assert 13 not in env.unit_pool_index

    env.player_units[1] = [
        u.tag for u in units
        if u.owner_id == 1 and CaptureTheFlagSingleEnv._is_squad_eligible_unit(u)
    ]
    env._rebuild_squad_unit_tags()
    tags = env.squad_unit_tags[0][1]
    assert set(tags) == {10}
