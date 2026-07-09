import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, call, patch
from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.data import Alliance


class _EmptyKillable:
    def __bool__(self):
        return False

    @property
    def amount(self):
        return 0

    def __iter__(self):
        return iter([])


def _fake_all_units(beacon_amount: int):
    """SC2Bot cleanup uses .filter; Level3 verification uses __call__(beacon_type)."""

    class _Fake:
        def filter(self, pred):
            return _EmptyKillable()

        def __call__(self, unit_type=None):
            coll = MagicMock()
            if unit_type == UnitTypeId.BEACON_TERRANSMALL:
                coll.amount = beacon_amount
                coll.exists = beacon_amount > 0
            else:
                coll.amount = 0
                coll.exists = False
            return coll

    return _Fake()


def test_level3_spawn_logic():
    asyncio.run(_test_level3_spawn_logic_impl())

async def _test_level3_spawn_logic_impl():
    # Mock bot and client
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 2})
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    
    # Mock pathing grid
    mock_grid = MagicMock()
    mock_grid.width = 100
    mock_grid.height = 100
    mock_grid.__getitem__.return_value = 1 
    bot.game_info.pathing_grid = mock_grid
    bot.game_info.playable_area = MagicMock(x=0, y=0, width=100, height=100)
    
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1

    bot.all_units = _fake_all_units(beacon_amount=2)

    z1, z2 = MagicMock(), MagicMock()
    z1.tag, z2.tag = 1, 2

    def units_side_effect(unit_type):
        coll = MagicMock()
        if unit_type == UnitTypeId.ZERGLING:
            coll.__iter__.return_value = [z1, z2]
        else:
            coll.exists = False
        return coll

    bot.units.side_effect = units_side_effect

    with patch.object(bot, "_refresh_state", new=AsyncMock()), patch.object(
        bot, "get_active_zerglings", return_value=[z1, z2]
    ), patch.object(bot, "get_optimal_assignments", return_value={}):
        await bot.reset_game_state()
    
    assert bot.client.debug_create_unit.call_count == 1
    
    spawn_batch = bot.client.debug_create_unit.call_args[0][0]
    zerglings = [c for c in spawn_batch if c[0] == UnitTypeId.ZERGLING]
    beacons = [c for c in spawn_batch if c[0] == UnitTypeId.BEACON_TERRANSMALL]
    
    assert len(zerglings) == 2
    assert len(beacons) == 2

def test_level3_spawn_cleanup():
    asyncio.run(_test_level3_spawn_cleanup_impl())

async def _test_level3_spawn_cleanup_impl():
    # Test that existing units are killed
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 1})
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    mock_grid = MagicMock()
    mock_grid.width = 100
    mock_grid.height = 100
    mock_grid.__getitem__.return_value = 1
    bot.game_info.pathing_grid = mock_grid
    bot.game_info.playable_area = MagicMock(x=0, y=0, width=100, height=100)
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1

    mock_z = MagicMock()
    mock_z.tag = 100
    mock_b = MagicMock()
    mock_b.tag = 200

    killable = MagicMock()
    killable.__iter__.return_value = [mock_z, mock_b]
    fake_all = _fake_all_units(beacon_amount=1)

    class _AllUnits:
        def filter(self, pred):
            return killable

        def __call__(self, unit_type=None):
            return fake_all(unit_type)

    bot.all_units = _AllUnits()

    def units_side_effect(unit_type):
        mock_collection = MagicMock()
        mock_collection.exists = True
        if unit_type == UnitTypeId.ZERGLING:
            mock_collection.__iter__.return_value = [mock_z]
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            mock_collection.__iter__.return_value = [mock_b]
        return mock_collection

    bot.units.side_effect = units_side_effect

    with patch.object(bot, "_refresh_state", new=AsyncMock()), patch.object(
        bot, "get_active_zerglings", return_value=[mock_z]
    ), patch.object(bot, "get_optimal_assignments", return_value={}):
        await bot.reset_game_state()
    
    assert bot.client.debug_kill_unit.called
    kill_calls = bot.client.debug_kill_unit.call_args_list
    all_killed_tags = []
    for c in kill_calls:
        all_killed_tags.extend(c[0][0])
        
    assert 100 in all_killed_tags
    assert 200 in all_killed_tags


def test_level3_spawn_cache_invalidates_on_geometry_change():
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 2})
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid = MagicMock(width=100, height=100)
    bot.game_info.pathing_grid.__getitem__.return_value = 1
    bot.game_info.playable_area = MagicMock(x=0, y=0, width=100, height=100)
    bot.game_info.map_size = MagicMock(width=100, height=100)

    bot._cached_spawn_points = [Point2((10, 10))]
    bot._spawn_cache_key = bot._spawn_geometry_key()

    bot.game_info.playable_area = MagicMock(x=50, y=50, width=200, height=200)
    bot.game_info.map_size = MagicMock(width=256, height=256)
    bot.game_info.pathing_grid = MagicMock(width=256, height=256)
    bot.game_info.pathing_grid.__getitem__.return_value = 1

    new_key = bot._spawn_geometry_key()
    assert new_key != bot._spawn_cache_key

    rebuilt = bot._rebuild_spawn_point_cache()
    assert len(rebuilt) > len(bot._cached_spawn_points)


def test_level3_cleanup_predicate_skips_enemy_map_units():
    bot = Level3Bot(conn=MagicMock(), training_conf={"targets": 2})
    bot.player_id = 1

    own_z = MagicMock(tag=1, owner_id=1, type_id=UnitTypeId.ZERGLING, alliance=Alliance.Self)
    own_b = MagicMock(tag=2, owner_id=1, type_id=UnitTypeId.BEACON_TERRANSMALL, alliance=Alliance.Self)
    enemy_pylon = MagicMock(tag=3, owner_id=2, type_id=UnitTypeId.PYLON, alliance=Alliance.Enemy)

    assert bot.unit_is_level3_scenario_cleanup(own_z) is True
    assert bot.unit_is_level3_scenario_cleanup(own_b) is True
    assert bot.unit_is_level3_scenario_cleanup(enemy_pylon) is False