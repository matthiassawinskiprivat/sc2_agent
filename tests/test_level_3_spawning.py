import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, call
from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId

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
    
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1
    
    # Mock units
    bot.units = MagicMock()
    bot.units.return_value.exists = False
    
    await bot.reset_game_state()
    
    assert bot.client.debug_create_unit.call_count == 4
    
    calls = bot.client.debug_create_unit.call_args_list
    zerglings = [c for c in calls if c[0][0][0][0] == UnitTypeId.ZERGLING]
    beacons = [c for c in calls if c[0][0][0][0] == UnitTypeId.BEACON_TERRANSMALL]
    
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
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1
    
    bot.units = MagicMock()
    
    mock_z = MagicMock()
    mock_z.tag = 100
    mock_b = MagicMock()
    mock_b.tag = 200
    
    def units_side_effect(unit_type):
        mock_collection = MagicMock()
        mock_collection.exists = True
        if unit_type == UnitTypeId.ZERGLING:
            mock_collection.__iter__.return_value = [mock_z]
        elif unit_type == UnitTypeId.BEACON_TERRANSMALL:
            mock_collection.__iter__.return_value = [mock_b]
        return mock_collection
        
    bot.units.side_effect = units_side_effect
    
    await bot.reset_game_state()
    
    assert bot.client.debug_kill_unit.called
    kill_calls = bot.client.debug_kill_unit.call_args_list
    all_killed_tags = []
    for c in kill_calls:
        all_killed_tags.extend(c[0][0])
        
    assert 100 in all_killed_tags
    assert 200 in all_killed_tags