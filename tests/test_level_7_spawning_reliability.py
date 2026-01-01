import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, call
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_level_7_zergling_ratio():
    training_conf = {
        "unit_type": "ROACH",
        "enemy_type": "ZERGLING",
        "targets": 2
    }
    bot = Level7Bot(conn=AsyncMock(), training_conf=training_conf)
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid.width = 100
    bot.game_info.pathing_grid.height = 100
    bot.game_info.pathing_grid.__getitem__.return_value = 1
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1
    
    # Mock units behavior to satisfy verification loop
    bot.all_units = MagicMock()
    bot.all_units.amount = 0
    bot.all_units.__bool__.return_value = False # For cleanup loop
    
    bot.units = MagicMock()
    bot.units.return_value.amount = 2
    
    bot.enemy_units = MagicMock()
    bot.enemy_units.return_value.amount = 4
    
    # Mock observation to avoid errors in verification loop
    mock_obs = MagicMock()
    bot.client.observation.return_value = mock_obs
    
    await bot.reset_game_state()
    
    # Verify debug_create_unit calls
    calls = bot.client.debug_create_unit.call_args_list
    allied_created = sum(c[0][0][0][1] for c in calls if c[0][0][0][3] == 1)
    enemy_created = sum(c[0][0][0][1] for c in calls if c[0][0][0][3] != 1)
            
    assert allied_created == 2
    assert enemy_created == 4

@pytest.mark.asyncio
async def test_level_7_non_zergling_ratio():
    training_conf = {
        "unit_type": "ROACH",
        "enemy_type": "MARINE",
        "targets": 2
    }
    bot = Level7Bot(conn=AsyncMock(), training_conf=training_conf)
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid.width = 100
    bot.game_info.pathing_grid.__getitem__.return_value = 1
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1
    
    bot.all_units = MagicMock()
    bot.all_units.amount = 0
    bot.all_units.__bool__.return_value = False
    
    bot.units = MagicMock()
    bot.units.return_value.amount = 2
    bot.enemy_units = MagicMock()
    bot.enemy_units.return_value.amount = 2
    
    bot.client.observation.return_value = MagicMock()
    
    await bot.reset_game_state()
    
    calls = bot.client.debug_create_unit.call_args_list
    enemy_created = sum(c[0][0][0][1] for c in calls if c[0][0][0][3] != 1)
    assert enemy_created == 2

@pytest.mark.asyncio
async def test_level_7_cleanup_step():
    training_conf = {"targets": 1}
    bot = Level7Bot(conn=AsyncMock(), training_conf=training_conf)
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid.width = 100
    bot.game_info.pathing_grid.__getitem__.return_value = 1
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1
    
    # Simulate one unit that needs killing
    mock_unit = MagicMock()
    mock_unit.tag = 101
    
    # We need to make all_units dynamic so it changes after cleanup
    all_units_list = [mock_unit]
    
    def all_units_side_effect():
        return all_units_list
    
    # Mocking properties is tricky, let's just make it return empty after the first check
    bot.all_units = MagicMock()
    # First check: returns True (has units)
    # Second check: returns False (empty)
    bot.all_units.__bool__.side_effect = [True, False]
    bot.all_units.__iter__.return_value = [mock_unit]
    
    bot.units = MagicMock()
    bot.units.return_value.amount = 1
    bot.enemy_units = MagicMock()
    bot.enemy_units.return_value.amount = 1
    
    bot.client.observation.return_value = MagicMock()
    
    await bot.reset_game_state()
    
    # Verify sequence
    method_names = [m[0] for m in bot.client.method_calls]
    assert "debug_kill_unit" in method_names
    assert "step" in method_names
    assert method_names.index("debug_kill_unit") < method_names.index("step")