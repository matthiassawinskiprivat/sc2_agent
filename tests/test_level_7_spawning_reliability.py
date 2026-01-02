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
    bot.client.observation.return_value = MagicMock()
    
    # Use patch or similar to avoid complex prepare_step
    bot._prepare_step = MagicMock()
    
    await bot.reset_game_state()
    
    # Verify debug_create_unit calls
    calls = bot.client.debug_create_unit.call_args_list
    allied_created = sum(c[0][0][0][1] for c in calls if c[0][0][0][3] == 1)
    enemy_created = sum(c[0][0][0][1] for c in calls if c[0][0][0][3] != 1)
            
    assert allied_created == 2
    assert enemy_created == 4

@pytest.mark.asyncio
async def test_level_7_cleanup_step():
    training_conf = {"targets": 1, "enemy_type": "MARINE"} # Expected 1 allied, 1 enemy
    bot = Level7Bot(conn=AsyncMock(), training_conf=training_conf)
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid.width = 100
    bot.game_info.pathing_grid.height = 100
    bot.game_info.pathing_grid.__getitem__.return_value = 1
    bot.game_info.map_size = (100, 100)
    bot.player_id = 1
    
    mock_unit = MagicMock()
    mock_unit.tag = 101
    
    bot.all_units = MagicMock()
    # Initial vision step, then cleanup loop (1st true, 2nd false)
    bot.all_units.__bool__.side_effect = [False, True, False, False, False, False]
    bot.all_units.__iter__.return_value = [mock_unit]
    
    bot.units = MagicMock()
    bot.units.return_value.amount = 1
    bot.enemy_units = MagicMock()
    bot.enemy_units.return_value.amount = 1
    
    bot.client.observation.return_value = MagicMock()
    bot._prepare_step = MagicMock()
    
    await bot.reset_game_state()
    
    method_names = [m[0] for m in bot.client.method_calls]
    assert "debug_show_map" in method_names
    assert "debug_kill_unit" in method_names
    assert "step" in method_names
    
    # After vision guarantee step (index 1), cleanup loop starts
    assert method_names.index("debug_kill_unit") > 1
