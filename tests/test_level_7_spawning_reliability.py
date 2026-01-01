import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, call
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_level_7_zergling_ratio():
    # Setup: 2 Roaches vs 4 Zerglings (flavor-matched doubling)
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
    
    # Mock units for cleanup
    bot.all_units = []
    
    await bot.reset_game_state()
    
    # Verify debug_create_unit calls
    calls = bot.client.debug_create_unit.call_args_list
    
    allied_created = 0
    enemy_created = 0
    
    for c in calls:
        unit_data = c[0][0][0] # [[type, amount, pos, player]]
        amount = unit_data[1]
        player = unit_data[3]
        
        if player == 1:
            allied_created += amount
        else:
            enemy_created += amount
            
    assert allied_created == 2, f"Expected 2 allies, got {allied_created}"
    assert enemy_created == 4, f"Expected 4 enemies (2 per target for Zerglings), got {enemy_created}"

@pytest.mark.asyncio
async def test_level_7_non_zergling_ratio():
    # Setup: 2 Roaches vs 2 Marines (No doubling)
    training_conf = {
        "unit_type": "ROACH",
        "enemy_type": "MARINE",
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
    
    bot.all_units = []
    
    await bot.reset_game_state()
    
    calls = bot.client.debug_create_unit.call_args_list
    enemy_created = 0
    for c in calls:
        unit_data = c[0][0][0]
        amount = unit_data[1]
        player = unit_data[3]
        if player != 1:
            enemy_created += amount
            
    assert enemy_created == 2, f"Expected 2 enemies (1 per target for Marines), got {enemy_created}"

@pytest.mark.asyncio
async def test_level_7_cleanup_step():
    training_conf = {"targets": 1}
    bot = Level7Bot(conn=AsyncMock(), training_conf=training_conf)
    bot.client = AsyncMock()
    bot.game_info = MagicMock()
    bot.game_info.pathing_grid.width = 100
    bot.game_info.pathing_grid.height = 100
    bot.game_info.pathing_grid.__getitem__.return_value = 1
    bot.game_info.map_size = (100, 100)
    
    # Mock some existing units to kill
    u1 = MagicMock()
    u1.tag = 101
    bot.all_units = [u1]
    bot.player_id = 1
    
    await bot.reset_game_state()
    
    # Verify sequence: debug_kill_unit -> client.step() -> debug_create_unit
    kill_idx = -1
    step_idx = -1
    create_idx = -1
    
    for i, method_call in enumerate(bot.client.method_calls):
        name = method_call[0]
        if name == "debug_kill_unit":
            kill_idx = i
        elif name == "step":
            step_idx = i
        elif name == "debug_create_unit":
            if create_idx == -1:
                create_idx = i
                
    assert kill_idx != -1, "debug_kill_unit not called"
    assert step_idx != -1, "client.step() not called after cleanup"
    assert create_idx != -1, "debug_create_unit not called"
    assert kill_idx < step_idx < create_idx, "Cleanup step must occur between kill and create"
