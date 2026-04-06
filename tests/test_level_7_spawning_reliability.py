import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, call
from src.level_7_bot import Level7Bot
from sc2.data import Alliance
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
    bot.all_units.__bool__.return_value = False  # For cleanup loop
    bot.all_units.__iter__.return_value = []  # post-cleanup must be empty for MapCleanupFailedError check
    
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

class _UnitsUntilKilled:
    """Simulates one unit until debug_kill_unit runs, then empty (matches real SC2 cleanup)."""

    def __init__(self, mock_unit):
        self._mock_unit = mock_unit
        self._alive = True

    def __iter__(self):
        if self._alive:
            return iter([self._mock_unit])
        return iter([])

    def __bool__(self):
        return self._alive

    def clear(self):
        self._alive = False


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
    mock_unit.alliance = Alliance.Self
    mock_unit.type_id = UnitTypeId.MARINE

    cleanup_units = _UnitsUntilKilled(mock_unit)
    bot.all_units = cleanup_units

    real_kill = bot.client.debug_kill_unit

    async def kill_then_clear(*args, **kwargs):
        await real_kill(*args, **kwargs)
        cleanup_units.clear()

    bot.client.debug_kill_unit = kill_then_clear
    
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
